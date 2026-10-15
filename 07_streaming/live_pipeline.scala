// NeuroTraffic - Live Streaming Pipeline (Spark Structured Streaming)
// ====================================================================
// Reads the live JSON batches that realtime_fetcher.py drops into HDFS,
// scores every zone against its normal level for that hour, and writes:
//   /data/streaming/output/latest.csv        newest score for each zone
//   /data/streaming/output/live_history.csv  one snapshot per minute, last 24 hours only
//
// Run: docker exec -it spark-master /spark/bin/spark-shell --master spark://spark-master:7077 -i /project/07_streaming/live_pipeline.scala

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.streaming.Trigger
import org.apache.hadoop.fs.Path
import java.time.{LocalDateTime, ZoneId}
import java.time.format.DateTimeFormatter

spark.sparkContext.setLogLevel("WARN")
spark.conf.set("spark.sql.shuffle.partitions", "4")   // data is small (263 zones), default 200 is wasteful
val HDFS    = "hdfs://namenode:9000"
val LATEST  = s"$HDFS/data/streaming/output/latest.csv"
val HISTORY = s"$HDFS/data/streaming/output/live_history.csv"
val fs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(HDFS), spark.sparkContext.hadoopConfiguration)

// 1. Normal level per zone x hour x weekday/weekend (built from the 2023 history)
val profile = spark.read.option("header", "true").option("inferSchema", "true")
  .csv(s"$HDFS/data/neurotraffic/lookups/zone_hour_profile.csv")
  .select(col("zone_id"), col("hour_of_day"), col("is_weekend"),
    col("taxi_pickups").as("avg_taxi"), col("subway_ridership").as("avg_subway"), col("bike_starts").as("avg_bike"))
  .cache()

// 2. The fields we use from each live record
val schema = new StructType()
  .add("zone_id", IntegerType).add("timestamp", StringType)
  .add("hour_of_day", IntegerType).add("is_weekend", IntegerType)
  .add("taxi_pickups", DoubleType).add("subway_ridership", DoubleType).add("bike_starts", DoubleType)
  .add("precipitation_mm", DoubleType).add("traffic_speed_mph", DoubleType)
  .add("affected_line", StringType).add("disruption_severity", DoubleType)

// 3. Anomaly score
// z = how far the live value is from normal, in "std-devs" (std-dev taken as 30% of the average)
def z(live: String, avg: String) =
  coalesce(abs(col(live) - col(avg)) / greatest(col(avg) * 0.3, lit(1.0)), lit(0.0))

def score(batch: DataFrame): DataFrame = {
  val withZ = batch.join(profile, Seq("zone_id", "hour_of_day", "is_weekend"), "left")
    .withColumn("z_pickups", z("taxi_pickups", "avg_taxi"))
    .withColumn("z_subway", z("subway_ridership", "avg_subway"))
    .withColumn("z_bike", z("bike_starts", "avg_bike"))
  val maxZ    = greatest(col("z_pickups"), col("z_subway"), col("z_bike"))
  val avgZ    = (col("z_pickups") + col("z_subway") + col("z_bike")) / 3
  val weather = when(col("precipitation_mm") > 5, 0.15).when(col("precipitation_mm") > 0, 0.05).otherwise(0.0)
  // An active MTA alert on a line serving the zone is the strongest signal
  val alert   = when(col("affected_line").isNull || col("affected_line") === "-", 0.0)
    .otherwise(lit(0.45) + coalesce(col("disruption_severity"), lit(0.0)) * 0.45)
  withZ
    .withColumn("anomaly_score", least(lit(1.0), maxZ * 0.04 + avgZ * 0.03 + weather + alert))
    .withColumn("is_disruption", when(col("anomaly_score") >= 0.8, 1).otherwise(0))
    .select(col("zone_id"), col("timestamp"), col("anomaly_score"), col("is_disruption"),
      coalesce(col("affected_line"), lit("-")).as("affected_line"),
      coalesce(col("disruption_severity"), lit(0.0)).as("disruption_severity"),
      col("traffic_speed_mph"), col("z_subway"), col("z_pickups"),
      col("taxi_pickups").cast("int").as("taxi_pickups"),
      col("subway_ridership").cast("int").as("subway_ridership"),
      col("bike_starts").cast("int").as("bike_starts"))
}

// 4. Write a DataFrame as ONE csv file (Spark normally writes a folder of part-files)
def writeCsv(df: DataFrame, dest: String): Unit = {
  val tmp = dest + "_tmp"
  df.coalesce(1).write.mode("overwrite").option("header", "true").csv(tmp)
  val part = fs.globStatus(new Path(tmp + "/part-*"))(0).getPath
  fs.delete(new Path(dest), false)
  fs.rename(part, new Path(dest))
  fs.delete(new Path(tmp), true)
}

// 5. History = old rows from the last 24 hours + this snapshot (anything older is dropped)
val fmt = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss")
def saveHistory(latest: DataFrame): Unit = {
  val cutoff = LocalDateTime.now(ZoneId.of("America/New_York")).minusHours(24).format(fmt)
  val old =
    if (fs.exists(new Path(HISTORY)))
      spark.read.option("header", "true").schema(latest.schema).csv(HISTORY).filter(col("timestamp") >= cutoff)
    else latest.limit(0)
  writeCsv(old.unionByName(latest), HISTORY)
}

// 6. Run for every micro-batch
var lastMinute = ""
def process(batch: DataFrame, id: Long): Unit = {
  if (batch.isEmpty) return
  val newest = batch.agg(max("timestamp")).head.getString(0)
  val latest = score(batch.filter(col("timestamp") === newest).dropDuplicates("zone_id")).cache()
  writeCsv(latest, LATEST)
  val minute = newest.take(16)                    // e.g. 2026-10-15T09:41
  val saved  = minute != lastMinute
  if (saved) { saveHistory(latest); lastMinute = minute }
  val peak = latest.agg(max("anomaly_score")).head.getDouble(0)
  println(f"Batch $id%4d | $newest | ${latest.count()} zones | peak=$peak%.3f | latest.csv" +
    (if (saved) " + history" else ""))
  latest.unpersist()
}

// 7. Start: check HDFS for new files every 15 s; processed input files are deleted
val query = spark.readStream.schema(schema)
  .option("cleanSource", "delete")
  .json(s"$HDFS/data/streaming/input")
  .writeStream
  .option("checkpointLocation", s"$HDFS/data/streaming/checkpoint")
  .trigger(Trigger.ProcessingTime("15 seconds"))
  .foreachBatch(process _)
  .start()

println("Live pipeline running - waiting for data from realtime_fetcher.py (Ctrl+C to stop)")
query.awaitTermination()
