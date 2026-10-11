// NeuroTraffic — Spark Structured Streaming Pipeline
// ===================================================
// Reads JSON micro-batches from HDFS /data/streaming/input/
// Computes anomaly scores using z-scores against historical baselines
// Writes latest.csv to HDFS /data/streaming/output/ for the dashboard
//
// Run: docker exec spark-master /spark/bin/spark-shell --master "local[2]" \
//        --driver-memory 2g --conf spark.sql.shuffle.partitions=8 \
//        -i /project/07_streaming/streaming_pipeline.scala

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.streaming.Trigger

val HDFS = "hdfs://namenode:9000"
val INPUT_DIR = s"$HDFS/data/streaming/input"
val OUTPUT_CSV = "/data/streaming/output/latest.csv"
val CHECKPOINT = s"$HDFS/data/streaming/checkpoint"
val BASELINES_PATH = s"$HDFS/data/neurotraffic/combined_v3"

println("=" * 60)
println("  NeuroTraffic — Spark Structured Streaming Pipeline")
println("=" * 60)

// ── Step 1: Load historical baselines for z-score computation ──
// Baselines are per zone x hour-of-day x weekday/weekend, so normal rush-hour volume
// is not mistaken for an anomaly.
println("\n[1/3] Loading historical baselines from combined_v3...")

val combined = spark.read.parquet(BASELINES_PATH)
  .withColumn("hour_of_day", hour(col("hourly_timestamp")))
  .withColumn("is_weekend", when(dayofweek(col("hourly_timestamp")).isin(1, 7), 1).otherwise(0))

val baselines = combined.groupBy("zone_id", "hour_of_day", "is_weekend").agg(
  avg("taxi_pickups").as("mean_pickups"),
  stddev("taxi_pickups").as("std_pickups"),
  avg("taxi_dropoffs").as("mean_dropoffs"),
  stddev("taxi_dropoffs").as("std_dropoffs"),
  avg("subway_ridership").as("mean_subway"),
  stddev("subway_ridership").as("std_subway"),
  avg("bike_starts").as("mean_bikes"),
  stddev("bike_starts").as("std_bikes"),
  avg("complaint_count").as("mean_complaints"),
  stddev("complaint_count").as("std_complaints")
).na.fill(1.0)

val baselinesBroadcast = broadcast(baselines)
println(s"  Loaded ${baselines.count()} zone x hour x day-type baselines")

// ── Step 2: Define input schema (matches 07_streaming/realtime_fetcher.py output) ──
println("\n[2/3] Setting up streaming schema...")

val inputSchema = new StructType()
  .add("zone_id", IntegerType)
  .add("timestamp", StringType)
  .add("zone_name", StringType)
  .add("zone_borough", StringType)
  .add("taxi_pickups", IntegerType)
  .add("taxi_dropoffs", IntegerType)
  .add("subway_ridership", IntegerType)
  .add("bike_starts", IntegerType)
  .add("bike_ends", IntegerType)
  .add("complaint_count", IntegerType)
  .add("event_count", IntegerType)
  .add("affected_line", StringType)
  .add("hour_of_day", IntegerType)
  .add("day_of_week", IntegerType)
  .add("is_weekend", IntegerType)
  .add("is_rush_hour", IntegerType)
  .add("temperature_c", DoubleType)
  .add("precipitation_mm", DoubleType)
  .add("humidity_pct", DoubleType)
  .add("wind_speed_kmh", DoubleType)
  .add("visibility_m", DoubleType)
  .add("is_rain", IntegerType)
  .add("is_snow", IntegerType)
  .add("is_heavy_rain", IntegerType)
  .add("is_extreme_cold", IntegerType)
  .add("is_extreme_heat", IntegerType)
  .add("is_low_visibility", IntegerType)
  .add("is_high_wind", IntegerType)
  .add("disruption_severity", DoubleType)
  .add("traffic_speed_mph", DoubleType)

// ── Step 3: Start streaming query ──
println("\n[3/3] Starting streaming query...")
println(s"  Input:      {INPUT_DIR}")
println(s"  Output:     {OUTPUT_CSV}")
println(s"  Checkpoint: {CHECKPOINT}")
println(s"  Trigger:    every 10 seconds")

val hdfsFs = org.apache.hadoop.fs.FileSystem.get(
  new java.net.URI(HDFS), spark.sparkContext.hadoopConfiguration)

def writeSingleCsv(df: DataFrame, destPath: String): Unit = {
  val stagingPath = s"$HDFS/data/streaming/_staging_latest"
  df.coalesce(1).write.mode("overwrite").option("header", "true").csv(stagingPath)
  val dir = new org.apache.hadoop.fs.Path(stagingPath)
  val partFiles = hdfsFs.listStatus(dir).filter(_.getPath.getName.startsWith("part-"))
  if (partFiles.nonEmpty) {
    val dest = new org.apache.hadoop.fs.Path(destPath)
    try { hdfsFs.delete(dest, false) } catch { case _: Exception => }
    hdfsFs.rename(partFiles.head.getPath, dest)
  }
  hdfsFs.delete(dir, true)
}

val rawStream = spark.readStream
  .schema(inputSchema)
  .option("maxFilesPerTrigger", 5)
  .json(INPUT_DIR)

val query = rawStream.writeStream
  .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
    if (!batchDF.isEmpty) {
      val batchCount = batchDF.count()

      // Join with baselines
      val joined = batchDF.join(baselinesBroadcast, Seq("zone_id", "hour_of_day", "is_weekend"), "left")

      // Compute z-scores (capped at 3.0 to handle distribution mismatch)
      val CAP = lit(3.0)
      val withZScores = joined
        .withColumn("z_pickups", least(CAP,
          when(col("std_pickups") > 0,
            abs(col("taxi_pickups") - col("mean_pickups")) / col("std_pickups"))
          .otherwise(0.0)))
        .withColumn("z_subway", least(CAP,
          when(col("std_subway") > 0,
            abs(col("subway_ridership") - col("mean_subway")) / col("std_subway"))
          .otherwise(0.0)))
        // One-sided: live bike counts are a lower bound, and only a surge (riders switching
        // from the subway) indicates disruption.
        .withColumn("z_bikes", least(CAP,
          when(col("std_bikes") > 0,
            greatest(lit(0.0), (col("bike_starts") - col("mean_bikes")) / col("std_bikes")))
          .otherwise(0.0)))
        .withColumn("z_complaints", least(CAP,
          when(col("std_complaints") > 0,
            abs(col("complaint_count") - col("mean_complaints")) / col("std_complaints"))
          .otherwise(0.0)))

      // Composite anomaly score.
      // The main signal is an active unplanned MTA alert on a line serving the zone, weighted by
      // severity (0.4 reduced service / reroute, 0.6 delays, 0.8 suspension): a suspension alone
      // reaches "disrupted" (0.80); a delay needs another live signal such as a bike surge or bad weather.
      val complaintExcess = when(col("std_complaints") > 0,
        greatest(lit(0.0), (col("complaint_count") - col("mean_complaints")) / col("std_complaints")))
        .otherwise(0.0)
      val scored = withZScores
        .withColumn("max_z", greatest(col("z_pickups"), col("z_subway"), col("z_bikes"), col("z_complaints")))
        .withColumn("avg_z", (col("z_pickups") + col("z_subway") + col("z_bikes") + col("z_complaints")) / 4.0)
        .withColumn("weather_penalty",
          col("is_rain") * 0.03 + col("is_heavy_rain") * 0.06 +
          col("is_extreme_cold") * 0.03 + col("is_extreme_heat") * 0.03 +
          col("is_high_wind") * 0.05 + col("is_low_visibility") * 0.03)
        .withColumn("complaint_signal", least(lit(0.2), complaintExcess * 0.07))
        .withColumn("disruption_indicator",
          when(col("affected_line") === "-", lit(0.0))
            .otherwise(lit(0.45) + col("disruption_severity") * 0.45))
        .withColumn("raw_score",
          col("max_z") * 0.04 + col("avg_z") * 0.03 +
          col("weather_penalty") + col("complaint_signal") + col("disruption_indicator"))
        .withColumn("anomaly_score",
          least(lit(1.0), greatest(lit(0.0), col("raw_score"))))
        .withColumn("is_disruption",
          when(col("anomaly_score") >= 0.80, 1).otherwise(0))

      // Keep only the latest timestamp (most recent batch)
      val maxTs = scored.agg(max("timestamp")).head().getString(0)
      val latestOnly = scored.filter(col("timestamp") === maxTs)

      // Select output columns matching dashboard's expected format
      val output = latestOnly.select(
        col("zone_id"),
        col("timestamp"),
        col("anomaly_score"),
        col("is_disruption"),
        col("affected_line"),
        col("disruption_severity"),
        col("traffic_speed_mph"),
        col("z_subway"),
        col("z_pickups"),
        col("taxi_pickups"),
        col("subway_ridership"),
        col("bike_starts")
      )

      // Write latest.csv as single file
      writeSingleCsv(output, OUTPUT_CSV)

      val nDisruptions = latestOnly.filter(col("is_disruption") === 1).count()
      val peakScore = latestOnly.agg(max("anomaly_score")).head().getDouble(0)

      val flag = if (nDisruptions > 0) "[ALERT]" else "[OK]   "
      println(f"  $flag Batch $batchId%04d | $batchCount%4d zones | " +
        f"$nDisruptions%3d disruptions | peak=$peakScore%.3f | latest.csv updated")
    }
  }
  .trigger(Trigger.ProcessingTime("10 seconds"))
  .option("checkpointLocation", CHECKPOINT)
  .start()

println("\n  Streaming pipeline is RUNNING")
println("  Watching for new data in: " + INPUT_DIR)
println("  Output: " + OUTPUT_CSV)
println("  Press Ctrl+C to stop\n")

query.awaitTermination()
