import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

val HDFS = "hdfs://namenode:9000"
spark.conf.set("spark.sql.parquet.enableVectorizedReader", "false")
spark.conf.set("spark.sql.caseSensitive", "false")

// Read each parquet file individually and union — avoids the bigint/int VendorID
// schema conflict that `mergeSchema` fails on.
def readTaxi(subdir: String, pickCol: String, dropCol: String) = {
  val fs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(HDFS), spark.sparkContext.hadoopConfiguration)
  val files = fs.listStatus(new org.apache.hadoop.fs.Path(s"$HDFS/data/neurotraffic/raw/taxi/$subdir"))
    .filter(_.getPath.getName.endsWith(".parquet")).map(_.getPath.toString).sorted
  println(s"  $subdir: ${files.length} files")
  files.map { f =>
    spark.read.parquet(f).select(
      col("PULocationID").cast(IntegerType).as("src"),
      col("DOLocationID").cast(IntegerType).as("dst"),
      col(pickCol).cast(TimestampType).as("pickup"),
      col(dropCol).cast(TimestampType).as("dropoff")
    )
  }.reduce(_ union _)
}

def readTaxiStub(subdir: String, pickCol: String, dropCol: String) =
  spark.read.parquet(s"$HDFS/data/neurotraffic/raw/taxi/$subdir")
    .select(
      col("PULocationID").cast(IntegerType).as("src"),
      col("DOLocationID").cast(IntegerType).as("dst"),
      col(pickCol).cast(TimestampType).as("pickup"),
      col(dropCol).cast(TimestampType).as("dropoff")
    )

val yellow = readTaxi("yellow", "tpep_pickup_datetime", "tpep_dropoff_datetime")
val green = readTaxi("green", "lpep_pickup_datetime", "lpep_dropoff_datetime")
val trips = yellow.union(green)
  .filter(col("src").isNotNull && col("dst").isNotNull)
  .filter(col("pickup").isNotNull && col("dropoff").isNotNull)
  .filter(col("src").between(1, 263) && col("dst").between(1, 263))
  .filter(col("src") =!= col("dst"))
  .filter(col("pickup") >= to_timestamp(lit("2022-01-01")))
  .filter(col("pickup") <= to_timestamp(lit("2024-12-31")))

val edges = trips
  .withColumn("hour_of_day", hour(col("pickup")))
  .withColumn("day_type", when(dayofweek(col("pickup")).isin(1, 7), "weekend").otherwise("weekday"))
  .withColumn("travel_sec", unix_timestamp(col("dropoff")) - unix_timestamp(col("pickup")))
  .filter(col("travel_sec").between(60, 7200))
  .groupBy("src", "dst", "hour_of_day", "day_type")
  .agg(count(lit(1)).as("trip_count"), avg("travel_sec").as("avg_travel_time"))
  .filter(col("trip_count") >= 5)
  .withColumn("src_node_id", col("src").cast(LongType))
  .withColumn("dst_node_id", col("dst").cast(LongType))
  .withColumn("src_type", lit("taxi"))
  .withColumn("dst_type", lit("taxi"))
  .select("src_node_id", "dst_node_id", "src_type", "dst_type",
          "trip_count", "avg_travel_time", "hour_of_day", "day_type")

edges.write.mode("overwrite").parquet(s"$HDFS/data/graph/combined/trips.parquet")
val n = spark.read.parquet(s"$HDFS/data/graph/combined/trips.parquet").count
println(s"EDGES_WRITTEN=$n")
System.exit(0)
