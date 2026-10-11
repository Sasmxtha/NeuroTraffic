// NeuroTraffic — Scala Spark Citi Bike Data Cleaning
// Run: spark-shell -i /project/02_cleaning/clean_bike.scala

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

val HDFS = "hdfs://namenode:9000/data/neurotraffic"

println("=" * 60)
println("NeuroTraffic — Citi Bike Cleaning (Scala)")
println("=" * 60)

var df = spark.read
  .option("header", "true").option("inferSchema", "true")
  .csv(s"$HDFS/raw/citibike")

val rawCount = df.count()
println(s"  Raw rows: $rawCount")

// Standardize columns
df = df.select(
  col("ride_id"),
  col("rideable_type"),
  col("started_at").cast("timestamp").as("start_time"),
  col("ended_at").cast("timestamp").as("end_time"),
  trim(initcap(col("start_station_name"))).as("start_station_name"),
  col("start_station_id").cast("string"),
  trim(initcap(col("end_station_name"))).as("end_station_name"),
  col("end_station_id").cast("string"),
  col("start_lat").cast(DoubleType),
  col("start_lng").cast(DoubleType),
  col("end_lat").cast(DoubleType),
  col("end_lng").cast(DoubleType),
  col("member_casual")
)

// Dedup
val beforeDedup = df.count()
df = df.dropDuplicates("ride_id")
println(s"  Duplicates removed: ${beforeDedup - df.count()}")

// Filter: coords, duration, date range
df = df
  .na.drop(Seq("start_lat", "start_lng", "end_lat", "end_lng"))
  .filter(
    col("start_lat").between(40.4774, 40.9176) && col("start_lng").between(-74.2591, -73.7004) &&
    col("end_lat").between(40.4774, 40.9176)   && col("end_lng").between(-74.2591, -73.7004)
  )
  .withColumn("trip_duration_sec",
    col("end_time").cast("long") - col("start_time").cast("long"))
  .filter(col("trip_duration_sec").between(60, 86400))
  // Remove dock errors (same station, < 2 min)
  .filter(!(col("start_station_id") === col("end_station_id") && col("trip_duration_sec") < 120))
  .filter(col("start_time").between("2022-01-01", "2025-01-01"))
  .withColumn("start_hour", date_trunc("hour", col("start_time")))
  .withColumn("year",  year(col("start_time")))
  .withColumn("month", month(col("start_time")))

val cleanCount = df.count()
println(s"  Cleaned rows: $cleanCount")
println(s"  Removed: ${rawCount - cleanCount}")

df.repartition(col("year"), col("month"))
  .write.partitionBy("year", "month")
  .mode("overwrite")
  .parquet(s"$HDFS/cleaned/citibike")

println("  [OK] Citi Bike cleaning complete!")

System.exit(0)
