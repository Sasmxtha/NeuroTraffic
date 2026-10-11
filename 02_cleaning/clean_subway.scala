// NeuroTraffic — Scala Spark MTA Subway Ridership Cleaning
// Run: spark-shell -i /project/02_cleaning/clean_subway.scala

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, LongType}

val HDFS = "hdfs://namenode:9000/data/neurotraffic"

println("=" * 60)
println("NeuroTraffic — Subway Ridership Cleaning (Scala)")
println("=" * 60)

var df = spark.read
  .option("header", "true").option("inferSchema", "true")
  .csv(s"$HDFS/raw/subway")

val rawCount = df.count()
println(s"  Raw rows: $rawCount")

// Standardize columns
df = df.select(
  col("transit_timestamp").cast("timestamp").as("transit_hour"),
  col("station_complex_id").cast("string"),
  trim(col("station_complex")).as("station_name"),
  col("borough"),
  col("ridership").cast(LongType),
  col("transfers").cast(LongType),
  col("latitude").cast(DoubleType),
  col("longitude").cast(DoubleType),
  col("payment_method"),
  col("fare_class_category")
)

// Drop nulls, negatives, filter dates
df = df
  .na.drop(Seq("transit_hour", "ridership"))
  .filter(col("ridership") >= 0)
  .filter(col("transit_hour").between("2022-01-01", "2025-01-01"))

// Aggregate by station + hour (raw data has rows per payment method)
var agg = df.groupBy("transit_hour", "station_complex_id", "station_name",
    "borough", "latitude", "longitude")
  .agg(
    sum("ridership").as("total_ridership"),
    sum("transfers").as("total_transfers")
  )

// Remove outliers (> 5 std devs from station mean)
val stats = agg.groupBy("station_complex_id").agg(
  mean("total_ridership").as("mean_r"),
  stddev("total_ridership").as("std_r")
)

agg = agg.join(stats, Seq("station_complex_id"), "left")
  .filter(col("total_ridership") <= col("mean_r") + lit(5) * col("std_r"))
  .drop("mean_r", "std_r")
  .withColumn("year",  year(col("transit_hour")))
  .withColumn("month", month(col("transit_hour")))

val cleanCount = agg.count()
println(s"  Raw rows (multi-row per station-hour): $rawCount")
println(s"  After aggregation + outlier removal:   $cleanCount")

agg.repartition(col("year"), col("month"))
  .write.partitionBy("year", "month")
  .mode("overwrite")
  .parquet(s"$HDFS/cleaned/subway")

println("  [OK] Subway cleaning complete!")
agg.printSchema()
agg.show(5, truncate = false)

System.exit(0)
