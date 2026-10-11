// NeuroTraffic — Scala Spark 311 Complaints Cleaning
// Run: spark-shell -i /project/02_cleaning/clean_311.scala

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

val HDFS = "hdfs://namenode:9000/data/neurotraffic"

println("=" * 60)
println("NeuroTraffic — 311 Complaints Cleaning (Scala)")
println("=" * 60)

val transportTypes = Seq(
  "Street Condition", "Traffic Signal Condition", "Street Light Condition",
  "Blocked Driveway", "Illegal Parking", "Noise - Street/Sidewalk",
  "Water Main Break", "Sewer", "Bus Stop Shelter Complaint",
  "Traffic", "Taxi Complaint", "Bike/Roller/Skate Condition"
)

var df = spark.read
  .option("header", "true").option("inferSchema", "true")
  .csv(s"$HDFS/raw/311")

val rawCount = df.count()
println(s"  Raw rows: $rawCount")

// Standardize columns
df = df.select(
  col("created_date").cast("timestamp").as("complaint_time"),
  col("closed_date").cast("timestamp").as("closed_time"),
  trim(col("complaint_type")).as("complaint_type"),
  trim(col("descriptor")).as("descriptor"),
  col("incident_zip").as("zip_code"),
  col("borough"),
  col("latitude").cast(DoubleType),
  col("longitude").cast(DoubleType),
  col("status")
)

// Filter to transport-relevant types, valid dates, valid coords
df = df
  .filter(col("complaint_type").isin(transportTypes: _*))
  .na.drop(Seq("complaint_time"))
  .filter(col("complaint_time").between("2022-01-01", "2025-01-01"))
  .na.drop(Seq("latitude", "longitude"))
  .filter(col("latitude").between(40.4774, 40.9176) && col("longitude").between(-74.2591, -73.7004))

// Severity score
df = df.withColumn("severity_score",
  when(col("complaint_type") === "Water Main Break", 5)
  .when(col("complaint_type") === "Traffic Signal Condition", 4)
  .when(col("complaint_type") === "Street Condition", 3)
  .when(col("complaint_type") === "Sewer", 3)
  .when(col("complaint_type") === "Blocked Driveway", 2)
  .when(col("complaint_type") === "Street Light Condition", 2)
  .when(col("complaint_type") === "Traffic", 2)
  .otherwise(1)
)

// Resolution time + derived columns
df = df
  .withColumn("resolution_hours",
    (col("closed_time").cast("long") - col("complaint_time").cast("long")) / 3600.0)
  .withColumn("complaint_hour", date_trunc("hour", col("complaint_time")))
  .withColumn("year",  year(col("complaint_time")))
  .withColumn("month", month(col("complaint_time")))

val cleanCount = df.count()
println(s"  Cleaned rows: $cleanCount")
println(s"  Removed: ${rawCount - cleanCount}")

df.groupBy("complaint_type").count().orderBy(desc("count")).show(15, truncate = false)

df.repartition(col("year"), col("month"))
  .write.partitionBy("year", "month")
  .mode("overwrite")
  .parquet(s"$HDFS/cleaned/311")

println("  [OK] 311 cleaning complete!")

System.exit(0)
