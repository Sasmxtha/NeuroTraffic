// NeuroTraffic — Scala Spark Taxi Data Cleaning (Yellow + Green + HVFHV)
// Run: spark-shell -i /project/02_cleaning/clean_taxi.scala

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, DoubleType}

val HDFS = "hdfs://namenode:9000/data/neurotraffic"

println("=" * 60)
println("NeuroTraffic — Taxi Cleaning (Scala)")
println("=" * 60)

val hdfsFs = org.apache.hadoop.fs.FileSystem.get(
  new java.net.URI(HDFS), spark.sparkContext.hadoopConfiguration)

// Common cleaning filters applied to all taxi types
def applyFilters(df: DataFrame, checkPassengers: Boolean = true): DataFrame = {
  var result = df
    .na.drop(Seq("pickup_datetime", "dropoff_datetime", "pu_location_id", "do_location_id"))

  if (checkPassengers) {
    result = result.filter(col("passenger_count") > 0 || col("passenger_count").isNull)
  }

  result
    .filter(col("trip_distance") > 0)
    .filter(col("fare_amount") > 0)
    .filter(col("pu_location_id").between(1, 263) && col("do_location_id").between(1, 263))
    .withColumn("trip_duration_sec",
      col("dropoff_datetime").cast("long") - col("pickup_datetime").cast("long"))
    .filter(col("trip_duration_sec").between(30, 43200))
    .filter(col("pickup_datetime").between("2022-01-01", "2025-01-01"))
    .withColumn("pickup_hour", date_trunc("hour", col("pickup_datetime")))
    .withColumn("year",  year(col("pickup_datetime")))
    .withColumn("month", month(col("pickup_datetime")))
}

def readFilesIndividually(dir: String, taxiType: String,
    pickupCol: String, dropoffCol: String): DataFrame = {
  val files = hdfsFs.listStatus(new org.apache.hadoop.fs.Path(dir))
    .map(_.getPath.toString).filter(_.endsWith(".parquet")).sorted
  println(s"  Found ${files.length} parquet files")
  val dfs = files.flatMap { f =>
    try {
      Some(spark.read.parquet(f).select(
        lit(taxiType).as("taxi_type"),
        col(pickupCol).cast("timestamp").as("pickup_datetime"),
        col(dropoffCol).cast("timestamp").as("dropoff_datetime"),
        col("PULocationID").cast(IntegerType).as("pu_location_id"),
        col("DOLocationID").cast(IntegerType).as("do_location_id"),
        col("passenger_count").cast(IntegerType),
        col("trip_distance").cast(DoubleType),
        col("fare_amount").cast(DoubleType),
        col("total_amount").cast(DoubleType),
        col("payment_type").cast(IntegerType)
      ))
    } catch { case e: Exception => println(s"    [SKIP] ${f.split("/").last}: $e"); None }
  }
  dfs.reduce(_ unionByName _)
}

// --- Yellow Taxi ---
println("\n--- Yellow Taxi ---")
val yellowRaw = readFilesIndividually(
  s"$HDFS/raw/taxi/yellow", "yellow", "tpep_pickup_datetime", "tpep_dropoff_datetime")
val yellowRawCount = yellowRaw.count()
val yellowClean = applyFilters(yellowRaw)
val yellowCleanCount = yellowClean.count()
println(s"  Raw: $yellowRawCount -> Cleaned: $yellowCleanCount")

yellowClean.write.mode("overwrite").parquet(s"$HDFS/cleaned/taxi/yellow")
println("  [OK] Yellow written")

// --- Green Taxi ---
println("\n--- Green Taxi ---")
val greenRaw = readFilesIndividually(
  s"$HDFS/raw/taxi/green", "green", "lpep_pickup_datetime", "lpep_dropoff_datetime")
val greenRawCount = greenRaw.count()
val greenClean = applyFilters(greenRaw)
val greenCleanCount = greenClean.count()
println(s"  Raw: $greenRawCount -> Cleaned: $greenCleanCount")

greenClean.write.mode("overwrite").parquet(s"$HDFS/cleaned/taxi/green")
println("  [OK] Green written")

// --- HVFHV (Uber/Lyft) — batch-read to avoid OOM on 684M rows ---
println("\n--- HVFHV (Uber/Lyft) ---")
val hvfhvStagingPath = s"$HDFS/staging/hvfhv_scala"

// List individual parquet files in HDFS
val hvfhvFs = org.apache.hadoop.fs.FileSystem.get(
  new java.net.URI(s"$HDFS/raw/taxi/hvfhv"), spark.sparkContext.hadoopConfiguration)
val hvfhvFiles = hvfhvFs.listStatus(new org.apache.hadoop.fs.Path(s"$HDFS/raw/taxi/hvfhv"))
  .map(_.getPath.toString).filter(_.endsWith(".parquet")).sorted
println(s"  Found ${hvfhvFiles.length} HVFHV parquet files")

// Delete old staging if exists
val stageFs = org.apache.hadoop.fs.FileSystem.get(
  new java.net.URI(hvfhvStagingPath), spark.sparkContext.hadoopConfiguration)
val stagePath = new org.apache.hadoop.fs.Path(hvfhvStagingPath)
if (stageFs.exists(stagePath)) stageFs.delete(stagePath, true)

// Process in batches of 6 files
val BATCH = 6
for (i <- 0 until hvfhvFiles.length by BATCH) {
  val batch = hvfhvFiles.slice(i, i + BATCH)
  val batchDfs = batch.flatMap { f =>
    try {
      Some(spark.read.parquet(f).select(
        lit("hvfhv").as("taxi_type"),
        col("pickup_datetime").cast("timestamp").as("pickup_datetime"),
        col("dropoff_datetime").cast("timestamp").as("dropoff_datetime"),
        col("PULocationID").cast(IntegerType).as("pu_location_id"),
        col("DOLocationID").cast(IntegerType).as("do_location_id"),
        lit(null).cast(IntegerType).as("passenger_count"),
        col("trip_miles").cast(DoubleType).as("trip_distance"),
        col("base_passenger_fare").cast(DoubleType).as("fare_amount"),
        col("driver_pay").cast(DoubleType).as("total_amount"),
        lit(null).cast(IntegerType).as("payment_type")
      ))
    } catch { case e: Exception => println(s"    [SKIP] ${f.split("/").last}: $e"); None }
  }
  if (batchDfs.nonEmpty) {
    batchDfs.reduce(_ unionByName _).write.mode("append").parquet(hvfhvStagingPath)
    println(s"    Batch ${i / BATCH + 1} written (files ${i + 1}-${Math.min(i + BATCH, hvfhvFiles.length)})")
  }
}

val hvfhvRaw = spark.read.parquet(hvfhvStagingPath)
val hvfhvRawCount = hvfhvRaw.count()
val hvfhvClean = applyFilters(hvfhvRaw, checkPassengers = false)
val hvfhvCleanCount = hvfhvClean.count()
println(s"  Raw: $hvfhvRawCount -> Cleaned: $hvfhvCleanCount")

hvfhvClean.write.mode("overwrite").parquet(s"$HDFS/cleaned/taxi/hvfhv")
println("  [OK] HVFHV written")

// Summary
println("\n" + "=" * 60)
println("TAXI CLEANING SUMMARY")
println("=" * 60)
val totalRaw = yellowRawCount + greenRawCount + hvfhvRawCount
val totalClean = yellowCleanCount + greenCleanCount + hvfhvCleanCount
println(s"  Yellow:  $yellowRawCount -> $yellowCleanCount")
println(s"  Green:   $greenRawCount -> $greenCleanCount")
println(s"  HVFHV:   $hvfhvRawCount -> $hvfhvCleanCount")
println(s"  TOTAL:   $totalRaw -> $totalClean (removed ${totalRaw - totalClean})")

System.exit(0)
