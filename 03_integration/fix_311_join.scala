// NeuroTraffic — Fix 311 Complaint Borough Case Mismatch
// Run: spark-shell -i /project/03_integration/fix_311_join.scala

import org.apache.spark.sql.functions._

val HDFS = "hdfs://namenode:9000/data/neurotraffic"

println("=" * 60)
println("Fixing 311 complaint join (Scala)")
println("=" * 60)

// Load combined_v2
var combined = spark.read.parquet(s"$HDFS/combined_v2")
println(s"  Rows: ${combined.count()}")

// Drop old (broken) complaint columns
combined = combined.drop("complaint_count", "max_complaint_severity")

// Re-aggregate 311 with fixed borough casing
val complaints = spark.read.parquet(s"$HDFS/cleaned/311")
  .withColumn("borough", initcap(col("borough")))

val complaintsAgg = complaints.groupBy(
    col("borough"),
    col("complaint_hour").as("hourly_timestamp")
  ).agg(
    count("*").as("complaint_count"),
    max("severity_score").as("max_complaint_severity")
  )
println(s"  311 aggregated rows: ${complaintsAgg.count()}")

// Re-join
combined = combined.join(complaintsAgg,
  combined("zone_borough") === complaintsAgg("borough") &&
  combined("hourly_timestamp") === complaintsAgg("hourly_timestamp"), "left")
  .drop(complaintsAgg("borough")).drop(complaintsAgg("hourly_timestamp"))
  .na.fill(Map("complaint_count" -> 0, "max_complaint_severity" -> 0))

// Verify
combined.select(
  sum("complaint_count").as("total_complaints"),
  max("max_complaint_severity").as("max_severity")
).show()

combined.repartition(col("year"), col("month"))
  .write.partitionBy("year", "month")
  .mode("overwrite")
  .parquet(s"$HDFS/combined_v3")

println(s"  [OK] Written to combined_v3 (${combined.count()} rows)")

System.exit(0)
