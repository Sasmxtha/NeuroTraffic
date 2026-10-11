// NeuroTraffic — Historical zone × hour × day-type profile for the live fetcher
// Averages combined_v3 per (zone_id, hour_of_day, is_weekend) so live estimates
// for sources without a real-time feed (taxi, subway ridership) match the scale
// of the streaming pipeline's baselines.
//
// Run: docker exec spark-master /spark/bin/spark-shell -i /project/07_streaming/build_live_profiles.scala
// Out: /data/neurotraffic/lookups/zone_hour_profile.csv (~12.6K rows)

import org.apache.spark.sql.functions._

val HDFS = "hdfs://namenode:9000"
val OUT = "/data/neurotraffic/lookups/zone_hour_profile.csv"
val STAGING = s"$HDFS/data/neurotraffic/lookups/_staging_profile"

val combined = spark.read.parquet(s"$HDFS/data/neurotraffic/combined_v3")

val metrics = Seq("taxi_pickups", "taxi_dropoffs", "subway_ridership",
  "bike_starts", "bike_ends", "complaint_count", "event_count")
  .filter(combined.columns.contains)

val profile = combined
  .withColumn("hour_of_day", hour(col("hourly_timestamp")))
  .withColumn("is_weekend", when(dayofweek(col("hourly_timestamp")).isin(1, 7), 1).otherwise(0))
  .groupBy("zone_id", "hour_of_day", "is_weekend")
  .agg(count(lit(1)).as("n_hours"), metrics.map(m => round(avg(col(m)), 3).as(m)): _*)
  .orderBy("zone_id", "is_weekend", "hour_of_day")

profile.coalesce(1).write.mode("overwrite").option("header", "true").csv(STAGING)

val fs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(HDFS), spark.sparkContext.hadoopConfiguration)
val part = fs.listStatus(new org.apache.hadoop.fs.Path(STAGING)).filter(_.getPath.getName.startsWith("part-")).head
fs.delete(new org.apache.hadoop.fs.Path(OUT), false)
fs.rename(part.getPath, new org.apache.hadoop.fs.Path(OUT))
fs.delete(new org.apache.hadoop.fs.Path(STAGING), true)

println(s"[OK] ${profile.count()} profile rows (metrics: ${metrics.mkString(", ")}) -> $OUT")
profile.describe(metrics: _*).show()
System.exit(0)
