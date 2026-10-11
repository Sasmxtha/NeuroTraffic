// Pre-filter the training set in-cluster so the host trainer only has to pull
// a small file over WebHDFS. Writes one small Parquet with all FEATURE_COLS +
// is_disruption, undersampling negatives so the file is a few hundred thousand
// rows instead of 6.9 million.
//
// Run: docker exec spark-master /spark/bin/spark-shell --master "local[2]" \
//        --driver-memory 3g -i /project/06_ml/prepare_ml_training.scala

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

val HDFS = "hdfs://namenode:9000"
val d = spark.read.parquet(s"$HDFS/data/neurotraffic/disruptions")
val f = spark.read.parquet(s"$HDFS/data/graph/features/features.parquet")
  .filter(col("node_type") === "taxi_zone")
  .select(col("vertex_id").cast(IntegerType).as("zone_id"),
          col("page_rank"), col("community_id"), col("in_degree"), col("out_degree"))

val sub = when(col("subway_ridership") > 0, col("subway_ridership")).otherwise(lit(1))
val taxi = when(col("taxi_pickups") > 0, col("taxi_pickups")).otherwise(lit(1))

val base = d.join(f, Seq("zone_id"), "left")
  .withColumn("bike_ends", col("bike_starts"))
  .na.fill(Map("page_rank" -> 0.0, "community_id" -> 0L, "in_degree" -> 0, "out_degree" -> 0))
  .withColumn("taxi_to_subway_ratio", col("taxi_pickups").cast(DoubleType) / sub)
  .withColumn("bike_to_subway_ratio", col("bike_starts").cast(DoubleType) / sub)
  .withColumn("total_surface_transport", col("taxi_pickups") + col("bike_starts"))
  .withColumn("complaint_density", col("complaint_count").cast(DoubleType) / taxi)
  .filter((col("taxi_pickups") > 0) || (col("subway_ridership") > 0))

val pos = base.filter(col("is_disruption") === 1)
val neg = base.filter(col("is_disruption") === 0).sample(0.05, 42L)

val training = pos.union(neg).select(
  "taxi_pickups", "taxi_dropoffs", "subway_ridership", "bike_starts", "bike_ends",
  "taxi_to_subway_ratio", "bike_to_subway_ratio", "total_surface_transport", "complaint_density",
  "temperature_c", "precipitation_mm", "is_rain", "is_snow",
  "hour_of_day", "is_rush_hour", "is_weekend", "day_of_week",
  "complaint_count", "event_count",
  "page_rank", "community_id", "in_degree", "out_degree",
  "is_disruption")

println(s"  pos=${pos.count}  neg_sampled=${neg.count}  total=${training.count}")

training.coalesce(1).write.mode("overwrite")
  .parquet(s"$HDFS/data/neurotraffic/ml/training_sample.parquet")

val fs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(HDFS), spark.sparkContext.hadoopConfiguration)
val dir = new org.apache.hadoop.fs.Path(s"$HDFS/data/neurotraffic/ml/training_sample.parquet")
val part = fs.listStatus(dir).filter(_.getPath.getName.startsWith("part-")).head
val dest = new org.apache.hadoop.fs.Path(s"$HDFS/data/neurotraffic/ml/training.parquet")
try { fs.delete(dest, false) } catch { case _: Exception => }
fs.rename(part.getPath, dest)
fs.delete(dir, true)
println(s"  wrote /data/neurotraffic/ml/training.parquet")
System.exit(0)
