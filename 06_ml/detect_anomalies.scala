// NeuroTraffic — Disruption Detection via Spark MLlib
// Detects anomalous hours per zone using statistical + clustering approach
// Run: spark-shell -i /project/06_ml/detect_anomalies.scala

import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.clustering.KMeans

val HDFS = "hdfs://namenode:9000/data/neurotraffic"

println("=" * 60)
println("  NEUROTRAFFIC DISRUPTION DETECTOR")
println("=" * 60)

val raw = spark.read.parquet(s"$HDFS/combined_v3")
println(s"  Input rows: ${raw.count()}")

val featureCols = Array(
  "taxi_pickups", "taxi_dropoffs", "avg_fare", "avg_trip_distance",
  "subway_ridership", "bike_starts", "bike_ends",
  "complaint_count", "event_count", "precipitation_mm",
  "wind_speed_kmh", "temperature_c"
)

// --- Step 1: Z-score anomaly detection per zone ---
println("\n--- Step 1: Computing z-scores per zone ---")

val statExprs = featureCols.flatMap { c =>
  Seq(avg(c).as(s"${c}_mean"), stddev(c).as(s"${c}_std"))
}
val zoneStats = raw.groupBy("zone_id").agg(statExprs.head, statExprs.tail: _*)

var df = raw.join(zoneStats, Seq("zone_id"))

for (c <- featureCols) {
  df = df.withColumn(s"${c}_z",
    when(col(s"${c}_std") > 0, (col(c) - col(s"${c}_mean")) / col(s"${c}_std")).otherwise(0.0))
}

val statCols = featureCols.flatMap(c => Seq(s"${c}_mean", s"${c}_std"))
df = df.drop(statCols: _*)

val zCols = featureCols.map(_ + "_z")
df = df.withColumn("max_abs_z", greatest(zCols.map(c => abs(col(c))): _*))
df = df.withColumn("avg_abs_z", zCols.map(c => abs(col(c))).reduce(_ + _) / lit(zCols.length))

val zThreshold = 3.0
val zAnomalyCount = df.filter(col("max_abs_z") > zThreshold).count()
println(s"  Z-score anomalies (|z| > $zThreshold): $zAnomalyCount")

// --- Step 2: KMeans clustering for pattern-based anomalies ---
println("\n--- Step 2: KMeans clustering ---")

val assembler = new VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("features_raw")
  .setHandleInvalid("skip")

val assembled = assembler.transform(df)

val scaler = new StandardScaler()
  .setInputCol("features_raw")
  .setOutputCol("features")
  .setWithStd(true)
  .setWithMean(true)

val scalerModel = scaler.fit(assembled)
val scaled = scalerModel.transform(assembled)

val kmeans = new KMeans()
  .setK(10)
  .setSeed(42)
  .setMaxIter(20)
  .setFeaturesCol("features")
  .setPredictionCol("cluster")

val kModel = kmeans.fit(scaled)
val clustered = kModel.transform(scaled)

val clusterSizes = clustered.groupBy("cluster").count().orderBy("count")
clusterSizes.show()

val smallClusterIds = clusterSizes.filter(col("count") < 50000)
  .select("cluster").collect().map(_.getInt(0)).toSeq

println(s"  Small clusters (< 50k members): ${smallClusterIds.mkString(", ")}")

// --- Step 3: Combine signals into disruption score ---
println("\n--- Step 3: Computing disruption scores ---")

val withSmallFlag = clustered
  .withColumn("is_small_cluster",
    when(col("cluster").isin(smallClusterIds: _*), 1.0).otherwise(0.0))

val disruptions = withSmallFlag
  .withColumn("disruption_score",
    col("avg_abs_z") * 0.4 +
    col("max_abs_z") * 0.3 +
    col("is_small_cluster") * 2.0 +
    col("complaint_count") / 100.0 * 0.1 +
    col("event_count") / 10.0 * 0.1 +
    (col("is_rain") + col("is_heavy_rain") * 2 + col("is_snow") * 2 +
     col("is_extreme_cold") + col("is_extreme_heat") + col("is_high_wind")) * 0.1
  )
  .withColumn("is_disruption", when(col("disruption_score") > 3.0, 1).otherwise(0))

val disruptionCount = disruptions.filter(col("is_disruption") === 1).count()
val totalCount = disruptions.count()
val pct = disruptionCount * 100.0 / totalCount
println(f"  Disruptions detected: $disruptionCount / $totalCount ($pct%.2f%%)")

// --- Step 4: Save results ---
println("\n--- Step 4: Saving results ---")

val output = disruptions.select(
  col("zone_id"), col("hourly_timestamp"), col("zone_borough"), col("zone_name"),
  col("day_of_week"), col("hour_of_day"), col("is_rush_hour"), col("is_weekend"),
  col("taxi_pickups"), col("taxi_dropoffs"), col("subway_ridership"),
  col("bike_starts"), col("complaint_count"), col("event_count"),
  col("temperature_c"), col("precipitation_mm"), col("is_rain"), col("is_snow"),
  col("cluster"), col("disruption_score"), col("is_disruption"),
  col("max_abs_z"), col("avg_abs_z")
)

output.write.mode("overwrite").parquet(s"$HDFS/disruptions")
println("  [OK] Disruption results saved to HDFS")

// --- Summary stats ---
println("\n" + "=" * 60)
println("  DISRUPTION SUMMARY")
println("=" * 60)

println("\nDisruptions by borough:")
output.filter(col("is_disruption") === 1)
  .groupBy("zone_borough")
  .agg(count("*").as("disruption_hours"), avg("disruption_score").as("avg_score"))
  .orderBy(desc("disruption_hours"))
  .show()

println("Top 10 most disrupted zones:")
output.filter(col("is_disruption") === 1)
  .groupBy("zone_id", "zone_name")
  .agg(count("*").as("disruption_hours"), avg("disruption_score").as("avg_score"))
  .orderBy(desc("disruption_hours"))
  .limit(10).show(truncate = false)

println("Disruptions by hour of day:")
output.filter(col("is_disruption") === 1)
  .groupBy("hour_of_day")
  .agg(count("*").as("disruption_count"))
  .orderBy("hour_of_day")
  .show(24)

println("Disruptions by day of week:")
output.filter(col("is_disruption") === 1)
  .groupBy("day_of_week")
  .agg(count("*").as("disruption_count"))
  .orderBy("day_of_week")
  .show()

println("\n  DISRUPTION DETECTION COMPLETE")

System.exit(0)
