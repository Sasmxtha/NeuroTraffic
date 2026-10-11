// tune_contamination.scala
// NeuroTraffic — Anomaly Detection Threshold Tuning (Scala)
// Replaces tune_contamination.py — uses z-score + KMeans instead of sklearn Isolation Forest.
//
// Tests multiple anomaly thresholds against combined_v3 to find the optimal
// z-score cutoff for disruption detection. Evaluates cross-modal signature
// strength (subway DOWN + taxi UP + bike UP).
//
// Run: spark-shell --driver-memory 4g -i /project/06_ml/tune_contamination.scala

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.clustering.KMeans
import org.apache.hadoop.fs.{FileSystem, Path}

val HDFS_COMBINED = "hdfs://namenode:9000/data/neurotraffic/combined_v3"
val HDFS_GRAPH    = "hdfs://namenode:9000/data/neurotraffic/graph_features"
val HDFS_MODELS   = "hdfs://namenode:9000/data/neurotraffic/models"

val IF_FEATURE_COLS = Array(
  "subway_ridership", "taxi_pickups", "bike_starts",
  "temperature_c", "is_rush_hour", "day_of_week",
  "is_weekend", "page_rank"
)

// Z-score thresholds to test (analogous to contamination values)
val THRESHOLDS = Array(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)

println("=" * 60)
println("  NeuroTraffic — Anomaly Threshold Tuning (Scala)")
println("=" * 60)

// Load data
println(s"\nLoading combined table from: ${HDFS_COMBINED}")
var df = spark.read.parquet(HDFS_COMBINED)

// Join graph features
try {
  val graph = spark.read.parquet(HDFS_GRAPH)
  val graphCols = graph.select(
    col("vertex_id").alias("gf_zone_id"),
    col("page_rank")
  )
  df = df.join(graphCols, df("zone_id") === graphCols("gf_zone_id"), "left")
    .drop("gf_zone_id")
    .na.fill(Map("page_rank" -> 0.0))
} catch {
  case _: Exception =>
    df = df.withColumn("page_rank", lit(0.0))
    println("  Graph features not available, using page_rank=0.0")
}

val total = df.count()
println(s"  Total rows: ${total}")

// Compute z-score baselines
println("\nComputing z-score baselines...")
val baselines = df.groupBy("zone_id", "hour_of_day", "is_weekend").agg(
  mean("subway_ridership").alias("mean_subway"),
  stddev("subway_ridership").alias("std_subway"),
  mean("taxi_pickups").alias("mean_pickups"),
  stddev("taxi_pickups").alias("std_pickups"),
  mean("bike_starts").alias("mean_bikes"),
  stddev("bike_starts").alias("std_bikes")
)

var scored = df.join(baselines, Seq("zone_id", "hour_of_day", "is_weekend"), "left")
scored = scored
  .withColumn("z_subway",
    when(col("std_subway") > 0,
      (col("subway_ridership") - col("mean_subway")) / col("std_subway"))
    .otherwise(0.0))
  .withColumn("z_pickups",
    when(col("std_pickups") > 0,
      (col("taxi_pickups") - col("mean_pickups")) / col("std_pickups"))
    .otherwise(0.0))
  .withColumn("z_bikes",
    when(col("std_bikes") > 0,
      (col("bike_starts") - col("mean_bikes")) / col("std_bikes"))
    .otherwise(0.0))

scored.cache()

// Test each threshold
println("\n" + "=" * 75)
println("THRESHOLD COMPARISON")
println("=" * 75)
println(f"${"Threshold"}%10s | ${"Anomalies"}%9s | ${"%"}%6s | " +
        f"${"Subway"}%7s | ${"Taxi"}%6s | ${"Bike"}%6s | ${"SigScore"}%8s")
println("-" * 75)

case class ThresholdResult(
  threshold: Double, nAnomalies: Long, pct: Double,
  subwayRatio: Double, taxiRatio: Double, bikeRatio: Double,
  sigScore: Double
)

val results = THRESHOLDS.map { thresh =>
  // Anomaly = subway significantly DOWN + (taxi UP or bike UP)
  val anomalyDf = scored.filter(
    (col("z_subway") < -thresh) &&
    ((col("z_pickups") > thresh) || (col("z_bikes") > thresh))
  )
  val normalDf = scored.filter(
    !((col("z_subway") < -thresh) &&
      ((col("z_pickups") > thresh) || (col("z_bikes") > thresh)))
  )

  val nAnomalies = anomalyDf.count()
  val pct = nAnomalies * 100.0 / total

  // Cross-modal signature ratios
  val aMeanSubway = anomalyDf.agg(mean("subway_ridership")).head().getAs[Double](0)
  val nMeanSubway = normalDf.agg(mean("subway_ridership")).head().getAs[Double](0)
  val aMeanTaxi = anomalyDf.agg(mean("taxi_pickups")).head().getAs[Double](0)
  val nMeanTaxi = normalDf.agg(mean("taxi_pickups")).head().getAs[Double](0)
  val aMeanBike = anomalyDf.agg(mean("bike_starts")).head().getAs[Double](0)
  val nMeanBike = normalDf.agg(mean("bike_starts")).head().getAs[Double](0)

  val subwayRatio = if (nMeanSubway > 0) aMeanSubway / nMeanSubway else 0.0
  val taxiRatio = if (nMeanTaxi > 0) aMeanTaxi / nMeanTaxi else 0.0
  val bikeRatio = if (nMeanBike > 0) aMeanBike / nMeanBike else 0.0

  // Signature score: reward low subway + high taxi + high bike
  val sigScore = math.max(0, 1.0 - subwayRatio) +
                 math.max(0, taxiRatio - 1.0) +
                 math.max(0, bikeRatio - 1.0)

  println(f"${thresh}%10.1f | ${nAnomalies}%9d | ${pct}%5.2f%% | " +
          f"${subwayRatio}%7.3f | ${taxiRatio}%6.3f | ${bikeRatio}%6.3f | " +
          f"${sigScore}%8.4f")

  ThresholdResult(thresh, nAnomalies, pct, subwayRatio, taxiRatio, bikeRatio, sigScore)
}

val best = results.maxBy(_.sigScore)
println(s"\nBEST: threshold = ${best.threshold} (sig_score = ${best.sigScore})")

// Save the best threshold configuration
println(s"\nSaving best threshold configuration...")
val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

val configJson = s"""{
  "method": "z-score",
  "best_threshold": ${best.threshold},
  "sig_score": ${best.sigScore},
  "anomaly_count": ${best.nAnomalies},
  "anomaly_pct": ${best.pct},
  "subway_ratio": ${best.subwayRatio},
  "taxi_ratio": ${best.taxiRatio},
  "bike_ratio": ${best.bikeRatio},
  "feature_cols": [${IF_FEATURE_COLS.map(c => s""""$c"""").mkString(", ")}],
  "note": "Uses z-score anomaly detection (Spark MLlib) instead of sklearn Isolation Forest"
}"""

val configPath = new Path(s"${HDFS_MODELS}/anomaly_config.json")
fs.mkdirs(configPath.getParent)
val out = fs.create(configPath, true)
out.writeBytes(configJson)
out.close()
println(s"  Config saved to ${configPath}")

// Also run KMeans to find anomalous clusters (complementary method)
println("\nRunning KMeans (k=10) for cluster-based anomaly detection...")
val kFeatures = Array("subway_ridership", "taxi_pickups", "bike_starts", "temperature_c")
  .filter(scored.columns.contains)

val kAssembler = new VectorAssembler()
  .setInputCols(kFeatures)
  .setOutputCol("km_features")
  .setHandleInvalid("skip")

val kScaler = new StandardScaler()
  .setInputCol("km_features")
  .setOutputCol("km_scaled")
  .setWithStd(true)
  .setWithMean(true)

val km = new KMeans()
  .setK(10)
  .setSeed(42)
  .setFeaturesCol("km_scaled")
  .setPredictionCol("cluster")

val kmPipeline = new org.apache.spark.ml.Pipeline()
  .setStages(Array(kAssembler, kScaler, km))

val kmModel = kmPipeline.fit(scored)
val clustered = kmModel.transform(scored)

// Find small clusters (anomalous patterns)
val clusterCounts = clustered.groupBy("cluster").count().orderBy("count")
clusterCounts.show(10)

val smallThreshold = total / 20  // clusters < 5% are anomalous
val smallClusters = clusterCounts.filter(col("count") < smallThreshold)
  .select("cluster").collect().map(_.getInt(0)).toSet
println(s"  Small clusters (< ${smallThreshold} members): ${smallClusters.mkString(", ")}")

scored.unpersist()

println("\n" + "=" * 60)
println("  TUNING COMPLETE")
println(s"  Best z-score threshold: ${best.threshold}")
println(s"  Dashboard thresholds: Score >= 0.80 = YELLOW, Score >= 0.90 = RED")
println("=" * 60)

System.exit(0)
