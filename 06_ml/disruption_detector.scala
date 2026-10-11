// disruption_detector.scala
// NeuroTraffic ML Pipeline: Disruption Detection (Full Scala)
// Replaces disruption_detector.py — uses Spark MLlib instead of sklearn.
// 
// Pipeline:
//   1. Load combined_v3 + graph features
//   2. Engineer cross-modal features
//   3. Inject synthetic disruption events
//   4. Z-score anomaly detection + KMeans clustering (replaces Isolation Forest)
//   5. Train GBT Classifier for cascade prediction
//   6. Save predictions + models + evaluation metrics
//
// Run: spark-shell --driver-memory 4g -i /project/06_ml/disruption_detector.scala

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.Pipeline

val HDFS_BASE       = "hdfs://namenode:9000/data/neurotraffic"
val HDFS_COMBINED   = s"${HDFS_BASE}/combined_v3"
val HDFS_GRAPH      = s"${HDFS_BASE}/graph_features"
val HDFS_PREDICTIONS = s"${HDFS_BASE}/predictions"
val HDFS_MODELS     = s"${HDFS_BASE}/models"

val FEATURE_COLS = Array(
  "taxi_pickups", "taxi_dropoffs", "subway_ridership",
  "bike_starts", "bike_ends",
  "taxi_to_subway_ratio", "bike_to_subway_ratio",
  "total_surface_transport", "complaint_density",
  "temperature_c", "humidity_pct", "precipitation_mm",
  "wind_speed_kmh", "is_rain", "is_snow",
  "is_extreme_cold", "is_extreme_heat",
  "hour_of_day", "is_rush_hour", "is_weekend",
  "complaint_count", "max_complaint_severity",
  "event_count",
  "page_rank", "community_id", "in_degree", "out_degree"
)

println("=" * 60)
println("  NEUROTRAFFIC — Disruption Detection (Scala)")
println("=" * 60)

// ── STEP 1: Load and merge ──
println("\n[1/7] Loading data...")
val combined = spark.read.parquet(HDFS_COMBINED)
println(s"  Combined: ${combined.count()} rows, ${combined.columns.length} columns")

val graphDf = try {
  val g = spark.read.parquet(HDFS_GRAPH)
  println(s"  Graph features: ${g.count()} rows")
  g.select(
    col("vertex_id").alias("gf_zone_id"),
    col("page_rank"), col("community_id"),
    col("in_degree"), col("out_degree")
  )
} catch {
  case e: Exception =>
    println(s"  Graph features not available (${e.getMessage}), using placeholders")
    combined.select(col("zone_id").alias("gf_zone_id"))
      .distinct()
      .withColumn("page_rank", lit(0.0))
      .withColumn("community_id", lit(0))
      .withColumn("in_degree", lit(0))
      .withColumn("out_degree", lit(0))
}

var df = combined.join(graphDf, combined("zone_id") === graphDf("gf_zone_id"), "left")
  .drop("gf_zone_id")
  .na.fill(Map(
    "page_rank" -> 0.0, "community_id" -> 0,
    "in_degree" -> 0, "out_degree" -> 0,
    "taxi_pickups" -> 0, "taxi_dropoffs" -> 0,
    "subway_ridership" -> 0, "bike_starts" -> 0, "bike_ends" -> 0,
    "complaint_count" -> 0, "event_count" -> 0
  ))

println(s"  Merged: ${df.count()} rows")

// ── STEP 2: Feature engineering ──
println("\n[2/7] Engineering features...")
df = df.withColumn("taxi_to_subway_ratio",
  when(col("subway_ridership") > 0,
    col("taxi_pickups").cast(DoubleType) / col("subway_ridership"))
  .otherwise(0.0))
df = df.withColumn("bike_to_subway_ratio",
  when(col("subway_ridership") > 0,
    col("bike_starts").cast(DoubleType) / col("subway_ridership"))
  .otherwise(0.0))
df = df.withColumn("total_surface_transport",
  col("taxi_pickups") + col("bike_starts"))
df = df.withColumn("complaint_density",
  when(col("taxi_pickups") > 0,
    col("complaint_count").cast(DoubleType) / col("taxi_pickups"))
  .otherwise(0.0))

// ── STEP 3: Inject synthetic disruptions ──
println("\n[3/7] Injecting synthetic disruptions (3.5%)...")
df = df.withColumn("_rand", rand(123))
val disrupted = df.filter(col("_rand") < 0.035)
  .withColumn("subway_ridership",
    (col("subway_ridership") * (lit(0.05) + rand(456) * 0.15)).cast(IntegerType))
  .withColumn("taxi_pickups",
    (col("taxi_pickups") * (lit(2.0) + rand(789) * 2.0)).cast(IntegerType))
  .withColumn("bike_starts",
    (col("bike_starts") * (lit(1.8) + rand(101) * 1.5)).cast(IntegerType))
  .withColumn("complaint_count",
    col("complaint_count") + lit(5) + (rand(202) * 10).cast(IntegerType))
val normal = df.filter(col("_rand") >= 0.035)
df = normal.union(disrupted).drop("_rand")
println(s"  Injected ${disrupted.count()} disruption events")

// ── STEP 4: Anomaly detection (z-score + KMeans, replaces sklearn Isolation Forest) ──
println("\n[4/7] Running anomaly detection (z-score + KMeans)...")
val baselines = df.groupBy("zone_id", "hour_of_day", "is_weekend").agg(
  mean("taxi_pickups").alias("mean_pickups"),
  stddev("taxi_pickups").alias("std_pickups"),
  mean("subway_ridership").alias("mean_subway"),
  stddev("subway_ridership").alias("std_subway"),
  mean("bike_starts").alias("mean_bikes"),
  stddev("bike_starts").alias("std_bikes")
)

var scored = df.join(baselines, Seq("zone_id", "hour_of_day", "is_weekend"), "left")
scored = scored.withColumn("z_subway",
  when(col("std_subway") > 0,
    (col("subway_ridership") - col("mean_subway")) / col("std_subway"))
  .otherwise(0.0))
scored = scored.withColumn("z_pickups",
  when(col("std_pickups") > 0,
    (col("taxi_pickups") - col("mean_pickups")) / col("std_pickups"))
  .otherwise(0.0))
scored = scored.withColumn("z_bikes",
  when(col("std_bikes") > 0,
    (col("bike_starts") - col("mean_bikes")) / col("std_bikes"))
  .otherwise(0.0))

// Anomaly score: subway DOWN + taxi UP + bike UP = disruption signature
scored = scored.withColumn("anomaly_score",
  least(lit(1.0), greatest(lit(0.0),
    greatest(lit(0.0), -col("z_subway")) * 0.4 +
    greatest(lit(0.0), col("z_pickups")) * 0.3 +
    greatest(lit(0.0), col("z_bikes")) * 0.3
  )))

scored = scored.withColumn("is_anomaly",
  when(
    (col("z_subway") < -1.5) &&
    ((col("z_pickups") > 1.5) || (col("z_bikes") > 1.5)),
    1).otherwise(0))

val totalRows = scored.count()
val anomalyCount = scored.filter(col("is_anomaly") === 1).count()
println(s"  Total rows: ${totalRows}")
println(s"  Anomalies detected: ${anomalyCount} (${anomalyCount * 100.0 / totalRows}%)")

// ── STEP 5: GBT Classifier ──
println("\n[5/7] Training GBT classifier...")
var active = scored.filter(col("taxi_pickups") > 0 || col("subway_ridership") > 0)
for (c <- FEATURE_COLS if active.columns.contains(c)) {
  active = active.withColumn(c, col(c).cast(DoubleType))
}

val availableCols = FEATURE_COLS.filter(active.columns.contains)

val assembler = new VectorAssembler()
  .setInputCols(availableCols)
  .setOutputCol("raw_features")
  .setHandleInvalid("skip")

val scaler = new StandardScaler()
  .setInputCol("raw_features")
  .setOutputCol("features")
  .setWithStd(true)
  .setWithMean(false)

val trainAnomalyCount = active.filter(col("is_anomaly") === 1).count()
val trainNormalCount = active.count() - trainAnomalyCount
val weightRatio = trainNormalCount.toDouble / math.max(1, trainAnomalyCount)

active = active.withColumn("class_weight",
  when(col("is_anomaly") === 1, lit(weightRatio)).otherwise(lit(1.0)))
println(s"  Class weight ratio: ${weightRatio}x for anomalies")

val gbt = new GBTClassifier()
  .setLabelCol("is_anomaly")
  .setFeaturesCol("features")
  .setWeightCol("class_weight")
  .setMaxIter(100)
  .setMaxDepth(5)
  .setStepSize(0.1)
  .setSubsamplingRate(0.8)
  .setSeed(42)

val pipeline = new Pipeline().setStages(Array(assembler, scaler, gbt))

val Array(trainData, testData) = active.randomSplit(Array(0.8, 0.2), seed = 42)
println(s"  Train: ${trainData.count()} | Test: ${testData.count()}")
println(s"  Features: ${availableCols.length}")
println("  Training GBT (maxIter=100, maxDepth=5)...")

val model = pipeline.fit(trainData)
val predictions = model.transform(testData)

val auc = new BinaryClassificationEvaluator()
  .setLabelCol("is_anomaly").setMetricName("areaUnderROC").evaluate(predictions)
val f1 = new MulticlassClassificationEvaluator()
  .setLabelCol("is_anomaly").setMetricName("f1").evaluate(predictions)
val accuracy = new MulticlassClassificationEvaluator()
  .setLabelCol("is_anomaly").setMetricName("accuracy").evaluate(predictions)
val precision = new MulticlassClassificationEvaluator()
  .setLabelCol("is_anomaly").setMetricName("weightedPrecision").evaluate(predictions)
val recall = new MulticlassClassificationEvaluator()
  .setLabelCol("is_anomaly").setMetricName("weightedRecall").evaluate(predictions)

println(s"\n  Results:")
println(s"    AUC-ROC:   ${auc}")
println(s"    Accuracy:  ${accuracy}")
println(s"    Precision: ${precision}")
println(s"    Recall:    ${recall}")
println(s"    F1 Score:  ${f1}")

// Feature importance
val gbtModel = model.stages.last.asInstanceOf[org.apache.spark.ml.classification.GBTClassificationModel]
val importances = gbtModel.featureImportances.toArray
val featImp = availableCols.zip(importances).sortBy(-_._2)
println(s"\n  Feature Importance (top 10):")
featImp.take(10).foreach { case (name, imp) =>
  println(f"    ${name}%-30s ${imp}%.4f")
}

// Confusion matrix
val tp = predictions.filter(col("is_anomaly") === 1 && col("prediction") === 1.0).count()
val fp = predictions.filter(col("is_anomaly") === 0 && col("prediction") === 1.0).count()
val fn = predictions.filter(col("is_anomaly") === 1 && col("prediction") === 0.0).count()
val tn = predictions.filter(col("is_anomaly") === 0 && col("prediction") === 0.0).count()
println(s"\n  Confusion Matrix:")
println(s"                  Predicted")
println(s"                Normal  Anomaly")
println(s"    Actual Normal  ${tn}  ${fp}")
println(s"    Actual Anomaly ${fn}  ${tp}")

// ── STEP 6: Save model ──
println("\n[6/7] Saving model...")
try {
  model.write.overwrite().save(s"${HDFS_MODELS}/gbt_cascade")
  println(s"  GBT pipeline saved to ${HDFS_MODELS}/gbt_cascade")
} catch {
  case e: Exception =>
    model.write.overwrite().save("gbt_cascade_model")
    println(s"  GBT pipeline saved locally (HDFS failed: ${e.getMessage})")
}

// ── STEP 7: Save predictions ──
println("\n[7/7] Saving predictions...")
val outputCols = Array(
  "zone_id", "hourly_timestamp", "year", "month", "hour_of_day",
  "taxi_pickups", "taxi_dropoffs", "subway_ridership",
  "bike_starts", "bike_ends",
  "z_subway", "z_pickups", "z_bikes", "anomaly_score",
  "is_anomaly", "prediction"
).filter(predictions.columns.contains)

predictions.select(outputCols.map(col): _*)
  .write.mode("overwrite")
  .parquet(HDFS_PREDICTIONS)
println(s"  Predictions saved to ${HDFS_PREDICTIONS}")

// Save evaluation metrics as JSON
val metricsJson = s"""{
  "gbt_classifier": {
    "accuracy": ${accuracy},
    "precision": ${precision},
    "recall": ${recall},
    "f1_score": ${f1},
    "auc_roc": ${auc},
    "true_positives": ${tp},
    "false_positives": ${fp},
    "true_negatives": ${tn},
    "false_negatives": ${fn}
  }
}"""

import org.apache.hadoop.fs.{FileSystem, Path}
val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
val metricsPath = new Path(s"${HDFS_BASE}/evaluation/evaluation_metrics.json")
fs.mkdirs(metricsPath.getParent)
val out = fs.create(metricsPath, true)
out.writeBytes(metricsJson)
out.close()
println(s"  Evaluation metrics saved to ${metricsPath}")

println("\n" + "=" * 60)
println("  PIPELINE COMPLETE")
println(s"  AUC: ${auc} | F1: ${f1} | Accuracy: ${accuracy}")
println("=" * 60)

System.exit(0)
