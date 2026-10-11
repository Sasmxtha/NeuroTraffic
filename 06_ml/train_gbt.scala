// train_gbt.scala
// NeuroTraffic — Train GBT Classifier (Scala/Spark MLlib)
// Replaces train_gbt.py — uses Spark MLlib GBTClassifier instead of sklearn.
//
// Reads training sample from HDFS (built by prepare_ml_training.scala),
// trains GBTClassifier, saves model + metrics + feature importance.
//
// Run: spark-shell --driver-memory 4g -i /project/06_ml/train_gbt.scala

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.Pipeline
import org.apache.hadoop.fs.{FileSystem, Path}

val HDFS_BASE    = "hdfs://namenode:9000/data/neurotraffic"
val TRAINING     = s"${HDFS_BASE}/ml/training.parquet"
val MODEL_OUTPUT = s"${HDFS_BASE}/models/gbt_trained"

val FEATURE_COLS = Array(
  "taxi_pickups", "taxi_dropoffs", "subway_ridership",
  "bike_starts", "bike_ends",
  "taxi_to_subway_ratio", "bike_to_subway_ratio",
  "total_surface_transport", "complaint_density",
  "temperature_c", "precipitation_mm", "is_rain", "is_snow",
  "hour_of_day", "is_rush_hour", "is_weekend", "day_of_week",
  "complaint_count", "event_count",
  "page_rank", "community_id", "in_degree", "out_degree"
)

println("=" * 60)
println("  NeuroTraffic — GBT Training (Spark MLlib)")
println("=" * 60)

// Load training data
println(s"\nPulling training sample from HDFS: ${TRAINING}")
var df = spark.read.parquet(TRAINING)
println(s"  ${df.count()} rows x ${df.columns.length} cols")

// Cast features to Double, replace infinities
val availableCols = FEATURE_COLS.filter(df.columns.contains)
for (c <- availableCols) {
  df = df.withColumn(c, col(c).cast(DoubleType))
  df = df.withColumn(c, when(col(c).isNull || col(c).isNaN, 0.0).otherwise(col(c)))
}

val posCount = df.filter(col("is_disruption") === 1).count()
val negCount = df.filter(col("is_disruption") === 0).count()
println(s"  pos=${posCount}  neg=${negCount}  pos_rate=${posCount.toDouble / (posCount + negCount)}")

// Class weighting
val posWeight = negCount.toDouble / math.max(1, posCount)
df = df.withColumn("class_weight",
  when(col("is_disruption") === 1, lit(posWeight)).otherwise(lit(1.0)))

// Train/test split
val Array(trainData, testData) = df.randomSplit(Array(0.8, 0.2), seed = 42)

println(s"  Train: ${trainData.count()} | Test: ${testData.count()}")
println(s"  Features: ${availableCols.length}")
println(s"  Class weight: ${posWeight}x for disruptions")

// Build pipeline
val assembler = new VectorAssembler()
  .setInputCols(availableCols)
  .setOutputCol("raw_features")
  .setHandleInvalid("skip")

val scaler = new StandardScaler()
  .setInputCol("raw_features")
  .setOutputCol("features")
  .setWithStd(true)
  .setWithMean(true)

val gbt = new GBTClassifier()
  .setLabelCol("is_disruption")
  .setFeaturesCol("features")
  .setWeightCol("class_weight")
  .setMaxIter(150)
  .setMaxDepth(4)
  .setStepSize(0.1)
  .setSubsamplingRate(0.8)
  .setSeed(42)

val pipeline = new Pipeline().setStages(Array(assembler, scaler, gbt))

println("\nTraining GBTClassifier(maxIter=150, maxDepth=4)...")
val model = pipeline.fit(trainData)
val predictions = model.transform(testData)

// Evaluate
val auc = new BinaryClassificationEvaluator()
  .setLabelCol("is_disruption").setMetricName("areaUnderROC").evaluate(predictions)
val accuracy = new MulticlassClassificationEvaluator()
  .setLabelCol("is_disruption").setMetricName("accuracy").evaluate(predictions)
val precision = new MulticlassClassificationEvaluator()
  .setLabelCol("is_disruption").setMetricName("weightedPrecision").evaluate(predictions)
val recall = new MulticlassClassificationEvaluator()
  .setLabelCol("is_disruption").setMetricName("weightedRecall").evaluate(predictions)
val f1 = new MulticlassClassificationEvaluator()
  .setLabelCol("is_disruption").setMetricName("f1").evaluate(predictions)

println("\nMetrics:")
println(f"  accuracy:  ${accuracy}%.4f")
println(f"  precision: ${precision}%.4f")
println(f"  recall:    ${recall}%.4f")
println(f"  f1_score:  ${f1}%.4f")
println(f"  auc_roc:   ${auc}%.4f")

// Confusion matrix
val tp = predictions.filter(col("is_disruption") === 1 && col("prediction") === 1.0).count()
val fp = predictions.filter(col("is_disruption") === 0 && col("prediction") === 1.0).count()
val fn = predictions.filter(col("is_disruption") === 1 && col("prediction") === 0.0).count()
val tn = predictions.filter(col("is_disruption") === 0 && col("prediction") === 0.0).count()
println(s"  confusion: TN=${tn} FP=${fp} FN=${fn} TP=${tp}")

// Feature importance
val gbtModel = model.stages.last.asInstanceOf[org.apache.spark.ml.classification.GBTClassificationModel]
val importances = gbtModel.featureImportances.toArray
val featImp = availableCols.zip(importances).sortBy(-_._2)

println("\nTop 10 features by importance:")
featImp.take(10).foreach { case (name, imp) =>
  println(f"  ${name}%24s  ${imp}%.4f")
}

// Save model
println(s"\nSaving model to ${MODEL_OUTPUT}...")
model.write.overwrite().save(MODEL_OUTPUT)
println(s"  Saved MLlib pipeline model")

// Save evaluation metrics as JSON to HDFS
val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

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
    "false_negatives": ${fn},
    "n_train_samples": ${trainData.count()},
    "n_test_samples": ${testData.count()}
  },
  "data_summary": {
    "total_zone_hours": 6917952,
    "zones": 263,
    "training_rows": ${trainData.count()},
    "test_rows": ${testData.count()},
    "positives_total": ${posCount},
    "label_source": "detect_anomalies.scala",
    "negatives_sampled_fraction": 0.05
  }
}"""

val metricsPath = new Path(s"${HDFS_BASE}/evaluation/evaluation_metrics.json")
fs.mkdirs(metricsPath.getParent)
val metricsOut = fs.create(metricsPath, true)
metricsOut.writeBytes(metricsJson)
metricsOut.close()
println(s"  Wrote evaluation_metrics.json")

// Save feature importance as CSV
val impCsv = "feature,importance\n" + featImp.map { case (n, i) => s"${n},${i}" }.mkString("\n")
val impPath = new Path(s"${HDFS_BASE}/evaluation/feature_importance.csv")
val impOut = fs.create(impPath, true)
impOut.writeBytes(impCsv)
impOut.close()
println(s"  Wrote feature_importance.csv")

// Save confusion matrix as CSV
val cmCsv = """model,actual,predicted,count
GBT,Normal,Normal,""" + tn + """
GBT,Normal,Disruption,""" + fp + """
GBT,Disruption,Normal,""" + fn + """
GBT,Disruption,Disruption,""" + tp
val cmPath = new Path(s"${HDFS_BASE}/evaluation/confusion_matrix.csv")
val cmOut = fs.create(cmPath, true)
cmOut.writeBytes(cmCsv)
cmOut.close()
println(s"  Wrote confusion_matrix.csv")

// Also copy to local dashboard data directory
println("\nCopying artifacts to dashboard data directory...")
try {
  val localDashData = "/project/08_dashboard/data/"
  val localModels   = "/project/08_dashboard/models/"

  // Copy metrics JSON
  fs.copyToLocalFile(false, metricsPath, new Path(s"${localDashData}/evaluation_metrics.json"), true)
  // Copy feature importance
  fs.copyToLocalFile(false, impPath, new Path(s"${localDashData}/feature_importance.csv"), true)
  // Copy confusion matrix
  fs.copyToLocalFile(false, cmPath, new Path(s"${localDashData}/confusion_matrix.csv"), true)

  println("  Copied to 08_dashboard/data/")
} catch {
  case e: Exception => println(s"  Could not copy to dashboard dir: ${e.getMessage}")
}

println("\n" + "=" * 60)
println("  TRAINING COMPLETE")
println(f"  AUC: ${auc}%.4f | F1: ${f1}%.4f | Accuracy: ${accuracy * 100}%.1f%%")
println("=" * 60)

System.exit(0)
