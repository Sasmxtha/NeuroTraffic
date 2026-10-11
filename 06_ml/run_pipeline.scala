// run_pipeline.scala
// NeuroTraffic ML Pipeline — Master Runner (Scala)
// Runs the ML pipeline stages in sequence inside Docker.
//
// Run: spark-shell -i /project/06_ml/run_pipeline.scala
//
// This script loads and executes each stage sequentially.
// For individual stages, run each .scala file directly with spark-shell.

println("=" * 60)
println("  NeuroTraffic ML Pipeline — Master Runner")
println("=" * 60)

println("""
Pipeline stages (run in order with spark-shell):

  1. Contamination/Threshold Tuning:
     spark-shell --driver-memory 4g -i /project/06_ml/tune_contamination.scala

  2. Disruption Detection (z-score + KMeans + GBT):
     spark-shell --driver-memory 4g -i /project/06_ml/disruption_detector.scala

  3. Anomaly Detection (detailed labels):
     spark-shell --driver-memory 4g -i /project/06_ml/detect_anomalies.scala

  4. Prepare ML Training Data:
     spark-shell --driver-memory 4g -i /project/06_ml/prepare_ml_training.scala

  5. Train GBT Classifier:
     spark-shell --driver-memory 4g -i /project/06_ml/train_gbt.scala

  Or run the full pipeline with a shell script:
     for f in tune_contamination.scala disruption_detector.scala train_gbt.scala; do
       spark-shell --driver-memory 4g -i /project/06_ml/$f || exit 1
     done

Note: Each .scala file includes System.exit(0) at the end so spark-shell
      exits cleanly after execution. Run them one at a time or via the
      shell loop above.

The scoring_module.scala is loaded at runtime by the streaming pipeline:
     :load /project/06_ml/scoring_module.scala
     val scorer = new NeuroTrafficScorer(spark)
""")

// If running interactively, execute stages in sequence
println("To run all stages now, paste this into spark-shell:")
println("""
  :load /project/06_ml/tune_contamination.scala
  :load /project/06_ml/disruption_detector.scala
  :load /project/06_ml/train_gbt.scala
""")

System.exit(0)
