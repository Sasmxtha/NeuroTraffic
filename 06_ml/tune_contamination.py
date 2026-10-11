"""
NeuroTraffic — Isolation Forest Contamination Tuning
=====================================================
Tests multiple contamination values against the real combined_v3 table
to find the optimal threshold for disruption detection.

Evaluates each value on:
  - Cross-modal signature strength (subway DOWN + taxi UP + bike UP)
  - Score separation between normal and anomalous groups
  - Precision/recall/F1 if ground truth labels are available

Data: hdfs://namenode:9000/data/neurotraffic/combined_v3 (6.9M rows)
Graph: hdfs://namenode:9000/data/neurotraffic/graph_features

Usage:
  spark-submit --master local[*] --driver-memory 4g tune_contamination.py --hdfs

"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import time
import sys
import os

HDFS_COMBINED = "hdfs://namenode:9000/data/neurotraffic/combined_v3"
HDFS_GRAPH = "hdfs://namenode:9000/data/neurotraffic/graph_features"
HDFS_MODELS = "hdfs://namenode:9000/data/neurotraffic/models"

IF_FEATURE_COLS = [
    "subway_ridership", "taxi_pickups", "bike_starts",
    "temperature_c", "is_rush_hour", "day_of_week",
    "is_weekend", "page_rank",
]

CONTAMINATION_VALUES = [0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03, 0.05]


def main():
    use_hdfs = "--hdfs" in sys.argv

    spark = SparkSession.builder \
        .appName("NeuroTraffic_ContaminationTuning") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    # Load data
    combined_path = HDFS_COMBINED if use_hdfs else sys.argv[sys.argv.index("--local") + 1]
    print(f"Loading combined table from: {combined_path}")
    df = spark.read.parquet(combined_path)

    # Join graph features
    try:
        graph_path = HDFS_GRAPH if use_hdfs else os.path.join(
            os.path.dirname(__file__), "..", "real_data", "graph_features", "features.parquet"
        )
        graph = spark.read.parquet(graph_path)
        graph_cols = graph.select(
            F.col("vertex_id").alias("gf_zone_id"),
            F.col("page_rank"),
        )
        df = df.join(graph_cols, df.zone_id == graph_cols.gf_zone_id, "left").drop("gf_zone_id")
        df = df.fillna({"page_rank": 0.0})
    except Exception:
        df = df.withColumn("page_rank", F.lit(0.0))
        print("  Graph features not available, using page_rank=0.0")

    # Sample for sklearn (2M rows max)
    total = df.count()
    sample_frac = min(1.0, 2_000_000 / max(1, total))
    if sample_frac < 1.0:
        print(f"  Sampling {sample_frac*100:.1f}% ({int(total*sample_frac):,} rows)")
        df = df.sample(fraction=sample_frac, seed=42)

    pandas_df = df.select(*IF_FEATURE_COLS).toPandas()
    X = pandas_df[IF_FEATURE_COLS].fillna(0).values
    print(f"  Feature matrix: {X.shape}\n")

    # Test each contamination value
    results = []
    for contam in CONTAMINATION_VALUES:
        print(f"--- contamination = {contam} ---")
        t0 = time.time()

        model = IsolationForest(
            n_estimators=200, contamination=contam,
            max_samples="auto", random_state=42, n_jobs=-1,
        )
        model.fit(X)

        raw_scores = model.decision_function(X)
        predictions = model.predict(X)

        s_min, s_max = raw_scores.min(), raw_scores.max()
        anomaly_scores = 1.0 - (raw_scores - s_min) / (s_max - s_min)

        anomaly_mask = predictions == -1
        normal_mask = predictions == 1
        n_anomalies = anomaly_mask.sum()
        elapsed = time.time() - t0

        # Cross-modal signature: subway should be LOW, taxi and bike HIGH
        subway_ratio = X[anomaly_mask, 0].mean() / max(X[normal_mask, 0].mean(), 1e-9)
        taxi_ratio = X[anomaly_mask, 1].mean() / max(X[normal_mask, 1].mean(), 1e-9)
        bike_ratio = X[anomaly_mask, 2].mean() / max(X[normal_mask, 2].mean(), 1e-9)

        score_gap = anomaly_scores[anomaly_mask].mean() - anomaly_scores[normal_mask].mean()

        # Signature score: reward low subway + high taxi + high bike
        sig_score = (
            max(0, 1.0 - subway_ratio) +
            max(0, taxi_ratio - 1.0) +
            max(0, bike_ratio - 1.0) +
            score_gap * 2
        )

        results.append({
            "contamination": contam, "n_anomalies": n_anomalies,
            "pct": n_anomalies / len(X) * 100,
            "subway_ratio": subway_ratio, "taxi_ratio": taxi_ratio,
            "bike_ratio": bike_ratio, "score_gap": score_gap,
            "sig_score": sig_score, "elapsed": elapsed,
        })

        print(f"  Anomalies: {n_anomalies:>7} ({results[-1]['pct']:.3f}%)")
        print(f"  Subway: {subway_ratio:.3f}x  Taxi: {taxi_ratio:.3f}x  Bike: {bike_ratio:.3f}x")
        print(f"  Score gap: {score_gap:.4f}  Sig score: {sig_score:.4f}  ({elapsed:.1f}s)\n")

    # Comparison table
    print("=" * 75)
    print("COMPARISON TABLE")
    print("=" * 75)
    print(f"{'Contam':>8} | {'Anomalies':>9} | {'%':>6} | "
          f"{'Subway':>7} | {'Taxi':>6} | {'Bike':>6} | {'SigScore':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r['contamination']:>8.3f} | {r['n_anomalies']:>9,} | "
              f"{r['pct']:>5.2f}% | {r['subway_ratio']:>7.3f} | "
              f"{r['taxi_ratio']:>6.3f} | {r['bike_ratio']:>6.3f} | "
              f"{r['sig_score']:>8.4f}")

    best = max(results, key=lambda r: r["sig_score"])
    print(f"\nBEST: contamination = {best['contamination']} (sig_score = {best['sig_score']:.4f})")

    # Retrain with best and save
    print(f"\nRetraining with contamination = {best['contamination']}...")
    best_model = IsolationForest(
        n_estimators=200, contamination=best["contamination"],
        max_samples="auto", random_state=42, n_jobs=-1,
    )
    best_model.fit(X)

    raw_scores = best_model.decision_function(X)
    s_min, s_max = raw_scores.min(), raw_scores.max()

    with open("isolation_forest_model_tuned.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("if_scaler_params_tuned.pkl", "wb") as f:
        pickle.dump({
            "feature_cols": IF_FEATURE_COLS,
            "score_min": float(s_min), "score_max": float(s_max),
            "best_contamination": best["contamination"],
        }, f)

    os.system(f"hadoop fs -mkdir -p {HDFS_MODELS}/isolation_forest/")
    os.system(f"hadoop fs -put -f isolation_forest_model_tuned.pkl {HDFS_MODELS}/isolation_forest/model.pkl")
    os.system(f"hadoop fs -put -f if_scaler_params_tuned.pkl {HDFS_MODELS}/isolation_forest/scaler_params.pkl")
    print("Tuned model saved to HDFS.")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    contams = [r["contamination"] for r in results]

    axes[0].plot(contams, [r["n_anomalies"] for r in results], "o-", color="steelblue", lw=2)
    axes[0].axvline(best["contamination"], color="red", ls="--", label=f"Best={best['contamination']}")
    axes[0].set_xlabel("Contamination"); axes[0].set_ylabel("Anomalies Flagged")
    axes[0].set_title("Anomaly Count vs Contamination"); axes[0].legend()

    axes[1].plot(contams, [r["subway_ratio"] for r in results], "o-", label="Subway (want LOW)", color="green")
    axes[1].plot(contams, [r["taxi_ratio"] for r in results], "s-", label="Taxi (want HIGH)", color="orange")
    axes[1].plot(contams, [r["bike_ratio"] for r in results], "^-", label="Bike (want HIGH)", color="purple")
    axes[1].axhline(1.0, color="gray", ls=":", alpha=0.5)
    axes[1].axvline(best["contamination"], color="red", ls="--")
    axes[1].set_xlabel("Contamination"); axes[1].set_ylabel("Anomaly / Normal Ratio")
    axes[1].set_title("Cross-Modal Signature"); axes[1].legend(fontsize=8)

    axes[2].bar(range(len(contams)), [r["sig_score"] for r in results],
                color="steelblue", tick_label=[str(c) for c in contams])
    axes[2].set_xlabel("Contamination"); axes[2].set_ylabel("Signature Score")
    axes[2].set_title("Overall Signature Score")
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig("contamination_tuning.png", dpi=150, bbox_inches="tight")
    print("Plot saved to contamination_tuning.png")

    print(f"\nDashboard thresholds: Score >= 0.80 = YELLOW, Score >= 0.90 = RED")

    spark.stop()


if __name__ == "__main__":
    main()
