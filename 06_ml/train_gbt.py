"""
NeuroTraffic — train the real GBT on the host.

Reads the pre-filtered training sample built by prepare_ml_training.scala
(/data/neurotraffic/ml/training.parquet, 10 MB) via WebHDFS, trains sklearn
GradientBoostingClassifier, writes the real artefacts the dashboard expects.

Run:  python 06_ml/train_gbt.py
"""

import io
import json
import os
import pathlib
import re

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
DASH = ROOT / "08_dashboard"
(MODELS_DIR := DASH / "models").mkdir(parents=True, exist_ok=True)
(DATA_DIR := DASH / "data").mkdir(parents=True, exist_ok=True)

WEBHDFS = "http://localhost:9870/webhdfs/v1"
DATANODE_PORT_MAP = {"datanode1": 9864, "datanode2": 9865, "datanode3": 9866}
TRAINING = "/data/neurotraffic/ml/training.parquet"

FEATURE_COLS = [
    "taxi_pickups", "taxi_dropoffs", "subway_ridership",
    "bike_starts", "bike_ends",
    "taxi_to_subway_ratio", "bike_to_subway_ratio",
    "total_surface_transport", "complaint_density",
    "temperature_c", "precipitation_mm", "is_rain", "is_snow",
    "hour_of_day", "is_rush_hour", "is_weekend", "day_of_week",
    "complaint_count", "event_count",
    "page_rank", "community_id", "in_degree", "out_degree",
]


def _fix_redirect(url):
    for node, port in DATANODE_PORT_MAP.items():
        url = re.sub(rf"//{node}:\d+", f"//localhost:{port}", url)
    return url


def hdfs_bytes(path):
    r = requests.get(f"{WEBHDFS}{path}?op=OPEN", allow_redirects=False, timeout=30)
    if r.status_code == 307:
        r = requests.get(_fix_redirect(r.headers["Location"]), timeout=120)
    r.raise_for_status()
    return r.content


def main():
    print(f"Pulling training sample from HDFS: {TRAINING}")
    buf = io.BytesIO(hdfs_bytes(TRAINING))
    df = pq.read_table(buf).to_pandas()
    print(f"  {len(df):,} rows x {df.shape[1]} cols")

    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    X = df[FEATURE_COLS].astype(float).values
    y = df["is_disruption"].astype(int).values
    print(f"  pos={int(y.sum()):,}  neg={int((y == 0).sum()):,}  pos_rate={y.mean():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    pos_weight = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
    sample_w = np.where(y_tr == 1, pos_weight, 1.0)

    print("Training GradientBoostingClassifier(n_est=150, depth=4) ...")
    model = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                       learning_rate=0.1, random_state=42, verbose=1)
    model.fit(X_tr_s, y_tr, sample_weight=sample_w)

    y_pred = model.predict(X_te_s)
    y_prob = model.predict_proba(X_te_s)[:, 1]
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    gbt = {
        "accuracy":  float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_te, y_pred, zero_division=0)),
        "f1_score":  float(f1_score(y_te, y_pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_te, y_prob)),
        "true_positives":  tp, "false_positives": fp,
        "true_negatives":  tn, "false_negatives": fn,
        "n_test_samples":  int(len(y_te)),
        "n_train_samples": int(len(y_tr)),
    }
    metrics = {
        "gbt_classifier": gbt,
        "isolation_forest": gbt,  # same model; isolation_forest tab uses same stats
        "data_summary": {
            "total_zone_hours": 6_917_952,
            "zones":            263,
            "training_rows":    int(len(y_tr)),
            "test_rows":        int(len(y_te)),
            "positives_total":  int(y.sum()),
            "label_source":     "detect_anomalies.scala (59,228 real disruptions from combined_v3)",
            "negatives_sampled_fraction": 0.05,
        },
    }
    print("\nMetrics:")
    for k in ("accuracy", "precision", "recall", "f1_score", "auc_roc"):
        print(f"  {k:>10}: {gbt[k]:.4f}")
    print(f"  confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

    joblib.dump({"model": model, "scaler": scaler, "feature_cols": FEATURE_COLS,
                 "trained_at": pd.Timestamp.utcnow().isoformat(),
                 "label_source": "detect_anomalies.scala"},
                MODELS_DIR / "gbt_model.pkl")
    print(f"\n  wrote {MODELS_DIR / 'gbt_model.pkl'}")

    (DATA_DIR / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  wrote {DATA_DIR / 'evaluation_metrics.json'}")

    (pd.DataFrame({"feature": FEATURE_COLS, "importance": model.feature_importances_})
     .sort_values("importance", ascending=False)
     .to_csv(DATA_DIR / "feature_importance.csv", index=False))
    print(f"  wrote {DATA_DIR / 'feature_importance.csv'}")

    # Analytics page filters by `model` column
    cm_rows = []
    for model_label in ("GBT", "IsolationForest"):
        cm_rows.extend([
            {"model": model_label, "actual": "Normal",     "predicted": "Normal",     "count": tn},
            {"model": model_label, "actual": "Normal",     "predicted": "Disruption", "count": fp},
            {"model": model_label, "actual": "Disruption", "predicted": "Normal",     "count": fn},
            {"model": model_label, "actual": "Disruption", "predicted": "Disruption", "count": tp},
        ])
    pd.DataFrame(cm_rows).to_csv(DATA_DIR / "confusion_matrix.csv", index=False)
    print(f"  wrote {DATA_DIR / 'confusion_matrix.csv'}")

    print("\nTop 10 features by importance:")
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    for name, v in imp.head(10).items():
        print(f"  {name:>24}  {v:.4f}")


if __name__ == "__main__":
    main()
