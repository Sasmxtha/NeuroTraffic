"""
NeuroTraffic — Central Configuration
======================================

All file paths, constants, and shared settings live here.
"""

import os

# ── Base directory (dashboard/ folder) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── HDFS Configuration ──
# Toggle this to True if you want the app to connect to Hadoop HDFS
USE_HDFS = os.environ.get("NEUROTRAFFIC_USE_HDFS", "true").lower() == "true"

# Host address of Hadoop Namenode
HDFS_HOST = os.environ.get("NEUROTRAFFIC_HDFS_HOST", "localhost")
HDFS_PORT_RPC = 9000
HDFS_PORT_WEB = 9870

HDFS_RPC_PREFIX = f"hdfs://{HDFS_HOST}:{HDFS_PORT_RPC}"
HDFS_WEB_PREFIX = f"http://{HDFS_HOST}:{HDFS_PORT_WEB}/webhdfs/v1"

# ── HDFS Absolute Paths (match real cluster layout) ──
# Combined table (6.9M rows, 38 cols)
HDFS_COMBINED_V3             = "/data/neurotraffic/combined_v3"
# Graph features
HDFS_GRAPH_FEATURES_PARQUET  = "/data/neurotraffic/graph_features"
# ML predictions
HDFS_PREDICTIONS_PARQUET     = "/data/neurotraffic/predictions"

# Dashboard-ready files (produced by generate_dashboard_data.scala)
HDFS_GRAPH_NODES_CSV         = "/data/neurotraffic/dashboard/graph_nodes.csv"
HDFS_EVALUATION_METRICS_JSON = "/data/neurotraffic/dashboard/evaluation_metrics.json"
HDFS_FEATURE_IMPORTANCE_CSV  = "/data/neurotraffic/dashboard/feature_importance.csv"
HDFS_CONFUSION_MATRIX_CSV    = "/data/neurotraffic/dashboard/confusion_matrix.csv"
HDFS_HISTORICAL_PARQUET      = "/data/neurotraffic/dashboard/historical_predictions.parquet"

HDFS_STREAMING_INPUT_DIR     = "/data/streaming/input"
HDFS_STREAMING_OUTPUT_DIR    = "/data/streaming/output"
HDFS_LATEST_CSV              = "/data/streaming/output/latest.csv"
HDFS_LIVE_HISTORY_CSV        = "/data/streaming/output/live_history.csv"
HDFS_SOURCE_STATUS_JSON      = "/data/streaming/output/source_status.json"

# ── Local Data paths ──
DATA_DIR = os.path.join(BASE_DIR, "data")

GRAPH_NODES_CSV         = os.path.join(DATA_DIR, "graph_nodes.csv")
EVALUATION_METRICS_JSON = os.path.join(DATA_DIR, "evaluation_metrics.json")
FEATURE_IMPORTANCE_CSV  = os.path.join(DATA_DIR, "feature_importance.csv")
CONFUSION_MATRIX_CSV    = os.path.join(DATA_DIR, "confusion_matrix.csv")
HISTORICAL_PARQUET      = os.path.join(DATA_DIR, "historical_predictions.parquet")

# ── Local Streaming paths ──
STREAMING_DIR        = os.path.join(BASE_DIR, "streaming")
STREAMING_INPUT_DIR  = os.path.join(STREAMING_DIR, "input")
STREAMING_OUTPUT_DIR = os.path.join(STREAMING_DIR, "output")
LATEST_CSV           = os.path.join(STREAMING_OUTPUT_DIR, "latest.csv")
SOURCE_STATUS_JSON   = os.path.join(STREAMING_OUTPUT_DIR, "source_status.json")

# ── Model paths ──
MODELS_DIR           = os.path.join(BASE_DIR, "models")
GBT_MODEL_PKL        = os.path.join(MODELS_DIR, "gbt_model.pkl")

# ── App settings ──
REFRESH_INTERVAL_SECONDS = 5          # How often Live Monitor refreshes
MAP_CENTER               = {"lat": 40.7380, "lon": -73.9800}
MAP_ZOOM                 = 10
ANOMALY_THRESHOLD_YELLOW = 0.50       # anomaly_score above this → yellow
ANOMALY_THRESHOLD_RED    = 0.80       # anomaly_score above this → red

# ── Subway lines ──
SUBWAY_LINES = {
    "L":     {"color": "#A7A9AC", "boroughs": ["Manhattan", "Brooklyn"]},
    "A/C/E": {"color": "#0039A6", "boroughs": ["Manhattan", "Brooklyn", "Queens"]},
    "1/2/3": {"color": "#EE352E", "boroughs": ["Manhattan", "Bronx", "Brooklyn"]},
    "4/5/6": {"color": "#00933C", "boroughs": ["Manhattan", "Bronx", "Brooklyn"]},
    "N/Q/R/W":{"color": "#FCCC0A","boroughs": ["Manhattan", "Brooklyn", "Queens"]},
    "J/Z":   {"color": "#996633", "boroughs": ["Manhattan", "Brooklyn", "Queens"]},
    "7":     {"color": "#B933AD", "boroughs": ["Manhattan", "Queens"]},
    "G":     {"color": "#6CBE45", "boroughs": ["Brooklyn", "Queens"]},
    "B/D/F/M":{"color": "#FF6319","boroughs": ["Manhattan", "Brooklyn", "Queens"]},
}

# ── Borough colours (for map) ──
BOROUGH_COLORS = {
    "Manhattan":     "#00d4ff",
    "Brooklyn":      "#a78bfa",
    "Queens":        "#34d399",
    "Bronx":         "#fb923c",
    "Staten Island": "#f472b6",
    "EWR":           "#94a3b8",
}

# ── Anomaly colour scale ──
ANOMALY_COLOR_SCALE = [
    [0.00, "#22c55e"],   # green  — normal
    [0.50, "#facc15"],   # yellow — mild anomaly
    [0.80, "#f97316"],   # orange — high anomaly
    [1.00, "#ef4444"],   # red    — disruption
]

# ── App metadata ──
APP_TITLE       = "NeuroTraffic"
APP_SUBTITLE    = "NYC Multi-Modal Transport Disruption Intelligence"
APP_ICON        = "🚊"
