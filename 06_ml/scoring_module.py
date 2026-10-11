"""
NeuroTraffic — Scoring Module (Integration API for Streaming)
==============================================================
Imported by the Spark Structured Streaming pipeline.
Loads trained models from HDFS and provides scoring functions.

Usage:
    from scoring_module import NeuroTrafficScorer
    scorer = NeuroTrafficScorer(spark)
    result = scorer.full_pipeline(incoming_batch)

Column names match the combined_v3 schema exactly.

"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import PipelineModel
import pickle
import numpy as np
import os


class NeuroTrafficScorer:

    # Core features for Isolation Forest (per zone per hour)
    IF_FEATURE_COLS = [
        "subway_ridership", "taxi_pickups", "bike_starts",
        "temperature_c", "is_rush_hour", "day_of_week",
        "is_weekend", "page_rank",
    ]

    # Full feature set for GBT classifier
    GBT_FEATURE_COLS = [
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
        "page_rank", "community_id", "in_degree", "out_degree",
    ]

    THRESHOLD_YELLOW = 0.80
    THRESHOLD_RED = 0.90

    def __init__(self, spark: SparkSession,
                 model_base="hdfs://namenode:9000/data/neurotraffic/models"):
        self.spark = spark
        self.model_base = model_base
        self._load_isolation_forest()
        self._load_gbt()

    def _load_isolation_forest(self):
        local_model = "/tmp/nt_if_model.pkl"
        local_scaler = "/tmp/nt_if_scaler.pkl"

        os.system(f"hadoop fs -get -f {self.model_base}/isolation_forest/model.pkl {local_model}")
        os.system(f"hadoop fs -get -f {self.model_base}/isolation_forest/scaler_params.pkl {local_scaler}")

        with open(local_model, "rb") as f:
            self.if_model = pickle.load(f)
        with open(local_scaler, "rb") as f:
            self.if_params = pickle.load(f)

        print(f"[Scorer] Loaded Isolation Forest "
              f"(contamination={self.if_params.get('best_contamination', 'unknown')})")

    def _load_gbt(self):
        gbt_path = f"{self.model_base}/gbt_cascade"
        try:
            self.gbt_model = PipelineModel.load(gbt_path)
            print(f"[Scorer] Loaded GBT pipeline from {gbt_path}")
        except Exception as e:
            print(f"[Scorer] GBT not available ({e}), cascade prediction disabled")
            self.gbt_model = None

    def _ensure_columns(self, df: DataFrame) -> DataFrame:
        defaults = {
            "taxi_pickups": 0, "taxi_dropoffs": 0,
            "subway_ridership": 0, "subway_transfers": 0,
            "bike_starts": 0, "bike_ends": 0,
            "temperature_c": 15.0, "humidity_pct": 50.0,
            "precipitation_mm": 0.0, "wind_speed_kmh": 10.0,
            "is_rain": 0, "is_snow": 0,
            "is_extreme_cold": 0, "is_extreme_heat": 0,
            "complaint_count": 0, "max_complaint_severity": 0,
            "event_count": 0,
            "page_rank": 0.0, "community_id": 0,
            "in_degree": 0, "out_degree": 0,
        }

        for col_name, default_val in defaults.items():
            if col_name not in df.columns:
                df = df.withColumn(col_name, F.lit(default_val))

        if "hourly_timestamp" in df.columns:
            if "hour_of_day" not in df.columns:
                df = df.withColumn("hour_of_day", F.hour("hourly_timestamp"))
            if "day_of_week" not in df.columns:
                df = df.withColumn("day_of_week", F.dayofweek("hourly_timestamp"))
            if "is_rush_hour" not in df.columns:
                df = df.withColumn(
                    "is_rush_hour",
                    F.when(
                        F.col("hour_of_day").between(7, 9) |
                        F.col("hour_of_day").between(17, 19), 1
                    ).otherwise(0),
                )
            if "is_weekend" not in df.columns:
                df = df.withColumn(
                    "is_weekend",
                    F.when(F.col("day_of_week").isin(1, 7), 1).otherwise(0),
                )

        return df

    def _engineer_features(self, df: DataFrame) -> DataFrame:
        df = df.withColumn(
            "taxi_to_subway_ratio",
            F.when(F.col("subway_ridership") > 0,
                   F.col("taxi_pickups").cast(DoubleType()) / F.col("subway_ridership"))
            .otherwise(0.0),
        )
        df = df.withColumn(
            "bike_to_subway_ratio",
            F.when(F.col("subway_ridership") > 0,
                   F.col("bike_starts").cast(DoubleType()) / F.col("subway_ridership"))
            .otherwise(0.0),
        )
        df = df.withColumn(
            "total_surface_transport",
            F.col("taxi_pickups") + F.col("bike_starts"),
        )
        df = df.withColumn(
            "complaint_density",
            F.when(F.col("taxi_pickups") > 0,
                   F.col("complaint_count").cast(DoubleType()) / F.col("taxi_pickups"))
            .otherwise(0.0),
        )
        return df

    def score_anomaly(self, df: DataFrame) -> DataFrame:
        """
        Score each zone-hour for disruption anomaly using Isolation Forest.

        Input:  DataFrame with at minimum: subway_ridership, taxi_pickups,
                bike_starts, temperature_c (+ time features)
        Output: Same DataFrame + anomaly_score (0-1) + anomaly_level (normal/yellow/red)
        """
        df = self._ensure_columns(df)

        pandas_df = df.select(*self.IF_FEATURE_COLS).toPandas()
        X = pandas_df[self.IF_FEATURE_COLS].fillna(0).values

        raw_scores = self.if_model.decision_function(X)
        s_min = self.if_params["score_min"]
        s_max = self.if_params["score_max"]
        anomaly_scores = 1.0 - (raw_scores - s_min) / (s_max - s_min)
        anomaly_scores = np.clip(anomaly_scores, 0.0, 1.0)

        score_df = self.spark.createDataFrame(
            [(float(s),) for s in anomaly_scores],
            ["anomaly_score"],
        )

        df = df.withColumn("_row_id", F.monotonically_increasing_id())
        score_df = score_df.withColumn("_row_id", F.monotonically_increasing_id())
        df = df.join(score_df, "_row_id").drop("_row_id")

        df = df.withColumn(
            "anomaly_level",
            F.when(F.col("anomaly_score") >= self.THRESHOLD_RED, "red")
             .when(F.col("anomaly_score") >= self.THRESHOLD_YELLOW, "yellow")
             .otherwise("normal"),
        )

        return df

    def predict_cascade(self, df: DataFrame) -> DataFrame:
        """
        Predict disruption cascade using GBT on zones flagged as anomalous.

        Input:  DataFrame with full feature set (all GBT_FEATURE_COLS)
        Output: Same DataFrame + prediction (0=normal, 1=disrupted) + probability
        """
        if self.gbt_model is None:
            return df.withColumn("prediction", F.lit(-1))

        df = self._ensure_columns(df)
        df = self._engineer_features(df)

        for col in self.GBT_FEATURE_COLS:
            if col in df.columns:
                df = df.withColumn(col, F.col(col).cast(DoubleType()))

        return self.gbt_model.transform(df)

    def full_pipeline(self, df: DataFrame) -> DataFrame:
        """
        Complete scoring pipeline for streaming micro-batches:
        1. Score anomalies (all zones)
        2. Predict cascades (red zones only)
        3. Return combined results
        """
        scored = self.score_anomaly(df)

        normal = scored.filter(F.col("anomaly_level") == "normal")
        yellow = scored.filter(F.col("anomaly_level") == "yellow")
        red = scored.filter(F.col("anomaly_level") == "red")

        n_red = red.count()
        print(f"[Scorer] {normal.count()} normal, {yellow.count()} yellow, {n_red} red")

        if n_red > 0 and self.gbt_model is not None:
            red = self.predict_cascade(red)

        for col_name in ["prediction"]:
            if col_name not in normal.columns:
                normal = normal.withColumn(col_name, F.lit(None).cast(DoubleType()))
            if col_name not in yellow.columns:
                yellow = yellow.withColumn(col_name, F.lit(None).cast(DoubleType()))

        common = list(set(normal.columns) & set(yellow.columns) & set(red.columns))

        return (normal.select(common)
                .union(yellow.select(common))
                .union(red.select(common)))


# ── Quick test ──
if __name__ == "__main__":
    spark = SparkSession.builder.appName("ScoringTest").getOrCreate()
    scorer = NeuroTrafficScorer(spark)

    test = spark.createDataFrame([
        # Normal Tuesday 9AM zone
        (100, "2023-03-14 09:00:00", 85000, 150, 40, 12.0),
        # Disrupted zone: subway crashed, taxi/bike spiked
        (101, "2023-03-14 09:00:00", 200, 580, 190, 12.0),
    ], ["zone_id", "hourly_timestamp", "subway_ridership",
        "taxi_pickups", "bike_starts", "temperature_c"])

    test = test.withColumn("hourly_timestamp", F.to_timestamp("hourly_timestamp"))

    result = scorer.score_anomaly(test)
    result.select("zone_id", "subway_ridership", "taxi_pickups",
                  "bike_starts", "anomaly_score", "anomaly_level").show(truncate=False)

    spark.stop()
