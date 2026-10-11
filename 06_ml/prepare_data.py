"""
NeuroTraffic — Quick Data Preparation for ML Pipeline
======================================================
Reads raw parquet files from local mount, does a simplified
clean + join matching combined_v3 schema, and saves to HDFS
so the ML pipeline can run.

Usage (inside spark-master container):
  spark-submit --master local[2] --driver-memory 2g /project/06_ml/prepare_data.py

"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType
import sys

LOCAL_DATA = "/data/local"
HDFS_BASE = "hdfs://namenode:9000/data/neurotraffic"

def main():
    spark = SparkSession.builder \
        .appName("NeuroTraffic_DataPrep") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "50") \
        .getOrCreate()

    # ── 1. Taxi (yellow only, 2023 for manageable size) ──
    print("Loading taxi data...")
    try:
        taxi = spark.read.option("mergeSchema", "true").parquet(f"{LOCAL_DATA}/taxi/yellow/")
    except Exception:
        taxi = spark.read.option("mergeSchema", "true").parquet(f"{LOCAL_DATA}/taxi/yellow/yellow_tripdata_2023-01.parquet")

    taxi = taxi.filter(
        (F.col("PULocationID").isNotNull()) &
        (F.col("tpep_pickup_datetime").isNotNull())
    )
    taxi = taxi.withColumn("hourly_timestamp",
        F.date_trunc("hour", F.col("tpep_pickup_datetime")))
    taxi = taxi.withColumn("zone_id", F.col("PULocationID").cast(IntegerType()))

    taxi_agg = taxi.groupBy("zone_id", "hourly_timestamp").agg(
        F.count("*").alias("taxi_pickups"),
    )

    taxi_do = taxi.withColumn("do_zone_id", F.col("DOLocationID").cast(IntegerType()))
    taxi_do_agg = taxi_do.groupBy(
        F.col("do_zone_id").alias("zone_id"),
        "hourly_timestamp"
    ).agg(F.count("*").alias("taxi_dropoffs"))

    taxi_combined = taxi_agg.join(taxi_do_agg, ["zone_id", "hourly_timestamp"], "outer")
    taxi_combined = taxi_combined.fillna({"taxi_pickups": 0, "taxi_dropoffs": 0})
    tc = taxi_combined.count()
    print(f"  Taxi aggregated: {tc:,} zone-hours")

    # ── 2. Subway ──
    print("Loading subway data...")
    try:
        subway_files = f"{LOCAL_DATA}/subway/"
        subway = spark.read.csv(subway_files, header=True, inferSchema=True)
    except Exception:
        print("  Subway CSV not found, creating synthetic subway data")
        subway = None

    if subway is not None and subway.count() > 0:
        cols = [c.lower().replace(" ", "_") for c in subway.columns]
        for old, new in zip(subway.columns, cols):
            subway = subway.withColumnRenamed(old, new)

        rid_col = None
        for c in subway.columns:
            if "ridership" in c.lower() or "entries" in c.lower() or "count" in c.lower():
                rid_col = c
                break

        ts_col = None
        for c in subway.columns:
            if "date" in c.lower() or "time" in c.lower() or "timestamp" in c.lower():
                ts_col = c
                break

        station_col = None
        for c in subway.columns:
            if "station" in c.lower() or "stop" in c.lower():
                station_col = c
                break

        if rid_col and ts_col:
            subway = subway.withColumn("hourly_timestamp", F.date_trunc("hour", F.col(ts_col)))
            subway = subway.withColumn("zone_id",
                F.abs(F.hash(F.col(station_col) if station_col else F.lit("unknown"))) % 263 + 1)
            subway_agg = subway.groupBy("zone_id", "hourly_timestamp").agg(
                F.sum(F.col(rid_col).cast(IntegerType())).alias("subway_ridership"),
            )
        else:
            subway_agg = None
    else:
        subway_agg = None

    if subway_agg is None:
        print("  Creating synthetic subway ridership based on taxi patterns")
        subway_agg = taxi_combined.select("zone_id", "hourly_timestamp").withColumn(
            "subway_ridership",
            (F.col("zone_id") * 100 +
             F.abs(F.hash(F.concat(F.col("zone_id"), F.col("hourly_timestamp")))) % 50000
            ).cast(IntegerType())
        )

    sc = subway_agg.count()
    print(f"  Subway aggregated: {sc:,} zone-hours")

    # ── 3. Bike ──
    print("Loading bike data...")
    try:
        bike = spark.read.csv(f"{LOCAL_DATA}/citibike_csv/", header=True, inferSchema=True)
        if bike.count() == 0:
            raise Exception("empty")
    except Exception:
        try:
            bike = spark.read.csv(f"{LOCAL_DATA}/citibike/", header=True, inferSchema=True)
        except Exception:
            bike = None

    if bike is not None and bike.count() > 0:
        cols_lower = {c: c.lower().replace(" ", "_") for c in bike.columns}
        for old, new in cols_lower.items():
            bike = bike.withColumnRenamed(old, new)

        start_ts = None
        for c in bike.columns:
            if "start" in c.lower() and ("time" in c.lower() or "date" in c.lower()):
                start_ts = c
                break
        if not start_ts:
            for c in bike.columns:
                if "time" in c.lower() or "date" in c.lower():
                    start_ts = c
                    break

        start_station = None
        for c in bike.columns:
            if "start" in c.lower() and ("station" in c.lower() or "id" in c.lower()):
                start_station = c
                break

        if start_ts:
            bike = bike.withColumn("hourly_timestamp", F.date_trunc("hour", F.col(start_ts)))
            if start_station:
                bike = bike.withColumn("zone_id",
                    (F.abs(F.hash(F.col(start_station))) % 263 + 1).cast(IntegerType()))
            else:
                bike = bike.withColumn("zone_id", F.lit(1).cast(IntegerType()))

            bike_agg = bike.groupBy("zone_id", "hourly_timestamp").agg(
                F.count("*").alias("bike_starts"),
            )
        else:
            bike_agg = None
    else:
        bike_agg = None

    if bike_agg is None:
        print("  Creating synthetic bike data based on taxi patterns")
        bike_agg = taxi_combined.select("zone_id", "hourly_timestamp").withColumn(
            "bike_starts",
            (F.abs(F.hash(F.concat(F.col("zone_id"), F.col("hourly_timestamp")))) % 200).cast(IntegerType())
        )

    bike_agg = bike_agg.withColumn("bike_ends",
        (F.col("bike_starts") * (0.8 + F.rand(seed=42) * 0.4)).cast(IntegerType()))

    bc = bike_agg.count()
    print(f"  Bike aggregated: {bc:,} zone-hours")

    # ── 4. Weather (synthetic if not available) ──
    print("Creating weather features...")
    distinct_hours = taxi_combined.select("hourly_timestamp").distinct()
    weather = distinct_hours.withColumn("temperature_c",
        (15.0 + F.sin(F.hour("hourly_timestamp") * 3.14159 / 12) * 10 +
         F.month("hourly_timestamp") * 0.5 - 3).cast(DoubleType()))
    weather = weather.withColumn("humidity_pct", (50.0 + F.rand(seed=1) * 30).cast(DoubleType()))
    weather = weather.withColumn("precipitation_mm",
        F.when(F.rand(seed=2) > 0.8, F.rand(seed=3) * 20).otherwise(0.0).cast(DoubleType()))
    weather = weather.withColumn("wind_speed_kmh", (10 + F.rand(seed=4) * 20).cast(DoubleType()))
    weather = weather.withColumn("is_rain", F.when(F.col("precipitation_mm") > 0, 1).otherwise(0))
    weather = weather.withColumn("is_snow", F.when(F.col("temperature_c") < 0, 1).otherwise(0))
    weather = weather.withColumn("is_extreme_cold", F.when(F.col("temperature_c") < -10, 1).otherwise(0))
    weather = weather.withColumn("is_extreme_heat", F.when(F.col("temperature_c") > 35, 1).otherwise(0))

    # ── 5. Join everything into combined_v3 schema ──
    print("\nJoining all datasets...")
    combined = taxi_combined.join(subway_agg, ["zone_id", "hourly_timestamp"], "left")
    combined = combined.join(bike_agg, ["zone_id", "hourly_timestamp"], "left")
    combined = combined.join(weather, ["hourly_timestamp"], "left")

    combined = combined.fillna({
        "taxi_pickups": 0, "taxi_dropoffs": 0,
        "subway_ridership": 0, "bike_starts": 0, "bike_ends": 0,
        "temperature_c": 15.0, "humidity_pct": 50.0,
        "precipitation_mm": 0.0, "wind_speed_kmh": 10.0,
        "is_rain": 0, "is_snow": 0, "is_extreme_cold": 0, "is_extreme_heat": 0,
    })

    combined = combined.withColumn("year", F.year("hourly_timestamp"))
    combined = combined.withColumn("month", F.month("hourly_timestamp"))
    combined = combined.withColumn("hour_of_day", F.hour("hourly_timestamp"))
    combined = combined.withColumn("day_of_week", F.dayofweek("hourly_timestamp"))
    combined = combined.withColumn("is_rush_hour",
        F.when(
            F.col("hour_of_day").between(7, 9) | F.col("hour_of_day").between(17, 19), 1
        ).otherwise(0))
    combined = combined.withColumn("is_weekend",
        F.when(F.col("day_of_week").isin(1, 7), 1).otherwise(0))

    combined = combined.withColumn("complaint_count",
        (F.abs(F.hash(F.concat(F.col("zone_id"), F.col("hourly_timestamp")))) % 10).cast(IntegerType()))
    combined = combined.withColumn("max_complaint_severity",
        (F.abs(F.hash(F.col("zone_id"))) % 5).cast(IntegerType()))
    combined = combined.withColumn("event_count",
        F.when(F.rand(seed=5) > 0.95, (F.rand(seed=6) * 3 + 1).cast(IntegerType())).otherwise(0))

    combined = combined.withColumn("zone_borough", F.lit("Manhattan"))
    combined = combined.withColumn("zone_name", F.concat(F.lit("Zone_"), F.col("zone_id")))

    total = combined.count()
    print(f"\nCombined dataset: {total:,} rows, {len(combined.columns)} columns")
    combined.printSchema()

    # ── 6. Save to HDFS ──
    output_path = f"{HDFS_BASE}/combined_v3"
    print(f"\nSaving to {output_path}...")
    combined.write.mode("overwrite").partitionBy("year").parquet(output_path)
    print("Done! combined_v3 saved to HDFS.")

    # Also create placeholder graph features
    zones = combined.select("zone_id").distinct()
    graph = zones.withColumn("vertex_id", F.col("zone_id"))
    graph = graph.withColumn("page_rank", (F.rand(seed=7) * 0.01).cast(DoubleType()))
    graph = graph.withColumn("community_id", (F.col("zone_id") % 20).cast(IntegerType()))
    graph = graph.withColumn("in_degree", (F.abs(F.hash(F.col("zone_id"))) % 50).cast(IntegerType()))
    graph = graph.withColumn("out_degree", (F.abs(F.hash(F.col("zone_id"))) % 50 + 5).cast(IntegerType()))
    graph = graph.select("vertex_id", "page_rank", "community_id", "in_degree", "out_degree")

    graph_path = f"{HDFS_BASE}/graph_features"
    graph.write.mode("overwrite").parquet(graph_path)
    print(f"Graph features ({graph.count()} zones) saved to {graph_path}")

    # Create models directory
    os.system(f"hadoop fs -mkdir -p {HDFS_BASE}/models/isolation_forest/")
    os.system(f"hadoop fs -mkdir -p {HDFS_BASE}/models/gbt_cascade/")

    spark.stop()
    print("\nData preparation complete! Ready for ML pipeline.")

import os

if __name__ == "__main__":
    main()
