"""
=============================================================================
NeuroTraffic — PySpark Taxi Data Cleaning (Yellow + Green + HVFHV)
=============================================================================
Reads raw TLC Parquet files from HDFS, cleans them, and writes a unified
taxi trip table partitioned by year and month.

Cleaning steps:
  1. Standardize column names across yellow/green/HVFHV schemas
  2. Remove GPS outliers (coordinates outside NYC bounding box)
  3. Remove zero-passenger trips
  4. Remove zero-distance trips
  5. Remove trips with negative or zero fare
  6. Remove trips with impossible durations (< 30s or > 12 hours)
  7. Add zone_id mapping and hourly_timestamp for downstream joins

Usage:
  spark-submit --master yarn --deploy-mode client \
    --num-executors 4 --executor-memory 4g \
    clean_taxi.py

  # Or for local testing:
  spark-submit --master local[*] clean_taxi.py
=============================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, DoubleType,
    StringType, TimestampType, LongType
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HDFS_BASE = "hdfs://namenode:9000/data/neurotraffic"
RAW_TAXI = f"{HDFS_BASE}/raw/taxi"
CLEANED_TAXI = f"{HDFS_BASE}/cleaned/taxi"

# NYC bounding box (generous — covers all 5 boroughs + some buffer)
NYC_LAT_MIN, NYC_LAT_MAX = 40.4774, 40.9176
NYC_LON_MIN, NYC_LON_MAX = -74.2591, -73.7004

# Trip duration bounds
MIN_TRIP_SECONDS = 30
MAX_TRIP_SECONDS = 12 * 3600  # 12 hours


def create_spark():
    return (
        SparkSession.builder
        .appName("NeuroTraffic_CleanTaxi")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def _read_and_select_parquet_files(spark, hdfs_dir, select_exprs, staging_path=None):
    """Read each Parquet file individually, apply select, then union — avoids schema merge.
    For large directories, set staging_path to write intermediate results and avoid OOM."""
    hadoop_conf = spark._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
        spark._jvm.java.net.URI(hdfs_dir), hadoop_conf
    )
    status_list = fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir))
    files = []
    for status in status_list:
        path_str = status.getPath().toString()
        if path_str.endswith(".parquet"):
            files.append(path_str)
    files.sort()
    print(f"  Found {len(files)} Parquet files")

    if staging_path:
        fs_out = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jvm.java.net.URI(staging_path), hadoop_conf
        )
        out_path = spark._jvm.org.apache.hadoop.fs.Path(staging_path)
        if fs_out.exists(out_path):
            fs_out.delete(out_path, True)

        BATCH = 6
        for i in range(0, len(files), BATCH):
            batch_files = files[i:i+BATCH]
            dfs = []
            for f in batch_files:
                fname = f.split("/")[-1]
                try:
                    one = spark.read.parquet(f).select(*select_exprs)
                    dfs.append(one)
                    print(f"    [OK] {fname}")
                except Exception as e:
                    print(f"    [SKIP] {fname}: {e}")
            if dfs:
                batch_df = dfs[0]
                for d in dfs[1:]:
                    batch_df = batch_df.unionByName(d)
                batch_df.write.mode("append").parquet(staging_path)
                print(f"    Wrote batch {i//BATCH + 1}")

        return spark.read.parquet(staging_path)

    dfs = []
    for f in files:
        fname = f.split("/")[-1]
        try:
            one = spark.read.parquet(f).select(*select_exprs)
            dfs.append(one)
            print(f"    [OK] {fname}")
        except Exception as e:
            print(f"    [SKIP] {fname}: {e}")

    if not dfs:
        raise RuntimeError(f"No files could be read from {hdfs_dir}")

    combined = dfs[0]
    for d in dfs[1:]:
        combined = combined.unionByName(d)
    return combined


def clean_yellow(spark):
    print("=" * 60)
    print("CLEANING: Yellow Taxi")
    print("=" * 60)

    select_exprs = [
        F.lit("yellow").alias("taxi_type"),
        F.col("tpep_pickup_datetime").cast("timestamp").alias("pickup_datetime"),
        F.col("tpep_dropoff_datetime").cast("timestamp").alias("dropoff_datetime"),
        F.col("PULocationID").cast(IntegerType()).alias("pu_location_id"),
        F.col("DOLocationID").cast(IntegerType()).alias("do_location_id"),
        F.col("passenger_count").cast(IntegerType()),
        F.col("trip_distance").cast(DoubleType()),
        F.col("fare_amount").cast(DoubleType()),
        F.col("total_amount").cast(DoubleType()),
        F.col("payment_type").cast(IntegerType()),
    ]

    df = _read_and_select_parquet_files(spark, f"{RAW_TAXI}/yellow/", select_exprs)

    raw_count = df.count()
    print(f"  Raw rows: {raw_count:,}")

    df = _apply_common_filters(df)
    clean_count = df.count()
    _print_stats("Yellow Taxi", raw_count, clean_count)
    return df, raw_count, clean_count


def clean_green(spark):
    print("=" * 60)
    print("CLEANING: Green Taxi")
    print("=" * 60)

    select_exprs = [
        F.lit("green").alias("taxi_type"),
        F.col("lpep_pickup_datetime").cast("timestamp").alias("pickup_datetime"),
        F.col("lpep_dropoff_datetime").cast("timestamp").alias("dropoff_datetime"),
        F.col("PULocationID").cast(IntegerType()).alias("pu_location_id"),
        F.col("DOLocationID").cast(IntegerType()).alias("do_location_id"),
        F.col("passenger_count").cast(IntegerType()),
        F.col("trip_distance").cast(DoubleType()),
        F.col("fare_amount").cast(DoubleType()),
        F.col("total_amount").cast(DoubleType()),
        F.col("payment_type").cast(IntegerType()),
    ]

    df = _read_and_select_parquet_files(spark, f"{RAW_TAXI}/green/", select_exprs)

    raw_count = df.count()
    print(f"  Raw rows: {raw_count:,}")

    df = _apply_common_filters(df)
    clean_count = df.count()
    _print_stats("Green Taxi", raw_count, clean_count)
    return df, raw_count, clean_count


def clean_hvfhv(spark):
    print("=" * 60)
    print("CLEANING: HVFHV (Uber/Lyft)")
    print("=" * 60)

    select_exprs = [
        F.lit("hvfhv").alias("taxi_type"),
        F.col("pickup_datetime").cast("timestamp").alias("pickup_datetime"),
        F.col("dropoff_datetime").cast("timestamp").alias("dropoff_datetime"),
        F.col("PULocationID").cast(IntegerType()).alias("pu_location_id"),
        F.col("DOLocationID").cast(IntegerType()).alias("do_location_id"),
        F.lit(None).cast(IntegerType()).alias("passenger_count"),
        F.col("trip_miles").cast(DoubleType()).alias("trip_distance"),
        F.col("base_passenger_fare").cast(DoubleType()).alias("fare_amount"),
        F.col("driver_pay").cast(DoubleType()).alias("total_amount"),
        F.lit(None).cast(IntegerType()).alias("payment_type"),
    ]

    staging = f"{HDFS_BASE}/staging/hvfhv_selected"
    df = _read_and_select_parquet_files(spark, f"{RAW_TAXI}/hvfhv/", select_exprs, staging_path=staging)

    raw_count = df.count()
    print(f"  Raw rows: {raw_count:,}")

    df = _apply_common_filters(df, check_passengers=False)
    clean_count = df.count()
    _print_stats("HVFHV", raw_count, clean_count)
    return df, raw_count, clean_count


def _apply_common_filters(df, check_passengers=True):
    """Apply shared cleaning filters to any taxi dataframe."""

    # 1. Remove nulls in critical columns
    df = df.dropna(subset=["pickup_datetime", "dropoff_datetime",
                            "pu_location_id", "do_location_id"])

    # 2. Remove zero-passenger trips (yellow/green only)
    if check_passengers:
        df = df.filter(
            (F.col("passenger_count") > 0) | F.col("passenger_count").isNull()
        )

    # 3. Remove zero or negative distance trips
    df = df.filter(F.col("trip_distance") > 0)

    # 4. Remove negative or zero fare
    df = df.filter(F.col("fare_amount") > 0)

    # 5. Remove location IDs outside valid range (1-263 for NYC taxi zones)
    df = df.filter(
        (F.col("pu_location_id").between(1, 263)) &
        (F.col("do_location_id").between(1, 263))
    )

    # 6. Remove trips with impossible durations
    df = df.withColumn(
        "trip_duration_sec",
        (F.col("dropoff_datetime").cast("long") -
         F.col("pickup_datetime").cast("long"))
    )
    df = df.filter(
        F.col("trip_duration_sec").between(MIN_TRIP_SECONDS, MAX_TRIP_SECONDS)
    )

    # 7. Filter to 2022-2024 date range only
    df = df.filter(
        F.col("pickup_datetime").between("2022-01-01", "2025-01-01")
    )

    # 8. Add derived columns for downstream joins
    df = df.withColumn(
        "pickup_hour", F.date_trunc("hour", F.col("pickup_datetime"))
    )
    df = df.withColumn("year", F.year("pickup_datetime"))
    df = df.withColumn("month", F.month("pickup_datetime"))

    return df


def _print_stats(name, raw, clean):
    removed = raw - clean
    pct = (removed / raw * 100) if raw > 0 else 0
    print(f"  Cleaned rows: {clean:,}")
    print(f"  Removed:      {removed:,} ({pct:.2f}%)")
    print()


def _write_cleaned(df, path, label):
    print(f"  Writing {label} to {path}...")
    (
        df
        .write
        .mode("overwrite")
        .parquet(path)
    )
    print(f"  [OK] {label} written.")


def main():
    spark = create_spark()

    print("\n" + "=" * 60)
    print("NeuroTraffic — Taxi Data Cleaning Pipeline")
    print("=" * 60 + "\n")

    stats = {}

    yellow_df, yellow_raw, yellow_clean = clean_yellow(spark)
    _write_cleaned(yellow_df, f"{CLEANED_TAXI}/yellow", "Yellow Taxi")
    stats["Yellow"] = (yellow_raw, yellow_clean)
    yellow_df.unpersist()
    spark.catalog.clearCache()

    green_df, green_raw, green_clean = clean_green(spark)
    _write_cleaned(green_df, f"{CLEANED_TAXI}/green", "Green Taxi")
    stats["Green"] = (green_raw, green_clean)
    green_df.unpersist()
    spark.catalog.clearCache()

    hvfhv_df, hvfhv_raw, hvfhv_clean = clean_hvfhv(spark)
    _write_cleaned(hvfhv_df, f"{CLEANED_TAXI}/hvfhv", "HVFHV")
    stats["HVFHV"] = (hvfhv_raw, hvfhv_clean)
    hvfhv_df.unpersist()
    spark.catalog.clearCache()

    # Print data quality summary
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT — TAXI")
    print("=" * 60)
    total_raw = total_clean = 0
    for name, (raw, clean) in stats.items():
        print(f"  {name:8s} {raw:>14,} → {clean:>14,}  "
              f"(removed {raw - clean:,})")
        total_raw += raw
        total_clean += clean
    print(f"  ─────────────────────────────────────────────")
    print(f"  TOTAL:   {total_raw:>14,} → {total_clean:>14,}  "
          f"(removed {total_raw - total_clean:,}, "
          f"{(total_raw - total_clean) / total_raw * 100:.2f}%)")
    print()

    spark.stop()


if __name__ == "__main__":
    main()
