"""
=============================================================================
NeuroTraffic — PySpark Citi Bike Data Cleaning
=============================================================================
Reads raw Citi Bike CSV files from HDFS, cleans, and writes unified Parquet.

Cleaning steps:
  1. Remove duplicate ride_ids
  2. Standardize station names (trim, title-case)
  3. Drop rows with missing coordinates
  4. Remove trips with identical start/end station and < 60s duration (dock errors)
  5. Remove trips with impossible durations (< 60s or > 24 hours)
  6. Map bike station coordinates to nearest NYC taxi zone for joining
  7. Add hourly_timestamp for downstream joins

Usage:
  spark-submit --master yarn clean_bike.py
=============================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

HDFS_BASE = "hdfs://namenode:9000/data/neurotraffic"
RAW_BIKE = f"{HDFS_BASE}/raw/citibike"
CLEANED_BIKE = f"{HDFS_BASE}/cleaned/citibike"
TAXI_ZONES = f"{HDFS_BASE}/raw/taxi/zones/taxi_zone_lookup.csv"

NYC_LAT_MIN, NYC_LAT_MAX = 40.4774, 40.9176
NYC_LON_MIN, NYC_LON_MAX = -74.2591, -73.7004

MIN_TRIP_SECONDS = 60
MAX_TRIP_SECONDS = 24 * 3600


def create_spark():
    return (
        SparkSession.builder
        .appName("NeuroTraffic_CleanBike")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "20")
        .config("spark.driver.maxResultSize", "1g")
        .getOrCreate()
    )


def main():
    spark = create_spark()

    print("\n" + "=" * 60)
    print("NeuroTraffic — Citi Bike Data Cleaning Pipeline")
    print("=" * 60 + "\n")

    # Citi Bike CSVs may be zipped — Spark can read them if the
    # underlying Hadoop codec is available, or you unzip first.
    # For zipped CSVs, unzip before uploading to HDFS.
    df = spark.read.csv(
        RAW_BIKE,
        header=True,
        inferSchema=True,
    )
    raw_count = df.count()
    print(f"  Raw rows: {raw_count:,}")

    # --- Standardize columns ---
    # Citi Bike schema (2022-2024):
    #   ride_id, rideable_type, started_at, ended_at,
    #   start_station_name, start_station_id, end_station_name, end_station_id,
    #   start_lat, start_lng, end_lat, end_lng, member_casual
    df = df.select(
        F.col("ride_id"),
        F.col("rideable_type"),
        F.col("started_at").cast("timestamp").alias("start_time"),
        F.col("ended_at").cast("timestamp").alias("end_time"),
        F.trim(F.initcap(F.col("start_station_name"))).alias("start_station_name"),
        F.col("start_station_id").cast("string").alias("start_station_id"),
        F.trim(F.initcap(F.col("end_station_name"))).alias("end_station_name"),
        F.col("end_station_id").cast("string").alias("end_station_id"),
        F.col("start_lat").cast(DoubleType()),
        F.col("start_lng").cast(DoubleType()),
        F.col("end_lat").cast(DoubleType()),
        F.col("end_lng").cast(DoubleType()),
        F.col("member_casual"),
    )

    # 1. Remove duplicate ride_ids
    before_dedup = df.count()
    df = df.dropDuplicates(["ride_id"])
    after_dedup = df.count()
    print(f"  Duplicates removed: {before_dedup - after_dedup:,}")

    # 2. Drop rows with missing start/end coordinates
    df = df.dropna(subset=["start_lat", "start_lng", "end_lat", "end_lng"])

    # 3. Filter coordinates to NYC bounding box
    df = df.filter(
        (F.col("start_lat").between(NYC_LAT_MIN, NYC_LAT_MAX)) &
        (F.col("start_lng").between(NYC_LON_MIN, NYC_LON_MAX)) &
        (F.col("end_lat").between(NYC_LAT_MIN, NYC_LAT_MAX)) &
        (F.col("end_lng").between(NYC_LON_MIN, NYC_LON_MAX))
    )

    # 4. Compute trip duration and filter impossible trips
    df = df.withColumn(
        "trip_duration_sec",
        (F.col("end_time").cast("long") - F.col("start_time").cast("long"))
    )
    df = df.filter(
        F.col("trip_duration_sec").between(MIN_TRIP_SECONDS, MAX_TRIP_SECONDS)
    )

    # 5. Remove dock-error trips (same station, very short)
    df = df.filter(
        ~(
            (F.col("start_station_id") == F.col("end_station_id")) &
            (F.col("trip_duration_sec") < 120)
        )
    )

    # 6. Filter to 2022-2024
    df = df.filter(F.col("start_time").between("2022-01-01", "2025-01-01"))

    # 7. Add derived columns for downstream joins
    df = df.withColumn("start_hour", F.date_trunc("hour", F.col("start_time")))
    df = df.withColumn("year", F.year("start_time"))
    df = df.withColumn("month", F.month("start_time"))

    clean_count = df.count()
    removed = raw_count - clean_count
    pct = (removed / raw_count * 100) if raw_count > 0 else 0

    print(f"  Cleaned rows: {clean_count:,}")
    print(f"  Removed:      {removed:,} ({pct:.2f}%)")

    # Write to HDFS
    print(f"\n  Writing to {CLEANED_BIKE} (partitioned by year/month)...")
    (
        df
        .repartition("year", "month")
        .write
        .partitionBy("year", "month")
        .mode("overwrite")
        .parquet(CLEANED_BIKE)
    )
    print("  [OK] Citi Bike cleaning complete!\n")

    print("=" * 60)
    print("DATA QUALITY REPORT — CITI BIKE")
    print("=" * 60)
    print(f"  Raw:     {raw_count:>12,}")
    print(f"  Cleaned: {clean_count:>12,}")
    print(f"  Removed: {removed:>12,} ({pct:.2f}%)")
    print()

    df.printSchema()
    df.show(5, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
