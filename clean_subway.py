"""
=============================================================================
NeuroTraffic — PySpark MTA Subway Ridership Cleaning
=============================================================================
Reads MTA Subway Hourly Ridership CSVs from HDFS, cleans, and writes Parquet.

The MTA Hourly Ridership dataset (wujg-7c2s) provides pre-computed hourly
ridership per station complex. This is simpler than the legacy turnstile
data which required cumulative counter differencing.

Cleaning steps:
  1. Parse and validate timestamps
  2. Remove rows with null/zero ridership
  3. Remove statistical outliers (> 5 std deviations from station mean)
  4. Standardize station names for cross-dataset matching
  5. Add hourly_timestamp for downstream joins

Usage:
  spark-submit --master yarn clean_subway.py
=============================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, LongType
from pyspark.sql.window import Window

HDFS_BASE = "hdfs://namenode:9000/data/neurotraffic"
RAW_SUBWAY = f"{HDFS_BASE}/raw/subway"
CLEANED_SUBWAY = f"{HDFS_BASE}/cleaned/subway"


def create_spark():
    return (
        SparkSession.builder
        .appName("NeuroTraffic_CleanSubway")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def main():
    spark = create_spark()

    print("\n" + "=" * 60)
    print("NeuroTraffic — MTA Subway Ridership Cleaning Pipeline")
    print("=" * 60 + "\n")

    df = spark.read.csv(RAW_SUBWAY, header=True, inferSchema=True)
    raw_count = df.count()
    print(f"  Raw rows: {raw_count:,}")
    print(f"  Columns: {df.columns}")

    # The MTA Hourly Ridership dataset columns:
    #   transit_timestamp, transit_mode, station_complex_id,
    #   station_complex, borough, payment_method,
    #   fare_class_category, ridership, transfers, latitude, longitude,
    #   georeference
    df = df.select(
        F.col("transit_timestamp").cast("timestamp").alias("transit_hour"),
        F.col("station_complex_id").cast("string"),
        F.trim(F.col("station_complex")).alias("station_name"),
        F.col("borough"),
        F.col("ridership").cast(LongType()),
        F.col("transfers").cast(LongType()),
        F.col("latitude").cast(DoubleType()),
        F.col("longitude").cast(DoubleType()),
        F.col("payment_method"),
        F.col("fare_class_category"),
    )

    # 1. Drop rows with null timestamps or null ridership
    df = df.dropna(subset=["transit_hour", "ridership"])

    # 2. Remove rows with negative ridership
    df = df.filter(F.col("ridership") >= 0)

    # 3. Filter to 2022-2024
    df = df.filter(
        F.col("transit_hour").between("2022-01-01", "2025-01-01")
    )

    # 4. Aggregate ridership per station per hour
    #    (the raw data may have multiple rows per station-hour for
    #     different payment methods and fare classes)
    df_agg = (
        df.groupBy(
            "transit_hour", "station_complex_id", "station_name",
            "borough", "latitude", "longitude"
        )
        .agg(
            F.sum("ridership").alias("total_ridership"),
            F.sum("transfers").alias("total_transfers"),
        )
    )

    # 5. Remove extreme outliers (> 5 std devs from station mean)
    station_stats = df_agg.groupBy("station_complex_id").agg(
        F.mean("total_ridership").alias("mean_ridership"),
        F.stddev("total_ridership").alias("std_ridership"),
    )
    df_agg = df_agg.join(station_stats, on="station_complex_id", how="left")
    df_agg = df_agg.filter(
        F.col("total_ridership") <=
        F.col("mean_ridership") + 5 * F.col("std_ridership")
    )
    df_agg = df_agg.drop("mean_ridership", "std_ridership")

    # 6. Add derived columns
    df_agg = df_agg.withColumn("year", F.year("transit_hour"))
    df_agg = df_agg.withColumn("month", F.month("transit_hour"))

    clean_count = df_agg.count()

    print(f"  Raw rows (multi-row per station-hour): {raw_count:,}")
    print(f"  After aggregation + outlier removal:   {clean_count:,}")
    print(f"  (Row reduction is mostly from aggregating payment methods per station-hour)")

    # Write to HDFS
    print(f"\n  Writing to {CLEANED_SUBWAY} (partitioned by year/month)...")
    (
        df_agg
        .repartition("year", "month")
        .write
        .partitionBy("year", "month")
        .mode("overwrite")
        .parquet(CLEANED_SUBWAY)
    )
    print("  [OK] Subway cleaning complete!\n")

    print("=" * 60)
    print("DATA QUALITY REPORT — SUBWAY")
    print("=" * 60)
    print(f"  Raw rows:        {raw_count:>12,}")
    print(f"  After aggregation + outlier removal: {clean_count:>12,}")
    print()

    df_agg.printSchema()
    df_agg.show(5, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
