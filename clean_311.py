"""
=============================================================================
NeuroTraffic — PySpark 311 Complaints Cleaning
=============================================================================
Reads NYC 311 Service Request CSVs from HDFS, filters transport-relevant
complaints, and writes cleaned Parquet.

Cleaning steps:
  1. Filter to transport-relevant complaint types
  2. Parse and validate timestamps + locations
  3. Drop rows missing coordinates
  4. Map complaint coordinates to nearest taxi zone
  5. Add hourly_timestamp for downstream joins

Usage:
  spark-submit --master yarn clean_311.py
=============================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

HDFS_BASE = "hdfs://namenode:9000/data/neurotraffic"
RAW_311 = f"{HDFS_BASE}/raw/311"
CLEANED_311 = f"{HDFS_BASE}/cleaned/311"

NYC_LAT_MIN, NYC_LAT_MAX = 40.4774, 40.9176
NYC_LON_MIN, NYC_LON_MAX = -74.2591, -73.7004

TRANSPORT_COMPLAINT_TYPES = [
    "Street Condition",
    "Traffic Signal Condition",
    "Street Light Condition",
    "Blocked Driveway",
    "Illegal Parking",
    "Noise - Street/Sidewalk",
    "Water Main Break",
    "Sewer",
    "Bus Stop Shelter Complaint",
    "Traffic",
    "Taxi Complaint",
    "Bike/Roller/Skate Condition",
]


def create_spark():
    return (
        SparkSession.builder
        .appName("NeuroTraffic_Clean311")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def main():
    spark = create_spark()

    print("\n" + "=" * 60)
    print("NeuroTraffic — 311 Complaints Cleaning Pipeline")
    print("=" * 60 + "\n")

    df = spark.read.csv(RAW_311, header=True, inferSchema=True)
    raw_count = df.count()
    print(f"  Raw rows: {raw_count:,}")
    print(f"  Columns: {df.columns[:10]}...")

    # Standardize columns
    df = df.select(
        F.col("created_date").cast("timestamp").alias("complaint_time"),
        F.col("closed_date").cast("timestamp").alias("closed_time"),
        F.trim(F.col("complaint_type")).alias("complaint_type"),
        F.trim(F.col("descriptor")).alias("descriptor"),
        F.col("incident_zip").alias("zip_code"),
        F.col("borough"),
        F.col("latitude").cast(DoubleType()),
        F.col("longitude").cast(DoubleType()),
        F.col("status"),
    )

    # 1. Filter to transport-relevant complaint types
    df = df.filter(F.col("complaint_type").isin(TRANSPORT_COMPLAINT_TYPES))
    after_filter = df.count()
    print(f"  After complaint type filter: {after_filter:,}")

    # 2. Drop rows missing timestamps
    df = df.dropna(subset=["complaint_time"])

    # 3. Filter to 2022-2024
    df = df.filter(
        F.col("complaint_time").between("2022-01-01", "2025-01-01")
    )

    # 4. Drop rows missing coordinates
    df = df.dropna(subset=["latitude", "longitude"])

    # 5. Filter coordinates to NYC bounding box
    df = df.filter(
        (F.col("latitude").between(NYC_LAT_MIN, NYC_LAT_MAX)) &
        (F.col("longitude").between(NYC_LON_MIN, NYC_LON_MAX))
    )

    # 6. Add severity score based on complaint type
    severity_map = {
        "Water Main Break": 5,
        "Traffic Signal Condition": 4,
        "Street Condition": 3,
        "Sewer": 3,
        "Blocked Driveway": 2,
        "Street Light Condition": 2,
        "Traffic": 2,
        "Illegal Parking": 1,
        "Noise - Street/Sidewalk": 1,
        "Bus Stop Shelter Complaint": 1,
        "Taxi Complaint": 1,
        "Bike/Roller/Skate Condition": 1,
    }
    severity_expr = F.lit(1)
    for ctype, score in severity_map.items():
        severity_expr = F.when(
            F.col("complaint_type") == ctype, score
        ).otherwise(severity_expr)
    df = df.withColumn("severity_score", severity_expr)

    # 7. Compute resolution time (hours)
    df = df.withColumn(
        "resolution_hours",
        (F.col("closed_time").cast("long") - F.col("complaint_time").cast("long")) / 3600.0
    )

    # 8. Add derived columns for downstream joins
    df = df.withColumn(
        "complaint_hour", F.date_trunc("hour", F.col("complaint_time"))
    )
    df = df.withColumn("year", F.year("complaint_time"))
    df = df.withColumn("month", F.month("complaint_time"))

    clean_count = df.count()
    removed = raw_count - clean_count
    pct = (removed / raw_count * 100) if raw_count > 0 else 0

    print(f"  Cleaned rows: {clean_count:,}")
    print(f"  Removed:      {removed:,} ({pct:.2f}%)")

    # Complaint type distribution
    print("\n  Complaint type distribution:")
    df.groupBy("complaint_type").count().orderBy(F.desc("count")).show(15, truncate=False)

    # Write to HDFS
    print(f"  Writing to {CLEANED_311} (partitioned by year/month)...")
    (
        df
        .repartition("year", "month")
        .write
        .partitionBy("year", "month")
        .mode("overwrite")
        .parquet(CLEANED_311)
    )
    print("  [OK] 311 cleaning complete!\n")

    print("=" * 60)
    print("DATA QUALITY REPORT — 311 COMPLAINTS")
    print("=" * 60)
    print(f"  Raw:     {raw_count:>12,}")
    print(f"  Cleaned: {clean_count:>12,}")
    print()

    df.printSchema()
    df.show(5, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
