"""
=============================================================================
NeuroTraffic — PySpark Event Permits Cleaning
=============================================================================
Reads NYC Event Permits + Film Permits CSVs from HDFS, extracts affected
streets using basic NLP on free-text descriptions, and writes cleaned Parquet.

The two source CSVs have different schemas:
  - event_permits_2022_2024.csv:
      event_id, event_name, start_date_time, end_date_time, event_agency,
      event_type, event_borough, event_location, event_street_side,
      street_closure_type, community_board, police_precinct, cemsid
  - film_permits_2022_2024.csv:
      eventid, eventtype, startdatetime, enddatetime, enteredon,
      eventagency, parkingheld, borough, communityboard_s,
      policeprecinct_s, category, subcategoryname, country, zipcode_s

We read them separately, normalize to a common schema, then union.

Cleaning steps:
  1. Parse event dates and filter to 2022-2024
  2. Extract street names and affected areas from free-text descriptions
  3. Classify event types (marathon, parade, construction, filming, etc.)
  4. Estimate disruption severity based on event type and duration
  5. Add hourly_timestamp for downstream joins

Usage:
  spark-submit --master local[*] --driver-memory 2g /scripts/clean_events.py
=============================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

HDFS_BASE = "hdfs://namenode:9000/data/neurotraffic"
RAW_EVENTS = f"{HDFS_BASE}/raw/events"
CLEANED_EVENTS = f"{HDFS_BASE}/cleaned/events"


def create_spark():
    return (
        SparkSession.builder
        .appName("NeuroTraffic_CleanEvents")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def extract_streets_udf():
    from pyspark.sql.functions import udf
    import re

    @udf(ArrayType(StringType()))
    def extract_streets(text):
        if not text:
            return []

        text = text.upper()
        streets = []

        patterns = [
            r'(\d+(?:ST|ND|RD|TH)\s+(?:ST(?:REET)?|AVE(?:NUE)?|BLVD|BOULEVARD|PL(?:ACE)?|DR(?:IVE)?|RD|ROAD|WAY|LN|LANE))',
            r'((?:BROADWAY|PARK\s+AVE|MADISON\s+AVE|LEXINGTON\s+AVE|5TH\s+AVE|FIFTH\s+AVE|AMSTERDAM\s+AVE|COLUMBUS\s+AVE|CENTRAL\s+PARK\s+(?:WEST|SOUTH|NORTH)))',
            r'(\b[A-Z]+\s+(?:AVENUE|STREET|BOULEVARD|PLACE|DRIVE|ROAD)\b)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            streets.extend(matches)

        return list(set(streets))[:10]

    return extract_streets


def read_event_permits(spark):
    """Read event_permits CSV with its known schema."""
    df = spark.read.csv(
        f"{RAW_EVENTS}/event_permits_2022_2024.csv",
        header=True, inferSchema=True,
    )
    print(f"  Event permits raw rows: {df.count():,}")

    return df.select(
        F.col("event_id").cast("string").alias("event_id"),
        F.col("event_name").alias("event_name"),
        F.col("start_date_time").cast("timestamp").alias("event_start"),
        F.col("end_date_time").cast("timestamp").alias("event_end"),
        F.col("event_type").alias("event_type"),
        F.col("event_borough").alias("borough"),
        F.col("event_location").cast("string").alias("street_description"),
        F.col("street_closure_type").cast("string").alias("closure_type"),
        F.lit("event_permit").alias("source"),
    )


def read_film_permits(spark):
    """Read film_permits CSV with its known schema."""
    df = spark.read.csv(
        f"{RAW_EVENTS}/film_permits_2022_2024.csv",
        header=True, inferSchema=True,
    )
    print(f"  Film permits raw rows: {df.count():,}")

    return df.select(
        F.col("eventid").cast("string").alias("event_id"),
        F.col("subcategoryname").alias("event_name"),
        F.col("startdatetime").cast("timestamp").alias("event_start"),
        F.col("enddatetime").cast("timestamp").alias("event_end"),
        F.col("eventtype").alias("event_type"),
        F.col("borough").alias("borough"),
        F.col("parkingheld").cast("string").alias("street_description"),
        F.lit(None).cast("string").alias("closure_type"),
        F.lit("film_permit").alias("source"),
    )


def main():
    spark = create_spark()

    print("\n" + "=" * 60)
    print("NeuroTraffic — Event Permits Cleaning Pipeline")
    print("=" * 60 + "\n")

    event_df = read_event_permits(spark)
    film_df = read_film_permits(spark)
    df = event_df.unionByName(film_df)

    raw_count = df.count()
    print(f"  Combined raw rows: {raw_count:,}")

    # 1. Drop rows with null event_start
    df = df.dropna(subset=["event_start"])

    # 2. Filter to 2022-2024
    df = df.filter(
        F.col("event_start").between("2022-01-01", "2025-01-01")
    )

    # 3. Extract streets from free-text descriptions
    extract_streets = extract_streets_udf()
    df = df.withColumn("affected_streets", extract_streets(F.col("street_description")))
    df = df.withColumn("num_streets_affected", F.size("affected_streets"))

    # 4. Compute event duration in hours
    df = df.withColumn(
        "event_duration_hours",
        (F.col("event_end").cast("long") - F.col("event_start").cast("long")) / 3600.0
    )
    df = df.filter(
        (F.col("event_duration_hours") > 0) &
        (F.col("event_duration_hours") < 720)
    )

    # 5. Estimate disruption severity
    df = df.withColumn(
        "disruption_severity",
        F.when(F.col("num_streets_affected") >= 5, 5)
        .when(F.col("num_streets_affected") >= 3, 4)
        .when(F.col("event_duration_hours") > 8, 3)
        .when(F.col("num_streets_affected") >= 1, 2)
        .otherwise(1)
    )

    # 6. Add derived columns
    df = df.withColumn("event_hour", F.date_trunc("hour", F.col("event_start")))
    df = df.withColumn("year", F.year("event_start"))
    df = df.withColumn("month", F.month("event_start"))

    clean_count = df.count()
    removed = raw_count - clean_count
    pct = (removed / raw_count * 100) if raw_count > 0 else 0

    print(f"  Cleaned rows: {clean_count:,}")
    print(f"  Removed:      {removed:,} ({pct:.2f}%)")

    print("\n  Event type distribution:")
    df.groupBy("event_type").count().orderBy(F.desc("count")).show(10, truncate=False)

    print(f"  Writing to {CLEANED_EVENTS} (partitioned by year/month)...")
    (
        df
        .repartition("year", "month")
        .write
        .partitionBy("year", "month")
        .mode("overwrite")
        .parquet(CLEANED_EVENTS)
    )
    print("  [OK] Events cleaning complete!\n")

    print("=" * 60)
    print("DATA QUALITY REPORT — EVENTS")
    print("=" * 60)
    print(f"  Raw:     {raw_count:>12,}")
    print(f"  Cleaned: {clean_count:>12,}")
    print()

    df.printSchema()
    df.show(5, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
