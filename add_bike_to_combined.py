"""
Add bike trip data to the combined table.
Uses the station-to-zone mapping CSV to aggregate bike trips
by zone_id + hourly_timestamp, then joins with existing combined data.
Runs in Docker (PySpark).
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

CLEANED_BIKE = "hdfs://namenode:9000/data/neurotraffic/cleaned/citibike"
COMBINED = "hdfs://namenode:9000/data/neurotraffic/combined"
COMBINED_V2 = "hdfs://namenode:9000/data/neurotraffic/combined_v2"
MAPPING_CSV = "/scripts/bike_station_zones.csv"

def create_spark():
    return (
        SparkSession.builder
        .appName("NeuroTraffic_AddBike")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "20")
        .config("spark.driver.maxResultSize", "1g")
        .getOrCreate()
    )

def main():
    spark = create_spark()

    print("=" * 60)
    print("STEP 1: Load station-to-zone mapping")
    print("=" * 60)

    mapping_schema = StructType([
        StructField("station_id", StringType()),
        StructField("station_name", StringType()),
        StructField("lat", DoubleType()),
        StructField("lng", DoubleType()),
        StructField("zone_id", IntegerType()),
    ])
    mapping = spark.read.csv(MAPPING_CSV, header=True, schema=mapping_schema)
    mapping_count = mapping.count()
    print(f"  Loaded {mapping_count:,} station-to-zone mappings")

    print("=" * 60)
    print("STEP 2: Read cleaned bike data and map to zones")
    print("=" * 60)

    bike = spark.read.parquet(CLEANED_BIKE)
    print(f"  Bike trips: {bike.count():,}")

    start_mapping = mapping.select(
        F.col("station_id").alias("start_station_id"),
        F.col("zone_id").alias("start_zone_id"),
    )
    end_mapping = mapping.select(
        F.col("station_id").alias("end_station_id"),
        F.col("zone_id").alias("end_zone_id"),
    )

    bike = bike.join(start_mapping, on="start_station_id", how="left")
    bike = bike.join(end_mapping, on="end_station_id", how="left")

    mapped = bike.filter(F.col("start_zone_id").isNotNull())
    unmapped = bike.filter(F.col("start_zone_id").isNull()).count()
    total = mapped.count()
    print(f"  Mapped to zones: {total:,}")
    print(f"  Unmapped (no zone): {unmapped:,}")

    print("=" * 60)
    print("STEP 3: Aggregate bike trips by zone + hour")
    print("=" * 60)

    bike_agg = mapped.groupBy(
        F.col("start_zone_id").alias("zone_id"),
        F.col("start_hour").alias("hourly_timestamp"),
    ).agg(
        F.count("*").alias("bike_starts"),
        F.avg("trip_duration_sec").alias("avg_bike_duration_sec"),
    )

    bike_ends = mapped.filter(F.col("end_zone_id").isNotNull()).groupBy(
        F.col("end_zone_id").alias("zone_id"),
        F.col("start_hour").alias("hourly_timestamp"),
    ).agg(
        F.count("*").alias("bike_ends"),
    )

    bike_combined = bike_agg.join(bike_ends, on=["zone_id", "hourly_timestamp"], how="outer")
    bike_combined = bike_combined.fillna(0, subset=["bike_starts", "bike_ends"])

    agg_count = bike_combined.count()
    print(f"  Aggregated bike rows: {agg_count:,}")

    print("=" * 60)
    print("STEP 4: Join with existing combined table")
    print("=" * 60)

    combined = spark.read.parquet(COMBINED)
    print(f"  Existing combined rows: {combined.count():,}")
    print(f"  Existing columns: {len(combined.columns)}")

    result = combined.join(
        bike_combined,
        on=["zone_id", "hourly_timestamp"],
        how="left",
    )

    result = result.fillna(0, subset=["bike_starts", "bike_ends"])
    result = result.fillna(0.0, subset=["avg_bike_duration_sec"])

    print(f"  Result rows: {result.count():,}")
    print(f"  Result columns: {len(result.columns)} ({', '.join(result.columns[-3:])} added)")

    print("=" * 60)
    print("STEP 5: Write combined_v2")
    print("=" * 60)

    result.write.mode("overwrite").partitionBy("year").parquet(COMBINED_V2)
    print(f"  Written to {COMBINED_V2}")

    spark.stop()
    print("=" * 60)
    print("ALL DONE — bike data added to combined table")
    print(f"  New columns: bike_starts, bike_ends, avg_bike_duration_sec")
    print(f"  Output: {COMBINED_V2}")
    print("=" * 60)

if __name__ == "__main__":
    main()
