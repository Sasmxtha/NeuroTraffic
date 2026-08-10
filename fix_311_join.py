"""
Fix the 311 complaint join in combined_v2.
The original join failed because 311 borough values are UPPERCASE
while zone lookup uses Title Case. This script:
1. Reads combined_v2
2. Drops the zero-filled complaint columns
3. Re-aggregates 311 data with initcap(borough)
4. Re-joins and writes combined_v3
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

HDFS = "hdfs://namenode:9000/data/neurotraffic"

spark = (
    SparkSession.builder
    .appName("Fix_311_Join")
    .config("spark.sql.shuffle.partitions", "20")
    .getOrCreate()
)

print("=" * 60)
print("Fixing 311 complaint join")
print("=" * 60)

print("\n1. Loading combined_v2...")
combined = spark.read.parquet(f"{HDFS}/combined_v2")
print(f"   Rows: {combined.count():,}")
print(f"   Columns: {len(combined.columns)}")

print("\n2. Checking current complaint values...")
combined.select(
    F.sum("complaint_count").alias("total_complaints"),
    F.max("max_complaint_severity").alias("max_severity"),
).show()

print("\n3. Loading and re-aggregating 311 data with fixed borough casing...")
complaints = spark.read.parquet(f"{HDFS}/cleaned/311")
complaints = complaints.withColumn("borough", F.initcap(F.col("borough")))

print("   311 borough values after fix:")
complaints.select("borough").distinct().orderBy("borough").show()

complaints_agg = (
    complaints.groupBy(
        F.col("borough"),
        F.col("complaint_hour").alias("hourly_timestamp"),
    )
    .agg(
        F.count("*").alias("complaint_count_new"),
        F.max("severity_score").alias("max_complaint_severity_new"),
    )
)
print(f"   311 aggregated rows: {complaints_agg.count():,}")

print("\n4. Loading zone lookup for borough mapping...")
zones = spark.read.csv(f"{HDFS}/raw/taxi/zones/taxi_zone_lookup.csv", header=True, inferSchema=True)
zones = zones.select(
    F.col("LocationID").cast("int").alias("zone_id"),
    F.col("Borough").alias("zone_borough"),
)

print("   Zone borough values:")
zones.select("zone_borough").distinct().orderBy("zone_borough").show()

print("\n5. Dropping old complaint columns...")
combined = combined.drop("complaint_count", "max_complaint_severity")

print("\n6. Joining fixed 311 data via zone_borough + hourly_timestamp...")
combined = combined.join(
    complaints_agg,
    on=[combined.zone_borough == complaints_agg.borough,
        combined.hourly_timestamp == complaints_agg.hourly_timestamp],
    how="left"
).drop(complaints_agg.borough).drop(complaints_agg.hourly_timestamp)

combined = combined.withColumnRenamed("complaint_count_new", "complaint_count")
combined = combined.withColumnRenamed("max_complaint_severity_new", "max_complaint_severity")

combined = combined.fillna({"complaint_count": 0, "max_complaint_severity": 0})

print("\n7. Verifying fix...")
combined.select(
    F.sum("complaint_count").alias("total_complaints"),
    F.max("max_complaint_severity").alias("max_severity"),
    F.sum(F.when(F.col("complaint_count") > 0, 1).otherwise(0)).alias("rows_with_complaints"),
).show()

print("\n8. Writing fixed table to combined_v3...")
total = combined.count()
print(f"   Total rows: {total:,}")
print(f"   Columns: {len(combined.columns)}")

combined.repartition("year", "month").write.partitionBy("year", "month").mode("overwrite").parquet(f"{HDFS}/combined_v3")

print("\n   [OK] Written to combined_v3")
print("=" * 60)

spark.stop()
