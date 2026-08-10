"""
=============================================================================
NeuroTraffic — PySpark Weather Data Cleaning
=============================================================================
Reads NOAA/Open-Meteo weather CSVs from HDFS, cleans, and writes Parquet.

Cleaning steps:
  1. Parse timestamps and standardize to hourly
  2. Standardize units to metric (Celsius, mm, km/h)
  3. Interpolate missing readings (forward-fill within 6-hour window)
  4. Add weather severity flags for downstream ML features
  5. Add hourly_timestamp for downstream joins

Usage:
  spark-submit --master yarn clean_weather.py
=============================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

HDFS_BASE = "hdfs://namenode:9000/data/neurotraffic"
RAW_WEATHER = f"{HDFS_BASE}/raw/weather"
CLEANED_WEATHER = f"{HDFS_BASE}/cleaned/weather"


def create_spark():
    return (
        SparkSession.builder
        .appName("NeuroTraffic_CleanWeather")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def clean_openmeteo(spark):
    """Clean Open-Meteo hourly weather data.

    Open-Meteo CSVs have a multi-header format:
      Row 1: latitude,longitude,elevation,...        (6 cols metadata header)
      Row 2: 40.808,-74.02,43.0,...                  (6 cols metadata values)
      Row 3: (blank)
      Row 4: time,temperature_2m (°C),...            (11 cols actual headers)
      Row 5+: data rows                             (11 cols)

    We define a fixed 11-column schema so Spark reads all columns correctly.
    Metadata rows (fewer columns) get nulls in extra cols and are filtered out.
    """
    print("  Attempting Open-Meteo format...")

    from pyspark.sql.types import StructType, StructField, StringType

    schema = StructType([
        StructField("c0", StringType()),
        StructField("c1", StringType()),
        StructField("c2", StringType()),
        StructField("c3", StringType()),
        StructField("c4", StringType()),
        StructField("c5", StringType()),
        StructField("c6", StringType()),
        StructField("c7", StringType()),
        StructField("c8", StringType()),
        StructField("c9", StringType()),
        StructField("c10", StringType()),
    ])

    df = spark.read.csv(
        RAW_WEATHER,
        header=False,
        schema=schema,
    )

    # Keep only data rows (timestamp pattern in first column)
    df = df.filter(F.col("c0").rlike(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}"))

    df = df.select(
        F.col("c0").cast("timestamp").alias("weather_hour"),
        F.col("c1").cast(DoubleType()).alias("temperature_c"),
        F.col("c2").cast(DoubleType()).alias("humidity_pct"),
        F.col("c3").cast(DoubleType()).alias("precipitation_mm"),
        F.col("c4").cast(DoubleType()).alias("rain_mm"),
        F.col("c5").cast(DoubleType()).alias("snowfall_cm"),
        F.col("c6").cast(DoubleType()).alias("snow_depth_m"),
        F.col("c7").cast("int").alias("wmo_weather_code"),
        F.col("c8").cast(DoubleType()).alias("wind_speed_kmh"),
        F.col("c9").cast(DoubleType()).alias("wind_gusts_kmh"),
        F.col("c10").cast(DoubleType()).alias("visibility_m"),
    )

    return df


def main():
    spark = create_spark()

    print("\n" + "=" * 60)
    print("NeuroTraffic — Weather Data Cleaning Pipeline")
    print("=" * 60 + "\n")

    df = clean_openmeteo(spark)
    raw_count = df.count()
    print(f"  Raw rows: {raw_count:,}")

    # 1. Drop rows with null timestamp
    df = df.dropna(subset=["weather_hour"])

    # 2. Filter to 2022-2024
    df = df.filter(
        F.col("weather_hour").between("2022-01-01", "2025-01-01")
    )

    # 3. Remove duplicate hours (keep first)
    df = df.dropDuplicates(["weather_hour"])

    # 4. Interpolate missing values using forward-fill within 6-hour windows
    #    PySpark doesn't have a native ffill, so we use last() with a window
    hour_window = (
        Window
        .orderBy("weather_hour")
        .rowsBetween(-6, 0)
    )

    fill_cols = [
        "temperature_c", "humidity_pct", "precipitation_mm",
        "wind_speed_kmh", "visibility_m"
    ]
    for col in fill_cols:
        df = df.withColumn(
            col,
            F.coalesce(F.col(col), F.last(F.col(col), ignorenulls=True).over(hour_window))
        )

    # 5. Add weather severity flags (useful as ML features)
    df = df.withColumn(
        "is_rain", (F.col("precipitation_mm") > 0.5).cast("int")
    )
    df = df.withColumn(
        "is_heavy_rain", (F.col("precipitation_mm") > 7.6).cast("int")
    )
    df = df.withColumn(
        "is_snow", (F.col("snowfall_cm") > 0).cast("int")
    )
    df = df.withColumn(
        "is_extreme_cold", (F.col("temperature_c") < -10).cast("int")
    )
    df = df.withColumn(
        "is_extreme_heat", (F.col("temperature_c") > 35).cast("int")
    )
    df = df.withColumn(
        "is_low_visibility", (F.col("visibility_m") < 1000).cast("int")
    )
    df = df.withColumn(
        "is_high_wind", (F.col("wind_speed_kmh") > 50).cast("int")
    )

    # 6. Add derived columns
    df = df.withColumn("year", F.year("weather_hour"))
    df = df.withColumn("month", F.month("weather_hour"))

    clean_count = df.count()
    removed = raw_count - clean_count
    pct = (removed / raw_count * 100) if raw_count > 0 else 0

    print(f"  Cleaned rows: {clean_count:,}")
    print(f"  Removed:      {removed:,} ({pct:.2f}%)")

    # Count interpolated values
    for col in fill_cols:
        null_count = df.filter(F.col(col).isNull()).count()
        if null_count > 0:
            print(f"  WARNING: {null_count} nulls remaining in {col} after interpolation")

    # Write to HDFS
    print(f"\n  Writing to {CLEANED_WEATHER} (partitioned by year/month)...")
    (
        df
        .repartition("year", "month")
        .write
        .partitionBy("year", "month")
        .mode("overwrite")
        .parquet(CLEANED_WEATHER)
    )
    print("  [OK] Weather cleaning complete!\n")

    print("=" * 60)
    print("DATA QUALITY REPORT — WEATHER")
    print("=" * 60)
    print(f"  Raw:     {raw_count:>12,}")
    print(f"  Cleaned: {clean_count:>12,}")
    print()

    df.printSchema()
    df.show(5, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
