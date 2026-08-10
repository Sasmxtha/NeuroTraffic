"""
Data Quality Report for the combined NeuroTraffic table.
Runs in Docker (PySpark). Prints statistics to stdout.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

COMBINED = "hdfs://namenode:9000/data/neurotraffic/combined"

def create_spark():
    return (
        SparkSession.builder
        .appName("NeuroTraffic_DataQuality")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "20")
        .config("spark.driver.maxResultSize", "1g")
        .getOrCreate()
    )

def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def main():
    spark = create_spark()

    print("=" * 60)
    print("  NEUROTRAFFIC DATA QUALITY REPORT")
    print("=" * 60)

    df = spark.read.parquet(COMBINED)

    section("1. BASIC STATS")
    total_rows = df.count()
    total_cols = len(df.columns)
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total columns: {total_cols}")
    print(f"  Columns: {', '.join(df.columns)}")

    section("2. NULL / ZERO COUNTS PER COLUMN")
    for col_name in df.columns:
        null_count = df.filter(F.col(col_name).isNull()).count()
        null_pct = (null_count / total_rows) * 100
        if df.schema[col_name].dataType.simpleString() in ("bigint", "int", "double", "long"):
            zero_count = df.filter(F.col(col_name) == 0).count()
            zero_pct = (zero_count / total_rows) * 100
            print(f"  {col_name}: nulls={null_count:,} ({null_pct:.1f}%), zeros={zero_count:,} ({zero_pct:.1f}%)")
        else:
            print(f"  {col_name}: nulls={null_count:,} ({null_pct:.1f}%)")

    section("3. ROWS PER YEAR")
    df.groupBy("year").count().orderBy("year").show()

    section("4. TAXI PICKUP STATISTICS")
    taxi_stats = df.agg(
        F.sum("taxi_pickups").alias("total_pickups"),
        F.sum("taxi_dropoffs").alias("total_dropoffs"),
        F.avg("taxi_pickups").alias("avg_pickups_per_zone_hour"),
        F.max("taxi_pickups").alias("max_pickups_single_zone_hour"),
        F.avg("avg_fare").alias("overall_avg_fare"),
        F.avg("avg_trip_distance").alias("overall_avg_distance"),
    ).collect()[0]
    print(f"  Total pickups: {int(taxi_stats['total_pickups']):,}")
    print(f"  Total dropoffs: {int(taxi_stats['total_dropoffs']):,}")
    print(f"  Avg pickups/zone/hour: {taxi_stats['avg_pickups_per_zone_hour']:.1f}")
    print(f"  Max pickups (single zone-hour): {int(taxi_stats['max_pickups_single_zone_hour']):,}")
    print(f"  Overall avg fare: ${taxi_stats['overall_avg_fare']:.2f}")
    print(f"  Overall avg distance: {taxi_stats['overall_avg_distance']:.2f} miles")

    section("5. WEATHER COVERAGE")
    weather_nulls = df.filter(F.col("temperature_c").isNull()).count()
    print(f"  Rows with weather data: {total_rows - weather_nulls:,} / {total_rows:,}")
    weather_stats = df.agg(
        F.min("temperature_c").alias("min_temp"),
        F.max("temperature_c").alias("max_temp"),
        F.avg("temperature_c").alias("avg_temp"),
        F.sum("is_rain").alias("rainy_zone_hours"),
        F.sum("is_snow").alias("snowy_zone_hours"),
        F.sum("is_extreme_heat").alias("extreme_heat_zone_hours"),
        F.sum("is_extreme_cold").alias("extreme_cold_zone_hours"),
    ).collect()[0]
    print(f"  Temperature range: {weather_stats['min_temp']:.1f}C to {weather_stats['max_temp']:.1f}C")
    print(f"  Average temperature: {weather_stats['avg_temp']:.1f}C")
    print(f"  Rainy zone-hours: {int(weather_stats['rainy_zone_hours']):,}")
    print(f"  Snowy zone-hours: {int(weather_stats['snowy_zone_hours']):,}")

    section("6. SUBWAY RIDERSHIP")
    subway_stats = df.agg(
        F.sum("subway_ridership").alias("total_ridership"),
        F.avg("subway_ridership").alias("avg_ridership"),
        F.max("subway_ridership").alias("max_ridership"),
    ).collect()[0]
    print(f"  Total ridership (across all zone-hours): {int(subway_stats['total_ridership']):,}")
    print(f"  Avg ridership per zone-hour: {subway_stats['avg_ridership']:.1f}")
    print(f"  Max ridership (single zone-hour): {int(subway_stats['max_ridership']):,}")

    section("7. 311 COMPLAINTS")
    complaint_stats = df.agg(
        F.sum("complaint_count").alias("total_complaints"),
        F.avg("complaint_count").alias("avg_complaints"),
        F.max("complaint_count").alias("max_complaints"),
        F.max("max_complaint_severity").alias("max_severity"),
    ).collect()[0]
    print(f"  Total complaints: {int(complaint_stats['total_complaints']):,}")
    print(f"  Avg complaints per zone-hour: {complaint_stats['avg_complaints']:.2f}")
    print(f"  Max complaints (single zone-hour): {int(complaint_stats['max_complaints']):,}")
    print(f"  Max severity seen: {int(complaint_stats['max_severity'])}")

    section("8. EVENTS")
    event_stats = df.agg(
        F.sum("event_count").alias("total_events"),
        F.avg("event_count").alias("avg_events"),
        F.max("event_count").alias("max_events"),
    ).collect()[0]
    print(f"  Total event zone-hours: {int(event_stats['total_events']):,}")
    print(f"  Avg events per zone-hour: {event_stats['avg_events']:.3f}")
    print(f"  Max events (single zone-hour): {int(event_stats['max_events']):,}")

    section("9. TIME FEATURES DISTRIBUTION")
    print("  Rush hour rows:")
    df.groupBy("is_rush_hour").count().orderBy("is_rush_hour").show()
    print("  Weekend rows:")
    df.groupBy("is_weekend").count().orderBy("is_weekend").show()

    section("10. TOP 10 BUSIEST ZONES (by total pickups)")
    df.groupBy("zone_id").agg(
        F.sum("taxi_pickups").alias("total_pickups")
    ).orderBy(F.desc("total_pickups")).limit(10).show()

    section("11. TOP 10 BUSIEST HOURS (by total pickups)")
    df.groupBy("hour_of_day").agg(
        F.sum("taxi_pickups").alias("total_pickups")
    ).orderBy(F.desc("total_pickups")).limit(10).show()

    spark.stop()
    print()
    print("=" * 60)
    print("  DATA QUALITY REPORT COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
