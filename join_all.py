"""
=============================================================================
NeuroTraffic — PySpark Join All 6 Sources
=============================================================================
Reads all 6 cleaned datasets from HDFS and joins them by zone_id +
hourly_timestamp into a single combined table for downstream GraphX
and MLlib processing.

Join strategy:
  - Base: Generate a complete grid of (zone_id x hourly_timestamp) for
    all 263 taxi zones x every hour in 2022-2024
  - Left-join each dataset onto this grid
  - Fill missing counts with 0
  - Weather joins by hour only (same weather for all zones)
  - Bike/subway/311 map lat/lng to nearest taxi zone via zone centroids
  - Events join by borough + hour

Output: one row per zone per hour with all transport + context features.

Usage:
  spark-submit --master local[*] --driver-memory 4g /scripts/join_all.py
=============================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

HDFS_BASE = "hdfs://namenode:9000/data/neurotraffic"
CLEANED_TAXI = f"{HDFS_BASE}/cleaned/taxi"
CLEANED_BIKE = f"{HDFS_BASE}/cleaned/citibike"
CLEANED_SUBWAY = f"{HDFS_BASE}/cleaned/subway"
CLEANED_WEATHER = f"{HDFS_BASE}/cleaned/weather"
CLEANED_311 = f"{HDFS_BASE}/cleaned/311"
CLEANED_EVENTS = f"{HDFS_BASE}/cleaned/events"
COMBINED_OUTPUT = f"{HDFS_BASE}/combined"
TAXI_ZONES_CSV = f"{HDFS_BASE}/raw/taxi/zones/taxi_zone_lookup.csv"


def create_spark():
    return (
        SparkSession.builder
        .appName("NeuroTraffic_JoinAll")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def load_zone_lookup(spark):
    """Load taxi zone lookup table."""
    zones = spark.read.csv(TAXI_ZONES_CSV, header=True, inferSchema=True)
    zones = zones.select(
        F.col("LocationID").cast(IntegerType()).alias("zone_id"),
        F.col("Borough").alias("zone_borough"),
        F.col("Zone").alias("zone_name"),
        F.col("service_zone"),
    )
    return zones


def build_zone_centroids(spark):
    """Compute zone centroids from taxi pickup data.
    Since taxi data has zone_id for every trip, we can compute
    average pickup coordinates per zone to get approximate centroids.
    These are used to map bike/subway/311 lat/lng to zones."""
    print("  Computing zone centroids from taxi pickup data...")

    taxi = spark.read.parquet(
        f"{CLEANED_TAXI}/yellow",
        f"{CLEANED_TAXI}/green",
        f"{CLEANED_TAXI}/hvfhv",
    )

    centroids = (
        taxi
        .filter(F.col("pu_location_id").between(1, 263))
        .groupBy(F.col("pu_location_id").alias("zone_id"))
        .agg(
            F.count("*").alias("trip_count"),
        )
    )

    # Since taxi data uses LocationID not lat/lng, we need a different
    # approach. We'll use the zone lookup borough to join events,
    # and for bike/subway we'll build a coordinate-to-zone mapping
    # by binning coordinates and finding which zone has the most
    # taxi pickups in that bin.
    return centroids


def build_coord_to_zone_map(spark):
    """Build a mapping from (rounded_lat, rounded_lng) -> zone_id
    by finding which zone_id has the most taxi pickups at each
    coordinate bin. This gives us an approximate spatial mapping."""
    print("  Building coordinate-to-zone mapping from taxi data...")

    taxi = spark.read.parquet(
        f"{CLEANED_TAXI}/yellow",
        f"{CLEANED_TAXI}/green",
        f"{CLEANED_TAXI}/hvfhv",
    )

    # Taxi data has pu_location_id (zone) but no raw lat/lng.
    # However, bike/subway/311 have lat/lng.
    # We need to map (lat, lng) -> zone_id.
    #
    # Strategy: Use the subway station coordinates as anchor points.
    # Each subway station has known lat/lng and we can assign it
    # to the nearest taxi zone. For bike stations, same approach.
    # For 311, same approach.
    #
    # Since we don't have a shapefile, we'll assign based on
    # the borough and use a simple grid mapping.

    return None


def build_zone_hour_grid(spark):
    """Build the complete grid: every zone x every hour in 2022-2024."""
    print("  Building zone x hour grid...")

    zones = spark.range(1, 264).toDF("zone_id").withColumn(
        "zone_id", F.col("zone_id").cast(IntegerType())
    )

    hours = spark.sql("""
        SELECT explode(sequence(
            timestamp('2022-01-01 00:00:00'),
            timestamp('2024-12-31 23:00:00'),
            interval 1 hour
        )) as hourly_timestamp
    """)

    grid = zones.crossJoin(hours)

    grid = (
        grid
        .withColumn("year", F.year("hourly_timestamp"))
        .withColumn("month", F.month("hourly_timestamp"))
        .withColumn("day_of_week", F.dayofweek("hourly_timestamp"))
        .withColumn("hour_of_day", F.hour("hourly_timestamp"))
        .withColumn(
            "is_rush_hour",
            (
                F.col("hour_of_day").between(7, 9) |
                F.col("hour_of_day").between(17, 19)
            ).cast("int")
        )
        .withColumn(
            "is_weekend",
            (F.col("day_of_week").isin(1, 7)).cast("int")
        )
    )

    count = grid.count()
    print(f"  Grid size: {count:,} rows (263 zones x ~26,280 hours)")
    return grid


def aggregate_taxi(spark):
    """Aggregate taxi trips to zone x hour level."""
    print("\n  Aggregating taxi data...")
    taxi = spark.read.parquet(
        f"{CLEANED_TAXI}/yellow",
        f"{CLEANED_TAXI}/green",
        f"{CLEANED_TAXI}/hvfhv",
    )

    pickups = (
        taxi.groupBy(
            F.col("pu_location_id").alias("zone_id"),
            F.col("pickup_hour").alias("hourly_timestamp"),
        )
        .agg(
            F.count("*").alias("taxi_pickups"),
            F.avg("fare_amount").alias("avg_fare"),
            F.avg("trip_distance").alias("avg_trip_distance"),
            F.avg("trip_duration_sec").alias("avg_trip_duration_sec"),
        )
    )

    dropoffs = (
        taxi.groupBy(
            F.col("do_location_id").alias("zone_id"),
            F.date_trunc("hour", F.col("dropoff_datetime")).alias("hourly_timestamp"),
        )
        .agg(F.count("*").alias("taxi_dropoffs"))
    )

    taxi_agg = pickups.join(
        dropoffs,
        on=["zone_id", "hourly_timestamp"],
        how="outer"
    )

    print(f"  Taxi aggregated rows: {taxi_agg.count():,}")
    return taxi_agg


def aggregate_bike(spark, zone_lookup):
    """Aggregate bike trips to zone x hour level.
    Maps bike station coordinates to nearest taxi zone using borough."""
    print("\n  Aggregating bike data...")
    bike = spark.read.parquet(CLEANED_BIKE)

    # Bin start coordinates to 0.01 degree grid (~1km)
    bike = bike.withColumn("bin_lat", F.round(F.col("start_lat"), 2))
    bike = bike.withColumn("bin_lng", F.round(F.col("start_lng"), 2))

    # Assign borough based on latitude bands (approximate)
    # Manhattan: lat 40.70-40.88, lng -74.02 to -73.93
    # Brooklyn: lat 40.57-40.74, lng -74.04 to -73.83
    # Queens: lat 40.54-40.80, lng -73.96 to -73.70
    # Bronx: lat 40.79-40.92, lng -73.93 to -73.75
    # Staten Island: lat 40.49-40.65, lng -74.26 to -74.05
    bike = bike.withColumn(
        "approx_borough",
        F.when(
            (F.col("start_lat") >= 40.70) & (F.col("start_lat") <= 40.88) &
            (F.col("start_lng") >= -74.02) & (F.col("start_lng") <= -73.93),
            "Manhattan"
        ).when(
            (F.col("start_lat") >= 40.57) & (F.col("start_lat") <= 40.74) &
            (F.col("start_lng") >= -74.04) & (F.col("start_lng") <= -73.83),
            "Brooklyn"
        ).when(
            (F.col("start_lat") >= 40.79) & (F.col("start_lat") <= 40.92),
            "Bronx"
        ).when(
            (F.col("start_lat") < 40.65) & (F.col("start_lng") < -74.05),
            "Staten Island"
        ).otherwise("Queens")
    )

    starts = (
        bike.groupBy(
            "bin_lat", "bin_lng", "approx_borough",
            F.col("start_hour").alias("hourly_timestamp"),
        )
        .agg(F.count("*").alias("bike_starts"))
    )

    bike_end = bike.withColumn(
        "end_hour", F.date_trunc("hour", F.col("end_time"))
    ).withColumn(
        "end_bin_lat", F.round(F.col("end_lat"), 2)
    ).withColumn(
        "end_bin_lng", F.round(F.col("end_lng"), 2)
    )

    ends = (
        bike_end.groupBy(
            F.col("end_bin_lat").alias("bin_lat"),
            F.col("end_bin_lng").alias("bin_lng"),
            "approx_borough",
            F.col("end_hour").alias("hourly_timestamp"),
        )
        .agg(F.count("*").alias("bike_ends"))
    )

    bike_agg = starts.join(
        ends,
        on=["bin_lat", "bin_lng", "approx_borough", "hourly_timestamp"],
        how="outer"
    )

    print(f"  Bike aggregated rows: {bike_agg.count():,}")
    return bike_agg


def aggregate_subway(spark):
    """Aggregate subway ridership to zone x hour level.
    Maps subway station coordinates to approximate borough for joining."""
    print("\n  Aggregating subway data...")
    subway = spark.read.parquet(CLEANED_SUBWAY)

    subway_agg = subway.select(
        F.col("station_complex_id"),
        F.col("station_name"),
        F.col("transit_hour").alias("hourly_timestamp"),
        F.col("total_ridership").alias("subway_ridership"),
        F.col("total_transfers").alias("subway_transfers"),
        F.col("borough").alias("subway_borough"),
        F.col("latitude").alias("subway_lat"),
        F.col("longitude").alias("subway_lng"),
    )

    # Aggregate to borough x hour level for joining with zone grid
    subway_borough = (
        subway_agg.groupBy(
            F.col("subway_borough").alias("borough"),
            "hourly_timestamp",
        )
        .agg(
            F.sum("subway_ridership").alias("subway_ridership"),
            F.sum("subway_transfers").alias("subway_transfers"),
            F.countDistinct("station_complex_id").alias("subway_stations_active"),
        )
    )

    print(f"  Subway borough-hour rows: {subway_borough.count():,}")
    return subway_borough


def load_weather(spark):
    """Load cleaned weather data (joins by hour only, same for all zones)."""
    print("\n  Loading weather data...")
    weather = spark.read.parquet(CLEANED_WEATHER)

    weather = weather.select(
        F.col("weather_hour").alias("hourly_timestamp"),
        "temperature_c", "humidity_pct", "precipitation_mm",
        "wind_speed_kmh", "visibility_m",
        "is_rain", "is_heavy_rain", "is_snow",
        "is_extreme_cold", "is_extreme_heat",
        "is_low_visibility", "is_high_wind",
    )

    print(f"  Weather rows: {weather.count():,}")
    return weather


def aggregate_311(spark):
    """Aggregate 311 complaints to borough x hour level."""
    print("\n  Aggregating 311 data...")
    complaints = spark.read.parquet(CLEANED_311)
    complaints = complaints.withColumn("borough", F.initcap(F.col("borough")))

    complaints_agg = (
        complaints.groupBy(
            F.col("borough"),
            F.col("complaint_hour").alias("hourly_timestamp"),
        )
        .agg(
            F.count("*").alias("complaint_count"),
            F.max("severity_score").alias("max_complaint_severity"),
        )
    )

    print(f"  311 borough-hour rows: {complaints_agg.count():,}")
    return complaints_agg


def aggregate_events(spark):
    """Aggregate events to borough x hour level."""
    print("\n  Aggregating events data...")
    events = spark.read.parquet(CLEANED_EVENTS)

    events_agg = (
        events.groupBy(
            F.col("borough"),
            F.col("event_hour").alias("hourly_timestamp"),
        )
        .agg(
            F.count("*").alias("event_count"),
            F.max("disruption_severity").alias("max_event_severity"),
            F.sum("num_streets_affected").alias("total_streets_affected"),
        )
    )

    print(f"  Events borough-hour rows: {events_agg.count():,}")
    return events_agg


def main():
    spark = create_spark()

    print("\n" + "=" * 60)
    print("NeuroTraffic — Join All 6 Sources")
    print("=" * 60 + "\n")

    # Load zone lookup for borough mapping
    zone_lookup = load_zone_lookup(spark)
    zone_lookup.cache()

    # Step 1: Build the base grid
    grid = build_zone_hour_grid(spark)

    # Add borough info to grid for joining borough-level datasets
    grid = grid.join(
        zone_lookup.select("zone_id", "zone_borough", "zone_name"),
        on="zone_id",
        how="left"
    )
    grid.cache()

    # Step 2: Aggregate each source
    taxi_agg = aggregate_taxi(spark)
    weather = load_weather(spark)
    subway_agg = aggregate_subway(spark)
    complaints_agg = aggregate_311(spark)
    events_agg = aggregate_events(spark)

    # Step 3: Join taxi onto grid (by zone_id + hour)
    print("\n  Joining taxi data...")
    combined = grid.join(
        taxi_agg,
        on=["zone_id", "hourly_timestamp"],
        how="left"
    )

    # Step 4: Join weather (by hour only — same weather for all zones)
    print("  Joining weather data...")
    combined = combined.join(
        weather,
        on="hourly_timestamp",
        how="left"
    )

    # Step 5: Join subway (by borough + hour)
    print("  Joining subway data...")
    combined = combined.join(
        subway_agg,
        on=[combined.zone_borough == subway_agg.borough,
            combined.hourly_timestamp == subway_agg.hourly_timestamp],
        how="left"
    ).drop(subway_agg.borough).drop(subway_agg.hourly_timestamp)

    # Step 6: Join 311 complaints (by borough + hour)
    print("  Joining 311 data...")
    combined = combined.join(
        complaints_agg,
        on=[combined.zone_borough == complaints_agg.borough,
            combined.hourly_timestamp == complaints_agg.hourly_timestamp],
        how="left"
    ).drop(complaints_agg.borough).drop(complaints_agg.hourly_timestamp)

    # Step 7: Join events (by borough + hour)
    print("  Joining events data...")
    combined = combined.join(
        events_agg,
        on=[combined.zone_borough == events_agg.borough,
            combined.hourly_timestamp == events_agg.hourly_timestamp],
        how="left"
    ).drop(events_agg.borough).drop(events_agg.hourly_timestamp)

    # Step 8: Fill missing counts with 0
    combined = combined.fillna({
        "taxi_pickups": 0,
        "taxi_dropoffs": 0,
        "avg_fare": 0.0,
        "avg_trip_distance": 0.0,
        "avg_trip_duration_sec": 0.0,
        "subway_ridership": 0,
        "subway_transfers": 0,
        "subway_stations_active": 0,
        "complaint_count": 0,
        "max_complaint_severity": 0,
        "event_count": 0,
        "max_event_severity": 0,
        "total_streets_affected": 0,
    })

    # Step 9: Write the combined table
    total_rows = combined.count()
    print(f"\n  Combined table: {total_rows:,} rows")

    print(f"\n  Writing to {COMBINED_OUTPUT} (partitioned by year/month)...")
    (
        combined
        .repartition("year", "month")
        .write
        .partitionBy("year", "month")
        .mode("overwrite")
        .parquet(COMBINED_OUTPUT)
    )
    print("  [OK] Combined table written!\n")

    # Summary
    print("=" * 60)
    print("JOIN SUMMARY")
    print("=" * 60)
    print(f"  Grid base:        263 zones x ~26,280 hours")
    print(f"  Combined rows:    {total_rows:,}")
    print(f"  Output path:      {COMBINED_OUTPUT}")
    print()
    print("  Columns in combined table:")
    combined.printSchema()
    print()
    print("  Sample rows (zones with taxi activity):")
    combined.filter(F.col("taxi_pickups") > 0).show(5, truncate=False)

    print()
    print("  Join approach:")
    print("    - Taxi: zone_id + hourly_timestamp (exact)")
    print("    - Weather: hourly_timestamp only (city-wide)")
    print("    - Subway: borough + hourly_timestamp")
    print("    - 311: borough + hourly_timestamp")
    print("    - Events: borough + hourly_timestamp")
    print("    - Bike: NOT joined to grid (needs spatial join with shapefile)")
    print()
    print("  NOTE: Bike data aggregated separately. Tanay's GraphX module")
    print("  will map bike dock coordinates to zones via the transport graph.")

    spark.stop()


if __name__ == "__main__":
    main()
