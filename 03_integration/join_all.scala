// NeuroTraffic — Scala Spark Join All 6 Sources
// Run: spark-shell -i /project/03_integration/join_all.scala

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

val HDFS = "hdfs://namenode:9000/data/neurotraffic"

println("=" * 60)
println("NeuroTraffic — Join All 6 Sources (Scala)")
println("=" * 60)

// Load zone lookup
val zones = spark.read
  .option("header", "true").option("inferSchema", "true")
  .csv(s"$HDFS/raw/taxi/zones/taxi_zone_lookup.csv")
  .select(
    col("LocationID").cast(IntegerType).as("zone_id"),
    col("Borough").as("zone_borough"),
    col("Zone").as("zone_name"),
    col("service_zone")
  )
zones.cache()

// Build the zone x hour grid: 263 zones x every hour in 2022-2024
println("  Building zone x hour grid...")
val zoneIds = spark.range(1, 264).toDF("zone_id")
  .withColumn("zone_id", col("zone_id").cast(IntegerType))

val hours = spark.sql("""
  SELECT explode(sequence(
    timestamp('2022-01-01 00:00:00'),
    timestamp('2024-12-31 23:00:00'),
    interval 1 hour
  )) as hourly_timestamp
""")

var grid = zoneIds.crossJoin(hours)
  .withColumn("year",        year(col("hourly_timestamp")))
  .withColumn("month",       month(col("hourly_timestamp")))
  .withColumn("day_of_week", dayofweek(col("hourly_timestamp")))
  .withColumn("hour_of_day", hour(col("hourly_timestamp")))
  .withColumn("is_rush_hour",
    (col("hour_of_day").between(7, 9) || col("hour_of_day").between(17, 19)).cast("int"))
  .withColumn("is_weekend",
    col("day_of_week").isin(1, 7).cast("int"))

// Add borough info
grid = grid.join(
  zones.select("zone_id", "zone_borough", "zone_name"),
  Seq("zone_id"), "left"
)
grid.cache()
println(s"  Grid size: ${grid.count()} rows")

// Aggregate taxi
println("  Aggregating taxi data...")
val taxi = spark.read.parquet(
  s"$HDFS/cleaned/taxi/yellow",
  s"$HDFS/cleaned/taxi/green",
  s"$HDFS/cleaned/taxi/hvfhv"
)

val pickups = taxi.groupBy(
    col("pu_location_id").as("zone_id"),
    col("pickup_hour").as("hourly_timestamp")
  ).agg(
    count("*").as("taxi_pickups"),
    avg("fare_amount").as("avg_fare"),
    avg("trip_distance").as("avg_trip_distance"),
    avg("trip_duration_sec").as("avg_trip_duration_sec")
  )

val dropoffs = taxi.groupBy(
    col("do_location_id").as("zone_id"),
    date_trunc("hour", col("dropoff_datetime")).as("hourly_timestamp")
  ).agg(count("*").as("taxi_dropoffs"))

val taxiAgg = pickups.join(dropoffs, Seq("zone_id", "hourly_timestamp"), "outer")

// Load weather
println("  Loading weather...")
val weather = spark.read.parquet(s"$HDFS/cleaned/weather")
  .select(
    col("weather_hour").as("hourly_timestamp"),
    col("temperature_c"), col("humidity_pct"), col("precipitation_mm"),
    col("wind_speed_kmh"), col("visibility_m"),
    col("is_rain"), col("is_heavy_rain"), col("is_snow"),
    col("is_extreme_cold"), col("is_extreme_heat"),
    col("is_low_visibility"), col("is_high_wind")
  )

// Aggregate subway by borough + hour
println("  Aggregating subway...")
val subway = spark.read.parquet(s"$HDFS/cleaned/subway")
val subwayAgg = subway.groupBy(
    col("borough"), col("transit_hour").as("hourly_timestamp")
  ).agg(
    sum("total_ridership").as("subway_ridership"),
    sum("total_transfers").as("subway_transfers"),
    countDistinct("station_complex_id").as("subway_stations_active")
  )

// Aggregate 311 by borough + hour
println("  Aggregating 311...")
val complaints = spark.read.parquet(s"$HDFS/cleaned/311")
  .withColumn("borough", initcap(col("borough")))
val complaintsAgg = complaints.groupBy(
    col("borough"), col("complaint_hour").as("hourly_timestamp")
  ).agg(
    count("*").as("complaint_count"),
    max("severity_score").as("max_complaint_severity")
  )

// Aggregate events by borough + hour
println("  Aggregating events...")
val events = spark.read.parquet(s"$HDFS/cleaned/events")
val eventsAgg = events.groupBy(
    col("borough"), col("event_hour").as("hourly_timestamp")
  ).agg(
    count("*").as("event_count"),
    max("disruption_severity").as("max_event_severity"),
    sum("num_streets_affected").as("total_streets_affected")
  )

// Join everything onto the grid
println("  Joining taxi...")
var combined = grid.join(taxiAgg, Seq("zone_id", "hourly_timestamp"), "left")

println("  Joining weather...")
combined = combined.join(weather, Seq("hourly_timestamp"), "left")

println("  Joining subway...")
combined = combined.join(subwayAgg,
  combined("zone_borough") === subwayAgg("borough") &&
  combined("hourly_timestamp") === subwayAgg("hourly_timestamp"), "left")
  .drop(subwayAgg("borough")).drop(subwayAgg("hourly_timestamp"))

println("  Joining 311...")
combined = combined.join(complaintsAgg,
  combined("zone_borough") === complaintsAgg("borough") &&
  combined("hourly_timestamp") === complaintsAgg("hourly_timestamp"), "left")
  .drop(complaintsAgg("borough")).drop(complaintsAgg("hourly_timestamp"))

println("  Joining events...")
combined = combined.join(eventsAgg,
  combined("zone_borough") === eventsAgg("borough") &&
  combined("hourly_timestamp") === eventsAgg("hourly_timestamp"), "left")
  .drop(eventsAgg("borough")).drop(eventsAgg("hourly_timestamp"))

// Fill nulls with 0
combined = combined.na.fill(Map(
  "taxi_pickups" -> 0, "taxi_dropoffs" -> 0,
  "avg_fare" -> 0.0, "avg_trip_distance" -> 0.0, "avg_trip_duration_sec" -> 0.0,
  "subway_ridership" -> 0, "subway_transfers" -> 0, "subway_stations_active" -> 0,
  "complaint_count" -> 0, "max_complaint_severity" -> 0,
  "event_count" -> 0, "max_event_severity" -> 0, "total_streets_affected" -> 0
))

val totalRows = combined.count()
println(s"\n  Combined table: $totalRows rows")

combined.repartition(col("year"), col("month"))
  .write.partitionBy("year", "month")
  .mode("overwrite")
  .parquet(s"$HDFS/combined")

println("  [OK] Combined table written!")
combined.printSchema()

System.exit(0)
