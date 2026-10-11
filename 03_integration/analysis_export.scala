// NeuroTraffic — Analysis & Visualization Data Export
// Generates summary CSVs for dashboards and charts
// Run: spark-shell -i /project/03_integration/analysis_export.scala

import org.apache.spark.sql.functions._

val HDFS = "hdfs://namenode:9000/data/neurotraffic"
val OUT = s"$HDFS/analysis"

println("=" * 60)
println("  NEUROTRAFFIC ANALYSIS EXPORT")
println("=" * 60)

val df = spark.read.parquet(s"$HDFS/combined_v3")

// 1. Hourly citywide aggregates (for time series charts)
println("\n--- 1. Hourly citywide time series ---")
val hourly = df.groupBy("hourly_timestamp", "hour_of_day", "day_of_week", "is_rush_hour", "is_weekend")
  .agg(
    sum("taxi_pickups").as("total_taxi_pickups"),
    sum("taxi_dropoffs").as("total_taxi_dropoffs"),
    sum("subway_ridership").as("total_subway"),
    sum("bike_starts").as("total_bike_starts"),
    sum("complaint_count").as("total_complaints"),
    sum("event_count").as("total_events"),
    avg("temperature_c").as("avg_temp"),
    max("precipitation_mm").as("max_precip"),
    max("is_rain").as("any_rain"),
    max("is_snow").as("any_snow")
  )
  .orderBy("hourly_timestamp")

hourly.coalesce(1).write.mode("overwrite").option("header", "true").csv(s"$OUT/hourly_citywide")
println(s"  [OK] ${hourly.count()} rows")

// 2. Daily borough aggregates
println("\n--- 2. Daily borough aggregates ---")
val dailyBorough = df
  .withColumn("date", to_date(col("hourly_timestamp")))
  .groupBy("date", "zone_borough")
  .agg(
    sum("taxi_pickups").as("taxi_pickups"),
    sum("subway_ridership").as("subway_ridership"),
    sum("bike_starts").as("bike_starts"),
    sum("complaint_count").as("complaints"),
    avg("temperature_c").as("avg_temp")
  )
  .orderBy("date", "zone_borough")

dailyBorough.coalesce(1).write.mode("overwrite").option("header", "true").csv(s"$OUT/daily_borough")
println(s"  [OK] ${dailyBorough.count()} rows")

// 3. Zone heatmap data (avg pickups per zone)
println("\n--- 3. Zone heatmap data ---")
val zoneHeat = df.groupBy("zone_id", "zone_name", "zone_borough")
  .agg(
    avg("taxi_pickups").as("avg_pickups"),
    sum("taxi_pickups").as("total_pickups"),
    avg("subway_ridership").as("avg_subway"),
    avg("bike_starts").as("avg_bike"),
    avg("complaint_count").as("avg_complaints")
  )
  .orderBy(desc("total_pickups"))

zoneHeat.coalesce(1).write.mode("overwrite").option("header", "true").csv(s"$OUT/zone_heatmap")
println(s"  [OK] ${zoneHeat.count()} rows")

// 4. Weather impact analysis
println("\n--- 4. Weather impact on ridership ---")
val weatherImpact = df.groupBy("is_rain", "is_snow", "is_heavy_rain", "is_extreme_cold", "is_extreme_heat")
  .agg(
    count("*").as("zone_hours"),
    avg("taxi_pickups").as("avg_taxi"),
    avg("subway_ridership").as("avg_subway"),
    avg("bike_starts").as("avg_bike")
  )
  .orderBy(desc("zone_hours"))

weatherImpact.coalesce(1).write.mode("overwrite").option("header", "true").csv(s"$OUT/weather_impact")
println(s"  [OK] ${weatherImpact.count()} rows")

// 5. Rush hour vs off-peak comparison
println("\n--- 5. Rush hour analysis ---")
val rushAnalysis = df.groupBy("is_rush_hour", "is_weekend", "zone_borough")
  .agg(
    avg("taxi_pickups").as("avg_pickups"),
    avg("subway_ridership").as("avg_subway"),
    avg("bike_starts").as("avg_bike"),
    avg("avg_fare").as("avg_fare"),
    avg("avg_trip_distance").as("avg_distance")
  )
  .orderBy("zone_borough", "is_weekend", "is_rush_hour")

rushAnalysis.coalesce(1).write.mode("overwrite").option("header", "true").csv(s"$OUT/rush_hour")
println(s"  [OK] ${rushAnalysis.count()} rows")

// 6. Monthly trends
println("\n--- 6. Monthly trends ---")
val monthly = df.groupBy("year", "month")
  .agg(
    sum("taxi_pickups").as("total_taxi"),
    sum("subway_ridership").as("total_subway"),
    sum("bike_starts").as("total_bike"),
    sum("complaint_count").as("total_complaints"),
    avg("temperature_c").as("avg_temp")
  )
  .orderBy("year", "month")

monthly.coalesce(1).write.mode("overwrite").option("header", "true").csv(s"$OUT/monthly_trends")
println(s"  [OK] ${monthly.count()} rows")

// 7. Correlation matrix data
println("\n--- 7. Correlation pairs ---")
val corrCols = Array("taxi_pickups", "subway_ridership", "bike_starts",
  "complaint_count", "temperature_c", "precipitation_mm", "wind_speed_kmh")

println("  Correlation matrix:")
for (a <- corrCols; b <- corrCols if a < b) {
  val corr = df.stat.corr(a, b)
  println(f"    $a%-20s x $b%-20s = $corr%.4f")
}

println("\n" + "=" * 60)
println("  ANALYSIS EXPORT COMPLETE")
println("=" * 60)

System.exit(0)
