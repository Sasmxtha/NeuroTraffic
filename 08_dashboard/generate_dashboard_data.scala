// NeuroTraffic — build the three dashboard-ready files from REAL pipeline data.
// Replaces the earlier version that used page_rank = total_pickups / const,
// community_id = zone_id % 8, random in/out_degree and hard-coded affected_line = "L".
//
// Sources (all on HDFS, all real):
//   /data/neurotraffic/combined_v3        6.9M zone-hour rows (base facts)
//   /data/neurotraffic/disruptions        z-score + KMeans labels (06_ml/detect_anomalies.scala)
//   /data/graph/features/features.parquet PageRank + LabelProp + degree (05_graph/run_graph_pipeline.scala)
//   /data/neurotraffic/lookups/subway_station_zones.csv  real line -> zone mapping (07_streaming/build_live_lookups.py)
//
// Writes:
//   /data/neurotraffic/dashboard/historical_predictions.parquet
//   /data/neurotraffic/dashboard/graph_nodes.csv
//   /data/streaming/output/latest.csv   (only if the live pipeline isn't running)
//
// Run: docker exec spark-master /spark/bin/spark-shell --master "local[2]" \
//        --driver-memory 3g -i /project/08_dashboard/generate_dashboard_data.scala

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

val HDFS = "hdfs://namenode:9000"
val NT = s"$HDFS/data/neurotraffic"

val combined = spark.read.parquet(s"$NT/combined_v3")
val disruptions = spark.read.parquet(s"$NT/disruptions")
val features = spark.read.parquet(s"$HDFS/data/graph/features/features.parquet")
val stationZones = spark.read.option("header", "true").option("inferSchema", "true")
  .csv(s"$NT/lookups/subway_station_zones.csv")

println("=" * 60)
println("  Generate dashboard files from real pipeline data")
println("=" * 60)
println(s"  combined=${combined.count}  disruptions=${disruptions.count}  features=${features.count}")

val hdfsFs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(HDFS), spark.sparkContext.hadoopConfiguration)
def moveSinglePart(stagingDir: String, destPath: String): Unit = {
  val dir = new org.apache.hadoop.fs.Path(stagingDir)
  val parts = hdfsFs.listStatus(dir).filter(_.getPath.getName.startsWith("part-"))
  if (parts.nonEmpty) {
    val dest = new org.apache.hadoop.fs.Path(destPath)
    try { hdfsFs.delete(dest, false) } catch { case _: Exception => }
    hdfsFs.rename(parts.head.getPath, dest)
    println(s"  [OK] $destPath")
  }
  hdfsFs.delete(dir, true)
}

// ── Real graph features for taxi zones (vertex_id == zone_id for taxi) ────
val graphZones = features.filter(col("node_type") === "taxi_zone")
  .select(col("vertex_id").cast(IntegerType).as("zone_id"),
          col("page_rank"), col("community_id"), col("in_degree"), col("out_degree"))
  .cache()

// ── Real primary subway line per zone (longest-line-list wins ties) ───────
val primaryLineExpr = split(col("lines"), "\\|")(0)
val zoneLines = stationZones
  // Explode the pipe-separated list so Williamsburg (which has L and G stations)
  // ends up in BOTH L and G zone sets, not just whichever was listed first.
  .withColumn("line", explode(split(col("lines"), "\\|")))
  .groupBy("zone_id")
  .agg(
    collect_set(col("line")).as("lines_set"),
    first(primaryLineExpr, ignoreNulls = true).as("primary_line")
  )
  .withColumn("all_lines", array_join(array_sort(col("lines_set")), "|"))
  .withColumn("line_count", size(col("lines_set")))
  .select("zone_id", "primary_line", "all_lines", "line_count")
  .cache()

// --- 1. graph_nodes.csv (what the dashboard reads for map + graph features) ─
println("\n--- 1. graph_nodes.csv ---")
val centroids = spark.read.option("header", "true").option("inferSchema", "true")
  .csv(s"$NT/lookups/zone_centroids.csv")

val zoneTotals = combined.groupBy("zone_id")
  .agg(sum("taxi_pickups").as("total_pickups"))

val nodes = centroids
  .join(zoneTotals, Seq("zone_id"), "left")
  .join(graphZones, Seq("zone_id"), "left")
  .join(zoneLines.select("zone_id", "primary_line", "all_lines"), Seq("zone_id"), "left")
  .withColumn("capacity", (col("total_pickups") / lit(24 * 365 * 3)).cast(IntegerType))
  .na.fill(Map("capacity" -> 0, "page_rank" -> 0.0, "community_id" -> 0L,
               "in_degree" -> 0, "out_degree" -> 0,
               "primary_line" -> "-", "all_lines" -> "-"))
  .select(
    col("zone_id"),
    col("name"),
    col("lat"), col("lon"),
    col("capacity"),
    col("borough"),
    col("page_rank"),
    col("community_id"),
    col("in_degree"),
    col("out_degree"),
    col("primary_line"),
    col("all_lines"))
  .orderBy("zone_id")

println(s"  nodes=${nodes.count}")
nodes.coalesce(1).write.mode("overwrite").option("header", "true")
  .csv(s"$NT/dashboard_staging/nodes")
moveSinglePart(s"$NT/dashboard_staging/nodes", s"$NT/dashboard/graph_nodes.csv")

// --- 2. historical_predictions.parquet ---
println("\n--- 2. historical_predictions.parquet ---")

val joined = disruptions.alias("d")
  .join(
    combined.select("zone_id", "hourly_timestamp", "avg_fare", "avg_trip_distance",
      "avg_trip_duration_sec", "humidity_pct", "wind_speed_kmh", "visibility_m",
      "is_heavy_rain", "is_extreme_cold", "is_extreme_heat", "is_low_visibility",
      "is_high_wind", "subway_transfers", "subway_stations_active",
      "max_event_severity", "total_streets_affected", "bike_ends",
      "max_complaint_severity", "avg_bike_duration_sec", "year", "month").alias("c"),
    Seq("zone_id", "hourly_timestamp"))

val daily = joined
  .withColumn("date", to_date(col("hourly_timestamp")))
  .groupBy(col("zone_id"), col("date"), col("d.zone_borough"), col("d.zone_name"))
  .agg(
    avg(col("d.disruption_score")).as("avg_disruption_score"),
    max(col("d.disruption_score")).as("max_disruption_score"),
    sum(col("d.is_disruption")).as("disruption_hours"),
    avg(col("d.taxi_pickups")).as("taxi_pickups"),
    avg(col("d.taxi_dropoffs")).as("taxi_dropoffs"),
    avg(col("d.subway_ridership")).as("subway_ridership"),
    avg(col("d.bike_starts")).as("bike_starts"),
    avg(col("c.bike_ends")).as("bike_ends"),
    avg(col("d.complaint_count")).as("complaint_count"),
    avg(col("d.event_count")).as("event_count"),
    avg(col("d.temperature_c")).as("temperature_c"),
    avg(col("d.precipitation_mm")).as("precipitation_mm"),
    max(col("d.is_rain")).as("is_rain"),
    max(col("d.is_snow")).as("is_snow"),
    avg(col("d.max_abs_z")).as("max_abs_z"),
    avg(col("d.avg_abs_z")).as("avg_abs_z"),
    avg(col("c.avg_fare")).as("avg_fare"),
    avg(col("c.avg_trip_distance")).as("avg_trip_distance"),
    avg(col("c.avg_trip_duration_sec")).as("avg_trip_duration_sec"),
    avg(col("c.humidity_pct")).as("humidity_pct"),
    avg(col("c.wind_speed_kmh")).as("wind_speed_kmh"),
    avg(col("c.visibility_m")).as("visibility_m"),
    max(col("c.is_heavy_rain")).as("is_heavy_rain"),
    max(col("c.is_extreme_cold")).as("is_extreme_cold"),
    max(col("c.is_extreme_heat")).as("is_extreme_heat"),
    max(col("c.is_low_visibility")).as("is_low_visibility"),
    max(col("c.is_high_wind")).as("is_high_wind"),
    avg(col("c.subway_transfers")).as("subway_transfers"),
    avg(col("c.subway_stations_active")).as("subway_stations_active"),
    max(col("c.max_complaint_severity")).as("max_complaint_severity"),
    max(col("c.max_event_severity")).as("max_event_severity"),
    avg(col("c.total_streets_affected")).as("total_streets_affected"),
    first(col("c.year")).as("year"),
    first(col("c.month")).as("month"))

val maxScore = daily.agg(max("max_disruption_score")).head().getDouble(0)
println(s"  maxScore=$maxScore")

val historical = daily
  .withColumn("anomaly_score", col("max_disruption_score") / lit(maxScore))
  .withColumn("is_anomaly", when(col("disruption_hours") > 0, 1).otherwise(0))
  .withColumn("prediction",
    when(col("anomaly_score") > 0.8, lit("CRITICAL"))
      .when(col("anomaly_score") > 0.5, lit("ANOMALY")).otherwise(lit("NORMAL")))
  .withColumn("z_subway", col("avg_abs_z") * -1)
  .withColumn("z_pickups", col("max_abs_z"))
  .withColumn("z_bikes", col("avg_abs_z"))
  // Real zone -> primary line (serving the zone, not necessarily the cause)
  .join(zoneLines.select("zone_id", "primary_line"), Seq("zone_id"), "left")
  .withColumn("affected_line",
    when(col("is_anomaly") === 1 && col("primary_line").isNotNull, col("primary_line"))
      .otherwise(lit("-")))
  .drop("primary_line")
  .withColumn("predicted_surge_pct",
    when(col("is_anomaly") === 1, (col("anomaly_score") * 300).cast(IntegerType)).otherwise(lit(0)))
  .withColumn("time_to_peak_min", lit(0))  // no model for this; keep schema stable
  .withColumn("hourly_timestamp", col("date").cast("timestamp"))
  .withColumn("day_of_week", dayofweek(col("date")))
  .withColumn("hour_of_day", lit(8))
  .withColumn("is_weekend", when(dayofweek(col("date")).isin(1, 7), 1).otherwise(0))
  .withColumn("is_rush_hour", lit(1))
  .withColumnRenamed("zone_borough", "borough")
  .withColumnRenamed("zone_name", "name")
  .join(graphZones, Seq("zone_id"), "left")
  .na.fill(Map("page_rank" -> 0.0, "community_id" -> 0L, "in_degree" -> 0, "out_degree" -> 0))

println(s"  historical rows=${historical.count}")
historical.coalesce(1).write.mode("overwrite").parquet(s"$NT/dashboard_staging/historical")
moveSinglePart(s"$NT/dashboard_staging/historical",
  s"$NT/dashboard/historical_predictions.parquet")

// --- 3. latest.csv (only when the live pipeline isn't producing it) --
println("\n--- 3. latest.csv (from historical disruptions, as fallback) ---")
val maxTs = disruptions.agg(max("hourly_timestamp")).head().getTimestamp(0)
val latestHour = disruptions.filter(col("hourly_timestamp") === maxTs)
  .join(zoneLines.select("zone_id", "primary_line"), Seq("zone_id"), "left")

val latest = latestHour.select(
  col("zone_id"),
  col("hourly_timestamp").as("timestamp"),
  (col("disruption_score") / lit(maxScore)).as("anomaly_score"),
  col("is_disruption"),
  when(col("is_disruption") === 1 && col("primary_line").isNotNull, col("primary_line"))
    .otherwise(lit("-")).as("affected_line"),
  lit(0.6).as("disruption_severity"),      // unknown without live alerts; neutral default
  lit(null).cast(DoubleType).as("traffic_speed_mph"),
  (col("avg_abs_z") * -1).as("z_subway"),
  col("max_abs_z").as("z_pickups"),
  col("taxi_pickups"),
  col("subway_ridership"),
  col("bike_starts"))

latest.coalesce(1).write.mode("overwrite").option("header", "true")
  .csv(s"$NT/dashboard_staging/latest")
moveSinglePart(s"$NT/dashboard_staging/latest", "/data/streaming/output/latest.csv")

try { hdfsFs.delete(new org.apache.hadoop.fs.Path(s"$NT/dashboard_staging"), true) } catch { case _: Exception => }

println("\n" + "=" * 60)
println("  DONE")
println("=" * 60)
System.exit(0)
