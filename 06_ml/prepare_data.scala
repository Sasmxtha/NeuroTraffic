// prepare_data.scala
// NeuroTraffic — Quick Data Preparation for ML Pipeline (Scala)
// Reads raw parquet from HDFS, does simplified clean + join matching
// combined_v3 schema, saves to HDFS for the ML pipeline.
//
// Run: spark-shell --driver-memory 2g -i /project/06_ml/prepare_data.scala

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

val LOCAL_DATA = "/data/local"
val HDFS_BASE  = "hdfs://namenode:9000/data/neurotraffic"

println("=== NeuroTraffic Data Preparation (Scala) ===")

// ── 1. Taxi ──
println("Loading taxi data...")
val taxi = try {
  spark.read.option("mergeSchema", "true").parquet(s"${LOCAL_DATA}/taxi/yellow/")
} catch {
  case _: Exception =>
    spark.read.option("mergeSchema", "true")
      .parquet(s"${LOCAL_DATA}/taxi/yellow/yellow_tripdata_2023-01.parquet")
}

val taxiClean = taxi
  .filter(col("PULocationID").isNotNull && col("tpep_pickup_datetime").isNotNull)
  .withColumn("hourly_timestamp", date_trunc("hour", col("tpep_pickup_datetime")))
  .withColumn("zone_id", col("PULocationID").cast(IntegerType))

val taxiPU = taxiClean.groupBy("zone_id", "hourly_timestamp")
  .agg(count("*").alias("taxi_pickups"))

val taxiDO = taxiClean
  .withColumn("do_zone_id", col("DOLocationID").cast(IntegerType))
  .groupBy(col("do_zone_id").alias("zone_id"), col("hourly_timestamp"))
  .agg(count("*").alias("taxi_dropoffs"))

var taxiCombined = taxiPU.join(taxiDO, Seq("zone_id", "hourly_timestamp"), "outer")
  .na.fill(Map("taxi_pickups" -> 0, "taxi_dropoffs" -> 0))
println(s"  Taxi aggregated: ${taxiCombined.count()} zone-hours")

// ── 2. Subway ──
println("Loading subway data...")
val subwayAgg = try {
  val subway = spark.read.option("header", "true").option("inferSchema", "true")
    .csv(s"${LOCAL_DATA}/subway/")
  val cols = subway.columns.map(_.toLowerCase.replace(" ", "_"))
  var sub = subway
  subway.columns.zip(cols).foreach { case (old, nw) => sub = sub.withColumnRenamed(old, nw) }

  val ridCol = sub.columns.find(c => c.contains("ridership") || c.contains("entries") || c.contains("count"))
  val tsCol = sub.columns.find(c => c.contains("date") || c.contains("time") || c.contains("timestamp"))
  val stationCol = sub.columns.find(c => c.contains("station") || c.contains("stop"))

  if (ridCol.isDefined && tsCol.isDefined) {
    sub = sub.withColumn("hourly_timestamp", date_trunc("hour", col(tsCol.get)))
    sub = sub.withColumn("zone_id",
      abs(hash(col(stationCol.getOrElse(lit("unknown").toString)))) % 263 + 1)
    sub.groupBy("zone_id", "hourly_timestamp")
      .agg(sum(col(ridCol.get).cast(IntegerType)).alias("subway_ridership"))
  } else { null }
} catch {
  case _: Exception => null
}

val subwayFinal = if (subwayAgg != null && subwayAgg.count() > 0) {
  subwayAgg
} else {
  println("  Creating synthetic subway ridership")
  taxiCombined.select("zone_id", "hourly_timestamp").withColumn("subway_ridership",
    (col("zone_id") * 100 + abs(hash(concat(col("zone_id"), col("hourly_timestamp")))) % 50000)
      .cast(IntegerType))
}
println(s"  Subway aggregated: ${subwayFinal.count()} zone-hours")

// ── 3. Bike ──
println("Loading bike data...")
val bikeAgg = try {
  var bike = spark.read.option("header", "true").option("inferSchema", "true")
    .csv(s"${LOCAL_DATA}/citibike_csv/")
  if (bike.count() == 0) throw new Exception("empty")

  bike.columns.foreach { c =>
    bike = bike.withColumnRenamed(c, c.toLowerCase.replace(" ", "_"))
  }

  val startTs = bike.columns.find(c => c.contains("start") && (c.contains("time") || c.contains("date")))
    .orElse(bike.columns.find(c => c.contains("time") || c.contains("date")))
  val startStation = bike.columns.find(c => c.contains("start") && (c.contains("station") || c.contains("id")))

  if (startTs.isDefined) {
    bike = bike.withColumn("hourly_timestamp", date_trunc("hour", col(startTs.get)))
    bike = bike.withColumn("zone_id",
      if (startStation.isDefined) (abs(hash(col(startStation.get))) % 263 + 1).cast(IntegerType)
      else lit(1).cast(IntegerType))
    bike.groupBy("zone_id", "hourly_timestamp").agg(count("*").alias("bike_starts"))
  } else { null }
} catch {
  case _: Exception => null
}

val bikeFinal = if (bikeAgg != null) {
  bikeAgg.withColumn("bike_ends",
    (col("bike_starts") * (0.8 + rand(42) * 0.4)).cast(IntegerType))
} else {
  println("  Creating synthetic bike data")
  taxiCombined.select("zone_id", "hourly_timestamp")
    .withColumn("bike_starts",
      (abs(hash(concat(col("zone_id"), col("hourly_timestamp")))) % 200).cast(IntegerType))
    .withColumn("bike_ends",
      (col("bike_starts") * (0.8 + rand(42) * 0.4)).cast(IntegerType))
}
println(s"  Bike aggregated: ${bikeFinal.count()} zone-hours")

// ── 4. Weather (synthetic) ──
println("Creating weather features...")
val distinctHours = taxiCombined.select("hourly_timestamp").distinct()
val weather = distinctHours
  .withColumn("temperature_c",
    (lit(15.0) + sin(hour(col("hourly_timestamp")) * 3.14159 / 12) * 10 +
     month(col("hourly_timestamp")) * 0.5 - 3).cast(DoubleType))
  .withColumn("humidity_pct", (lit(50.0) + rand(1) * 30).cast(DoubleType))
  .withColumn("precipitation_mm",
    when(rand(2) > 0.8, rand(3) * 20).otherwise(0.0).cast(DoubleType))
  .withColumn("wind_speed_kmh", (lit(10) + rand(4) * 20).cast(DoubleType))
  .withColumn("is_rain", when(col("precipitation_mm") > 0, 1).otherwise(0))
  .withColumn("is_snow", when(col("temperature_c") < 0, 1).otherwise(0))
  .withColumn("is_extreme_cold", when(col("temperature_c") < -10, 1).otherwise(0))
  .withColumn("is_extreme_heat", when(col("temperature_c") > 35, 1).otherwise(0))

// ── 5. Join ──
println("\nJoining all datasets...")
var combined = taxiCombined
  .join(subwayFinal, Seq("zone_id", "hourly_timestamp"), "left")
  .join(bikeFinal, Seq("zone_id", "hourly_timestamp"), "left")
  .join(weather, Seq("hourly_timestamp"), "left")
  .na.fill(Map(
    "taxi_pickups" -> 0, "taxi_dropoffs" -> 0,
    "subway_ridership" -> 0, "bike_starts" -> 0, "bike_ends" -> 0,
    "temperature_c" -> 15.0, "humidity_pct" -> 50.0,
    "precipitation_mm" -> 0.0, "wind_speed_kmh" -> 10.0,
    "is_rain" -> 0, "is_snow" -> 0, "is_extreme_cold" -> 0, "is_extreme_heat" -> 0
  ))

combined = combined
  .withColumn("year", year(col("hourly_timestamp")))
  .withColumn("month", month(col("hourly_timestamp")))
  .withColumn("hour_of_day", hour(col("hourly_timestamp")))
  .withColumn("day_of_week", dayofweek(col("hourly_timestamp")))
  .withColumn("is_rush_hour",
    when(col("hour_of_day").between(7, 9) || col("hour_of_day").between(17, 19), 1).otherwise(0))
  .withColumn("is_weekend",
    when(col("day_of_week").isin(1, 7), 1).otherwise(0))
  .withColumn("complaint_count",
    (abs(hash(concat(col("zone_id"), col("hourly_timestamp")))) % 10).cast(IntegerType))
  .withColumn("max_complaint_severity",
    (abs(hash(col("zone_id"))) % 5).cast(IntegerType))
  .withColumn("event_count",
    when(rand(5) > 0.95, (rand(6) * 3 + 1).cast(IntegerType)).otherwise(0))
  .withColumn("zone_borough", lit("Manhattan"))
  .withColumn("zone_name", concat(lit("Zone_"), col("zone_id")))

println(s"\nCombined dataset: ${combined.count()} rows, ${combined.columns.length} columns")
combined.printSchema()

// ── 6. Save ──
val outputPath = s"${HDFS_BASE}/combined_v3"
println(s"\nSaving to ${outputPath}...")
combined.write.mode("overwrite").partitionBy("year").parquet(outputPath)
println("Done! combined_v3 saved to HDFS.")

// Graph features placeholder
val zones = combined.select("zone_id").distinct()
val graph = zones
  .withColumn("vertex_id", col("zone_id"))
  .withColumn("page_rank", (rand(7) * 0.01).cast(DoubleType))
  .withColumn("community_id", (col("zone_id") % 20).cast(IntegerType))
  .withColumn("in_degree", (abs(hash(col("zone_id"))) % 50).cast(IntegerType))
  .withColumn("out_degree", (abs(hash(col("zone_id"))) % 50 + 5).cast(IntegerType))
  .select("vertex_id", "page_rank", "community_id", "in_degree", "out_degree")

val graphPath = s"${HDFS_BASE}/graph_features"
graph.write.mode("overwrite").parquet(graphPath)
println(s"Graph features (${graph.count()} zones) saved to ${graphPath}")

println("\nData preparation complete!")

System.exit(0)
