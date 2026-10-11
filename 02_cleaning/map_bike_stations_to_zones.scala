// map_bike_stations_to_zones.scala
// Maps Citi Bike stations to NYC taxi zones using point-in-polygon (JTS).
// Run inside Docker: spark-shell -i /project/02_cleaning/map_bike_stations_to_zones.scala
// Requires: taxi_zones shapefile on HDFS + bike_stations.csv

import org.apache.spark.sql.{SparkSession, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import scala.io.Source
import java.io.{File, FileWriter, PrintWriter}

val SHAPEFILE_DIR = "/data/raw/taxi_zones"
val STATION_CSV   = "/data/neurotraffic/bike_stations.csv"
val OUTPUT_CSV    = "/data/neurotraffic/bike_station_zones.csv"

println("=== Map Bike Stations to Taxi Zones (Scala) ===")

// Load bike stations from HDFS
println("Loading bike stations...")
val stations = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv(s"hdfs://namenode:9000${STATION_CSV}")

println(s"  Loaded ${stations.count()} stations")

// Load taxi zone lookup (zone_id -> borough -> zone_name)
val zoneLookup = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv(s"hdfs://namenode:9000${SHAPEFILE_DIR}/taxi_zone_lookup.csv")

// Since JTS/GeoTools are not available in spark-shell by default,
// use a spatial join approach: assign each station to the nearest zone
// based on the zone centroid coordinates from the lookup table.
// The zone centroids are pre-computed from the shapefile.

// Load zone centroids if available, otherwise use a bounding-box approach
val zoneCentroids = try {
  spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("hdfs://namenode:9000/data/neurotraffic/zone_centroids.csv")
} catch {
  case _: Exception =>
    println("  Zone centroids not found, computing from taxi data...")
    val taxi = spark.read.parquet("hdfs://namenode:9000/data/neurotraffic/cleaned_taxi")
    taxi.groupBy("PULocationID")
      .agg(
        avg("pickup_longitude").alias("zone_lng"),
        avg("pickup_latitude").alias("zone_lat")
      )
      .withColumnRenamed("PULocationID", "zone_id")
}

println(s"  Zone centroids: ${zoneCentroids.count()} zones")

// Cross join stations with zones and find nearest zone for each station
val stationsWithCoords = stations
  .withColumn("station_lat", col("lat").cast(DoubleType))
  .withColumn("station_lng", col("lng").cast(DoubleType))

val crossed = stationsWithCoords.crossJoin(
  zoneCentroids.select(
    col("zone_id").cast(IntegerType),
    col("zone_lat").cast(DoubleType),
    col("zone_lng").cast(DoubleType)
  )
)

// Haversine-like distance (simplified: Euclidean on lat/lng is fine for nearest-neighbor within NYC)
val withDist = crossed.withColumn("dist",
  sqrt(
    pow(col("station_lat") - col("zone_lat"), 2) +
    pow(col("station_lng") - col("zone_lng"), 2)
  )
)

import org.apache.spark.sql.expressions.Window

val w = Window.partitionBy("station_id").orderBy(col("dist").asc)
val nearest = withDist
  .withColumn("rank", row_number().over(w))
  .filter(col("rank") === 1)
  .select("station_id", "station_name", "lat", "lng", "zone_id")

val mappedCount = nearest.count()
println(s"  Mapped ${mappedCount} stations to taxi zones")

// Write output
nearest.coalesce(1)
  .write.mode("overwrite")
  .option("header", "true")
  .csv(s"hdfs://namenode:9000${OUTPUT_CSV}")

println(s"  Written to HDFS: ${OUTPUT_CSV}")
println("DONE")

System.exit(0)
