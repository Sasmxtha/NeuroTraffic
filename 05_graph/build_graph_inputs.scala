// NeuroTraffic — build real graph node & edge Parquet from source data
// Produces the layout GraphBuilder.scala expects at /data/graph/
//
// Run: docker exec spark-master /spark/bin/spark-shell --master "local[2]" \
//        --driver-memory 3g -i /project/05_graph/build_graph_inputs.scala

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

val HDFS = "hdfs://namenode:9000"
val GRAPH = s"$HDFS/data/graph"
val fs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(HDFS), spark.sparkContext.hadoopConfiguration)

println("=" * 60)
println("  BUILD REAL GRAPH INPUTS")
println("=" * 60)

// --- 1. Taxi zones (263 nodes) ----------------------------------------
println("\n[1/4] Taxi zone nodes")
val centroids = spark.read.option("header", "true").option("inferSchema", "true")
  .csv(s"$HDFS/data/neurotraffic/lookups/zone_centroids.csv")

// Capacity = total taxi pickups across history (real activity volume)
val zoneCapacity = spark.read.parquet(s"$HDFS/data/neurotraffic/combined_v3")
  .groupBy("zone_id").agg(sum("taxi_pickups").cast(IntegerType).as("capacity"))

val taxiZones = centroids
  .join(zoneCapacity, Seq("zone_id"), "left")
  .na.fill(Map("capacity" -> 0))
  .withColumnRenamed("name", "zone_name")
  .select("zone_id", "zone_name", "lat", "lon", "capacity", "borough")

taxiZones.write.mode("overwrite").parquet(s"$GRAPH/nodes/taxi_zones.parquet")
println(s"  wrote ${taxiZones.count} taxi zones")

// --- 2. Subway stations (one row per complex) -------------------------
println("\n[2/4] Subway station nodes")
val stationZones = spark.read.option("header", "true").option("inferSchema", "true")
  .csv(s"$HDFS/data/neurotraffic/lookups/subway_station_zones.csv")

// Capacity = total subway ridership in the station's zone
val zoneSubway = spark.read.parquet(s"$HDFS/data/neurotraffic/combined_v3")
  .groupBy("zone_id").agg(sum("subway_ridership").as("zone_subway"))

val subwayStations = stationZones
  .join(zoneSubway, Seq("zone_id"), "left")
  .join(centroids.select(col("zone_id"), col("borough")), Seq("zone_id"), "left")
  .withColumn("capacity", (col("zone_subway") / 10).cast(IntegerType))  // rough per-station share
  .na.fill(Map("capacity" -> 1000))
  .withColumnRenamed("complex_id", "station_id")
  .withColumnRenamed("name", "station_name")
  .select("station_id", "station_name", "lat", "lon", "capacity", "borough")

subwayStations.write.mode("overwrite").parquet(s"$GRAPH/nodes/subway_stations.parquet")
println(s"  wrote ${subwayStations.count} subway stations")

// --- 3. Bike docks -----------------------------------------------------
println("\n[3/4] Bike dock nodes")
val bikeRaw = spark.read.option("header", "true").option("inferSchema", "true")
  .csv(s"$HDFS/data/neurotraffic/bike_station_zones.csv")

// station_id in that file is a float like 6551.11; make it a stable integer
val bikeDocks = bikeRaw
  .withColumn("dock_id", (col("station_id") * 100).cast(LongType))
  .join(centroids.select(col("zone_id"), col("borough")), Seq("zone_id"), "left")
  .withColumn("capacity", lit(30))  // Citi Bike typical dock capacity
  .withColumnRenamed("station_name", "dock_name")
  .withColumnRenamed("lng", "lon")
  .select("dock_id", "dock_name", "lat", "lon", "capacity", "borough")
  .dropDuplicates("dock_id")

bikeDocks.write.mode("overwrite").parquet(s"$GRAPH/nodes/bike_docks.parquet")
println(s"  wrote ${bikeDocks.count} bike docks")

// --- 4. Trip edges from raw yellow + green taxi -----------------------
println("\n[4/4] Trip edges (taxi OD pairs)")

def readTaxi(subdir: String, pickCol: String, dropCol: String) =
  spark.read.parquet(s"$HDFS/data/neurotraffic/raw/taxi/$subdir")
    .select(
      col("PULocationID").cast(IntegerType).as("src"),
      col("DOLocationID").cast(IntegerType).as("dst"),
      col(pickCol).as("pickup"),
      col(dropCol).as("dropoff")
    )

val trips = readTaxi("yellow", "tpep_pickup_datetime", "tpep_dropoff_datetime")
  .union(readTaxi("green", "lpep_pickup_datetime", "lpep_dropoff_datetime"))
  .filter(col("src").between(1, 263) && col("dst").between(1, 263))
  .filter(col("src") =!= col("dst"))
  .filter(col("pickup").between("2022-01-01", "2024-12-31"))

val edges = trips
  .withColumn("hour_of_day", hour(col("pickup")))
  .withColumn("day_type", when(dayofweek(col("pickup")).isin(1, 7), "weekend").otherwise("weekday"))
  .withColumn("travel_sec", unix_timestamp(col("dropoff")) - unix_timestamp(col("pickup")))
  .filter(col("travel_sec").between(60, 7200))
  .groupBy(col("src"), col("dst"), col("hour_of_day"), col("day_type"))
  .agg(
    count(lit(1)).as("trip_count"),
    avg("travel_sec").as("avg_travel_time")
  )
  .filter(col("trip_count") >= 5)  // drop one-off noise
  .withColumn("src_node_id", col("src").cast(LongType))
  .withColumn("dst_node_id", col("dst").cast(LongType))
  .withColumn("src_type", lit("taxi"))
  .withColumn("dst_type", lit("taxi"))
  .select("src_node_id", "dst_node_id", "src_type", "dst_type",
          "trip_count", "avg_travel_time", "hour_of_day", "day_type")

edges.write.mode("overwrite").parquet(s"$GRAPH/combined/trips.parquet")
val edgeCount = spark.read.parquet(s"$GRAPH/combined/trips.parquet").count
println(s"  wrote $edgeCount edges")

println("\n" + "=" * 60)
println(s"  DONE: $GRAPH ready for NeuroTrafficApp")
println("=" * 60)
System.exit(0)
