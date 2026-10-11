// NeuroTraffic — Extract Unique Bike Stations
// Run: spark-shell -i /project/02_cleaning/extract_bike_stations.scala

import org.apache.spark.sql.functions._

val HDFS = "hdfs://namenode:9000/data/neurotraffic"

println("=" * 60)
println("Extract Bike Stations (Scala)")
println("=" * 60)

val bike = spark.read.parquet(s"$HDFS/cleaned/citibike")
println(s"  Total rows: ${bike.count()}")

val starts = bike.select(
  col("start_station_id").as("station_id"),
  col("start_station_name").as("station_name"),
  col("start_lat").as("lat"),
  col("start_lng").as("lng")
).filter(col("station_id").isNotNull)

val ends = bike.select(
  col("end_station_id").as("station_id"),
  col("end_station_name").as("station_name"),
  col("end_lat").as("lat"),
  col("end_lng").as("lng")
).filter(col("station_id").isNotNull)

val stations = starts.union(ends)
  .groupBy("station_id", "station_name")
  .agg(avg("lat").as("lat"), avg("lng").as("lng"))

println(s"  Unique stations: ${stations.count()}")

stations.coalesce(1).write
  .mode("overwrite")
  .option("header", "true")
  .csv("/tmp/bike_stations_out")

println("  [OK] Written to /tmp/bike_stations_out")
println("  Copy the CSV: find /tmp/bike_stations_out -name '*.csv'")

System.exit(0)
