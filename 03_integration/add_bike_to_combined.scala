// NeuroTraffic — Add Bike Trip Data to Combined Table
// Run: spark-shell -i /project/03_integration/add_bike_to_combined.scala

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, StructField, StringType, DoubleType, IntegerType}

val HDFS = "hdfs://namenode:9000/data/neurotraffic"

println("=" * 60)
println("NeuroTraffic — Add Bike to Combined (Scala)")
println("=" * 60)

// Load station-to-zone mapping
val mappingSchema = new StructType()
  .add("station_id", StringType)
  .add("station_name", StringType)
  .add("lat", DoubleType)
  .add("lng", DoubleType)
  .add("zone_id", IntegerType)

val mapping = spark.read
  .option("header", "true")
  .schema(mappingSchema)
  .csv("/data/local/bike_station_zones.csv")
println(s"  Station-to-zone mappings: ${mapping.count()}")

// Read cleaned bike data and map to zones
val bike = spark.read.parquet(s"$HDFS/cleaned/citibike")
println(s"  Bike trips: ${bike.count()}")

val startMap = mapping.select(
  col("station_id").as("start_station_id"),
  col("zone_id").as("start_zone_id")
)
val endMap = mapping.select(
  col("station_id").as("end_station_id"),
  col("zone_id").as("end_zone_id")
)

val mapped = bike
  .join(startMap, Seq("start_station_id"), "left")
  .join(endMap, Seq("end_station_id"), "left")
  .filter(col("start_zone_id").isNotNull)

println(s"  Mapped to zones: ${mapped.count()}")

// Aggregate by zone + hour
val bikeStarts = mapped.groupBy(
    col("start_zone_id").as("zone_id"),
    col("start_hour").as("hourly_timestamp")
  ).agg(
    count("*").as("bike_starts"),
    avg("trip_duration_sec").as("avg_bike_duration_sec")
  )

val bikeEnds = mapped.filter(col("end_zone_id").isNotNull)
  .groupBy(
    col("end_zone_id").as("zone_id"),
    col("start_hour").as("hourly_timestamp")
  ).agg(count("*").as("bike_ends"))

val bikeAgg = bikeStarts.join(bikeEnds, Seq("zone_id", "hourly_timestamp"), "outer")
  .na.fill(0, Seq("bike_starts", "bike_ends"))

// Join with existing combined table
val combined = spark.read.parquet(s"$HDFS/combined")
println(s"  Existing combined rows: ${combined.count()}")

val result = combined
  .join(bikeAgg, Seq("zone_id", "hourly_timestamp"), "left")
  .na.fill(0, Seq("bike_starts", "bike_ends"))
  .na.fill(0.0, Seq("avg_bike_duration_sec"))

result.write.mode("overwrite").partitionBy("year").parquet(s"$HDFS/combined_v2")
println(s"  [OK] Written to combined_v2 (${result.count()} rows)")

System.exit(0)
