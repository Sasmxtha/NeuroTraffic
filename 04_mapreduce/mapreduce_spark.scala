// NeuroTraffic — MapReduce equivalent via Spark
// Replicates mapper.sh + reducer.sh: aggregate taxi pickups by zone_id + date
// Run: spark-shell -i /project/04_mapreduce/mapreduce_spark.scala

import org.apache.spark.sql.functions._

val HDFS = "hdfs://namenode:9000/data/neurotraffic"

println("=" * 60)
println("MapReduce via Spark (Scala)")
println("=" * 60)

val df = spark.read.parquet(s"$HDFS/combined_v3")
println(s"  Input rows: ${df.count()}")

val result = df
  .filter(col("taxi_pickups") > 0)
  .withColumn("date", to_date(col("hourly_timestamp")))
  .groupBy("zone_id", "date")
  .agg(sum("taxi_pickups").as("total_pickups"))
  .orderBy("zone_id", "date")

println(s"  Output rows: ${result.count()}")

result.write
  .mode("overwrite")
  .option("sep", "\t")
  .option("header", "true")
  .csv(s"$HDFS/mapreduce_output")

println("  [OK] MapReduce output written")

result.show(20)

System.exit(0)
