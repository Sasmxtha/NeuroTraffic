// NeuroTraffic — Export Combined Parquet to TSV for MapReduce
// Run: spark-shell -i /project/04_mapreduce/export_for_mapreduce.scala

val HDFS = "hdfs://namenode:9000/data/neurotraffic"

println("=" * 60)
println("Export for MapReduce (Scala)")
println("=" * 60)

val df = spark.read.parquet(s"$HDFS/combined_v3")
println(s"  Rows: ${df.count()}")

println("  Writing TSV...")
df.write
  .mode("overwrite")
  .option("sep", "\t")
  .option("header", "true")
  .csv(s"$HDFS/mapreduce_input")

println("  [OK] Written to mapreduce_input")

System.exit(0)
