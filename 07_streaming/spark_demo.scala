println("=" * 60)
println("  NeuroTraffic - Spark Analytics Demo")
println("=" * 60)

println("\n[1/4] Reading NYC Film Permits from HDFS...")
val films = spark.read.option("header","true").option("inferSchema","true").csv("hdfs://namenode:9000/data/neurotraffic/raw/events/film_permits_2022_2024.csv")
println(s"  Loaded ${films.count()} film permit records")

println("\n[2/4] Schema:")
films.printSchema()

println("\n[3/4] Top 10 boroughs by permit count:")
import org.apache.spark.sql.functions._
films.groupBy("borough").agg(count("*").as("permits")).orderBy(desc("permits")).show(10, false)

println("\n[4/4] Permits by category:")
films.groupBy("category").agg(count("*").as("count")).orderBy(desc("count")).show(10, false)

println("\n" + "=" * 60)
println("  Spark UI live at http://localhost:4040")
println("  Keeping alive for 10 minutes...")
println("=" * 60)
Thread.sleep(600000)