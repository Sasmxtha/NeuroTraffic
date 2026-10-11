// NeuroTraffic — Scala Spark Event Permits Cleaning
// Run: spark-shell -i /project/02_cleaning/clean_events.scala

import org.apache.spark.sql.functions._

val HDFS = "hdfs://namenode:9000/data/neurotraffic"

println("=" * 60)
println("NeuroTraffic — Events Cleaning (Scala)")
println("=" * 60)

// Read event permits
val events = spark.read
  .option("header", "true").option("inferSchema", "true")
  .csv(s"$HDFS/raw/events/event_permits_2022_2024.csv")
  .select(
    col("event_id").cast("string").as("event_id"),
    col("event_name"),
    col("start_date_time").cast("timestamp").as("event_start"),
    col("end_date_time").cast("timestamp").as("event_end"),
    col("event_type"),
    col("event_borough").as("borough"),
    col("event_location").cast("string").as("street_description"),
    col("street_closure_type").cast("string").as("closure_type"),
    lit("event_permit").as("source")
  )
println(s"  Event permits: ${events.count()}")

// Read film permits
val films = spark.read
  .option("header", "true").option("inferSchema", "true")
  .csv(s"$HDFS/raw/events/film_permits_2022_2024.csv")
  .select(
    col("eventid").cast("string").as("event_id"),
    col("subcategoryname").as("event_name"),
    col("startdatetime").cast("timestamp").as("event_start"),
    col("enddatetime").cast("timestamp").as("event_end"),
    col("eventtype").as("event_type"),
    col("borough"),
    col("parkingheld").cast("string").as("street_description"),
    lit(null).cast("string").as("closure_type"),
    lit("film_permit").as("source")
  )
println(s"  Film permits: ${films.count()}")

// Union and clean
var df = events.unionByName(films)
val rawCount = df.count()
println(s"  Combined raw: $rawCount")

df = df
  .na.drop(Seq("event_start"))
  .filter(col("event_start").between("2022-01-01", "2025-01-01"))

// Extract street count using regex (no UDF needed — pure SQL)
df = df.withColumn("street_text", upper(col("street_description")))
  .withColumn("num_streets_affected",
    size(
      split(
        regexp_extract(col("street_text"), "((?:\\d+(?:ST|ND|RD|TH)\\s+(?:ST|AVE|BLVD|PL|DR|RD|WAY)\\s*,?\\s*)+)", 1),
        ","
      )
    )
  )
  .withColumn("num_streets_affected",
    when(col("num_streets_affected") < 0, lit(0)).otherwise(col("num_streets_affected"))
  )
  .drop("street_text")

// Event duration
df = df.withColumn("event_duration_hours",
    (col("event_end").cast("long") - col("event_start").cast("long")) / 3600.0
  )
  .filter(col("event_duration_hours") > 0 && col("event_duration_hours") < 720)

// Disruption severity
df = df.withColumn("disruption_severity",
    when(col("num_streets_affected") >= 5, 5)
    .when(col("num_streets_affected") >= 3, 4)
    .when(col("event_duration_hours") > 8, 3)
    .when(col("num_streets_affected") >= 1, 2)
    .otherwise(1)
  )
  .withColumn("event_hour", date_trunc("hour", col("event_start")))
  .withColumn("year",  year(col("event_start")))
  .withColumn("month", month(col("event_start")))

val cleanCount = df.count()
println(s"  Cleaned rows: $cleanCount")
println(s"  Removed: ${rawCount - cleanCount}")

df.groupBy("event_type").count().orderBy(desc("count")).show(10, truncate = false)

df.repartition(col("year"), col("month"))
  .write.partitionBy("year", "month")
  .mode("overwrite")
  .parquet(s"$HDFS/cleaned/events")

println("  [OK] Events cleaning complete!")

System.exit(0)
