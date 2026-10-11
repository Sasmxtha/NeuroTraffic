// NeuroTraffic — Scala Spark Weather Data Cleaning
// Run: spark-shell -i /project/02_cleaning/clean_weather.scala

import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types.DoubleType

val HDFS = "hdfs://namenode:9000/data/neurotraffic"

println("=" * 60)
println("NeuroTraffic — Weather Cleaning (Scala)")
println("=" * 60)

// Read CSV with fixed 11-column schema (Open-Meteo multi-header format)
// The CSV has metadata rows (6 cols) then data rows (11 cols).
// We force an 11-column schema so Spark reads all columns correctly.
import org.apache.spark.sql.types.{StructType, StructField, StringType}

val schema = new StructType(
  (0 to 10).map(i => StructField(s"c$i", StringType)).toArray
)

val raw = spark.read
  .option("header", "false")
  .schema(schema)
  .csv(s"$HDFS/raw/weather")

// Keep only data rows (timestamp pattern in c0)
var df = raw
  .filter(col("c0").rlike("^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}"))
  .select(
    col("c0").cast("timestamp").as("weather_hour"),
    col("c1").cast(DoubleType).as("temperature_c"),
    col("c2").cast(DoubleType).as("humidity_pct"),
    col("c3").cast(DoubleType).as("precipitation_mm"),
    col("c4").cast(DoubleType).as("rain_mm"),
    col("c5").cast(DoubleType).as("snowfall_cm"),
    col("c6").cast(DoubleType).as("snow_depth_m"),
    col("c7").cast("int").as("wmo_weather_code"),
    col("c8").cast(DoubleType).as("wind_speed_kmh"),
    col("c9").cast(DoubleType).as("wind_gusts_kmh"),
    col("c10").cast(DoubleType).as("visibility_m")
  )

val rawCount = df.count()
println(s"  Raw rows: $rawCount")

// Filter: drop nulls, date range, dedup
df = df
  .na.drop(Seq("weather_hour"))
  .filter(col("weather_hour").between("2022-01-01", "2025-01-01"))
  .dropDuplicates("weather_hour")

// Forward-fill missing values within 6-hour window
val hourWin = Window.orderBy("weather_hour").rowsBetween(-6, 0)
val fillCols = Seq("temperature_c", "humidity_pct", "precipitation_mm", "wind_speed_kmh", "visibility_m")

for (c <- fillCols) {
  df = df.withColumn(c, coalesce(col(c), last(col(c), ignoreNulls = true).over(hourWin)))
}

// Add weather severity flags
df = df
  .withColumn("is_rain",           (col("precipitation_mm") > 0.5).cast("int"))
  .withColumn("is_heavy_rain",     (col("precipitation_mm") > 7.6).cast("int"))
  .withColumn("is_snow",           (col("snowfall_cm") > 0).cast("int"))
  .withColumn("is_extreme_cold",   (col("temperature_c") < -10).cast("int"))
  .withColumn("is_extreme_heat",   (col("temperature_c") > 35).cast("int"))
  .withColumn("is_low_visibility", (col("visibility_m") < 1000).cast("int"))
  .withColumn("is_high_wind",      (col("wind_speed_kmh") > 50).cast("int"))
  .withColumn("year",  year(col("weather_hour")))
  .withColumn("month", month(col("weather_hour")))

val cleanCount = df.count()
println(s"  Cleaned rows: $cleanCount")
println(s"  Removed: ${rawCount - cleanCount}")

// Write to HDFS
df.repartition(col("year"), col("month"))
  .write.partitionBy("year", "month")
  .mode("overwrite")
  .parquet(s"$HDFS/cleaned/weather")

println("  [OK] Weather cleaning complete!")
df.printSchema()
df.show(5, truncate = false)

System.exit(0)
