// scoring_module.scala
// NeuroTraffic — Scoring Module (Integration API for Streaming)
// Loads trained GBT model from HDFS and provides scoring functions.
// Replaces scoring_module.py — uses Spark MLlib natively (no pickle/sklearn).
//
// Usage in spark-shell:
//   :load /project/06_ml/scoring_module.scala
//   val scorer = new NeuroTrafficScorer(spark)
//   val result = scorer.fullPipeline(incomingBatch)

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.PipelineModel

class NeuroTrafficScorer(spark: SparkSession,
                         modelBase: String = "hdfs://namenode:9000/data/neurotraffic/models") {

  val THRESHOLD_YELLOW = 0.80
  val THRESHOLD_RED    = 0.90

  private val gbtModel: Option[PipelineModel] = try {
    val m = PipelineModel.load(s"${modelBase}/gbt_cascade")
    println(s"[Scorer] Loaded GBT pipeline from ${modelBase}/gbt_cascade")
    Some(m)
  } catch {
    case e: Exception =>
      println(s"[Scorer] GBT not available (${e.getMessage}), cascade prediction disabled")
      None
  }

  // Load z-score baselines for anomaly scoring
  private val baselines: Option[DataFrame] = try {
    val b = spark.read.parquet("hdfs://namenode:9000/data/neurotraffic/baselines")
    println(s"[Scorer] Loaded baselines (${b.count()} zone-hour profiles)")
    Some(b)
  } catch {
    case _: Exception =>
      println("[Scorer] Baselines not available, will compute on-the-fly")
      None
  }

  private def ensureColumns(df: DataFrame): DataFrame = {
    val defaults: Map[String, Any] = Map(
      "taxi_pickups" -> 0, "taxi_dropoffs" -> 0,
      "subway_ridership" -> 0, "bike_starts" -> 0, "bike_ends" -> 0,
      "temperature_c" -> 15.0, "humidity_pct" -> 50.0,
      "precipitation_mm" -> 0.0, "wind_speed_kmh" -> 10.0,
      "is_rain" -> 0, "is_snow" -> 0,
      "is_extreme_cold" -> 0, "is_extreme_heat" -> 0,
      "complaint_count" -> 0, "max_complaint_severity" -> 0,
      "event_count" -> 0,
      "page_rank" -> 0.0, "community_id" -> 0,
      "in_degree" -> 0, "out_degree" -> 0
    )

    var result = df
    for ((colName, defaultVal) <- defaults if !result.columns.contains(colName)) {
      defaultVal match {
        case v: Double => result = result.withColumn(colName, lit(v))
        case v: Int    => result = result.withColumn(colName, lit(v))
        case _         => result = result.withColumn(colName, lit(0))
      }
    }

    if (result.columns.contains("hourly_timestamp")) {
      if (!result.columns.contains("hour_of_day"))
        result = result.withColumn("hour_of_day", hour(col("hourly_timestamp")))
      if (!result.columns.contains("day_of_week"))
        result = result.withColumn("day_of_week", dayofweek(col("hourly_timestamp")))
      if (!result.columns.contains("is_rush_hour"))
        result = result.withColumn("is_rush_hour",
          when(col("hour_of_day").between(7, 9) || col("hour_of_day").between(17, 19), 1).otherwise(0))
      if (!result.columns.contains("is_weekend"))
        result = result.withColumn("is_weekend",
          when(col("day_of_week").isin(1, 7), 1).otherwise(0))
    }

    result
  }

  private def engineerFeatures(df: DataFrame): DataFrame = {
    df.withColumn("taxi_to_subway_ratio",
        when(col("subway_ridership") > 0,
          col("taxi_pickups").cast(DoubleType) / col("subway_ridership"))
        .otherwise(0.0))
      .withColumn("bike_to_subway_ratio",
        when(col("subway_ridership") > 0,
          col("bike_starts").cast(DoubleType) / col("subway_ridership"))
        .otherwise(0.0))
      .withColumn("total_surface_transport",
        col("taxi_pickups") + col("bike_starts"))
      .withColumn("complaint_density",
        when(col("taxi_pickups") > 0,
          col("complaint_count").cast(DoubleType) / col("taxi_pickups"))
        .otherwise(0.0))
  }

  def scoreAnomaly(df: DataFrame): DataFrame = {
    var data = ensureColumns(df)

    // Z-score based anomaly scoring (replaces sklearn Isolation Forest)
    // Uses inline baselines if pre-computed baselines not available
    val bl = baselines.getOrElse(
      data.groupBy("zone_id", "hour_of_day", "is_weekend").agg(
        mean("taxi_pickups").alias("mean_pickups"),
        stddev("taxi_pickups").alias("std_pickups"),
        mean("subway_ridership").alias("mean_subway"),
        stddev("subway_ridership").alias("std_subway"),
        mean("bike_starts").alias("mean_bikes"),
        stddev("bike_starts").alias("std_bikes")
      )
    )

    data = data.join(broadcast(bl), Seq("zone_id", "hour_of_day", "is_weekend"), "left")

    data = data.withColumn("z_subway",
      when(col("std_subway") > 0,
        (col("subway_ridership") - col("mean_subway")) / col("std_subway"))
      .otherwise(0.0))
    data = data.withColumn("z_pickups",
      when(col("std_pickups") > 0,
        (col("taxi_pickups") - col("mean_pickups")) / col("std_pickups"))
      .otherwise(0.0))
    data = data.withColumn("z_bikes",
      when(col("std_bikes") > 0,
        (col("bike_starts") - col("mean_bikes")) / col("std_bikes"))
      .otherwise(0.0))

    data = data.withColumn("anomaly_score",
      least(lit(1.0), greatest(lit(0.0),
        greatest(lit(0.0), -col("z_subway")) * 0.4 +
        greatest(lit(0.0), col("z_pickups")) * 0.3 +
        greatest(lit(0.0), col("z_bikes")) * 0.3
      )))

    data.withColumn("anomaly_level",
      when(col("anomaly_score") >= THRESHOLD_RED, "red")
      .when(col("anomaly_score") >= THRESHOLD_YELLOW, "yellow")
      .otherwise("normal"))
  }

  def predictCascade(df: DataFrame): DataFrame = {
    gbtModel match {
      case Some(model) =>
        val data = engineerFeatures(ensureColumns(df))
        val casted = data.columns.foldLeft(data) { (d, c) =>
          if (d.schema(c).dataType == IntegerType) d.withColumn(c, col(c).cast(DoubleType))
          else d
        }
        model.transform(casted)
      case None =>
        df.withColumn("prediction", lit(-1.0))
    }
  }

  def fullPipeline(df: DataFrame): DataFrame = {
    val scored = scoreAnomaly(df)

    val normalDf = scored.filter(col("anomaly_level") === "normal")
    val yellowDf = scored.filter(col("anomaly_level") === "yellow")
    var redDf = scored.filter(col("anomaly_level") === "red")

    val nRed = redDf.count()
    println(s"[Scorer] ${normalDf.count()} normal, ${yellowDf.count()} yellow, ${nRed} red")

    if (nRed > 0 && gbtModel.isDefined) {
      redDf = predictCascade(redDf)
    }

    // Align columns
    val allCols = (normalDf.columns.toSet ++ yellowDf.columns.toSet ++ redDf.columns.toSet).toArray
    def alignDf(d: DataFrame): DataFrame = {
      allCols.foldLeft(d) { (df, c) =>
        if (!df.columns.contains(c)) df.withColumn(c, lit(null).cast(DoubleType))
        else df
      }.select(allCols.map(col): _*)
    }

    alignDf(normalDf).union(alignDf(yellowDf)).union(alignDf(redDf))
  }
}

// Quick test
println("=== NeuroTrafficScorer loaded ===")
println("Usage: val scorer = new NeuroTrafficScorer(spark)")
println("       val result = scorer.fullPipeline(incomingBatch)")
