package neurotraffic

import org.apache.spark.sql.{SparkSession, DataFrame, Row}
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object GraphBuilder {

  // ID offset ranges so taxi/subway/bike IDs never collide:
  //   Taxi zones:      1 — 263
  //   Subway stations: 1000 — 1999
  //   Bike docks:      5000+
  val SUBWAY_OFFSET = 1000L
  val BIKE_OFFSET   = 5000L

  // Default to HDFS for cluster runs; override with local path for testing
  var basePath: String = "hdfs:///data"

  def loadNodes(spark: SparkSession): RDD[(VertexId, NodeProperties)] = {

    // --- Taxi zones ---
    val taxiZones = spark.read.parquet(s"$basePath/nodes/taxi_zones.parquet")
    val taxiRDD: RDD[(VertexId, NodeProperties)] = taxiZones.rdd.map { row =>
      val zoneId = row.getAs[Number]("zone_id").longValue()
      (
        zoneId,
        NodeProperties(
          nodeId   = zoneId,
          nodeType = NodeType.TaxiZone,
          name     = row.getAs[String]("zone_name"),
          lat      = row.getAs[Number]("lat").doubleValue(),
          lon      = row.getAs[Number]("lon").doubleValue(),
          capacity = row.getAs[Number]("capacity").intValue(),
          borough  = row.getAs[String]("borough")
        )
      )
    }

    // --- Subway stations ---
    val subwayStations = spark.read.parquet(s"$basePath/nodes/subway_stations.parquet")
    val subwayRDD: RDD[(VertexId, NodeProperties)] = subwayStations.rdd.map { row =>
      val rawId = row.getAs[Number]("station_id").longValue()
      val vertexId = rawId + SUBWAY_OFFSET
      (
        vertexId,
        NodeProperties(
          nodeId   = vertexId,
          nodeType = NodeType.SubwayStation,
          name     = row.getAs[String]("station_name"),
          lat      = row.getAs[Number]("lat").doubleValue(),
          lon      = row.getAs[Number]("lon").doubleValue(),
          capacity = row.getAs[Number]("capacity").intValue(),
          borough  = row.getAs[String]("borough")
        )
      )
    }

    // --- Bike docks ---
    val bikeDocks = spark.read.parquet(s"$basePath/nodes/bike_docks.parquet")
    val bikeRDD: RDD[(VertexId, NodeProperties)] = bikeDocks.rdd.map { row =>
      val rawId = row.getAs[Number]("dock_id").longValue()
      val vertexId = rawId + BIKE_OFFSET
      (
        vertexId,
        NodeProperties(
          nodeId   = vertexId,
          nodeType = NodeType.BikeDock,
          name     = row.getAs[String]("dock_name"),
          lat      = row.getAs[Number]("lat").doubleValue(),
          lon      = row.getAs[Number]("lon").doubleValue(),
          capacity = row.getAs[Number]("capacity").intValue(),
          borough  = row.getAs[String]("borough")
        )
      )
    }

    // Union all three into a single vertex RDD
    taxiRDD.union(subwayRDD).union(bikeRDD)
  }

  /**
   * Load edges from the cleaned combined trips table.
   *
   * Expects the combined Parquet at /data/combined/trips.parquet with columns:
   *   src_node_id, dst_node_id, src_type, dst_type, trip_count,
   *   avg_travel_time, hour_of_day, day_type
   *
   * src_type/dst_type are "taxi", "subway", or "bike" — we use them to apply
   * the correct ID offset so vertex IDs match the node RDD.
   */
  def loadEdges(spark: SparkSession): RDD[Edge[EdgeProperties]] = {

    val trips = spark.read.parquet(s"$basePath/combined/trips.parquet")

    trips.rdd.map { row =>
      val srcRaw = row.getAs[Number]("src_node_id").longValue()
      val dstRaw = row.getAs[Number]("dst_node_id").longValue()
      val srcType = row.getAs[String]("src_type")
      val dstType = row.getAs[String]("dst_type")

      val srcId = applyOffset(srcRaw, srcType)
      val dstId = applyOffset(dstRaw, dstType)

      Edge(
        srcId,
        dstId,
        EdgeProperties(
          tripCount     = row.getAs[Number]("trip_count").longValue(),
          avgTravelTime = row.getAs[Number]("avg_travel_time").doubleValue(),
          hourOfDay     = row.getAs[Number]("hour_of_day").intValue(),
          dayType       = row.getAs[String]("day_type")
        )
      )
    }
  }

  private def applyOffset(rawId: Long, nodeType: String): Long = nodeType match {
    case "taxi"   => rawId
    case "subway" => rawId + SUBWAY_OFFSET
    case "bike"   => rawId + BIKE_OFFSET
    case other    => throw new IllegalArgumentException(s"Unknown node type: $other")
  }

  /**
   * Build the full multi-modal transport graph.
   *
   * The default vertex covers any edge that references a node ID not in the
   * vertex RDD (shouldn't happen with clean data, but keeps Spark from crashing).
   */
  def buildGraph(spark: SparkSession): Graph[NodeProperties, EdgeProperties] = {

    val vertices = loadNodes(spark)
    val edges = loadEdges(spark)

    val defaultNode = NodeProperties(
      nodeId   = -1L,
      nodeType = NodeType.TaxiZone,
      name     = "UNKNOWN",
      lat      = 0.0,
      lon      = 0.0,
      capacity = 0,
      borough  = "UNKNOWN"
    )

    Graph(vertices, edges, defaultNode)
  }

  /**
   * Build a time-windowed graph — only edges for a specific hour and day type.
   *
   * Example: buildTimeWindowedGraph(spark, 8, "weekday") gives you the
   * 8 AM weekday graph — rush hour Manhattan will have very different edge
   * weights than midnight Brooklyn.
   */
  def buildTimeWindowedGraph(
    spark: SparkSession,
    hour: Int,
    dayType: String
  ): Graph[NodeProperties, EdgeProperties] = {

    val fullGraph = buildGraph(spark)

    // Keep only edges matching this time window
    fullGraph.subgraph(
      epred = edge =>
        edge.attr.hourOfDay == hour && edge.attr.dayType == dayType
    )
  }
}
