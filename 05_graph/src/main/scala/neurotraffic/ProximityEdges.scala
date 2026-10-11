package neurotraffic

import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
 * Generates cross-modal proximity edges between nodes of different transport types
 * that are geographically close.
 *
 * Why this matters:
 *   Trip data alone only creates edges where trips actually happened. But cascade
 *   prediction needs to know that a subway station and a bike dock 200m apart are
 *   connected — when the subway fails, commuters walk to that dock.
 *
 *   Proximity edges fill this gap: if a subway station and a bike dock are within
 *   500m of each other, we create an edge with weight proportional to their closeness.
 *   Closer = stronger connection = more likely cascade path.
 */
object ProximityEdges {

  // Maximum distance (km) to create a proximity edge between different node types
  val MAX_TAXI_SUBWAY_KM = 1.5   // taxi zone centroid to subway station
  val MAX_TAXI_BIKE_KM   = 1.2   // taxi zone centroid to bike dock
  val MAX_SUBWAY_BIKE_KM = 0.8   // subway station to bike dock (walking distance)

  /**
   * Generate proximity edges for all cross-modal node pairs within distance thresholds.
   *
   * Each edge gets a synthetic trip count inversely proportional to distance:
   *   closer nodes = higher weight = stronger connection.
   *
   * The edges are bidirectional (A→B and B→A) because commuters can walk either way.
   */
  def generateProximityEdges(
    graph: Graph[NodeProperties, EdgeProperties]
  ): RDD[Edge[EdgeProperties]] = {

    val vertices = graph.vertices.collect()

    val taxiNodes   = vertices.filter(_._2.nodeType == NodeType.TaxiZone)
    val subwayNodes = vertices.filter(_._2.nodeType == NodeType.SubwayStation)
    val bikeNodes   = vertices.filter(_._2.nodeType == NodeType.BikeDock)

    val edges = scala.collection.mutable.ArrayBuffer[Edge[EdgeProperties]]()

    // --- Taxi ↔ Subway proximity ---
    for (taxi <- taxiNodes; subway <- subwayNodes) {
      val dist = haversine(taxi._2.lat, taxi._2.lon, subway._2.lat, subway._2.lon)
      if (dist <= MAX_TAXI_SUBWAY_KM && dist > 0.01) {
        val weight = proximityWeight(dist, MAX_TAXI_SUBWAY_KM, baseDemand = 100)
        val edgeProps = EdgeProperties(
          tripCount     = weight,
          avgTravelTime = dist * 12.0, // ~5 km/h walking speed → 12 min/km
          hourOfDay     = -1,          // -1 = aggregate (applies to all hours)
          dayType       = "all"
        )
        edges += Edge(taxi._1, subway._1, edgeProps)
        edges += Edge(subway._1, taxi._1, edgeProps)
      }
    }

    // --- Taxi ↔ Bike proximity ---
    for (taxi <- taxiNodes; bike <- bikeNodes) {
      val dist = haversine(taxi._2.lat, taxi._2.lon, bike._2.lat, bike._2.lon)
      if (dist <= MAX_TAXI_BIKE_KM && dist > 0.01) {
        val weight = proximityWeight(dist, MAX_TAXI_BIKE_KM, baseDemand = 60)
        val edgeProps = EdgeProperties(
          tripCount     = weight,
          avgTravelTime = dist * 12.0,
          hourOfDay     = -1,
          dayType       = "all"
        )
        edges += Edge(taxi._1, bike._1, edgeProps)
        edges += Edge(bike._1, taxi._1, edgeProps)
      }
    }

    // --- Subway ↔ Bike proximity (the critical last-mile link) ---
    for (subway <- subwayNodes; bike <- bikeNodes) {
      val dist = haversine(subway._2.lat, subway._2.lon, bike._2.lat, bike._2.lon)
      if (dist <= MAX_SUBWAY_BIKE_KM && dist > 0.01) {
        val weight = proximityWeight(dist, MAX_SUBWAY_BIKE_KM, baseDemand = 80)
        val edgeProps = EdgeProperties(
          tripCount     = weight,
          avgTravelTime = dist * 12.0,
          hourOfDay     = -1,
          dayType       = "all"
        )
        edges += Edge(subway._1, bike._1, edgeProps)
        edges += Edge(bike._1, subway._1, edgeProps)
      }
    }

    val sc = graph.vertices.sparkContext
    sc.parallelize(edges.toSeq)
  }

  /**
   * Build a graph with both trip edges AND proximity edges merged together.
   */
  def buildConnectedGraph(spark: SparkSession): Graph[NodeProperties, EdgeProperties] = {
    val baseGraph = GraphBuilder.buildGraph(spark)

    val proximityEdges = generateProximityEdges(baseGraph)
    val allEdges = baseGraph.edges.union(proximityEdges)

    val defaultNode = NodeProperties(
      nodeId   = -1L,
      nodeType = NodeType.TaxiZone,
      name     = "UNKNOWN",
      lat      = 0.0,
      lon      = 0.0,
      capacity = 0,
      borough  = "UNKNOWN"
    )

    val connectedGraph = Graph(baseGraph.vertices, allEdges, defaultNode)

    val tripEdgeCount = baseGraph.edges.count()
    val proxEdgeCount = proximityEdges.count()
    println(s"  Trip edges:      $tripEdgeCount")
    println(s"  Proximity edges: $proxEdgeCount")
    println(s"  Total edges:     ${tripEdgeCount + proxEdgeCount}")

    connectedGraph
  }

  /**
   * Inverse-distance weight: closer nodes get higher synthetic trip counts.
   * At distance 0, weight = baseDemand. At maxDist, weight = 1.
   */
  private def proximityWeight(dist: Double, maxDist: Double, baseDemand: Int): Long = {
    val ratio = 1.0 - (dist / maxDist)
    (baseDemand * ratio * ratio).toLong.max(1) // quadratic decay
  }

  private def haversine(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double = {
    val R = 6371.0
    val dLat = Math.toRadians(lat2 - lat1)
    val dLon = Math.toRadians(lon2 - lon1)
    val a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2)
    R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  }
}
