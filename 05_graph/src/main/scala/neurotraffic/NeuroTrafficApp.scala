package neurotraffic

import org.apache.spark.sql.SparkSession

/**
 * Main entry point — run with:
 *   spark-submit --class neurotraffic.NeuroTrafficApp neurotraffic-graphx.jar [mode]
 *
 * Modes:
 *   full      — build graph + run all analytics + save features (default)
 *   pagerank  — build graph + run PageRank only (quick test)
 *   timewindow — build time-windowed graphs for rush hour analysis
 *   stats     — build graph + print summary statistics
 */
object NeuroTrafficApp {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("NeuroTraffic-GraphX")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    // Parse --data <path> flag if present, otherwise default to HDFS
    val (mode, remainingArgs) = parseArgs(args)

    println("=" * 70)
    println("  NEUROTRAFFIC — Multi-Modal Transport Graph Engine")
    println(s"  Data path: ${GraphBuilder.basePath}")
    println("=" * 70)

    mode match {
      case "full"       => runFull(spark)
      case "pagerank"   => runPageRankOnly(spark)
      case "timewindow" => runTimeWindowAnalysis(spark)
      case "stats"      => runStats(spark)
      case other        => println(s"Unknown mode: $other. Use: full, pagerank, timewindow, stats, gendata")
    }

    spark.stop()
  }

  /**
   * Full pipeline: build graph → PageRank → Label Propagation → save features.
   */
  private def runFull(spark: SparkSession): Unit = {
    println("\n[1/4] Building multi-modal transport graph with proximity edges...")
    val graph = ProximityEdges.buildConnectedGraph(spark)
    printGraphSummary(graph)

    println("\n[2/4] Running PageRank + Label Propagation + Degree analysis...")
    val features = GraphAnalytics.extractAllFeatures(graph)

    println("\n[3/4] Saving graph features to HDFS...")
    GraphAnalytics.saveFeatures(spark, features)

    println("\n[4/4] Printing top nodes by PageRank...")
    printTopNodes(features, 20)

    println("\nFull pipeline complete.")
  }

  /**
   * Quick PageRank test — useful for verifying the graph is correct
   * before running the full pipeline.
   */
  private def runPageRankOnly(spark: SparkSession): Unit = {
    println("\n[1/2] Building graph with proximity edges...")
    val graph = ProximityEdges.buildConnectedGraph(spark)
    printGraphSummary(graph)

    println("\n[2/2] Running PageRank...")
    val ranks = GraphAnalytics.runPageRank(graph)

    // Join with node properties to print names
    val namedRanks = graph.vertices.join(ranks).map {
      case (vid, (props, rank)) => (props.name, props.nodeType.toString, rank)
    }

    println("\nTop 20 nodes by PageRank:")
    println("-" * 60)
    namedRanks
      .sortBy(_._3, ascending = false)
      .take(20)
      .foreach { case (name, ntype, rank) =>
        println(f"  $name%-40s [$ntype%-14s] PR = $rank%.6f")
      }
  }

  /**
   * Build time-windowed graphs and compare rush hour vs midnight.
   * This shows how the network's critical nodes shift by time of day.
   */
  private def runTimeWindowAnalysis(spark: SparkSession): Unit = {
    val timeWindows = Seq(
      (8, "weekday", "Weekday 8 AM (morning rush)"),
      (17, "weekday", "Weekday 5 PM (evening rush)"),
      (12, "weekday", "Weekday 12 PM (midday)"),
      (2, "weekend", "Weekend 2 AM (late night)")
    )

    for ((hour, dayType, label) <- timeWindows) {
      println(s"\n--- $label ---")
      val windowGraph = GraphBuilder.buildTimeWindowedGraph(spark, hour, dayType)
      println(s"  Vertices: ${windowGraph.vertices.count()}")
      println(s"  Edges:    ${windowGraph.edges.count()}")

      if (windowGraph.edges.count() > 0) {
        val ranks = GraphAnalytics.runPageRank(windowGraph, tol = 0.01)
        val namedRanks = windowGraph.vertices.join(ranks).map {
          case (_, (props, rank)) => (props.name, props.nodeType.toString, rank)
        }
        println("  Top 5 critical nodes:")
        namedRanks.sortBy(_._3, ascending = false).take(5).foreach {
          case (name, ntype, rank) =>
            println(f"    $name%-35s [$ntype%-14s] PR = $rank%.6f")
        }
      }
    }
  }

  /**
   * Print summary statistics about the graph — useful for sanity checking.
   */
  private def runStats(spark: SparkSession): Unit = {
    println("\nBuilding graph with proximity edges...")
    val graph = ProximityEdges.buildConnectedGraph(spark)
    printGraphSummary(graph)

    // Count by node type
    val typeCounts = graph.vertices.map(_._2.nodeType).countByValue()
    println("\nNodes by type:")
    typeCounts.foreach { case (ntype, count) =>
      println(f"  $ntype%-20s $count%,d")
    }

    // Count by borough
    val boroughCounts = graph.vertices.map(_._2.borough).countByValue()
    println("\nNodes by borough:")
    boroughCounts.toSeq.sortBy(-_._2).foreach { case (borough, count) =>
      println(f"  $borough%-20s $count%,d")
    }

    // Edge weight distribution
    val tripCounts = graph.edges.map(_.attr.tripCount)
    println("\nEdge trip count distribution:")
    println(f"  Min:    ${tripCounts.min()}%,d")
    println(f"  Max:    ${tripCounts.max()}%,d")
    println(f"  Mean:   ${tripCounts.mean()}%,.1f")
    println(f"  Median: ${tripCounts.take((tripCounts.count() / 2).toInt).last}%,d")

    // Connected components — are there disconnected parts of the graph?
    val cc = graph.connectedComponents().vertices.map(_._2).distinct().count()
    println(s"\nConnected components: $cc")
    if (cc > 1) {
      println("  WARNING: Graph has disconnected components!")
      println("  This means some transport nodes can't reach others.")
      println("  Check if cross-modal edges (taxi-to-subway, subway-to-bike) are present.")
    }
  }

  private def printGraphSummary(graph: org.apache.spark.graphx.Graph[NodeProperties, EdgeProperties]): Unit = {
    println(f"  Total vertices: ${graph.vertices.count()}%,d")
    println(f"  Total edges:    ${graph.edges.count()}%,d")
  }

  private def parseArgs(args: Array[String]): (String, Array[String]) = {
    var mode = "full"
    val remaining = scala.collection.mutable.ArrayBuffer[String]()
    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--data" if i + 1 < args.length =>
          GraphBuilder.basePath = args(i + 1)
          i += 2
        case other if mode == "full" && !other.startsWith("--") =>
          mode = other
          i += 1
        case other =>
          remaining += other
          i += 1
      }
    }
    (mode, remaining.toArray)
  }

  private def printTopNodes(
    features: org.apache.spark.rdd.RDD[(org.apache.spark.graphx.VertexId, NodeProperties, GraphFeatures)],
    n: Int
  ): Unit = {
    println(s"\nTop $n nodes by PageRank:")
    println("-" * 80)
    println(f"  ${"Name"}%-35s ${"Type"}%-16s ${"PageRank"}%-10s ${"Community"}%-10s ${"In"}%-5s ${"Out"}%-5s")
    println("-" * 80)

    features
      .sortBy(_._3.pageRank, ascending = false)
      .take(n)
      .foreach { case (_, props, feats) =>
        println(f"  ${props.name}%-35s ${props.nodeType}%-16s ${feats.pageRank}%.6f   ${feats.communityId}%-10d ${feats.inDegree}%-5d ${feats.outDegree}%-5d")
      }
  }
}
