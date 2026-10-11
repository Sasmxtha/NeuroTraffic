package neurotraffic

import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SparkSession, SaveMode}

object GraphAnalytics {

  /**
   * Run PageRank on the transport graph.
   *
   * PageRank answers: "Which nodes are most critical to the network?"
   * A node is critical if many high-traffic routes pass through it.
   *
   * In transport terms:
   *   - Times Square will rank high because tons of taxi, subway, and bike
   *     trips connect through it.
   *   - A quiet residential zone in Staten Island will rank low.
   *
   * @param graph  The multi-modal transport graph
   * @param tol    Convergence tolerance — smaller = more precise but slower.
   *               0.001 is good for production, 0.01 for quick testing.
   * @param resetProb  Probability of random jump (damping = 1 - resetProb).
   *                   Standard value is 0.15.
   * @return RDD of (vertexId, pageRankScore)
   */
  def runPageRank(
    graph: Graph[NodeProperties, EdgeProperties],
    tol: Double = 0.001,
    resetProb: Double = 0.15
  ): RDD[(VertexId, Double)] = {

    // GraphX PageRank needs numeric edge weights.
    // We use trip count so heavily-used routes contribute more influence.
    val weightedGraph: Graph[NodeProperties, Double] = graph.mapEdges { edge =>
      edge.attr.tripCount.toDouble
    }

    val ranks = weightedGraph.pageRank(tol, resetProb).vertices
    ranks
  }

  /**
   * Run Label Propagation to find transport communities.
   *
   * Label Propagation answers: "Which nodes naturally cluster together?"
   * Nodes that exchange lots of trips end up in the same community.
   *
   * In transport terms:
   *   - Midtown Manhattan taxi zones, Penn Station subway, and nearby bike docks
   *     will form one community because commuters flow between them.
   *   - Downtown Brooklyn zones will form a separate community.
   *
   * @param graph    The transport graph
   * @param maxIter  How many iterations to run. 5-10 is usually enough for
   *                 convergence. More = slower but potentially tighter communities.
   * @return RDD of (vertexId, communityId)
   */
  def runLabelPropagation(
    graph: Graph[NodeProperties, EdgeProperties],
    maxIter: Int = 10
  ): RDD[(VertexId, VertexId)] = {

    // LPA needs numeric edge weights — use trip count again
    val weightedGraph: Graph[NodeProperties, Double] = graph.mapEdges { edge =>
      edge.attr.tripCount.toDouble
    }

    val communities = lib.LabelPropagation.run(weightedGraph, maxIter).vertices
    communities
  }

  /**
   * Compute in-degree and out-degree for every node.
   *
   * In-degree  = how many other nodes send trips TO this node
   * Out-degree = how many other nodes this node sends trips TO
   *
   * High in-degree = popular destination (e.g., Penn Station during morning rush)
   * High out-degree = popular origin (e.g., Penn Station during evening rush)
   */
  def computeDegrees(
    graph: Graph[NodeProperties, EdgeProperties]
  ): (RDD[(VertexId, Int)], RDD[(VertexId, Int)]) = {

    val inDeg  = graph.inDegrees   // RDD[(VertexId, Int)]
    val outDeg = graph.outDegrees  // RDD[(VertexId, Int)]
    (inDeg, outDeg)
  }

  /**
   * Combine all graph analytics into a single GraphFeatures object per node.
   *
   * This output is used as ML features:
   *   - pageRank      → how critical this node is
   *   - communityId   → which neighborhood cluster it belongs to
   *   - inDegree      → how many sources feed into it
   *   - outDegree     → how many destinations it feeds
   */
  def extractAllFeatures(
    graph: Graph[NodeProperties, EdgeProperties]
  ): RDD[(VertexId, NodeProperties, GraphFeatures)] = {

    val ranks       = runPageRank(graph)
    val communities = runLabelPropagation(graph)
    val (inDeg, outDeg) = computeDegrees(graph)

    // Join everything together by vertex ID.
    // leftOuterJoin handles nodes with no edges (degree = 0).
    val features: RDD[(VertexId, GraphFeatures)] =
      ranks
        .join(communities)
        .leftOuterJoin(inDeg)
        .leftOuterJoin(outDeg)
        .map { case (vid, ((((pr, comm), inD), outD))) =>
          (vid, GraphFeatures(
            pageRank    = pr,
            communityId = comm,
            inDegree    = inD.getOrElse(0),
            outDegree   = outD.getOrElse(0)
          ))
        }

    // Join features back with node properties so we have everything in one place
    graph.vertices.join(features).map {
      case (vid, (props, feats)) => (vid, props, feats)
    }
  }

  /**
   * Save the extracted features as a Parquet file on HDFS.
   * The ML pipeline reads this file to build feature vectors.
   */
  def saveFeatures(
    spark: SparkSession,
    features: RDD[(VertexId, NodeProperties, GraphFeatures)],
    outputPath: String = ""
  ): Unit = {

    import spark.implicits._

    val resolvedPath = if (outputPath.nonEmpty) outputPath
      else s"${GraphBuilder.basePath}/graph_features/features.parquet"

    val df = features.map { case (vid, props, feats) =>
      (
        vid,
        props.nodeType.toString,
        props.name,
        props.lat,
        props.lon,
        props.borough,
        props.capacity,
        feats.pageRank,
        feats.communityId,
        feats.inDegree,
        feats.outDegree
      )
    }.toDF(
      "vertex_id", "node_type", "name", "lat", "lon", "borough", "capacity",
      "page_rank", "community_id", "in_degree", "out_degree"
    )

    df.write.mode(SaveMode.Overwrite).parquet(resolvedPath)

    println(s"Graph features saved to $resolvedPath")
    println(s"Total nodes with features: ${df.count()}")
  }
}
