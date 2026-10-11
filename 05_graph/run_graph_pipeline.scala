// NeuroTraffic — run the full graph pipeline directly in spark-shell.
// Mirrors 05_graph/src/main/scala/neurotraffic/GraphAnalytics.extractAllFeatures
// but doesn't need sbt. Reads /data/graph/{nodes,combined} built earlier,
// writes /data/graph/features/features.parquet in the schema the ML trainer
// and dashboard generator expect.
//
// Run:
//   docker exec spark-master /spark/bin/spark-shell --master "local[2]" \
//     --driver-memory 4g --conf spark.sql.shuffle.partitions=16 \
//     -i /project/05_graph/run_graph_pipeline.scala

import org.apache.spark.graphx._
import org.apache.spark.graphx.lib.LabelPropagation
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

val HDFS = "hdfs://namenode:9000"
val GRAPH = s"$HDFS/data/graph"
val SUBWAY_OFFSET = 1000L
val BIKE_OFFSET   = 5000L

println("=" * 60)
println("  NeuroTraffic graph pipeline (PageRank + LPA + degree)")
println("=" * 60)

// ── 1. Load nodes from the three Parquet files ────────────────────────
val taxi = spark.read.parquet(s"$GRAPH/nodes/taxi_zones.parquet")
val subway = spark.read.parquet(s"$GRAPH/nodes/subway_stations.parquet")
val bike = spark.read.parquet(s"$GRAPH/nodes/bike_docks.parquet")

// Normalise each node source to (vertex_id, node_type, name, lat, lon, borough, capacity)
val taxiV = taxi.select(
  col("zone_id").cast(LongType).as("vertex_id"),
  lit("taxi_zone").as("node_type"),
  col("zone_name").as("name"),
  col("lat"), col("lon"), col("borough"), col("capacity"))

val subwayV = subway.select(
  (col("station_id").cast(LongType) + lit(SUBWAY_OFFSET)).as("vertex_id"),
  lit("subway_station").as("node_type"),
  col("station_name").as("name"),
  col("lat"), col("lon"), col("borough"), col("capacity"))

val bikeV = bike.select(
  (col("dock_id").cast(LongType) + lit(BIKE_OFFSET)).as("vertex_id"),
  lit("bike_dock").as("node_type"),
  col("dock_name").as("name"),
  col("lat"), col("lon"), col("borough"), col("capacity"))

val allNodes = taxiV.union(subwayV).union(bikeV)
  .filter(col("vertex_id").isNotNull)
  .dropDuplicates("vertex_id")
  .cache()
println(s"  loaded ${allNodes.count} nodes (after dedupe + null filter)")

// ── 2. Load edges from trips.parquet ──────────────────────────────────
val trips = spark.read.parquet(s"$GRAPH/combined/trips.parquet")
println(s"  loaded ${trips.count} edge rows")

def offsetFor(nodeType: String) = nodeType match {
  case "taxi"   => 0L
  case "subway" => SUBWAY_OFFSET
  case "bike"   => BIKE_OFFSET
}

// Aggregate time-windowed edges down to a single weight per OD pair so
// PageRank operates on the combined demand graph.
val edgesDF = trips
  .groupBy("src_node_id", "dst_node_id", "src_type", "dst_type")
  .agg(sum("trip_count").cast(LongType).as("total_trips"),
       avg("avg_travel_time").as("avg_time"))

val edgeRDD = edgesDF
  .filter(col("src_node_id").isNotNull && col("dst_node_id").isNotNull &&
          col("src_type").isNotNull && col("dst_type").isNotNull)
  .rdd.map { row =>
    val src = row.getAs[Long]("src_node_id") + offsetFor(row.getAs[String]("src_type"))
    val dst = row.getAs[Long]("dst_node_id") + offsetFor(row.getAs[String]("dst_type"))
    Edge(src, dst, row.getAs[Long]("total_trips").toDouble)
  }

// ── 3. Build the GraphX graph (vertices only need the id for analytics) ─
val vertexRDD = allNodes.select("vertex_id").rdd
  .filter(r => !r.isNullAt(0))
  .map(r => (r.getLong(0), ()))
val g = Graph(vertexRDD, edgeRDD, ())

println(s"  graph vertices=${g.vertices.count}, edges=${g.edges.count}")

// ── 4. Run analytics ──────────────────────────────────────────────────
println("\n[PageRank]")
val ranks = g.pageRank(0.001).vertices  // tol = 0.001

println("[Label Propagation — community detection]")
val communities = LabelPropagation.run(g, maxSteps = 5).vertices

println("[Degrees]")
val inDeg = g.inDegrees   // RDD[(VertexId, Int)]
val outDeg = g.outDegrees

// ── 5. Join everything together with node properties ──────────────────
import spark.implicits._
val ranksDF = ranks.toDF("vertex_id", "page_rank")
val commDF = communities.toDF("vertex_id", "community_id")
val inDF = inDeg.toDF("vertex_id", "in_degree")
val outDF = outDeg.toDF("vertex_id", "out_degree")

val features = allNodes
  .join(ranksDF, Seq("vertex_id"), "left")
  .join(commDF, Seq("vertex_id"), "left")
  .join(inDF, Seq("vertex_id"), "left")
  .join(outDF, Seq("vertex_id"), "left")
  .na.fill(Map("page_rank" -> 0.0, "community_id" -> 0L, "in_degree" -> 0, "out_degree" -> 0))
  .select("vertex_id", "node_type", "name", "lat", "lon", "borough", "capacity",
          "page_rank", "community_id", "in_degree", "out_degree")

features.write.mode("overwrite").parquet(s"$GRAPH/features/features.parquet")
println(s"\n  wrote ${features.count} rows to $GRAPH/features/features.parquet")

println("\nTop 10 taxi zones by PageRank:")
features.filter("node_type = 'taxi_zone'")
  .orderBy(desc("page_rank")).select("name", "borough", "page_rank", "community_id", "in_degree", "out_degree")
  .show(10, 40)

System.exit(0)
