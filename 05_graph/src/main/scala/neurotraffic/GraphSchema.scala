package neurotraffic

/**
 * Every node in the transport graph is one of these three types.
 * A taxi zone, a subway station, or a bike dock.
 */
object NodeType extends Enumeration {
  val TaxiZone, SubwayStation, BikeDock = Value
}

/**
 * Properties attached to each vertex in the graph.
 *
 * @param nodeId     Unique ID (taxi zone 1-263, subway stations 1000+, bike docks 5000+)
 * @param nodeType   Which transport mode this node belongs to
 * @param name       Human-readable name ("East Village", "14th St - Union Sq", "1 Ave & E 15 St")
 * @param lat        Latitude of the node center
 * @param lon        Longitude of the node center
 * @param capacity   Throughput capacity (passengers/hour for subway, trips/hour for taxi zones,
 *                   dock slots for bikes). Used to normalize demand metrics.
 * @param borough    NYC borough ("Manhattan", "Brooklyn", etc.)
 */
case class NodeProperties(
  nodeId:   Long,
  nodeType: NodeType.Value,
  name:     String,
  lat:      Double,
  lon:      Double,
  capacity: Int,
  borough:  String
)

/**
 * Properties attached to each edge in the graph.
 * An edge represents trips flowing from one node to another in a given time window.
 *
 * @param tripCount     Number of trips in this time window
 * @param avgTravelTime Average travel time in minutes
 * @param hourOfDay     Which hour this edge weight applies to (0-23). -1 means aggregate.
 * @param dayType       "weekday" or "weekend"
 */
case class EdgeProperties(
  tripCount:     Long,
  avgTravelTime: Double,
  hourOfDay:     Int,
  dayType:       String
)

/**
 * Output of graph analytics — attached back to each node after running PageRank
 * and community detection.
 *
 * @param pageRank      How "important" this node is in the network (higher = more critical)
 * @param communityId   Which transport community this node belongs to
 * @param inDegree      How many nodes send trips TO this node
 * @param outDegree     How many nodes this node sends trips TO
 */
case class GraphFeatures(
  pageRank:    Double,
  communityId: Long,
  inDegree:    Int,
  outDegree:   Int
)
