name := "neurotraffic-graphx"
version := "1.0.0"
scalaVersion := "2.12.18"

// Target Java 8 bytecode — Spark 3.5 doesn't support Java 21
javacOptions ++= Seq("-source", "1.8", "-target", "1.8")
scalacOptions += "-target:jvm-1.8"

// Must match your Spark cluster version — adjust if needed
val sparkVersion = "3.5.1"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"    % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql"     % sparkVersion % "provided",
  "org.apache.spark" %% "spark-graphx"  % sparkVersion % "provided"
)

// Build a fat JAR with all your code (Spark JARs are on the cluster already)
assembly / assemblyJarName := "neurotraffic-graphx.jar"

// Avoid duplicate-file conflicts when building the JAR
assembly / assemblyMergeStrategy := {
  case PathList("META-INF", _*) => MergeStrategy.discard
  case _                        => MergeStrategy.first
}
