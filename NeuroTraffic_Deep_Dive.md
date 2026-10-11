# NeuroTraffic — Complete Deep-Dive Guide

Everything you need to understand about the NeuroTraffic project: what it does, how every file works, what Docker/Spark/HDFS/DAG/RDD/GraphX/MLlib mean, and how all the pieces connect.

---

## 1. What Is NeuroTraffic?

NeuroTraffic is a **Big Data pipeline** that detects transport disruptions across New York City by analyzing six real-world datasets together. When a subway line breaks down, people switch to taxis and bikes — NeuroTraffic detects that cross-modal shift pattern and flags it as a disruption.

**The core idea:** A disruption is not just "subway ridership dropped." It is: subway DOWN + taxi UP + bike UP happening at the same time in the same area. That three-signal pattern is the "disruption signature" the ML model learns to detect.

**Technology stack:**
- **Storage:** HDFS (Hadoop Distributed File System) — stores all raw and processed data across 3 DataNodes
- **Processing:** Apache Spark 3.3.0 with Scala 2.12 — cleans, joins, and analyzes the data
- **Graph Analytics:** GraphX (Spark's graph library) — models NYC transport as a network
- **Machine Learning:** Spark MLlib — trains models to detect and predict disruptions
- **Streaming:** Spark Structured Streaming — processes live data every 10 seconds
- **Dashboard:** Streamlit (Python) — web UI showing live maps, graphs, and alerts
- **Infrastructure:** Docker Compose — runs the entire Hadoop+Spark cluster in containers

---

## 2. The Data — What We Use and What Year

### Time Range: 2022-2024

The download script (01_ingestion/download_datasets.py) is configured with START_YEAR = 2022 and END_YEAR = 2024. All six datasets cover this three-year window.

### The Six Datasets

| # | Dataset | Source | Format | What It Contains |
|---|---------|--------|--------|------------------|
| 1 | **NYC TLC Taxi** | NYC Taxi and Limousine Commission | Parquet | Yellow, Green, and HVFHV (Uber/Lyft) trip records. ~100M+ trips/year. |
| 2 | **Citi Bike** | Citi Bike System Data (S3) | CSV (zipped) | Every bike trip — start/end station, coordinates, duration. ~30M+ trips/year. |
| 3 | **MTA Subway** | NY State Open Data (Socrata API) | CSV | Hourly ridership per station — entries, exits, transfers. ~50-80M rows for 2022-2024. |
| 4 | **NOAA Weather** | NOAA CDO API / Open-Meteo fallback | CSV | Hourly NYC Central Park weather — temperature, precipitation, wind, visibility, snow. |
| 5 | **NYC 311 Complaints** | NYC Open Data (Socrata API) | CSV | Transport-related complaints — street conditions, traffic signals, blocked driveways. |
| 6 | **NYC Event Permits** | NYC Open Data (Socrata API) | CSV | Street closures, parades, film shoots — events that disrupt normal traffic flow. |

### Why These Six?

- **Taxi + Bike + Subway** = the three main modes people switch between during disruptions
- **Weather** = external factor that affects all modes (rainstorm = fewer bikes, more taxis)
- **311 Complaints** = ground-truth signal of problems (traffic lights out, roads blocked)
- **Events** = known causes of disruptions (marathons, parades, film shoots)

### Data Volume

- Raw download: ~50-65 GB
- After cleaning: the combined table has ~6.9 million rows (263 zones x every hour for 3 years)
- Stored as Parquet in HDFS (compressed, columnar format)
---

## 3. Docker — What It Is and How Our Cluster Works

### What Is Docker?

Docker is a tool that runs applications inside **containers** — lightweight, isolated environments that include everything the application needs (OS, libraries, config). Instead of installing Hadoop and Spark directly on your machine (which is painful on Windows), we run them inside Docker containers that already have everything set up.

### What Is Docker Compose?

Docker Compose is a tool that manages **multiple containers** at once. Our docker-compose.yml defines 7 containers that form a complete Hadoop+Spark cluster.

### Commands

```
docker-compose up -d     <- Start all 7 containers in detached mode
docker-compose down      <- Stop and remove all containers
docker-compose ps        <- See which containers are running
docker-compose logs -f   <- Follow live logs from all containers
```

### Every Service Explained

#### namenode (Hadoop NameNode)
- Image: bde2020/hadoop-namenode:2.0.0-hadoop3.2.1-java8
- Ports: 9870 (Web UI), 9000 (RPC)
- Volumes: namenode_data (metadata), ./neurotraffic_data mounted at /data/local

**What it does:** The NameNode is the "brain" of HDFS. It keeps track of WHERE every file block is stored across the DataNodes. It does not store actual data — only the metadata (file names, block locations, permissions). Think of it as the library catalog that tells you which shelf a book is on.

#### datanode1, datanode2, datanode3 (Hadoop DataNodes)
- Ports: 9864, 9865, 9866 respectively
- Wait for NameNode via SERVICE_PRECONDITION

**What they do:** DataNodes store the actual file blocks. When you write a 300 MB Parquet file to HDFS, it gets split into blocks (128 MB each by default — see dfs_blocksize=134217728 in hadoop.env). Each block is **replicated 3 times** (dfs_replication=3) across the three DataNodes for fault tolerance. If datanode2 dies, the data is still on datanode1 and datanode3.

#### spark-master (Spark Master)
- Ports: 8080 (Web UI), 7077 (RPC), 4040 (App UI)
- All project folders mounted at /project/

**What it does:** The Spark Master coordinates work. When you submit a Spark job, the master decides which workers run it. The volumes mount your Scala scripts inside the container at /project/, so spark-shell -i /project/02_cleaning/clean_taxi.scala finds the file.

#### spark-worker-1, spark-worker-2 (Spark Workers)
- SPARK_WORKER_CORES=2 (2 CPU cores per worker)
- SPARK_WORKER_MEMORY=4g (4 GB RAM per worker)

**What they do:** Workers are the muscle. They execute the actual computations — reading data, filtering rows, training ML models. When Spark distributes a job, parts run in parallel across both workers.

### The Network

All 7 containers share a private Docker network called hadoop_net. This means namenode, spark-master, datanode1 etc. are DNS hostnames that containers use to find each other. From your Windows host, you access them via localhost on the mapped ports.

### The hadoop.env File

| Setting | Value | Meaning |
|---------|-------|---------|
| CORE_CONF_fs_defaultFS | hdfs://namenode:9000 | All HDFS paths resolve against this |
| HDFS_CONF_dfs_webhdfs_enabled | true | Enables HTTP REST API on port 9870 |
| HDFS_CONF_dfs_replication | 3 | Every block stored on all 3 DataNodes |
| HDFS_CONF_dfs_blocksize | 134217728 | 128 MB per block |
| HDFS_CONF_dfs_permissions_enabled | false | No file permission checks (dev cluster) |
| MAPRED_CONF_mapreduce_map_memory_mb | 4096 | 4 GB per map task |
| MAPRED_CONF_mapreduce_reduce_memory_mb | 8192 | 8 GB per reduce task |

### What docker-compose up -d Actually Does

1. Pulls the Docker images (first time only, ~2-3 GB total)
2. Creates the hadoop_net network
3. Creates persistent volumes (namenode_data, datanode1_data, etc.) so data survives container restarts
4. Starts the NameNode first (other services wait via depends_on and SERVICE_PRECONDITION)
5. Starts the 3 DataNodes (they register with the NameNode)
6. Starts the Spark Master
7. Starts the 2 Spark Workers (they register with the Master)

After ~30 seconds, the full cluster is ready.

### What docker-compose down Actually Does

1. Sends SIGTERM to all 7 containers (graceful shutdown)
2. Removes the containers (but NOT the volumes — your data persists)
3. Removes the network

To destroy volumes too (lose all HDFS data): docker-compose down -v

---

## 4. HDFS — The Distributed File System

### What Is HDFS?

HDFS (Hadoop Distributed File System) is a file system designed to store very large files across multiple machines. Instead of storing a 500 MB file on one disk, HDFS splits it into blocks and distributes them across DataNodes.

### How HDFS Works in Our Project

```
Your Scala script: spark.read.parquet("hdfs://namenode:9000/data/neurotraffic/combined_v3")

1. Spark asks NameNode: "Where are the blocks for combined_v3?"
2. NameNode replies: "Block 1 is on datanode1+datanode3, Block 2 on datanode2+datanode1..."
3. Spark workers read blocks directly from DataNodes (parallel reads = fast)
4. Each worker processes its blocks locally (data locality — no network transfer needed)
```

### HDFS Directory Layout

```
hdfs://namenode:9000/data/
  neurotraffic/
    raw/           <- Original data uploaded from local (taxi, citibike, subway, weather, 311, events)
    cleaned/       <- After Stage 2 cleaning (Parquet)
    combined/      <- Stage 3: joined 6 sources
    combined_v3/   <- Final combined table (263 zones x hours, ~6.9M rows)
    graph_features/<- Stage 5: PageRank, communities
    disruptions/   <- Stage 6: anomaly detection output
    predictions/   <- Stage 6: GBT model predictions
    models/        <- Saved ML model (Spark Pipeline)
    evaluation/    <- Model evaluation metrics JSON
    mapreduce_output/ <- Stage 4 results
    dashboard/     <- Stage 8: dashboard-ready files
  streaming/
    input/         <- JSON micro-batches (from fetcher)
    output/        <- latest.csv (scoring results)
    checkpoint/    <- Spark streaming checkpoint
```

### WebHDFS — How the Dashboard Reads HDFS

The Streamlit dashboard runs on your Windows host (outside Docker), so it cannot use HDFS's native Java client. Instead, it uses **WebHDFS** — a REST API exposed on port 9870.

```
GET http://localhost:9870/webhdfs/v1/data/streaming/output/latest.csv?op=OPEN
-> 307 redirect to datanode1:9864/...
-> Dashboard follows redirect (with hostname fix: datanode1 -> localhost:9864)
-> Gets the CSV content
```

This is what hdfs_helper.py handles — it makes HTTP requests to WebHDFS and fixes Docker hostname redirects.

---

## 5. Apache Spark — The Processing Engine

### What Is Spark?

Apache Spark is a distributed computing engine. Instead of processing data on one machine, Spark splits the work across many machines (workers) and processes in parallel. This is how we handle ~6.9 million rows without waiting hours.

### Key Concepts

**SparkSession:** The entry point. In spark-shell, it is pre-created as `spark`.

**Driver vs Executors:**
- Driver = your spark-shell process. It plans the work and coordinates.
- Executors = processes on the workers that do the actual computation.

When you write df.filter(col("fare") > 0).count(), the driver sends instructions to executors, who process their partitions in parallel and send back partial results.

**Partitions:** Data in Spark is split into partitions — chunks that executors process independently. A 1 GB Parquet file might have 8 partitions. More partitions = more parallelism.

**Lazy Evaluation:** Spark is lazy — it does not do work until you ask for a result. Writing df.filter(...).withColumn(...) just builds a plan. Only when you call an action like .count(), .show(), .write.parquet(...) does Spark actually execute. This lets Spark optimize the whole chain.

---

## 6. RDD vs DataFrame — Two Ways to Handle Data in Spark

### RDD (Resilient Distributed Dataset)

The original Spark abstraction. An RDD is a distributed collection of objects. You process it with functional operations: map, filter, reduce, groupBy.

**Used in our project for:** GraphX. GraphX requires RDDs — you cannot use DataFrames for graph vertices and edges. This is why GraphBuilder.scala uses RDD[(VertexId, NodeProperties)] and RDD[Edge[EdgeProperties]].

**How RDD is "resilient":** If a partition is lost (worker crashes), Spark re-computes it from the original data using the lineage (the chain of transformations). No data is permanently lost.

### DataFrame

A higher-level abstraction built on top of RDD. A DataFrame is like a SQL table — named columns with types. Spark optimizes DataFrame operations using its **Catalyst optimizer** (query planning) and **Tungsten engine** (memory management).

**Used in our project for:** Everything except GraphX. All cleaning, joining, aggregation, and ML pipelines use DataFrames because they are faster and easier to write.

| Aspect | RDD | DataFrame |
|--------|-----|-----------|
| Speed | Slower (no optimizer) | Faster (Catalyst + Tungsten) |
| API | Functional (map/filter/reduce) | SQL-like (select/filter/groupBy) |
| Type safety | Compile-time (Scala case classes) | Runtime (column names as strings) |
| Used for | GraphX, custom low-level ops | ETL, joins, aggregations, ML |

---

## 7. DAG — How Spark Plans Its Work

### What Is a DAG?

DAG stands for **Directed Acyclic Graph**. It is the execution plan Spark creates when you trigger an action.

- **Directed** = edges have a direction (step A feeds into step B)
- **Acyclic** = no cycles (you never go back to a previous step)
- **Graph** = nodes (operations) connected by edges (data flow)

### Example

```scala
val result = spark.read.parquet("hdfs://...")   // (1) Read
  .filter(col("fare") > 0)                      // (2) Filter
  .groupBy("zone_id")                            // (3) Group
  .agg(count("*").as("trips"))                   // (4) Aggregate
  .orderBy(desc("trips"))                        // (5) Sort
```

Spark creates this DAG:
```
Read Parquet -> Filter -> Shuffle (redistribute by zone_id) -> Aggregate -> Sort
     Stage 1                              Stage 2                     Stage 3
```

### Stages and Shuffles

Spark groups operations into **stages**. A new stage starts at a **shuffle** — where data must be redistributed.

- **filter** = narrow transformation — each partition processed independently. No data moves.
- **groupBy** = wide transformation — all rows with same zone_id must go to same partition. Requires a shuffle (data sent across the network).

Shuffles are expensive. Our pipeline minimizes them by:
- Broadcast joins for small tables (weather baselines)
- Partitioning output by year/month
- Caching intermediate results (grid.cache() in join_all.scala)

### Viewing the DAG

Open http://localhost:4040 when a Spark job is active to see the DAG, stage durations, shuffle sizes, and task details.

---

## 8. The 8-Stage Pipeline — Full Walkthrough

```
Stage 1: Download (Python) -> Stage 2: Clean (6 Scala scripts) -> Stage 3: Join All (Scala)
-> Stage 4: MapReduce (Scala) -> Stage 5: Graph (GraphX) -> Stage 6: ML (MLlib)
-> Stage 7: Streaming (Structured Streaming) -> Stage 8: Dashboard (Streamlit)
```

### Stage 1: Data Ingestion (01_ingestion/download_datasets.py)
Downloads all 6 datasets (~50-65 GB). Supports resume. Run on your host machine.

### Stage 2: Data Cleaning (02_cleaning/)

**clean_taxi.scala** — Reads Yellow/Green/HVFHV Parquet. Standardizes columns, filters nulls, zero-distance trips, impossible durations (<30s or >12h), invalid zone IDs. HVFHV processed in batches of 6 files (684M+ rows).

**clean_subway.scala** — Aggregates raw data (one row per station x hour x payment method) into one row per station x hour. Removes outliers beyond 5 standard deviations.

**clean_bike.scala** — Deduplicates by ride_id. Filters NYC bounding box, duration 1min-24hrs. Removes dock errors (same station, <2 min).

**clean_weather.scala** — Computes binary flags: is_rain, is_heavy_rain, is_snow, is_extreme_cold, is_extreme_heat, is_low_visibility, is_high_wind.

**clean_311.scala** — Filters transport-relevant complaints. Assigns severity scores.

**clean_events.scala** — Calculates disruption severity from event type and streets affected.

### Stage 3: Data Integration (03_integration/join_all.scala)

The most important stage. Creates the **combined table**:

1. Build grid: 263 zones x every hour (2022-2024) = ~6.9M rows
2. Aggregate taxi by pickup/dropoff zone + hour
3. Join weather (city-wide, 1-to-many on hourly_timestamp)
4. Join subway by borough + hour
5. Join 311 and events by borough + hour
6. Fill nulls with 0
7. Write partitioned Parquet (by year/month)

Output: 38 columns including zone_id, hourly_timestamp, taxi_pickups, taxi_dropoffs, subway_ridership, bike_starts, temperature_c, precipitation_mm, complaint_count, event_count, and time features.

### Stage 4: MapReduce (04_mapreduce/mapreduce_spark.scala)

Demonstrates MapReduce pattern: total taxi pickups per zone per day. In Spark this is just:
```scala
df.groupBy("zone_id", "date").agg(sum("taxi_pickups"))
```

### Stages 5-8: See dedicated sections below.

---

## 9. GraphX — Modeling NYC as a Network

### What Is GraphX?

GraphX is Spark's library for graph computation. A graph has vertices (nodes) and edges (connections).

### Our Transport Graph

| Node Type | ID Range | Count | Example |
|-----------|----------|-------|---------|
| Taxi Zones | 1-263 | 263 | East Village, Times Square |
| Subway Stations | 1000-1999 | ~400+ | 14th St - Union Sq |
| Bike Docks | 5000+ | ~1800+ | 1 Ave & E 15 St |

Edges represent trips between nodes. An edge from Zone 161 (Midtown) to Station 1042 (Penn Station) with tripCount=50000 means 50,000 trips connected them.

Edges have time dimensions: hourOfDay (0-23) and dayType ("weekday"/"weekend").

### Node Properties (GraphSchema.scala)
nodeId, nodeType (TaxiZone/SubwayStation/BikeDock), name, lat, lon, capacity, borough

### Edge Properties
tripCount, avgTravelTime, hourOfDay, dayType

### Algorithms

**PageRank** — "Which nodes are most critical?" A node scores high if many important nodes connect to it. Times Square ranks high; quiet Staten Island zones rank low. Uses tripCount as edge weight. Parameters: tol=0.001, resetProb=0.15.

**Label Propagation** — "Which nodes cluster together?" Each node adopts the most common label among its neighbors. After 10 iterations, nodes exchanging lots of trips share a label = same community. Midtown zones + Penn Station + nearby bike docks form one community.

**Degree Computation** — In-degree = how many nodes send trips here (popular destination). Out-degree = how many nodes this sends trips to (popular origin).

### Graph Features as ML Input

Output: page_rank, community_id, in_degree, out_degree per node. These become ML features — a high-PageRank zone with sudden low ridership is a stronger disruption signal.

---

## 10. MLlib — Machine Learning on Spark

### What Is MLlib?

Spark's distributed ML library. Trains on millions of rows across a cluster — unlike sklearn which runs on one machine.

### The ML Pipeline (disruption_detector.scala) — 7 Steps

**Step 1: Load** — Read combined_v3 (6.9M rows) + graph features. Join on zone_id.

**Step 2: Feature Engineering** — Create cross-modal features:
- taxi_to_subway_ratio = taxi_pickups / subway_ridership (spikes during subway disruptions)
- bike_to_subway_ratio = bike_starts / subway_ridership
- total_surface_transport = taxi_pickups + bike_starts
- complaint_density = complaint_count / taxi_pickups
Total: 27 features including time, weather, and graph metrics.

**Step 3: Inject Synthetic Disruptions (3.5%)** — Subway ridership reduced to 5-20%, taxi pickups increased 200-400%, bike starts increased 180-330%, extra complaints added. Creates labeled training data.

**Step 4: Anomaly Detection (Z-Score + KMeans)** — Why not Isolation Forest? Spark MLlib does not have it. Instead:

Z-Score: For each zone x hour x weekday/weekend, compute mean and std. A row is anomalous if z_subway < -1.5 AND (z_pickups > 1.5 OR z_bikes > 1.5).

Anomaly Score (0-1):
```
score = max(0, -z_subway) * 0.4 + max(0, z_pickups) * 0.3 + max(0, z_bikes) * 0.3
```

**Step 5: GBT Classifier** — Gradient Boosted Trees. 100 rounds of decision trees, each correcting previous mistakes.
- maxIter=100, maxDepth=5, stepSize=0.1, subsamplingRate=0.8
- Class weighting: anomalies weighted ~28x to handle imbalance
- Pipeline: VectorAssembler -> StandardScaler -> GBTClassifier

**Step 6: Save Model** — Saved to HDFS as a Spark Pipeline (loadable without retraining).

**Step 7: Save Results** — Predictions Parquet, evaluation_metrics.json (accuracy, precision, recall, F1, AUC-ROC), feature importance.

### Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| Accuracy | % of all predictions correct |
| Precision | Of predicted disruptions, what % are real? |
| Recall | Of real disruptions, what % did we catch? |
| F1 Score | Balanced precision/recall |
| AUC-ROC | Ability to distinguish disrupted from normal (0.5=random, 1.0=perfect) |

---

## 11. Spark Structured Streaming — Real-Time Processing

### What Is Structured Streaming?

Treats a live data feed as an ever-growing table. Every few seconds, new rows arrive, get processed, and output is updated. Same DataFrame API as batch — the only difference is spark.readStream instead of spark.read.

### How It Works (streaming_pipeline.scala)

```
realtime_fetcher.py  ->  streaming_pipeline.scala  ->  Dashboard
writes JSON to HDFS      reads, scores, writes CSV      reads latest.csv
```

**Step 1:** Load historical baselines (per zone x hour x weekday/weekend means and std devs). Broadcast to all workers.

**Step 2:** Define input schema (35 columns: counts, weather, MTA alerts).

**Step 3:** Every 10 seconds, process new JSON files:
1. Join with baselines
2. Compute z-scores (capped at 3.0)
3. Calculate composite anomaly score:
   - max_z * 0.04 + avg_z * 0.03 (statistical deviation)
   - weather_penalty (rain/snow/wind boost)
   - complaint_signal (311 excess)
   - disruption_indicator (0.45-0.90 if active MTA alert)
4. Flag zones with score >= 0.80 as disrupted
5. Write latest.csv to HDFS

The **disruption indicator** is the strongest signal: an active MTA service alert pushes the score to 0.45 (reduced service) through 0.90 (suspension).

**Checkpoint** at /data/streaming/checkpoint ensures exactly-once processing on restart.

**streaming_simulator.py** generates fake data for testing without live MTA/weather feeds.

---

## 12. The Dashboard — Streamlit Web UI

### What Is Streamlit?

Python framework that turns scripts into web apps. No HTML/CSS/JS needed. Python-only — no Scala equivalent exists.

### Pages

| File | Page | What It Shows |
|------|------|---------------|
| app.py | Home | System status, disruption count, pipeline diagram, data checks |
| pages/1_live_monitor.py | Live Monitor | Real-time NYC map (green/yellow/red zones). Auto-refreshes 5s. |
| pages/2_simulator.py | Simulator | Pick subway line -> see cascade predictions via GBT model |
| pages/3_explorer.py | Explorer | Historical trends, disruption timeline, borough breakdown |
| pages/4_analytics.py | Analytics | Model metrics, feature importance, confusion matrix, graph stats |

### Data Flow

config.py defines all file paths (HDFS and local). hdfs_helper.py reads from HDFS via WebHDFS REST API (port 9870) and falls back to local files. It handles Docker hostname redirects (datanode1:9864 -> localhost:9864).

Run with: `streamlit run 08_dashboard/app.py` -> http://localhost:8501

---

## 13. File-by-File Reference

```
Big_Data_Project/
+-- docker-compose.yml              <- 7-container cluster definition
+-- hadoop.env                       <- Hadoop config for all containers
|
+-- 01_ingestion/
|   +-- download_datasets.py         <- Downloads all 6 datasets (Python, ~50-65 GB)
|
+-- 02_cleaning/
|   +-- clean_taxi.scala              <- Clean Yellow/Green/HVFHV taxi
|   +-- clean_subway.scala            <- Clean MTA subway ridership
|   +-- clean_bike.scala              <- Clean Citi Bike trips
|   +-- clean_weather.scala           <- Clean weather + compute flags
|   +-- clean_311.scala               <- Clean 311 complaints
|   +-- clean_events.scala            <- Clean event/film permits
|   +-- extract_bike_stations.scala   <- Extract unique bike stations
|   +-- map_bike_stations_to_zones.scala <- Map bike stations to taxi zones
|
+-- 03_integration/
|   +-- join_all.scala                <- Join 6 sources into combined table
|   +-- add_bike_to_combined.scala    <- Add bike columns
|   +-- fix_311_join.scala            <- Fix 311 join issues
|   +-- analysis_export.scala         <- Export analysis-ready subset
|
+-- 04_mapreduce/
|   +-- export_for_mapreduce.scala    <- Export for MapReduce
|   +-- mapreduce_spark.scala         <- MapReduce via Spark
|
+-- 05_graph/
|   +-- build_graph_inputs.scala      <- Prepare node/edge data
|   +-- build_edges_only.scala        <- Build edge table
|   +-- run_graph_pipeline.scala      <- Orchestrate graph pipeline
|   +-- src/main/scala/neurotraffic/
|       +-- GraphSchema.scala         <- NodeProperties, EdgeProperties, GraphFeatures
|       +-- GraphBuilder.scala        <- Load nodes/edges, build graph
|       +-- GraphAnalytics.scala      <- PageRank, Label Propagation, degrees
|       +-- NeuroTrafficApp.scala     <- Graph pipeline entry point
|       +-- ProximityEdges.scala      <- Edges from geographic proximity
|
+-- 06_ml/
|   +-- detect_anomalies.scala        <- Z-score + KMeans anomaly detection
|   +-- disruption_detector.scala     <- Full ML pipeline (GBT + eval)
|   +-- prepare_data.scala            <- Prepare ML training data
|   +-- prepare_ml_training.scala     <- Alternative ML data prep
|   +-- scoring_module.scala          <- Score new data with trained model
|   +-- train_gbt.scala               <- Train GBT separately
|   +-- tune_contamination.scala      <- Tune z-score thresholds
|   +-- run_pipeline.scala            <- ML pipeline orchestrator
|
+-- 07_streaming/
|   +-- streaming_pipeline.scala      <- Spark Structured Streaming
|   +-- build_live_profiles.scala     <- Per-zone streaming baselines
|   +-- realtime_fetcher.py           <- Fetch live MTA/weather (Python)
|   +-- build_live_lookups.py         <- Lookup tables for live scoring
|   +-- spark_live_processor.py       <- Alternative PySpark streaming
|   +-- live_processor.py             <- Lightweight live processor
|   +-- streaming_simulator.py        <- Generate fake streaming data
|
+-- 08_dashboard/
|   +-- app.py                        <- Streamlit home page
|   +-- config.py                     <- All paths, constants, settings
|   +-- hdfs_helper.py                <- WebHDFS REST API client
|   +-- generate_dashboard_data.scala <- Export dashboard files from HDFS
|   +-- pages/
|       +-- 1_live_monitor.py         <- Real-time NYC zone map
|       +-- 2_simulator.py            <- Disruption cascade simulator
|       +-- 3_explorer.py             <- Historical trend explorer
|       +-- 4_analytics.py            <- Model metrics and graph stats
|
+-- neurotraffic_data/                <- Raw data (gitignored, ~50-65 GB)
```

---

## 14. How to Run Everything (Step-by-Step)

### Prerequisites
- Docker Desktop installed and running
- Python 3.8+ with: pip install streamlit pandas requests pyarrow tqdm
- ~60 GB disk space for data + ~5 GB for Docker images

### Step 1: Download Data
```
python 01_ingestion/download_datasets.py
```

### Step 2: Start the Cluster
```
docker-compose up -d
```
Wait ~30 seconds. Verify at http://localhost:9870 (HDFS) and http://localhost:8080 (Spark).

### Step 3: Upload Data to HDFS
```
docker exec namenode hdfs dfs -mkdir -p /data/neurotraffic/raw
docker exec namenode hdfs dfs -put /data/local/taxi /data/neurotraffic/raw/taxi
docker exec namenode hdfs dfs -put /data/local/citibike_csv /data/neurotraffic/raw/citibike
docker exec namenode hdfs dfs -put /data/local/subway /data/neurotraffic/raw/subway
docker exec namenode hdfs dfs -put /data/local/weather /data/neurotraffic/raw/weather
docker exec namenode hdfs dfs -put /data/local/311 /data/neurotraffic/raw/311
docker exec namenode hdfs dfs -put /data/local/events /data/neurotraffic/raw/events
```

### Step 4: Run Cleaning
```
docker exec spark-master /spark/bin/spark-shell -i /project/02_cleaning/clean_taxi.scala
docker exec spark-master /spark/bin/spark-shell -i /project/02_cleaning/clean_subway.scala
docker exec spark-master /spark/bin/spark-shell -i /project/02_cleaning/clean_bike.scala
docker exec spark-master /spark/bin/spark-shell -i /project/02_cleaning/clean_weather.scala
docker exec spark-master /spark/bin/spark-shell -i /project/02_cleaning/clean_311.scala
docker exec spark-master /spark/bin/spark-shell -i /project/02_cleaning/clean_events.scala
```

### Step 5: Join All Sources
```
docker exec spark-master /spark/bin/spark-shell -i /project/03_integration/join_all.scala
```

### Step 6: MapReduce
```
docker exec spark-master /spark/bin/spark-shell -i /project/04_mapreduce/mapreduce_spark.scala
```

### Step 7: Graph Analytics
```
docker exec spark-master /spark/bin/spark-shell -i /project/05_graph/build_graph_inputs.scala
docker exec spark-master /spark/bin/spark-shell -i /project/05_graph/run_graph_pipeline.scala
```

### Step 8: ML Pipeline
```
docker exec spark-master /spark/bin/spark-shell --driver-memory 4g -i /project/06_ml/disruption_detector.scala
```

### Step 9: Dashboard Data
```
docker exec spark-master /spark/bin/spark-shell -i /project/08_dashboard/generate_dashboard_data.scala
```

### Step 10: Start Streaming
Terminal 1:
```
docker exec spark-master /spark/bin/spark-shell --master "local[2]" --driver-memory 2g --conf spark.sql.shuffle.partitions=8 -i /project/07_streaming/streaming_pipeline.scala
```
Terminal 2:
```
python 07_streaming/streaming_simulator.py
```

### Step 11: Launch Dashboard
```
streamlit run 08_dashboard/app.py
```

### Step 12: Stop Everything
```
docker-compose down
```

---

## Quick Reference — Port Map

| Port | Service | URL |
|------|---------|-----|
| 9870 | HDFS NameNode Web UI | http://localhost:9870 |
| 9000 | HDFS RPC | hdfs://namenode:9000 |
| 8080 | Spark Master Web UI | http://localhost:8080 |
| 4040 | Spark App UI (active jobs) | http://localhost:4040 |
| 7077 | Spark Master RPC | spark://spark-master:7077 |
| 9864 | DataNode 1 | http://localhost:9864 |
| 9865 | DataNode 2 | http://localhost:9865 |
| 9866 | DataNode 3 | http://localhost:9866 |
| 8501 | Streamlit Dashboard | http://localhost:8501 |

---

## 15. Live Data — Where It Comes From, How It Works

### Overview

The streaming layer has two modes:

1. **Real live data** (`realtime_fetcher.py`) — polls actual NYC public APIs every 10 seconds and writes JSON micro-batches to HDFS
2. **Simulated data** (`streaming_simulator.py`) — generates fake but realistic data for testing without API access

Both write to the same HDFS path (`/data/streaming/input/`), so the Spark Structured Streaming pipeline (`streaming_pipeline.scala`) consumes them identically.

### The 9 Live Data Sources

| # | Source | API URL | Refresh | What It Returns |
|---|--------|---------|---------|-----------------|
| 1 | Weather | `https://api.open-meteo.com/v1/forecast` | 5 min | Temperature, humidity, precipitation, wind speed, weather code for NYC |
| 2 | MTA Service Alerts | `https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/camsys%2Fsubway-alerts.json` | 60 s | Active subway alerts (delays, planned work, service changes) |
| 3 | MTA Subway Trains | `https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2F{feed}` | 30 s | Real-time train positions from 8 GTFS-RT feeds |
| 4 | NYC DOT Traffic Speeds | `https://data.cityofnewyork.us/resource/i4gi-tjb9.json` | 2 min | Link-level traffic speeds across NYC road segments |
| 5 | Citi Bike Station Status | `https://gbfs.citibikenyc.com/gbfs/en/station_status.json` | 60 s | Available bikes and docks at every station |
| 6 | Citi Bike Station Info | `https://gbfs.citibikenyc.com/gbfs/en/station_information.json` | Once | Station names, coordinates, capacity (static reference data) |
| 7 | NYC 311 Complaints | `https://data.cityofnewyork.us/resource/erm2-nwe9.json` | 30 min | Recent 311 complaints (noise, blocked driveway, etc.) |
| 8 | NYC Events/Film Permits | `https://data.cityofnewyork.us/resource/tvpp-9vvx.json` | 30 min | Active event and film permits across the city |
| 9 | Taxi Pickups + Subway Ridership | Historical profiles (no public real-time API) | 10 s | Estimated from per-zone hourly averages built from 2022-2024 data |

All APIs are free and require no API key (Open-Meteo, MTA GTFS, NYC Open Data, Citi Bike GBFS).

### MTA GTFS-RT Feed Details

The MTA publishes 8 separate GTFS-RT feeds, one per subway line group:

| Feed ID | Lines Covered |
|---------|---------------|
| `gtfs` | 1, 2, 3, 4, 5, 6, 7, S (numbered lines) |
| `gtfs-ace` | A, C, E |
| `gtfs-bdfm` | B, D, F, M |
| `gtfs-g` | G |
| `gtfs-jz` | J, Z |
| `gtfs-l` | L |
| `gtfs-nqrw` | N, Q, R, W |
| `gtfs-si` | SIR (Staten Island Railway) |

The fetcher polls all 8 feeds every 30 seconds, counts active trips near each taxi zone, and maps them to zones using a station-to-zone lookup table.

### How realtime_fetcher.py Works

The fetcher runs as a Python script with a main loop that executes every 10 seconds:

```
while running:
    1. ThreadPoolExecutor fetches all 7 API sources in parallel
    2. Each source class (LiveWeather, LiveMTA, LiveSubwayTrains, etc.) handles its own API
    3. Results are combined into a per-zone JSON record (one row per taxi zone)
    4. Taxi pickups and subway ridership are estimated from historical profiles
    5. JSON micro-batch is written to HDFS: /data/streaming/input/batch_<timestamp>.json
    6. Source health status is written to: /data/streaming/output/source_status.json
    7. Sleep 10 seconds, repeat
```

**Key classes in realtime_fetcher.py:**

| Class | Responsibility |
|-------|---------------|
| `LiveWeather` | Calls Open-Meteo API, returns temp/humidity/precip/wind/code |
| `LiveMTA` | Fetches subway alerts JSON, parses active alerts per line |
| `LiveSubwayTrains` | Fetches all 8 GTFS-RT feeds, counts trains per zone |
| `LiveTrafficSpeeds` | Fetches NYC DOT speed data, maps road segments to zones |
| `LiveCitiBike` | Joins station status + info, aggregates by zone |
| `Live311` | Fetches recent 311 complaints, counts per zone |
| `LiveEvents` | Fetches event/film permits, counts per zone |
| `ZoneLocator` | Point-in-polygon lookup: lat/lon -> taxi zone ID |

### Prerequisites for Live Data

Before running `realtime_fetcher.py`, two prep steps are needed:

**Step 1: Build lookup tables**
```
python 07_streaming/build_live_lookups.py --upload
```
This creates:
- `zone_centroids.csv` — centroid lat/lon for each of 263 taxi zones (from TLC shapefile)
- `subway_station_zones.csv` — maps each subway station to its taxi zone using Shapely point-in-polygon
- Fixes `graph_nodes.csv` coordinates and uploads everything to HDFS

**Step 2: Build historical profiles**
```
docker exec spark-master /spark/bin/spark-shell -i /project/07_streaming/build_live_profiles.scala
```
This computes per-zone, per-hour, per-day-type (weekday vs weekend) averages from the 2022-2024 combined dataset. These profiles are used to estimate taxi pickups and subway ridership in real-time (since no public real-time API exists for those).

### How Taxi Pickups and Subway Ridership Are Estimated

There is no public real-time API for NYC taxi pickups or MTA subway turnstile counts. The fetcher uses **historical hourly profiles**:

1. `build_live_profiles.scala` computes: for each (zone, hour, weekday/weekend), what is the average taxi pickup count and subway ridership from the 2022-2024 data?
2. At runtime, the fetcher looks up the current zone + hour + day-type and uses that average
3. A small random noise factor is added so the values are not perfectly static
4. This means the "live" taxi and subway numbers are estimates, not true real-time counts

### The JSON Micro-Batch Format

Each batch file written to HDFS (`/data/streaming/input/batch_<timestamp>.json`) contains one JSON record per taxi zone:

```json
{
  "zone_id": 161,
  "timestamp": "2024-03-15T14:30:00",
  "taxi_pickups": 245,
  "subway_ridership": 1820,
  "bike_starts": 34,
  "bike_ends": 28,
  "temperature_c": 12.5,
  "humidity_pct": 65,
  "precipitation_mm": 0.0,
  "wind_speed_kmh": 18.3,
  "weather_code": 0,
  "is_rain": 0,
  "is_snow": 0,
  "traffic_speed_mph": 22.4,
  "complaint_count": 3,
  "event_count": 1,
  "subway_trains_nearby": 5,
  "active_alerts": ["A","C"],
  "disruption_severity": 0.6,
  "affected_line": "A"
}
```

### How streaming_pipeline.scala Processes Live Data

Once JSON micro-batches land in HDFS, Spark Structured Streaming picks them up:

```
1. READ: Spark watches /data/streaming/input/ for new JSON files (FileStream source)
2. PARSE: Each JSON record is parsed into a DataFrame with the schema above
3. BASELINE LOOKUP: Pre-computed baselines (mean + stddev per zone/hour/day-type) are broadcast
4. Z-SCORE: For each row, compute z-scores:
   - z_subway = (subway_ridership - mean) / stddev
   - z_pickups = (taxi_pickups - mean) / stddev
   - z_bikes = (bike_starts - mean) / stddev
5. ANOMALY SCORE: Weighted combination (subway DOWN 40% + taxi UP 30% + bike UP 30%), clamped 0-1
6. DISRUPTION FLAG: is_disruption = 1 if z_subway < -1.5 AND (z_pickups > 1.5 OR z_bikes > 1.5)
7. OUTPUT: Write latest scores to /data/streaming/output/latest.csv (overwrite each trigger)
```

The pipeline triggers every 10 seconds (`Trigger.ProcessingTime("10 seconds")`), so there is at most a 10-second delay between data arriving and scores updating.

### How the Dashboard Displays Live Data

The Live Monitor page (`pages/1_live_monitor.py`) reads the streaming output:

1. **latest.csv** — read via WebHDFS every 3-5 seconds (configurable auto-refresh)
2. **source_status.json** — read to show which data sources are active and their freshness
3. **graph_nodes.csv** — provides lat/lon coordinates and zone names for the map

The page merges these into a Plotly Scattermap of NYC with 263 dots:
- **Green** (score < 0.50) — Normal operation
- **Yellow** (0.50 - 0.80) — Anomaly detected
- **Red** (> 0.80) — Critical disruption (with pulsing ring animation)

Below the map: anomaly score distribution chart, active disruptions table, top anomaly zones table, and a data source health panel showing each source's age and provider.

### The Simulator Alternative

If you cannot access the live APIs (no internet, APIs down, or just testing), use the simulator:

```
python 07_streaming/streaming_simulator.py
```

The simulator generates fake but realistic data:
- **WeatherState class** slowly drifts temperature, humidity, wind — no sudden jumps
- **DisruptionManager** randomly creates subway disruptions (~8% chance per tick) with realistic cascade effects (subway ridership drops, taxi/bike surge, complaints rise)
- **Rush hour patterns** apply multipliers based on hour of day
- Writes to the same HDFS path as the real fetcher, so the streaming pipeline and dashboard work identically

### End-to-End Data Flow Diagram

```
                    LIVE APIs                          SIMULATOR
                    --------                           ---------
  Open-Meteo ─┐                              streaming_simulator.py
  MTA Alerts ─┤                                   generates fake
  MTA Trains ─┤                                   zone-level data
  NYC DOT    ─┤── realtime_fetcher.py ──┐              │
  Citi Bike  ─┤   (polls every 10s)     │              │
  311        ─┤   (ThreadPoolExecutor)  │              │
  Events     ─┘                         │              │
                                        ▼              ▼
                              HDFS /data/streaming/input/
                                   batch_<ts>.json
                                        │
                                        ▼
                            streaming_pipeline.scala
                            (Spark Structured Streaming)
                            - z-score vs baselines
                            - anomaly score 0-1
                            - disruption flag
                                        │
                                        ▼
                              HDFS /data/streaming/output/
                              ├── latest.csv (263 rows)
                              └── source_status.json
                                        │
                                        ▼
                              Streamlit Dashboard
                              pages/1_live_monitor.py
                              (WebHDFS read every 3-5s)
                              - NYC map with colored dots
                              - Disruption alerts table
                              - Source health panel
```

### source_status.json

The fetcher writes a health report to HDFS after every cycle. The dashboard reads this to show which sources are fresh:

```json
{
  "generated_at": "2024-03-15T14:30:10-04:00",
  "sources": [
    {"name": "weather", "kind": "api", "provider": "Open-Meteo", "observed_at": "2024-03-15T14:30:08-04:00", "detail": "12.5C, clear"},
    {"name": "mta_alerts", "kind": "api", "provider": "MTA", "observed_at": "2024-03-15T14:30:09-04:00", "detail": "3 active alerts"},
    {"name": "subway_trains", "kind": "api", "provider": "MTA GTFS-RT", "observed_at": "2024-03-15T14:30:09-04:00", "detail": "847 active trips"}
  ],
  "mta_alerts": [
    {"lines": ["A","C"], "type": "Delays", "header": "Signal problems at 125 St"}
  ]
}
```

If a source errors, the `error` field is populated instead of `detail`, and the dashboard shows it in red.
