# NeuroTraffic — Project Documentation

## Overview

NeuroTraffic is a Big Data pipeline that detects and predicts multi-modal transport disruptions across New York City's 263 taxi zones. It ingests six public datasets, cleans and joins them using Apache Spark (Scala), runs graph analytics with GraphX, trains ML models for anomaly detection, and streams live NYC data through a real-time scoring pipeline — all visualised in an interactive Streamlit dashboard.

**Technology stack:** Apache Spark 3.3.0 (Scala 2.12) on Docker, HDFS (3 DataNodes), PySpark, scikit-learn, Streamlit, Plotly, GraphX.

---

## 1. Datasets

### 1.1 Historical Datasets (2022-2024) — Downloaded Once, Cleaned in Spark

| # | Dataset | Source | API / URL | Format | Volume |
|---|---------|--------|-----------|--------|--------|
| 1 | **NYC TLC Taxi Trips** | NYC Taxi and Limousine Commission | `https://d37ci6vzurychx.cloudfront.net/trip-data/{type}_tripdata_{YYYY-MM}.parquet` | Parquet | Yellow + Green + HVFHV, approx 36 months each |
| 2 | **Citi Bike Trips** | Lyft / Citi Bike | `https://s3.amazonaws.com/tripdata/{YYYYMM}-citibike-tripdata.zip` | CSV (zipped) | approx 100 monthly files, 2022-2024 |
| 3 | **MTA Subway Hourly Ridership** | MTA / NY Open Data | `https://data.ny.gov/resource/wujg-7c2s.csv` (Socrata API, paginated) | CSV | approx 50-80M rows total |
| 4 | **Weather (Hourly)** | Open-Meteo Archive | `https://archive-api.open-meteo.com/v1/archive?latitude=40.7831&longitude=-73.9712` | CSV | Hourly for 3 years |
| 5 | **NYC 311 Complaints** | NYC Open Data | `https://data.cityofnewyork.us/resource/erm2-nwe9.json` (historical bulk) | CSV | Transport-relevant types only |
| 6 | **Event and Film Permits** | NYC Open Data | Event permits + film permits CSVs | CSV | 2022-2024 |

**Reference files:**
- Taxi zone lookup table + shapefile: `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip`
- These define the 263 taxi zones that serve as the spatial unit for the entire pipeline.

### 1.2 Live Streaming Data Sources (Polled Every 30 Seconds)

The real-time fetcher (`07_streaming/realtime_fetcher.py`) polls **seven live APIs** and builds one JSON record per zone per cycle:

| # | Source | Live API URL | What It Provides |
|---|--------|-------------|------------------|
| 1 | **Weather** | `https://api.open-meteo.com/v1/forecast` | Current temperature, precipitation, wind, humidity, visibility |
| 2 | **MTA Service Alerts** | `https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/camsys%2Fsubway-alerts.json` | Unplanned delays, suspensions, schedule changes per subway line |
| 3 | **MTA Train Positions** | `https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2F{feed}` | Real-time GTFS feeds for all subway divisions (8 feeds) |
| 4 | **DOT Traffic Speeds** | `https://data.cityofnewyork.us/resource/i4gi-tjb9.json` | Real-time link speeds across NYC road segments |
| 5 | **Citi Bike Station Status** | `https://gbfs.citibikenyc.com/gbfs/en/station_status.json` | Available bikes/docks at every station (GBFS standard) |
| 6 | **NYC 311 Complaints** | `https://data.cityofnewyork.us/resource/erm2-nwe9.json` | Recent transport-related complaints (last 2 hours) |
| 7 | **NYC Event Permits** | `https://data.cityofnewyork.us/resource/tvpp-9vvx.json` | Active event permits that may affect traffic |

### 1.3 Saved Dataset in Docker (HDFS)

**Yes, the cleaned historical data is stored on HDFS inside Docker** and is used in multiple places:

- `/data/neurotraffic/combined_v3` — The fully joined historical dataset (approx 6.9M zone-hour rows). Used by:
  - `06_ml/detect_anomalies.scala` for disruption labeling
  - `06_ml/prepare_ml_training.scala` for ML training data
  - `07_streaming/streaming_pipeline.scala` for baseline computation for z-scores
  - `07_streaming/build_live_profiles.scala` for zone-hour profiles for the live fetcher
  - `08_dashboard/generate_dashboard_data.scala` for historical predictions for the dashboard

- `/data/graph/features/features.parquet` — Graph analytics output (PageRank, communities, degrees). Used by the ML trainer and dashboard.

- `/data/neurotraffic/disruptions` — Disruption labels from `detect_anomalies.scala`. Used as ground truth for GBT training.

- `/data/streaming/input/` — Where the live fetcher writes JSON micro-batches. Read by both streaming pipelines.

- `/data/streaming/output/latest.csv` — The live scoring output. Read by the dashboard.

The dashboard also keeps **local copies** in `08_dashboard/data/` so it works without Docker running:
- `graph_nodes.csv`, `historical_predictions.parquet`, `evaluation_metrics.json`, `feature_importance.csv`, `confusion_matrix.csv`

---

## 2. Pipeline Architecture — The DAG

The pipeline follows a Directed Acyclic Graph (DAG) structure where each stage depends on the outputs of previous stages:

```
STAGE 1: INGESTION
  01_ingestion/download_datasets.py
  Raw data -> neurotraffic_data/ (local) -> HDFS /data/raw/

STAGE 2: CLEANING  (all Scala, run on Spark inside Docker)
  02_cleaning/clean_taxi.scala ----+
  02_cleaning/clean_subway.scala --+
  02_cleaning/clean_bike.scala ----+---> HDFS /data/neurotraffic/cleaned_{source}
  02_cleaning/clean_weather.scala -+
  02_cleaning/clean_311.scala -----+
  02_cleaning/clean_events.scala --+

STAGE 3: INTEGRATION  (Scala)
  03_integration/join_all.scala -----------> combined (263 zones x hourly grid)
  03_integration/add_bike_to_combined.scala -> combined_v2
  03_integration/fix_311_join.scala -------> combined_v3 (final)

STAGE 4: MAPREDUCE  (Scala + Bash)
  04_mapreduce/mapreduce_spark.scala -----> zone x date taxi aggregations
  04_mapreduce/mapper.sh + reducer.sh ----> Hadoop Streaming equivalent

STAGE 5: GRAPH ANALYTICS  (Scala + GraphX)
  05_graph/build_graph_inputs.scala ------> /data/graph/nodes/ + edges
  05_graph/run_graph_pipeline.scala ------> PageRank + Communities
       -> /data/graph/features/features.parquet

STAGE 6: ML PIPELINE  (Scala + Python)
  06_ml/detect_anomalies.scala ----------> /data/neurotraffic/disruptions
       (z-score + KMeans -> disruption labels)
  06_ml/prepare_ml_training.scala -------> training.parquet (via WebHDFS)
  06_ml/train_gbt.py --------------------> gbt_model.pkl + metrics

STAGE 7: STREAMING  (Scala + Python)
  07_streaming/build_live_profiles.scala -> zone_hour_profile.csv
  07_streaming/realtime_fetcher.py ------> JSON batches -> HDFS
  07_streaming/streaming_pipeline.scala -> latest.csv (Spark Streaming)
  07_streaming/spark_live_processor.py --> latest.csv (PySpark alternative)

STAGE 8: DASHBOARD  (Scala + Python/Streamlit)
  08_dashboard/generate_dashboard_data.scala -> dashboard-ready files
  08_dashboard/app.py -----------------------> Streamlit multi-page app
```

**How the DAG works:**
- Each stage reads the output of the previous stage — there are no circular dependencies.
- Stages 2-6 run **once** on historical data inside the Docker Spark cluster.
- Stage 7 runs **continuously** for live monitoring.
- Stage 8 reads from both historical (HDFS/local) and live (latest.csv) data.

---

## 3. Where Spark Is Used

Spark is the backbone of every data-heavy operation. Here is every place it runs:

### 3.1 Spark in Docker (Scala — spark-shell)

All Scala files run inside the Docker Spark cluster using spark-shell:

```bash
docker exec spark-master /spark/bin/spark-shell --master "local[2]" \
  --driver-memory 4g --conf spark.sql.shuffle.partitions=16 \
  -i /project/02_cleaning/clean_taxi.scala
```

| File | Spark Features Used |
|------|-------------------|
| `02_cleaning/clean_*.scala` (6 files) | `spark.read.parquet/csv`, DataFrame API, `filter`, `groupBy`, `agg`, `withColumn`, `write.parquet` |
| `03_integration/join_all.scala` | Hourly zone grid generation (`explode`), multi-source `join`, `coalesce` |
| `03_integration/add_bike_to_combined.scala` | `broadcast` join for station-to-zone mapping |
| `03_integration/fix_311_join.scala` | `initcap()` for case normalization, re-join |
| `04_mapreduce/mapreduce_spark.scala` | `groupBy` + `agg(sum)` as Spark-native MapReduce |
| `05_graph/build_graph_inputs.scala` | DataFrame to Parquet for graph node/edge construction |
| `05_graph/run_graph_pipeline.scala` | **GraphX**: `PageRank(tol=0.001)`, `LabelPropagation(maxSteps=5)`, `inDegrees`, `outDegrees` |
| `06_ml/detect_anomalies.scala` | **MLlib**: `VectorAssembler`, `StandardScaler`, `KMeans(k=10)`, z-score computation |
| `06_ml/prepare_ml_training.scala` | Undersampling, feature engineering, WebHDFS export |
| `07_streaming/streaming_pipeline.scala` | **Structured Streaming**: `readStream.json()`, `writeStream.foreachBatch`, `Trigger.ProcessingTime("10 seconds")` |
| `07_streaming/build_live_profiles.scala` | `groupBy(zone_id, hour, is_weekend).agg(avg(...))` for baseline profiles |
| `08_dashboard/generate_dashboard_data.scala` | Multi-source join to build dashboard-ready CSV/Parquet |

### 3.2 PySpark (Python — runs locally or in Docker)

| File | What It Does |
|------|-------------|
| `07_streaming/spark_live_processor.py` | PySpark `local[2]` mode. Reads JSON batches, joins with zone profiles, computes z-scores, writes latest.csv. Alternative to the Scala streaming pipeline when Docker is not running. |

### 3.3 Hadoop Streaming MapReduce (Bash)

| File | What It Does |
|------|-------------|
| `04_mapreduce/mapper.sh` | Reads TSV rows, emits `(zone_id, date, pickups)` tuples |
| `04_mapreduce/reducer.sh` | Sums taxi pickups by `(zone_id, date)` key |
| `04_mapreduce/export_for_mapreduce.scala` | Converts Parquet to TSV for the shell scripts |

These demonstrate classic Hadoop Streaming alongside the Spark equivalent (`mapreduce_spark.scala`).

---

## 4. File-by-File Reference

### 4.1 `01_ingestion/` — Data Download

| File | Language | Description |
|------|----------|-------------|
| `download_datasets.py` | Python | Downloads all 6 raw datasets from public APIs. Handles pagination (Socrata), retry/resume, ZIP extraction. Writes to `neurotraffic_data/`. |

### 4.2 `02_cleaning/` — Spark Data Cleaning (Scala)

| File | Input (HDFS) | Output (HDFS) | Key Operations |
|------|-------------|---------------|----------------|
| `clean_taxi.scala` | Raw taxi parquet | `/data/neurotraffic/cleaned_taxi` | Filter valid coords/fares, add `pickup_hour`, batch HVFHV for memory |
| `clean_subway.scala` | Raw MTA CSV | `/data/neurotraffic/cleaned_subway` | Aggregate per-payment rows to station x hour totals, remove 5-sigma outliers |
| `clean_bike.scala` | Raw Citi Bike CSV | `/data/neurotraffic/cleaned_bike` | Deduplicate by `ride_id`, filter NYC bounding box, duration bounds |
| `clean_weather.scala` | Open-Meteo CSV | `/data/neurotraffic/cleaned_weather` | Forward-fill gaps (6h window), add severity flags (rain/snow/cold/heat) |
| `clean_311.scala` | Raw 311 CSV | `/data/neurotraffic/cleaned_311` | Filter transport-related types, validate NYC coords, add severity scores |
| `clean_events.scala` | Event/film permit CSV | `/data/neurotraffic/cleaned_events` | Union event+film permits, extract street count, compute disruption severity |
| `extract_bike_stations.scala` | Cleaned bike parquet | `bike_stations.csv` | Extract unique station IDs with averaged lat/lng |
| `map_bike_stations_to_zones.py` | bike_stations.csv + shapefile | `bike_station_zones.csv` | Point-in-polygon mapping of bike stations to taxi zone IDs |

### 4.3 `03_integration/` — Data Joining (Scala)

| File | What It Does |
|------|-------------|
| `join_all.scala` | Builds a 263-zone x hourly time grid (2022-2024). Aggregates taxi, subway, 311, events by zone+hour. Left-joins weather. Writes `combined` parquet. |
| `add_bike_to_combined.scala` | Joins bike trip counts (starts/ends per zone per hour) onto `combined` producing `combined_v2`. Uses `bike_station_zones.csv` for station-to-zone mapping. |
| `fix_311_join.scala` | Fixes a borough case mismatch (`MANHATTAN` vs `Manhattan`) in the 311 join. Re-aggregates with `initcap()`, writes `combined_v3`. |
| `analysis_export.scala` | Produces 7 summary CSVs from `combined_v3`: hourly citywide, daily borough, zone heatmap, weather impact, rush hour analysis, monthly trends, correlation pairs. |

### 4.4 `04_mapreduce/` — MapReduce (Scala + Bash)

| File | What It Does |
|------|-------------|
| `mapreduce_spark.scala` | Spark-native MapReduce: reads `combined_v3`, groups by `(zone_id, date)`, sums taxi pickups, writes TSV. |
| `export_for_mapreduce.scala` | Converts `combined_v3` parquet to TSV for the Hadoop Streaming scripts. |
| `mapper.sh` | Hadoop Streaming mapper: emits `(zone_id, date, pickups)` for non-zero pickup rows. |
| `reducer.sh` | Hadoop Streaming reducer: sums pickups per `(zone_id, date)` key. |

### 4.5 `05_graph/` — Graph Analytics (Scala + GraphX)

| File | What It Does |
|------|-------------|
| `build_graph_inputs.scala` | Builds graph node parquet (taxi zones, subway stations, bike docks) and trip edge parquet from raw taxi OD data. Writes `/data/graph/nodes/` and `/data/graph/combined/trips.parquet`. |
| `build_edges_only.scala` | Alternative edge builder. Aggregates taxi OD pairs by hour and day-type. Handles schema conflicts between yellow/green taxi parquets. |
| `run_graph_pipeline.scala` | **The main graph analytics script.** Loads nodes + edges into GraphX. Runs: (1) **PageRank** (tolerance 0.001) — identifies structurally critical zones, (2) **Label Propagation** (5 steps) — detects transport communities, (3) **In/Out Degree** computation. Joins all features, writes `features.parquet`. |

### 4.6 `06_ml/` — Machine Learning (Scala + Python)

| File | Language | What It Does |
|------|----------|-------------|
| `detect_anomalies.scala` | Scala | **Disruption labeling (unsupervised).** Computes z-scores per zone across 12 features, runs KMeans(k=10), flags small clusters + high z-score rows as disruptions. Composite score: `0.4 x avg_z + 0.3 x max_z + 2.0 x small_cluster + 0.1 x complaints + 0.1 x events + 0.1 x weather`. Writes `/data/neurotraffic/disruptions`. |
| `prepare_ml_training.scala` | Scala | Prepares training data: joins `combined_v3` + graph features + disruption labels. Engineers cross-modal features (taxi-to-subway ratio, bike-to-subway ratio). Undersamples negatives to 5%. Exports single-file `training.parquet` via WebHDFS. |
| `train_gbt.py` | Python | Trains a `GradientBoostingClassifier` (scikit-learn) on the training sample. 80/20 split, StandardScaler. Produces: `gbt_model.pkl`, `evaluation_metrics.json` (97.8% accuracy, 0.98 AUC-ROC), `feature_importance.csv`, `confusion_matrix.csv`. |
| `disruption_detector.py` | Python | Full alternative ML pipeline: loads data, injects synthetic disruptions, runs Isolation Forest + GBT, saves predictions. |
| `scoring_module.py` | Python | Integration API: loads trained IF and GBT models from HDFS, exposes `score_anomaly()` and `predict_cascade()` for streaming. |

### 4.7 `07_streaming/` — Live Streaming Pipeline

| File | Language | What It Does |
|------|----------|-------------|
| `build_live_profiles.scala` | Scala | Builds historical zone x hour x day-type profiles from `combined_v3`. Averages of taxi pickups, subway ridership, bike trips, complaints, events per zone per hour. Writes `zone_hour_profile.csv` to HDFS. |
| `build_live_lookups.py` | Python | Builds static lookup tables: zone centroids (from taxi zone shapefile), subway station-to-zone mapping. Optionally uploads to HDFS. |
| `realtime_fetcher.py` | Python | **The live data fetcher.** Polls 7 real-time APIs every N seconds. For each of the 263 zones, builds a JSON record with current weather, MTA disruption severity, estimated ridership (from profiles), Citi Bike availability, 311 complaints, events. Writes JSON micro-batches to HDFS `/data/streaming/input/`. |
| `streaming_pipeline.scala` | Scala | **Spark Structured Streaming pipeline.** Reads JSON micro-batches from HDFS. Computes anomaly scores using z-scores against historical baselines (per zone x hour x weekend). Joins with broadcast baselines. Writes `latest.csv` to HDFS. Trigger: every 10 seconds. |
| `spark_live_processor.py` | Python | **PySpark alternative** to the Scala streaming pipeline. Runs locally (no Docker needed). Same logic: reads JSON batches, computes z-scores against `zone_hour_profile.csv`, writes `latest.csv` to both HDFS and local disk. |
| `live_processor.py` | Python | **Lightweight pure-Python alternative** (no Spark). Reads JSON from HDFS via WebHDFS REST API, scores with pandas. |
| `streaming_simulator.py` | Python | Generates synthetic NYC transport data for testing/demo when live APIs are unavailable. |

**Lookup files** (`07_streaming/lookups/`):
- `zone_centroids.csv` — Lat/lng center of each taxi zone
- `subway_station_zones.csv` — Subway station to taxi zone mapping
- `zone_hour_profile.csv` — Historical average per zone per hour (the baseline for z-score scoring)

### 4.8 `08_dashboard/` — Streamlit Dashboard

| File | Language | What It Does |
|------|----------|-------------|
| `generate_dashboard_data.scala` | Scala | Builds dashboard-ready files from HDFS data: `graph_nodes.csv` (zone metadata + graph features + subway lines), `historical_predictions.parquet` (daily disruption summaries), fallback `latest.csv`. |
| `config.py` | Python | Central configuration: all HDFS/local paths, map center, anomaly thresholds, subway line metadata, color scales. |
| `hdfs_helper.py` | Python | WebHDFS REST client: reads/writes CSV, Parquet, JSON from HDFS. Falls back to local files when HDFS is offline. |
| `app.py` | Python | **Home page.** Hero section, live metrics, navigation cards, borough breakdown, pipeline architecture diagram, data status. |
| `pages/1_live_monitor.py` | Python | **Real-time map.** Auto-refreshes every 5s. Shows anomaly scores per zone on a dark map, active disruptions, MTA alerts, data source freshness. |
| `pages/2_simulator.py` | Python | **Disruption simulator.** User picks a subway line + time + day. GBT model predicts cascade effects across zones. Shows epicenter/cascade/normal zones on map with surge percentages. |
| `pages/3_explorer.py` | Python | **Explorer.** Live streaming insights (primary section) with anomaly map, top zones, rolling timeline with week/month history. Historical analysis (secondary). |
| `pages/4_analytics.py` | Python | **Analytics dashboard.** Model performance metrics (accuracy, precision, recall, F1, AUC-ROC), confusion matrix, feature importance chart, graph analytics (PageRank bar chart, community/borough map, degree distribution). |

---

## 5. How Live Streaming Works

### Step-by-step flow:

1. **`realtime_fetcher.py`** runs continuously, polling 7 APIs every 30 seconds.
2. For each cycle, it builds one JSON record per zone (263 records) containing current weather, MTA disruption severity, estimated transport volumes (scaled from historical profiles by time-of-day), Citi Bike station availability, recent 311 complaints, and active events.
3. Each batch is written as a JSON file to HDFS: `/data/streaming/input/batch_YYYYMMDD_HHMMSS.json`
4. **`streaming_pipeline.scala`** (Spark Structured Streaming) or **`spark_live_processor.py`** (PySpark) picks up these batches.
5. The processor joins each batch with **historical baselines** (mean and std dev per zone x hour x weekend/weekday) and computes z-scores for taxi, subway, bike, and complaints.
6. A composite anomaly score is computed: weighted combination of z-scores, weather severity, and disruption indicators.
7. The scored output is written to `latest.csv` (both HDFS and local disk).
8. The **Streamlit dashboard** reads `latest.csv` every 3-5 seconds and refreshes the map, metrics, and tables.

### Streaming options:

| Option | File | Requirements | When to Use |
|--------|------|-------------|-------------|
| **Scala (Docker)** | `streaming_pipeline.scala` | Docker Spark cluster running | Full production mode |
| **PySpark (Local)** | `spark_live_processor.py` | PySpark installed locally | Development / without Docker |
| **Pure Python** | `live_processor.py` | Just Python + pandas | Lightweight / testing |

---

## 6. Spark Scala Deep Dive — Key Files

### 6.1 `02_cleaning/clean_taxi.scala`

Reads raw TLC taxi parquet files from HDFS, filters to valid NYC coordinates (latitude 40.4-41.0, longitude -74.3 to -73.7), removes trips with negative fares or zero duration, adds a `pickup_hour` column extracted from the timestamp, and writes cleaned parquet. Processes HVFHV data in monthly batches to avoid memory issues (these files are much larger than yellow/green).

```scala
val raw = spark.read.parquet("hdfs://namenode:9000/data/raw/taxi/yellow_tripdata_*.parquet")
val cleaned = raw
  .filter(col("pickup_latitude").between(40.4, 41.0))
  .filter(col("fare_amount") > 0)
  .withColumn("pickup_hour", hour(col("tpep_pickup_datetime")))
cleaned.write.mode("overwrite").parquet("/data/neurotraffic/cleaned_taxi")
```

### 6.2 `03_integration/join_all.scala`

The most complex Scala file. Generates a complete time grid: every combination of 263 taxi zones x every hour from 2022-01-01 to 2024-12-31. Then left-joins aggregated taxi pickups/dropoffs, subway ridership, 311 complaints, event counts, and weather data onto this grid. Missing values are filled with 0 (counts) or null (weather). The result is the `combined` parquet — the foundation for all downstream analytics.

```scala
val zones = (1 to 263).toDF("zone_id")
val hours = spark.range(startTs, endTs, 3600).toDF("hour_ts")
  .withColumn("hour_timestamp", from_unixtime(col("hour_ts")))
val grid = zones.crossJoin(hours)  // 263 zones x ~26,280 hours = ~6.9M rows
```

### 6.3 `05_graph/run_graph_pipeline.scala`

Builds a GraphX graph where:
- **Nodes** = taxi zones, subway stations, bike docks (each with lat/lng/borough/capacity)
- **Edges** = trip volumes between locations (taxi OD pairs, subway transfers, bike trips)

Runs three algorithms:
1. **PageRank** (tolerance 0.001): Identifies structurally important nodes. High PageRank = hub zones like Midtown, Penn Station.
2. **Label Propagation** (5 iterations): Detects transport communities — groups of zones that are tightly connected by trips.
3. **Degree computation**: In-degree (trips arriving) and out-degree (trips departing) per node.

```scala
import org.apache.spark.graphx._
import org.apache.spark.graphx.lib.LabelPropagation

val g: Graph[NodeAttr, Long] = Graph(vertices, edges)
val ranks = g.pageRank(0.001).vertices
val communities = LabelPropagation.run(g, maxSteps = 5).vertices
val inDeg = g.inDegrees
val outDeg = g.outDegrees
```

Output: `features.parquet` with columns: vertex_id, node_type, name, lat, lon, borough, capacity, page_rank, community_id, in_degree, out_degree.

### 6.4 `06_ml/detect_anomalies.scala`

The unsupervised disruption labeling engine. Two-stage approach:

**Stage 1 — Z-score anomaly detection:**
Computes z-scores per zone across 12 feature columns (taxi pickups, subway ridership, bike trips, complaints, events, weather metrics). A zone-hour with any |z-score| > 3.0 is flagged.

**Stage 2 — KMeans clustering:**
Runs KMeans with k=10 on all zone-hour feature vectors. Small clusters (fewer than 50,000 members) represent unusual patterns and are flagged as anomalous.

**Composite disruption score:**
```scala
val score = avg_abs_z * 0.4 + max_abs_z * 0.3 + is_small_cluster * 2.0 +
            complaints * 0.1 + events * 0.1 + weather * 0.1
val is_disruption = score > 3.0
```

### 6.5 `07_streaming/streaming_pipeline.scala`

Spark Structured Streaming pipeline that processes live data:

```scala
val rawStream = spark.readStream
  .schema(inputSchema)
  .option("maxFilesPerTrigger", 5)
  .json("hdfs://namenode:9000/data/streaming/input/")

val baselines = spark.read.csv("zone_hour_profile.csv")  // broadcast
val scored = rawStream
  .join(broadcast(baselines), Seq("zone_id", "hour_of_day", "is_weekend"))
  .withColumn("z_taxi", (col("taxi_pickups") - col("mean_taxi")) / col("std_taxi"))

scored.writeStream
  .foreachBatch { (df, batchId) => df.coalesce(1).write.mode("overwrite").csv(OUTPUT) }
  .trigger(Trigger.ProcessingTime("10 seconds"))
  .start()
  .awaitTermination()
```

### 6.6 `08_dashboard/generate_dashboard_data.scala`

Bridges HDFS data to the dashboard. Reads graph features, zone metadata, and subway line information. Joins them into `graph_nodes.csv` (263 rows, one per zone). Also aggregates `combined_v3` into daily disruption summaries for the explorer's historical view, writing `historical_predictions.parquet`.

---

## 7. ML Model Details

### Training Data
- **Source:** `combined_v3` (6.9M zone-hour rows) + graph features + disruption labels from `detect_anomalies.scala`
- **Rows used for training:** approx 314K (after 5% undersampling of negatives)
- **Features:** 23 features across 6 categories:
  - Transport: taxi pickups/dropoffs, subway ridership, bike starts/ends
  - Cross-modal: taxi-to-subway ratio, bike-to-subway ratio, total surface transport
  - Graph: PageRank, community_id, in_degree, out_degree
  - Weather: temperature, precipitation, humidity, wind, rain/snow/extreme flags
  - Temporal: hour_of_day, is_rush_hour, is_weekend
  - External: complaint_count, complaint_density, event_count

### GBT Classifier Results
- **Accuracy:** 97.8%
- **Precision:** 92.7%
- **Recall:** 92.1%
- **F1 Score:** 92.4%
- **AUC-ROC:** 0.9808

---

## 8. Can Everything Be Done in Scala? — Analysis

This project uses Python in 6 areas. Below is a file-by-file assessment of whether each could be replaced with Scala, and if not, why.

### 8.1 Components That ARE Already in Scala

| Stage | Files | Status |
|-------|-------|--------|
| Cleaning (Stage 2) | `clean_taxi.scala`, `clean_subway.scala`, `clean_bike.scala`, `clean_weather.scala`, `clean_311.scala`, `clean_events.scala` | Already Scala |
| Integration (Stage 3) | `join_all.scala`, `add_bike_to_combined.scala`, `fix_311_join.scala`, `analysis_export.scala` | Already Scala |
| MapReduce (Stage 4) | `mapreduce_spark.scala`, `export_for_mapreduce.scala` | Already Scala |
| Graph Analytics (Stage 5) | `build_graph_inputs.scala`, `run_graph_pipeline.scala` | Already Scala |
| Anomaly Detection (Stage 6) | `detect_anomalies.scala`, `prepare_ml_training.scala` | Already Scala |
| Streaming Processor (Stage 7) | `streaming_pipeline.scala`, `build_live_profiles.scala` | Already Scala |
| Dashboard Data Gen (Stage 8) | `generate_dashboard_data.scala` | Already Scala |

### 8.2 Python Components — Can They Be Converted to Scala?

| Python File | Purpose | Scala? | Explanation |
|-------------|---------|:---:|-------------|
| `01_ingestion/download_datasets.py` | Downloads 6 datasets from public APIs | YES | Scala can use `java.net.http.HttpClient` (Java 11+) or `sttp`/`akka-http` for HTTP. `java.util.zip.ZipInputStream` handles ZIPs. More verbose (2-3x code) but fully functional. |
| `02_cleaning/map_bike_stations_to_zones.py` | Point-in-polygon geo mapping via GeoPandas | YES | Scala can use **JTS (Java Topology Suite)** or **GeoTools** for spatial joins. JTS provides `Point.within(Polygon)` — same logic as GeoPandas internally. |
| `06_ml/train_gbt.py` | Trains GBT classifier via scikit-learn | YES | Spark MLlib has `GBTClassifier` natively. Uses `Pipeline`, `VectorAssembler`, `BinaryClassificationEvaluator`. Model saves as Spark ML directory instead of `.pkl`. |
| `06_ml/disruption_detector.py` | Isolation Forest + GBT | PARTIAL | **GBT: YES** via MLlib. **Isolation Forest: NO** — Spark MLlib does not include Isolation Forest. Options: (a) third-party `spark-iforest` (not actively maintained), (b) replace with KMeans-based outlier detection (already in `detect_anomalies.scala`), (c) implement from scratch (~500-800 lines). |
| `06_ml/scoring_module.py` | Loads models, exposes scoring API | YES | Spark MLlib models save/load natively with `PipelineModel.load()`. No joblib/pickle needed. |
| `06_ml/tune_contamination.py` | Tunes Isolation Forest parameters | PARTIAL | Same Isolation Forest limitation. If replaced with MLlib alternative, use `CrossValidator` or `TrainValidationSplit`. |
| `06_ml/run_pipeline.py` | Orchestrates ML pipeline scripts | YES | Simple Scala or shell script can run stages in sequence. |
| `06_ml/prepare_data.py` | Fallback data prep with synthetic data | YES | DataFrame operations translate directly to Spark SQL. |
| `07_streaming/realtime_fetcher.py` | Polls 7 live APIs every 30s, writes JSON to HDFS | YES | Scala can use `java.net.http.HttpClient`. HDFS writes via Hadoop `FileSystem` API (already in Spark). GTFS protobuf uses `gtfs-realtime-bindings` (Java-native, easier from Scala). Python `requests` is simpler, but Scala is fully capable. |
| `07_streaming/build_live_lookups.py` | Builds zone/station lookup CSVs | YES | Use GeoTools for shapefiles + Spark or file I/O for CSVs. |
| `07_streaming/spark_live_processor.py` | PySpark streaming fallback | REDUNDANT | This is a Python copy of `streaming_pipeline.scala`. In an all-Scala project, this file is unnecessary. |
| `07_streaming/live_processor.py` | Pure Python processor (no Spark) | NO | Exists specifically for no-Spark environments. Converting to Scala would require JVM, which defeats the lightweight purpose. |
| `07_streaming/streaming_simulator.py` | Generates synthetic test data | YES | Trivial with `scala.util.Random`. |
| `08_dashboard/app.py` | Streamlit home page | **NO** | **Streamlit is Python-only. No Scala equivalent exists.** See Section 8.4 for details. |
| `08_dashboard/pages/1_live_monitor.py` | Real-time monitoring page | **NO** | Streamlit-only — same reason. |
| `08_dashboard/pages/2_simulator.py` | Disruption simulator page | **NO** | Streamlit-only — same reason. |
| `08_dashboard/pages/3_explorer.py` | Data explorer page | **NO** | Streamlit-only — same reason. |
| `08_dashboard/pages/4_analytics.py` | Analytics page | **NO** | Streamlit-only — same reason. |
| `08_dashboard/config.py` | Dashboard configuration | **NO** | Required by Streamlit pages — must be Python. |
| `08_dashboard/hdfs_helper.py` | WebHDFS client for dashboard | **NO** | Required by Streamlit pages. Scala pipeline uses Hadoop API directly; this exists because Streamlit needs its own HDFS access. |

### 8.3 Summary

| Category | Scala? | Count | Details |
|----------|:---:|:---:|---------|
| Already Scala | YES | **13 files** | Cleaning, integration, MapReduce, graph, anomaly detection, streaming, dashboard data gen |
| Could convert to Scala | YES | **8 files** | Data download, geo-mapping, GBT training, fetcher, lookups, simulator, scoring, orchestrator |
| Partially convertible | PARTIAL | **2 files** | Isolation Forest has no Spark MLlib equivalent |
| Cannot be Scala | **NO** | **8 files** | All dashboard/Streamlit files + lightweight Python-only processor |

### 8.4 Why the Dashboard Cannot Be Scala

The dashboard is the **only component that fundamentally cannot be Scala**:

1. **Streamlit has no Scala port.** Streamlit is Python-only, maintained by Snowflake. It re-runs a Python script on every user interaction. No JVM framework uses this reactive model.

2. **No Scala dashboard framework at this level.** JVM has web frameworks (Play, Akka HTTP, http4s, Vaadin) but none provide one-line data visualization like `st.plotly_chart()`, `st.pydeck_chart()`, `st.dataframe()`. Building the same dashboard in Scala requires:
   - A web server (Play or Akka HTTP)
   - A JavaScript frontend (React, Vue, or Angular)
   - Manual chart rendering (D3.js or Plotly.js)
   - WebSocket connections for live updates
   - Estimated **10-20x the development effort**

3. **The data pipeline IS Scala.** Everything that touches Spark, HDFS, GraphX, MLlib, and Structured Streaming is already Scala. The dashboard only reads outputs (CSV, Parquet, JSON) produced by the Scala pipeline. It displays data — it does not process it.

4. **Industry standard.** Python for the presentation layer + Scala/Spark for the data pipeline is the standard Big Data pattern. The two layers communicate through files on HDFS or local disk — a clean separation of concerns.

### 8.5 What Would a Fully-Scala Project Look Like?

If the requirement were 100% Scala (excluding the dashboard):

- **Stages 1-7 could be fully Scala** (13 existing + 8 converted = 21 Scala files)
- Replace scikit-learn GBT with Spark MLlib `GBTClassifier`
- Replace Isolation Forest with KMeans-based anomaly detection (already in `detect_anomalies.scala`)
- Replace Python `requests` HTTP calls with `java.net.http.HttpClient`
- Replace GeoPandas with JTS (Java Topology Suite)
- The dashboard remains Python/Streamlit — it is the presentation layer, not the data pipeline
- Alternative: replace dashboard with a **Scala backend + React frontend** web app — but that is a separate full-stack project

---

## 9. Running the Pipeline

### Prerequisites
- Docker with Spark 3.3.0 + HDFS cluster
- Python 3.10+ with: `streamlit`, `pandas`, `plotly`, `scikit-learn`, `joblib`, `pyspark`, `requests`

### Execution Order

```bash
# 1. Download raw data
python 01_ingestion/download_datasets.py

# 2. Upload to HDFS and clean (run each in spark-shell inside Docker)
docker exec spark-master spark-shell -i /project/02_cleaning/clean_taxi.scala
docker exec spark-master spark-shell -i /project/02_cleaning/clean_subway.scala
docker exec spark-master spark-shell -i /project/02_cleaning/clean_bike.scala
docker exec spark-master spark-shell -i /project/02_cleaning/clean_weather.scala
docker exec spark-master spark-shell -i /project/02_cleaning/clean_311.scala
docker exec spark-master spark-shell -i /project/02_cleaning/clean_events.scala
docker exec spark-master spark-shell -i /project/02_cleaning/extract_bike_stations.scala
python 02_cleaning/map_bike_stations_to_zones.py

# 3. Join all sources
docker exec spark-master spark-shell -i /project/03_integration/join_all.scala
docker exec spark-master spark-shell -i /project/03_integration/add_bike_to_combined.scala
docker exec spark-master spark-shell -i /project/03_integration/fix_311_join.scala

# 4. MapReduce
docker exec spark-master spark-shell -i /project/04_mapreduce/mapreduce_spark.scala

# 5. Graph analytics
docker exec spark-master spark-shell -i /project/05_graph/build_graph_inputs.scala
docker exec spark-master spark-shell -i /project/05_graph/run_graph_pipeline.scala

# 6. ML pipeline
docker exec spark-master spark-shell -i /project/06_ml/detect_anomalies.scala
docker exec spark-master spark-shell -i /project/06_ml/prepare_ml_training.scala
python 06_ml/train_gbt.py

# 7. Prepare streaming
docker exec spark-master spark-shell -i /project/07_streaming/build_live_profiles.scala
python 07_streaming/build_live_lookups.py

# 8. Generate dashboard data
docker exec spark-master spark-shell -i /project/08_dashboard/generate_dashboard_data.scala

# 9. Start live pipeline
python 07_streaming/realtime_fetcher.py --interval 30 &
python 07_streaming/spark_live_processor.py --interval 15 &

# 10. Launch dashboard
streamlit run 08_dashboard/app.py
```

