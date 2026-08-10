# NeuroTraffic — Data Pipeline Report

**Project:** NeuroTraffic — City-Scale Multi-Modal Transport Intelligence  
**University:** Amrita Vishwa Vidyapeetham  
**Prepared by:** Ravula Navaneeth (CB.AI.U4AID24044) — Data Pipeline Developer  
**Date Range:** January 2022 – December 2024 (3 years)  
**Pipeline Stage:** Acquisition, cleaning & join complete  
**Total raw data downloaded:** 36.92 GB across 135 files

---

## 1. Overview

NeuroTraffic fuses 6 NYC datasets to detect urban transportation disruptions using a cross-modal behavioral signature: when a subway line fails, ridership drops while taxi pickups and bike checkouts surge in the same zone. The system uses Isolation Forest for anomaly detection and GBT Classifier for disruption classification.

All datasets have been downloaded, uploaded to HDFS (3x replication across 3 DataNodes), cleaned using PySpark, and joined into a single combined table of **6.9 million rows** ready for ML modeling.

**Infrastructure:**
- Docker-based Hadoop cluster: 7 containers (NameNode, DataNode x3, Spark Master, Spark Worker x2)
- HDFS capacity: 2.95 TB (77.4 GB used)
- Processing: PySpark in local mode (`local[2]`, 2g driver memory)
- Host: Windows 11, 15.2 GB RAM, Docker allocated 7.37 GB

---

## 2. Datasets — Acquisition, Schemas & Cleaning

### 2.1 NYC TLC Taxi Trip Data (Yellow + Green + HVFHV)

| Property | Value |
|----------|-------|
| Source | NYC Taxi & Limousine Commission |
| URL | https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page |
| Format | Parquet (one file per month per taxi type) |
| Files | Yellow (36, 1.94 GB) + Green (36, 52 MB) + HVFHV (36, 17.03 GB) = 108 files |
| Raw download | ~19 GB |

**Why we need it:** Taxi trips are one of the three core transport modes. Pickup/dropoff volumes per zone per hour are primary features for the Isolation Forest anomaly detector. When a subway fails, taxi pickups surge — this is the signal the system detects.

**Reference files:** `taxi_zone_lookup.csv` (263 zone IDs → borough/name), `taxi_zones.zip` (shapefile with zone boundary polygons)

**Cleaning results:**

| Taxi Type | Raw Rows | Cleaned Rows | Removed | HDFS Size |
|-----------|----------|-------------|---------|-----------|
| Yellow | 119,136,044 | 112,170,940 | 5.85% | 2.1 GB |
| Green | 2,287,680 | 2,107,944 | 7.86% | 54.5 MB |
| HVFHV (Uber/Lyft) | 684,376,551 | 655,579,700 | 4.21% | 12.1 GB |
| **Total** | **805,800,275** | **769,858,584** | **4.46%** | **14.3 GB** |

**Cleaned columns (14, unified across all types):**

| Column | Type | Description |
|--------|------|-------------|
| taxi_type | string | "yellow", "green", or "hvfhv" |
| pickup_datetime | timestamp | Trip pickup time |
| dropoff_datetime | timestamp | Trip dropoff time |
| pu_location_id | int | Pickup taxi zone ID (1–263) |
| do_location_id | int | Dropoff taxi zone ID (1–263) |
| passenger_count | int | Number of passengers (null for HVFHV) |
| trip_distance | double | Trip distance (miles) |
| fare_amount | double | Fare amount ($) |
| total_amount | double | Total charged ($) |
| payment_type | int | Payment method code (null for HVFHV) |
| trip_duration_sec | bigint | Trip duration in seconds |
| pickup_hour | timestamp | Hour-aligned pickup time for joining |
| year | int | Year (partition column) |
| month | int | Month (partition column) |

**Column mapping (how different taxi schemas were unified):**

| Unified Column | Yellow Source | Green Source | HVFHV Source |
|----------------|-------------|-------------|--------------|
| pickup_datetime | tpep_pickup_datetime | lpep_pickup_datetime | pickup_datetime |
| dropoff_datetime | tpep_dropoff_datetime | lpep_dropoff_datetime | dropoff_datetime |
| trip_distance | trip_distance | trip_distance | trip_miles |
| fare_amount | fare_amount | fare_amount | base_passenger_fare |
| total_amount | total_amount | total_amount | driver_pay |

**Cleaning applied:**
- Removed null pickup/dropoff times or location IDs
- Removed zero-passenger trips (Yellow/Green only; HVFHV doesn't report passenger count)
- Removed zero or negative distance/fare trips
- Validated location IDs are in range 1–263 (NYC taxi zones)
- Removed trips shorter than 30 seconds or longer than 12 hours
- Filtered to 2022–2024 date range

---

### 2.2 Citi Bike Trip Data

| Property | Value |
|----------|-------|
| Source | Citi Bike / Lyft |
| URL | https://citibikenyc.com/system-data |
| Format | Zipped CSV |
| Files | 2 yearly zips (2022, 2023) + 12 monthly zips (2024) = 14 files, 11.49 GB |
| Raw rows | 109,249,001 |
| Cleaned rows | 108,469,697 |
| Removed | 779,304 (0.71%) |
| HDFS size | 5.0 GB |
| Granularity | 1 row per bike trip |

**Why we need it:** Bike-share is the second alternative transport mode. When a subway station shuts down, nearby Citi Bike docks get emptied within minutes. Bike checkout spikes are a key component of the cross-modal disruption signature (turnstiles DOWN + taxis UP + bikes UP = subway failure detected).

**Cleaned columns (17):**

| Column | Type | Description |
|--------|------|-------------|
| ride_id | string | Unique ride identifier |
| rideable_type | string | Bike type (classic_bike, electric_bike, docked_bike) |
| start_time | timestamp | Trip start time |
| end_time | timestamp | Trip end time |
| start_station_name | string | Start station name (title-cased) |
| start_station_id | string | Start station ID |
| end_station_name | string | End station name (title-cased) |
| end_station_id | string | End station ID |
| start_lat | double | Start latitude |
| start_lng | double | Start longitude |
| end_lat | double | End latitude |
| end_lng | double | End longitude |
| member_casual | string | Rider type (member or casual) |
| trip_duration_sec | bigint | Trip duration in seconds |
| start_hour | timestamp | Hour-aligned start time for joining |
| year | int | Year (partition column) |
| month | int | Month (partition column) |

**Cleaning applied:**
- Removed duplicate ride IDs (0 found)
- Dropped rows with missing start/end coordinates
- Filtered to NYC bounding box (lat 40.48–40.92, lng -74.26 to -73.70)
- Removed trips shorter than 60 seconds or longer than 24 hours
- Removed dock errors (same start/end station with duration under 120 seconds)
- Filtered to 2022–2024 date range
- Standardized station names (trimmed, title-cased)

**Note:** Bike data is NOT in the combined join table yet — bike stations have lat/lng but no taxi zone ID. Spatial mapping (lat/lng → taxi zone) using the NYC taxi zone shapefile is pending.

---

### 2.3 MTA Subway Hourly Ridership

| Property | Value |
|----------|-------|
| Source | Metropolitan Transportation Authority (MTA) via NY Open Data |
| URL | https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s |
| Format | CSV (downloaded via Socrata API, paginated at 50K rows/request) |
| Files | 3 yearly CSVs — 2022 (9.55M rows), 2023 (6.80M rows), 2024 (24.45M rows), total 6.63 GB |
| Raw rows | 40,800,000 |
| Cleaned rows | 5,985,275 |
| Removed | 85.3%* |
| HDFS size | 46.7 MB |
| Granularity | 1 row per station per hour |

*The large reduction is from **aggregation**, not data removal. Raw data had one row per payment method per station per hour (OMNY, MetroCard, etc.). Cleaning aggregated these into one row per station per hour with total ridership.

**Why we need it:** Subway ridership is the core signal for disruption detection. When a subway line fails, the hourly ridership at affected stations drops to near zero. This drop, combined with simultaneous taxi and bike surges in the same zone, forms the cross-modal behavioral signature that the Isolation Forest model is trained to detect.

**Cleaned columns (10):**

| Column | Type | Description |
|--------|------|-------------|
| station_complex_id | string | Unique station complex identifier |
| transit_hour | timestamp | Hour of the ridership count |
| station_name | string | Station name |
| borough | string | NYC borough |
| latitude | double | Station latitude |
| longitude | double | Station longitude |
| total_ridership | bigint | Total riders (all payment methods combined) |
| total_transfers | bigint | Total transfer riders |
| year | int | Year (partition column) |
| month | int | Month (partition column) |

**Cleaning applied:** Aggregated payment methods (OMNY + MetroCard) into total ridership per station-hour. Removed stations with missing coordinates. Removed outlier ridership values. Standardized station names.

---

### 2.4 Weather (NYC Central Park)

| Property | Value |
|----------|-------|
| Source | Open-Meteo Historical Weather API |
| URL | https://archive-api.open-meteo.com/v1/archive |
| Location | NYC Central Park (40.7831, -73.9712) |
| Format | CSV (hourly observations) |
| Files | 3 yearly CSVs — ~8,763 rows each, total 1.5 MB |
| Raw rows | 26,304 |
| Cleaned rows | 26,304 |
| Removed | 0 (0%) |
| HDFS size | 564 KB |
| Granularity | 1 row per hour |

**Why we need it:** Weather directly affects transport demand. Rain increases taxi demand and decreases bike usage. Snow causes subway delays. Extreme heat or cold changes commuting patterns. Without weather context, the ML models would flag a rainy-day taxi surge as an anomaly when it's actually normal.

**Cleaned columns (20):**

| Column | Type | Description |
|--------|------|-------------|
| weather_hour | timestamp | Hour of the weather reading |
| temperature_c | double | Temperature in Celsius |
| humidity_pct | double | Relative humidity percentage |
| precipitation_mm | double | Total precipitation in mm |
| rain_mm | double | Rainfall in mm |
| snowfall_cm | double | Snowfall in cm |
| snow_depth_m | double | Snow depth in meters |
| wmo_weather_code | int | WMO standard weather condition code |
| wind_speed_kmh | double | Wind speed in km/h |
| wind_gusts_kmh | double | Wind gust speed in km/h |
| visibility_m | double | Visibility in meters |
| is_rain | int | Binary flag: raining (1) or not (0) |
| is_heavy_rain | int | Binary flag: heavy rain |
| is_snow | int | Binary flag: snowing |
| is_extreme_cold | int | Binary flag: extreme cold |
| is_extreme_heat | int | Binary flag: extreme heat |
| is_low_visibility | int | Binary flag: low visibility |
| is_high_wind | int | Binary flag: high wind |
| year | int | Year (partition column) |
| month | int | Month (partition column) |

**Cleaning applied:** No rows removed. Added binary weather flags (is_rain, is_snow, etc.) derived from raw weather codes and thresholds. These flags are used as features for disruption detection.

---

### 2.5 NYC 311 Service Requests

| Property | Value |
|----------|-------|
| Source | NYC Open Data — 311 Service Requests |
| URL | https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9 |
| Format | CSV (downloaded via Socrata API, paginated) |
| Files | 3 yearly CSVs — 2022 (923K rows), 2023 (850K rows), 2024 (1.01M rows), total 1.66 GB |
| Raw rows | 2,784,357 |
| Cleaned rows | 2,700,357 |
| Removed | 84,000 (3.02%) |
| HDFS size | 95.1 MB |
| Granularity | 1 row per complaint |

**Why we need it:** 311 complaints signal road-level disruptions that affect transport but aren't captured by the other datasets. A water main break closes streets. A broken traffic signal causes congestion. Filtered to 9 transport-relevant complaint types:

| Complaint Type | What it indicates |
|---|---|
| Street Condition | Potholes, road damage — slows taxis |
| Traffic Signal Condition | Broken traffic lights — causes congestion |
| Street Light Condition | Dark streets — safety concern affecting usage |
| Blocked Driveway | Obstruction — reroutes traffic |
| Illegal Parking | Lane blockage — reduces road capacity |
| Water Main Break | Major disruption — street closures |
| Sewer | Flooding risk — affects all modes |
| Bus Stop Shelter Complaint | Transit infrastructure issues |
| Noise - Street/Sidewalk | Construction or events causing disruptions |

**Cleaned columns (14):**

| Column | Type | Description |
|--------|------|-------------|
| complaint_time | timestamp | When the complaint was filed |
| closed_time | timestamp | When the complaint was resolved |
| complaint_type | string | Category (Noise, Street Condition, etc.) |
| descriptor | string | Detailed description |
| zip_code | int | ZIP code of complaint location |
| borough | string | NYC borough |
| latitude | double | Complaint location latitude |
| longitude | double | Complaint location longitude |
| status | string | Current status (Closed, Open, etc.) |
| severity_score | int | Computed severity (based on complaint type) |
| resolution_hours | double | Hours to resolve the complaint |
| complaint_hour | timestamp | Hour-aligned timestamp for joining |
| year | int | Year (partition column) |
| month | int | Month (partition column) |

**Cleaning applied:** Removed rows with missing borough, coordinates, or timestamps. Filtered to transportation-related complaint types. Computed severity scores based on complaint type. Removed duplicates. Aligned to hourly timestamps.

---

### 2.6 NYC Event Permits (Street Closures)

| Property | Value |
|----------|-------|
| Source | NYC Open Data — Street Event Permits + Film Permits |
| URL | https://data.cityofnewyork.us/City-Government/ |
| Format | CSV (downloaded via Socrata API) |
| Files | event_permits_2022_2024.csv (844,567 rows) + film_permits_2022_2024.csv (10,918 rows), total 223 MB |
| Raw rows | 855,485 |
| Cleaned rows | 853,828 |
| Removed | 1,657 (0.19%) |
| HDFS size | 12.6 MB |
| Granularity | 1 row per event per hour |

**Why we need it:** Permitted events cause planned disruptions — marathons close major avenues, film shoots block streets, parades reroute traffic for hours. The street closure descriptions are parsed to extract affected streets and estimate disruption severity. Events explain demand shifts that would otherwise appear as anomalies.

**Cleaned columns (16):**

| Column | Type | Description |
|--------|------|-------------|
| event_id | string | Unique event identifier |
| event_name | string | Name/description of the event |
| event_start | timestamp | Event start time |
| event_end | timestamp | Event end time |
| event_type | string | Type of event (street closure, block party, etc.) |
| borough | string | NYC borough (Manhattan, Brooklyn, etc.) |
| street_description | string | Streets affected (text) |
| closure_type | string | Type of closure |
| source | string | Data source/agency |
| affected_streets | array\<string\> | List of individual streets affected |
| num_streets_affected | int | Count of streets affected |
| event_duration_hours | double | Duration of event in hours |
| disruption_severity | int | Severity score (computed from type + streets + duration) |
| event_hour | timestamp | Hour-aligned timestamp for joining |
| year | int | Year (partition column) |
| month | int | Month (partition column) |

**Cleaning applied:** Removed rows with missing borough or event times. Parsed street descriptions into individual street lists. Computed disruption severity scores. Aligned event times to hourly timestamps.

---

## 3. Grand Summary

### 3.1 Raw Data Downloaded

| Dataset | Files | Download Size | Rows (approx) | Format |
|---------|-------|---------------|----------------|--------|
| Yellow Taxi | 36 | 1.94 GB | ~119M | Parquet |
| Green Taxi | 36 | 52 MB | ~2.3M | Parquet |
| HVFHV Taxi | 36 | 17.03 GB | ~684M | Parquet |
| Taxi Zones | 2 | 1 MB | 263 | CSV/Shapefile |
| Citi Bike | 14 | 11.49 GB | ~109M | Zipped CSV |
| Subway | 3 | 6.63 GB | 40.8M | CSV |
| Weather | 3 | 1.5 MB | 26.3K | CSV |
| 311 | 3 | 1.66 GB | 2.78M | CSV |
| Events | 2 | 223 MB | 855K | CSV |
| **TOTAL** | **135** | **36.92 GB** | **~959M** | |

### 3.2 After Cleaning

| # | Dataset | Raw Rows | Cleaned Rows | Removed | HDFS Size |
|---|---------|----------|-------------|---------|-----------|
| 1 | Taxi (all) | 805,800,275 | 769,858,584 | 4.46% | 14.3 GB |
| 2 | Citi Bike | 109,249,001 | 108,469,697 | 0.71% | 5.0 GB |
| 3 | Subway | 40,800,000 | 5,985,275 | 85.3%* | 46.7 MB |
| 4 | 311 Complaints | 2,784,357 | 2,700,357 | 3.02% | 95.1 MB |
| 5 | Events | 855,485 | 853,828 | 0.19% | 12.6 MB |
| 6 | Weather | 26,304 | 26,304 | 0% | 564 KB |
| | **TOTAL** | **~959M** | **~887M** | | **~19.4 GB** |

*Subway's 85% reduction is from aggregation (combining payment methods), not data removal.

---

## 4. Join Strategy

### 4.1 The Problem

All 6 datasets have different schemas, different granularity, and different location formats:

| Dataset | Location Format | Time Format |
|---------|----------------|-------------|
| Taxi | zone_id (1–263) | pickup_hour |
| Bike | lat/lng coordinates | start_hour |
| Subway | station name + lat/lng + borough | transit_hour |
| 311 | borough + lat/lng | complaint_hour |
| Events | borough | event_hour |
| Weather | city-wide (no location) | weather_hour |

### 4.2 The Solution — Zone x Hour Grid

We build a **base grid** of every possible combination:
- **263 NYC taxi zones** x **~26,280 hours** (3 years) = **~6.9 million rows**

Each dataset is aggregated and left-joined onto this grid:

```
Grid (zone_id x hourly_timestamp)
  |
  |-- LEFT JOIN taxi       ON zone_id + hourly_timestamp
  |-- LEFT JOIN weather    ON hourly_timestamp only (city-wide)
  |-- LEFT JOIN subway     ON borough + hourly_timestamp
  |-- LEFT JOIN 311        ON borough + hourly_timestamp
  |-- LEFT JOIN events     ON borough + hourly_timestamp
  |-- Bike: pending spatial mapping (lat/lng → taxi zone)
```

### 4.3 How Each Dataset Connects

**Taxi → Grid (direct match):**
- Taxi data already uses `pu_location_id` which IS a taxi zone ID (1–263)
- Aggregated to: pickup count, dropoff count, avg fare, avg distance, avg duration per zone per hour

**Weather → Grid (hour only):**
- Same weather for all 263 zones in a given hour
- Joins on `hourly_timestamp` only — weather is city-wide

**Subway → Grid (via borough):**
- Subway stations are aggregated to borough level per hour
- Each zone inherits its borough's subway ridership
- Provides: total ridership, transfers, active stations per borough per hour

**311 Complaints → Grid (via borough):**
- Complaints aggregated to borough + hour
- Provides: complaint count, max severity per borough per hour

**Events → Grid (via borough):**
- Events aggregated to borough + hour
- Provides: event count, max severity, streets affected per borough per hour

**Bike → Grid (via spatial mapping):**
- 2,609 unique bike stations mapped to taxi zones using NYC taxi zone shapefile (point-in-polygon with pyproj CRS transformation from WGS84 to NAD83 State Plane)
- 2,520 stations successfully mapped (96.6%), 89 unmapped (outside zone boundaries)
- 122M bike trips assigned zone IDs, aggregated to: bike_starts, bike_ends, avg_bike_duration_sec per zone per hour
- Added to combined table as `combined_v2` (36 columns)

### 4.4 What One Row Looks Like After Join

```
zone_id: 161 (Midtown Center, Manhattan)
hourly_timestamp: 2023-07-15 18:00:00

-- Time features --
year: 2023, month: 7, day_of_week: 7 (Saturday)
hour_of_day: 18, is_rush_hour: 1, is_weekend: 0

-- Taxi features --
taxi_pickups: 342, taxi_dropoffs: 289
avg_fare: 14.50, avg_trip_distance: 2.3, avg_trip_duration_sec: 720

-- Weather features --
temperature_c: 31.2, humidity_pct: 65, precipitation_mm: 0
wind_speed_kmh: 15, visibility_m: 16000
is_rain: 0, is_snow: 0, is_extreme_heat: 1

-- Subway features (Manhattan total) --
subway_ridership: 85000, subway_transfers: 12000
subway_stations_active: 147

-- Bike features --
bike_starts: 85, bike_ends: 72, avg_bike_duration_sec: 780

-- 311 features (Manhattan total) --
complaint_count: 45, max_complaint_severity: 3

-- Event features (Manhattan total) --
event_count: 3, max_event_severity: 4, total_streets_affected: 12
```

### 4.5 Missing Values

Zones/hours with no activity get 0 (not null). For example, a residential zone at 3 AM may have `taxi_pickups: 0`, `complaint_count: 0`. Weather is never missing (continuous hourly data).

---

## 5. Join Results

| Property | Value |
|----------|-------|
| Script | join_all.py + add_bike_to_combined.py + fix_311_join.py |
| Rows | 6,917,952 |
| Columns | 38 |
| HDFS Size | 274.4 MB (823.2 MB with 3x replication) |
| Partitioned by | year |
| Output path | `/data/neurotraffic/combined_v3/` |

**Combined table columns (36):**

| Column | Type | Source |
|--------|------|--------|
| zone_id | int | Grid (1–263 taxi zones) |
| hourly_timestamp | timestamp | Grid (every hour, 2022–2024) |
| year | int | Derived from timestamp |
| month | int | Derived from timestamp |
| day_of_week | int | Derived (1=Mon, 7=Sun) |
| hour_of_day | int | Derived (0–23) |
| is_rush_hour | int | Flag: 7–9 AM or 5–7 PM weekdays |
| is_weekend | int | Flag: Saturday/Sunday |
| taxi_pickups | long | Taxi |
| taxi_dropoffs | long | Taxi |
| avg_fare | double | Taxi |
| avg_trip_distance | double | Taxi |
| avg_trip_duration_sec | double | Taxi |
| temperature_c | double | Weather |
| humidity_pct | double | Weather |
| precipitation_mm | double | Weather |
| rain_mm | double | Weather |
| snowfall_cm | double | Weather |
| wind_speed_kmh | double | Weather |
| wind_gusts_kmh | double | Weather |
| visibility_m | double | Weather |
| is_rain | int | Weather |
| is_snow | int | Weather |
| is_extreme_cold | int | Weather |
| is_extreme_heat | int | Weather |
| is_low_visibility | int | Weather |
| is_high_wind | int | Weather |
| subway_ridership | long | Subway |
| subway_transfers | long | Subway |
| subway_stations_active | long | Subway |
| complaint_count | long | 311 |
| max_complaint_severity | int | 311 |
| event_count | long | Events |
| bike_starts | long | Citi Bike |
| bike_ends | long | Citi Bike |
| avg_bike_duration_sec | double | Citi Bike |

---

## 6. Hadoop MapReduce Job

A Hadoop Streaming MapReduce job was run to compute **daily taxi pickup counts per zone** — this satisfies the professor's requirement for Hadoop MapReduce (as opposed to only PySpark).

| Property | Value |
|----------|-------|
| Job type | Hadoop Streaming (bash mapper + reducer) |
| Input | Combined table exported as TSV (1.2 GB, 6.9M rows) |
| Output | 200.8 MB (602.3 MB replicated) |
| Map tasks | 10 |
| Reduce tasks | 1 |
| Output path | `/data/neurotraffic/mapreduce_output/` |
| Output format | TSV: `zone_id \t date \t total_daily_pickups` |

**Scripts:** `mapper.sh` (emits zone_id + date + pickup count), `reducer.sh` (sums pickups per zone per day)

**Command:**
```powershell
docker exec namenode hadoop jar /opt/hadoop-3.2.1/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar -D mapreduce.framework.name=local -D mapreduce.jobtracker.address=local -input /data/neurotraffic/mapreduce_input -output /data/neurotraffic/mapreduce_output -mapper /scripts/mapper.sh -reducer /scripts/reducer.sh
```

---

## 7. Data Quality Report

| Metric | Value |
|--------|-------|
| Total rows | 6,917,952 |
| Total columns | 36 |
| Null values | 0 across all columns |
| Zones with zero pickups | 311,385 zone-hours (4.5%) |

**Taxi statistics:**
- Total pickups: 769,858,584
- Total dropoffs: 769,851,721
- Avg pickups per zone-hour: 111.3
- Max pickups (single zone-hour): 3,265
- Overall avg fare: $20.41
- Overall avg trip distance: 4.92 miles

**Weather coverage:**
- 100% coverage (no missing weather rows)
- Temperature range: -17.0C to 36.9C (avg 12.8C)
- Rainy zone-hours: 436,580 (6.3%)
- Snowy zone-hours: 90,209 (1.3%)

**Subway ridership:**
- Total ridership: 114.7 billion (across all zone-hours)
- Avg per zone-hour: 16,588
- Max single zone-hour: 268,324
- Zero ridership in 47.9% of zone-hours (zones without subway stations)

**Events:**
- Total event zone-hours: 52,047,470
- Max events in single zone-hour: 156

**311 complaints (fixed):**
- Total complaints across all zone-hours: 161,869,813
- Zone-hours with complaints: 6,523,863 (94.3% of all zone-hours)
- Max complaint severity: 4
- Bug fix: 311 data had UPPERCASE borough names (`BRONX`) while zone lookup used Title Case (`Bronx`). Added `initcap()` normalization in join_all.py.

---

## 8. HDFS Storage

All data is stored with **3x replication** across 3 DataNodes for fault tolerance.

**Total HDFS usage: 77.4 GB actual / 232.3 GB replicated**

### By directory

| Directory | Actual Size | Replicated (3x) | Notes |
|-----------|-------------|------------------|-------|
| raw/ | 46.1 GB | 138.4 GB | Original uploaded data |
| cleaned/ | 19.4 GB | 58.3 GB | Cleaned parquet files |
| combined/ | 178.4 MB | 535.3 MB | Joined output |
| staging/ | 11.7 GB | 35.1 GB | Leftover HVFHV staging (can be deleted) |
| **TOTAL** | **77.4 GB** | **232.3 GB** | |

### Raw data on HDFS

| Dataset | Actual Size | Replicated (3x) |
|---------|-------------|------------------|
| Citi Bike | 19.9 GB | 59.7 GB |
| Taxi | 17.7 GB | 53.2 GB |
| Subway | 6.6 GB | 19.9 GB |
| 311 | 1.7 GB | 5.0 GB |
| Events | 212.9 MB | 638.7 MB |
| Weather | 1.5 MB | 4.6 MB |

### Cleaned data on HDFS

| Dataset | Actual Size | Replicated (3x) |
|---------|-------------|------------------|
| Taxi (all) | 14.3 GB | 42.8 GB |
| Citi Bike | 5.0 GB | 15.0 GB |
| 311 | 95.1 MB | 285.3 MB |
| Subway | 46.7 MB | 140.2 MB |
| Events | 12.6 MB | 37.7 MB |
| Weather | 564.3 KB | 1.7 MB |

---

## 7. Directory Structure

### Local (E:\Big_Data_Project\)

```
E:\Big_Data_Project\
├── docker-compose.yml          # Docker cluster config (7 containers)
├── hadoop.env                  # Hadoop environment variables
├── neurotraffic_download.py    # Dataset download script
├── clean_weather.py            # Weather cleaning
├── clean_events.py             # Events cleaning
├── clean_311.py                # 311 complaints cleaning
├── clean_subway.py             # Subway ridership cleaning
├── clean_bike.py               # Citi Bike cleaning
├── clean_taxi.py               # Taxi cleaning (Yellow + Green + HVFHV)
├── join_all.py                 # Join all datasets into combined table
├── extract_bike_stations.py    # Extract unique bike stations from HDFS
├── bike_zone_mapping.py        # Map bike stations to taxi zones (local, pyshp)
├── add_bike_to_combined.py     # Add bike data to combined table
├── fix_311_join.py             # Fix 311 borough case mismatch
├── export_for_mapreduce.py     # Export combined to TSV for MapReduce
├── mapper.sh                   # Hadoop MapReduce mapper (bash)
├── reducer.sh                  # Hadoop MapReduce reducer (bash)
├── data_quality_report.py      # Data quality analysis script
├── NEUROTRAFFIC_DATA_REPORT.md # This file
└── neurotraffic_data/          # Raw downloaded data (36.92 GB)
    ├── taxi/yellow/             36 Parquet files
    ├── taxi/green/              36 Parquet files
    ├── taxi/hvfhv/              36 Parquet files
    ├── taxi/zones/              taxi_zone_lookup.csv + taxi_zones.zip
    ├── citibike/                14 zip files
    ├── citibike_csv/            126 extracted CSVs
    ├── subway/                  3 yearly CSVs
    ├── weather/                 3 yearly CSVs
    ├── 311/                     3 yearly CSVs
    └── events/                  2 CSV files
```

### HDFS (/data/neurotraffic/)

```
/data/neurotraffic/
├── raw/                          # Original uploaded data (46.1 GB)
│   ├── weather/
│   ├── events/
│   ├── 311/
│   ├── subway/
│   ├── citibike/
│   └── taxi/
│       ├── yellow/
│       ├── green/
│       ├── hvfhv/
│       └── zones/
│
├── cleaned/                      # Cleaned parquet data (19.4 GB)
│   ├── weather/
│   ├── events/
│   ├── 311/
│   ├── subway/
│   ├── citibike/
│   └── taxi/
│       ├── yellow/
│       ├── green/
│       └── hvfhv/
│
├── staging/                      # HVFHV staging leftovers (11.7 GB, can be deleted)
│
├── combined/                     # Initial join output (178.4 MB, 33 columns)
│   ├── year=2022/
│   ├── year=2023/
│   └── year=2024/
│
├── combined_v2/                  # Join with bike data (259.8 MB, 36 columns)
│   ├── year=2022/
│   ├── year=2023/
│   └── year=2024/
│
├── combined_v3/                  # Final: 311 fix applied (274.4 MB, 38 columns)
│   ├── year=2022/
│   ├── year=2023/
│   └── year=2024/
│
├── mapreduce_input/              # Combined data as TSV (1.2 GB)
│
└── mapreduce_output/             # MapReduce results (200.8 MB)
```

---

## 8. How to Run

**Start the cluster:**
```powershell
cd E:\Big_Data_Project
docker-compose up -d
docker exec namenode hdfs dfsadmin -safemode wait
```

**Run cleaning (use PowerShell, NOT Git Bash):**
```powershell
docker exec spark-master /spark/bin/spark-submit --master "local[2]" --driver-memory 2g --conf spark.driver.maxResultSize=1g --conf spark.sql.shuffle.partitions=20 /scripts/clean_weather.py
docker exec spark-master /spark/bin/spark-submit --master "local[2]" --driver-memory 2g --conf spark.driver.maxResultSize=1g --conf spark.sql.shuffle.partitions=20 /scripts/clean_events.py
docker exec spark-master /spark/bin/spark-submit --master "local[2]" --driver-memory 2g --conf spark.driver.maxResultSize=1g --conf spark.sql.shuffle.partitions=20 /scripts/clean_311.py
docker exec spark-master /spark/bin/spark-submit --master "local[2]" --driver-memory 2g --conf spark.driver.maxResultSize=1g --conf spark.sql.shuffle.partitions=20 /scripts/clean_subway.py
docker exec spark-master /spark/bin/spark-submit --master "local[2]" --driver-memory 2g --conf spark.driver.maxResultSize=1g --conf spark.sql.shuffle.partitions=20 /scripts/clean_bike.py
docker exec spark-master /spark/bin/spark-submit --master "local[2]" --driver-memory 2g --conf spark.driver.maxResultSize=1g --conf spark.sql.shuffle.partitions=20 /scripts/clean_taxi.py
```

**Run join:**
```powershell
docker exec spark-master /spark/bin/spark-submit --master "local[2]" --driver-memory 2g --conf spark.driver.maxResultSize=1g --conf spark.sql.shuffle.partitions=20 /scripts/join_all.py
```

**Stop containers (preserves data):**
```powershell
docker-compose stop
```

**Important:** Use `docker-compose stop` (NOT `down`) to preserve HDFS volumes. Use `local[2]` and `2g` memory to avoid Docker Desktop memory crashes (system has 15.2 GB RAM, Docker allocated 7.37 GB). Always use PowerShell — Git Bash mangles Unix paths inside containers.

---

## 9. Current Status & Next Steps

| Step | Script | Owner | Status |
|------|--------|-------|--------|
| 1. Download all 6 datasets | neurotraffic_download.py | Navaneeth | Done |
| 2. Upload to HDFS | — | Navaneeth | Done |
| 3. Clean all 6 datasets | clean_*.py | Navaneeth | Done |
| 4. Join 5 datasets (zone x hour grid) | join_all.py | Navaneeth | Done |
| 5. Bike spatial mapping (lat/lng → taxi zone) | bike_zone_mapping.py | Navaneeth | Done |
| 6. Add bike data to combined table | add_bike_to_combined.py | Navaneeth | Done |
| 7. Hadoop MapReduce job | mapper.sh + reducer.sh | Navaneeth | Done |
| 8. Data quality report | data_quality_report.py | Navaneeth | Done |
| 9. Fix 311 complaint join | fix_311_join.py | Navaneeth | Done |
| 10. GraphX transport network | — | Tanay | Pending |
| 11. MLlib disruption model | — | Team | Pending |

**Known limitations:**
- Subway, 311, and events are joined at **borough level** (not zone level). All zones in Manhattan share the same subway ridership / complaint count for a given hour.
- "Unspecified" borough 311 complaints (~small fraction) are dropped since they can't be mapped to any zone.
