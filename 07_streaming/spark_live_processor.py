"""
NeuroTraffic — Spark Live Processor
====================================
Reads streaming JSON from HDFS, computes anomaly scores using
PySpark against zone_hour_profile, writes latest.csv back to HDFS.
"""
import os, sys, time, json
from datetime import datetime
import pytz

NYC = pytz.timezone("America/New_York")

SPARK_MASTER = os.environ.get("SPARK_MASTER", "local[2]")
HDFS_PREFIX = "hdfs://localhost:9000"
INPUT_DIR = f"{HDFS_PREFIX}/data/streaming/input"
OUTPUT_CSV = "/data/streaming/output/latest.csv"
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "lookups", "zone_hour_profile.csv")
LOCAL_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "08_dashboard", "streaming", "output", "latest.csv")
HISTORY_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "08_dashboard", "streaming", "output", "live_history.csv")
HISTORY_MAX_DAYS = 30

WEBHDFS = "http://localhost:9870/webhdfs/v1"
DATANODE_PORTS = {"datanode1": 9864, "datanode2": 9865, "datanode3": 9866}

import requests, re, io
import pandas as pd

def hdfs_list(path):
    try:
        r = requests.get(f"{WEBHDFS}{path}?op=LISTSTATUS", timeout=3)
        if r.status_code == 200:
            return [f["pathSuffix"] for f in r.json()["FileStatuses"]["FileStatus"] if f["type"] == "FILE"]
    except Exception:
        pass
    return []

def _fix_redirect(url):
    for node, port in DATANODE_PORTS.items():
        if f"//{node}:" in url:
            url = re.sub(rf'//{node}:\d+', f'//localhost:{port}', url)
    return url

def hdfs_read(path):
    try:
        r = requests.get(f"{WEBHDFS}{path}?op=OPEN", allow_redirects=False, timeout=3)
        if r.status_code == 307:
            r = requests.get(_fix_redirect(r.headers["Location"]), timeout=5)
        return r.text if r.status_code == 200 else None
    except Exception:
        return None

def hdfs_write(path, data):
    try:
        r = requests.put(f"{WEBHDFS}{path}?op=CREATE&overwrite=true", allow_redirects=False, timeout=3)
        if r.status_code == 307:
            r2 = requests.put(_fix_redirect(r.headers["Location"]),
                              data=data.encode("utf-8"),
                              headers={"Content-Type": "application/octet-stream"}, timeout=5)
            return r2.status_code in [200, 201]
    except Exception:
        pass
    return False

def hdfs_delete(path):
    try:
        requests.delete(f"{WEBHDFS}{path}?op=DELETE", timeout=2)
    except Exception:
        pass

def run(interval):
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, IntegerType

    print("=" * 60)
    print("  NeuroTraffic - Spark Live Processor")
    print("=" * 60)

    spark = SparkSession.builder \
        .appName("NeuroTraffic-SparkLiveProcessor") \
        .master(SPARK_MASTER) \
        .config("spark.executor.memory", "512m") \
        .config("spark.driver.memory", "512m") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print(f"  Spark session: {spark.sparkContext.master}")

    # Load profile — rename value columns to avg_ prefix to avoid ambiguity after join
    profile_pdf = pd.read_csv(PROFILE_PATH)
    val_cols = ["taxi_pickups", "taxi_dropoffs", "subway_ridership",
                "bike_starts", "bike_ends", "complaint_count", "event_count"]
    for c in val_cols:
        if c in profile_pdf.columns:
            profile_pdf.rename(columns={c: f"avg_{c}"}, inplace=True)
            profile_pdf[f"std_{c}"] = (profile_pdf[f"avg_{c}"] * 0.3).clip(lower=1.0)
    profile_df = spark.createDataFrame(profile_pdf)
    profile_df = profile_df.withColumnRenamed("zone_id", "p_zone_id") \
        .withColumnRenamed("hour_of_day", "p_hour") \
        .withColumnRenamed("is_weekend", "p_is_weekend")
    profile_df.cache()
    print(f"  Profile: {profile_df.count()} rows (cached in Spark)")

    seen = set()
    batch_num = 0

    while True:
        try:
            files = hdfs_list("/data/streaming/input")
            new_files = sorted(set(files) - seen)
            if new_files:
                fname = new_files[-1]
                text = hdfs_read(f"/data/streaming/input/{fname}")
                seen.update(new_files)

                if text:
                    records = []
                    for line in text.strip().split(chr(10)):
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

                    if records:
                        batch_num += 1
                        # Deduplicate by zone_id
                        pdf = pd.DataFrame(records).drop_duplicates(subset=["zone_id"], keep="last")
                        now = datetime.now(NYC)

                        # Create Spark DF from incoming data
                        sdf = spark.createDataFrame(pdf)

                        # Add time columns
                        hour_val = now.hour
                        is_wknd = 1 if now.weekday() >= 5 else 0
                        sdf = sdf.withColumn("hour_of_day", F.lit(hour_val)) \
                                  .withColumn("is_weekend", F.lit(is_wknd))

                        # Ensure numeric columns
                        for col in ["taxi_pickups", "taxi_dropoffs", "subway_ridership",
                                    "bike_starts", "bike_ends", "complaint_count", "event_count"]:
                            if col in sdf.columns:
                                sdf = sdf.withColumn(col, F.col(col).cast(DoubleType()))
                            else:
                                sdf = sdf.withColumn(col, F.lit(0.0))

                        # Join with profile to get historical means/stds
                        joined = sdf.join(
                            profile_df,
                            (sdf["zone_id"] == profile_df["p_zone_id"]) &
                            (sdf["hour_of_day"] == profile_df["p_hour"]) &
                            (sdf["is_weekend"] == profile_df["p_is_weekend"]),
                            "left"
                        )

                        # Compute z-scores — use sdf[] for streaming cols to avoid ambiguity
                        z_cols = [
                            ("taxi_pickups", "avg_taxi_pickups", "std_taxi_pickups", "z_taxi"),
                            ("subway_ridership", "avg_subway_ridership", "std_subway_ridership", "z_subway"),
                            ("bike_starts", "avg_bike_starts", "std_bike_starts", "z_bike"),
                        ]
                        for val_col, mean_col, std_col, z_name in z_cols:
                            if mean_col in profile_pdf.columns:
                                joined = joined.withColumn(
                                    z_name,
                                    F.when(
                                        (F.col(std_col).isNotNull()) & (F.col(std_col) > 0),
                                        F.abs((sdf[val_col] - F.col(mean_col)) / F.col(std_col))
                                    ).otherwise(0.0)
                                )
                            else:
                                joined = joined.withColumn(z_name, F.lit(0.0))

                        # Ensure z columns exist
                        for zc in ["z_taxi", "z_subway", "z_bike"]:
                            if zc not in [f.name for f in joined.schema.fields]:
                                joined = joined.withColumn(zc, F.lit(0.0))

                        # Compute anomaly score
                        joined = joined.withColumn("max_z",
                            F.greatest(
                                F.coalesce(F.col("z_taxi"), F.lit(0.0)),
                                F.coalesce(F.col("z_subway"), F.lit(0.0)),
                                F.coalesce(F.col("z_bike"), F.lit(0.0))
                            )
                        )
                        joined = joined.withColumn("avg_z",
                            (F.coalesce(F.col("z_taxi"), F.lit(0.0)) +
                             F.coalesce(F.col("z_subway"), F.lit(0.0)) +
                             F.coalesce(F.col("z_bike"), F.lit(0.0))) / 3.0
                        )

                        # Weather penalty
                        if "precipitation_mm" in sdf.columns:
                            joined = joined.withColumn("weather_penalty",
                                F.when(sdf["precipitation_mm"] > 5, 0.15)
                                 .when(sdf["precipitation_mm"] > 0, 0.05)
                                 .otherwise(0.0)
                            )
                        else:
                            joined = joined.withColumn("weather_penalty", F.lit(0.0))

                        # Disruption indicator from MTA alerts
                        if "disruption_severity" in sdf.columns:
                            has_alert = F.when(
                                (sdf["affected_line"].isNotNull()) & (sdf["affected_line"] != "-"),
                                F.lit(1.0)
                            ).otherwise(F.lit(0.0))
                            joined = joined.withColumn("disruption_indicator",
                                has_alert * (0.45 + F.coalesce(sdf["disruption_severity"], F.lit(0.0)) * 0.45)
                            )
                        else:
                            joined = joined.withColumn("disruption_indicator", F.lit(0.0))

                        # Complaint signal
                        if "z_complaints" in [f.name for f in joined.schema.fields]:
                            joined = joined.withColumn("complaint_signal",
                                F.least(F.col("z_complaints") * 0.07, F.lit(0.2)))
                        else:
                            joined = joined.withColumn("complaint_signal", F.lit(0.0))

                        # Final score
                        joined = joined.withColumn("anomaly_score",
                            F.least(F.lit(1.0), F.greatest(F.lit(0.0),
                                F.col("max_z") * 0.04 +
                                F.col("avg_z") * 0.03 +
                                F.col("weather_penalty") +
                                F.col("complaint_signal") +
                                F.col("disruption_indicator")
                            ))
                        )
                        joined = joined.withColumn("is_disruption",
                            F.when(F.col("anomaly_score") >= 0.80, 1).otherwise(0)
                        )

                        # Select output columns
                        ts_str = now.strftime("%Y-%m-%dT%H:%M:%S")
                        out = joined.select(
                            sdf["zone_id"],
                            F.lit(ts_str).alias("timestamp"),
                            F.col("anomaly_score"),
                            F.col("is_disruption"),
                            F.coalesce(sdf["affected_line"], F.lit("-")).alias("affected_line"),
                            F.coalesce(sdf["disruption_severity"], F.lit(0.0)).alias("disruption_severity"),
                            F.coalesce(sdf["traffic_speed_mph"], F.lit(0.0)).alias("traffic_speed_mph") if "traffic_speed_mph" in sdf.columns else F.lit(0.0).alias("traffic_speed_mph"),
                            F.coalesce(F.col("z_subway"), F.lit(0.0)).alias("z_subway"),
                            F.coalesce(F.col("z_taxi"), F.lit(0.0)).alias("z_pickups"),
                            F.coalesce(sdf["taxi_pickups"], F.lit(0)).cast(IntegerType()).alias("taxi_pickups"),
                            F.coalesce(sdf["subway_ridership"], F.lit(0)).cast(IntegerType()).alias("subway_ridership"),
                            F.coalesce(sdf["bike_starts"], F.lit(0)).cast(IntegerType()).alias("bike_starts"),
                        )

                        # Collect and write
                        result_pdf = out.toPandas()
                        csv_text = result_pdf.to_csv(index=False)

                        # Write local copy
                        os.makedirs(os.path.dirname(LOCAL_OUTPUT), exist_ok=True)
                        with open(LOCAL_OUTPUT, "w") as lf:
                            lf.write(csv_text)

                        # Append to rolling history every ~60s (4 batches)
                        if batch_num % 4 == 1:
                            append_header = not os.path.exists(HISTORY_OUTPUT)
                            with open(HISTORY_OUTPUT, "a") as hf:
                                if append_header:
                                    hf.write(csv_text)
                                else:
                                    lines = csv_text.split(chr(10))
                                    hf.write(chr(10).join(lines[1:]))
                            # Trim history to HISTORY_MAX_DAYS
                            if batch_num % 240 == 1:  # check every ~hour
                                try:
                                    hist = pd.read_csv(HISTORY_OUTPUT)
                                    if "timestamp" in hist.columns:
                                        hist["_ts"] = pd.to_datetime(hist["timestamp"])
                                        cutoff = datetime.now(NYC) - pd.Timedelta(days=HISTORY_MAX_DAYS)
                                        trimmed = hist[hist["_ts"] >= cutoff].drop(columns=["_ts"])
                                        if len(trimmed) < len(hist):
                                            trimmed.to_csv(HISTORY_OUTPUT, index=False)
                                            print(f"  [INFO] Trimmed history: {len(hist)} -> {len(trimmed)} rows (kept {HISTORY_MAX_DAYS}d)")
                                except Exception:
                                    pass

                        # Write to HDFS
                        if hdfs_write(OUTPUT_CSV, csv_text):
                            nd = int(result_pdf["is_disruption"].sum())
                            peak = result_pdf["anomaly_score"].max()
                            now_s = now.strftime("%H:%M:%S")
                            flag = "[ALERT]" if nd > 0 else "[OK]   "
                            print(f"  {flag} Batch {batch_num:04d} | {now_s} | "
                                  f"{len(result_pdf)} zones | {nd} disruptions | peak={peak:.3f} [Spark]")
                        else:
                            print("  [ERR]  HDFS write failed")

                # Cleanup old input files
                for old in sorted(seen)[:-10]:
                    hdfs_delete(f"/data/streaming/input/{old}")
                    seen.discard(old)

        except Exception as e:
            print(f"  [ERR]  {e}")

        time.sleep(interval)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=15)
    run(ap.parse_args().interval)
