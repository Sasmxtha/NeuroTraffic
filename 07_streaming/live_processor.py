"""
NeuroTraffic - Lightweight Live Processor
Reads JSON batches from HDFS /data/streaming/input/, computes anomaly scores
using z-scores against zone_hour_profile baselines, writes latest.csv to HDFS.
Usage: python 07_streaming/live_processor.py --interval 15
"""
import argparse, io, json, os, re, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import requests

NYC = ZoneInfo("America/New_York")
HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE_LOCAL = os.path.join(HERE, "lookups", "zone_hour_profile.csv")
WEBHDFS_HOST = "localhost"
WEBHDFS_BASE = f"http://{WEBHDFS_HOST}:9870/webhdfs/v1"
DATANODE_PORT_MAP = {"datanode1": 9864, "datanode2": 9865, "datanode3": 9866}
INPUT_DIR = "/data/streaming/input"
OUTPUT_CSV = "/data/streaming/output/latest.csv"

def fix_redirect(url):
    for node, port in DATANODE_PORT_MAP.items():
        url = re.sub(rf"//{node}:\d+", f"//{WEBHDFS_HOST}:{port}", url)
    return url

def hdfs_write(path, text):
    r = requests.put(f"{WEBHDFS_BASE}{path}?op=CREATE&overwrite=true", allow_redirects=False, timeout=10)
    if r.status_code != 307: return False
    r = requests.put(fix_redirect(r.headers["Location"]), data=text.encode("utf-8"),
                     headers={"Content-Type": "application/octet-stream"}, timeout=10)
    return r.status_code in (200, 201)

def hdfs_list(path):
    r = requests.get(f"{WEBHDFS_BASE}{path}?op=LISTSTATUS", timeout=10)
    if r.status_code != 200: return []
    return [f["pathSuffix"] for f in r.json()["FileStatuses"]["FileStatus"]
            if f["type"] == "FILE" and f["pathSuffix"].endswith(".json")]

def hdfs_read(path):
    r = requests.get(f"{WEBHDFS_BASE}{path}?op=OPEN", allow_redirects=False, timeout=10)
    if r.status_code == 307:
        r = requests.get(fix_redirect(r.headers["Location"]), timeout=30)
    return r.text if r.status_code == 200 else None

def hdfs_delete(path):
    requests.delete(f"{WEBHDFS_BASE}{path}?op=DELETE&recursive=false", timeout=5)

def load_profile():
    if not os.path.exists(PROFILE_LOCAL):
        sys.exit("Missing zone_hour_profile.csv")
    return pd.read_csv(PROFILE_LOCAL)

def compute_scores(batch_df, profile):
    df = batch_df.copy()
    key_cols = ["zone_id", "hour_of_day", "is_weekend"]
    df = df.merge(profile, on=key_cols, how="left", suffixes=("", "_prof"))
    for metric, z_col, prof_col in [
        ("taxi_pickups", "z_pickups", "taxi_pickups_prof"),
        ("subway_ridership", "z_subway", "subway_ridership_prof"),
        ("bike_starts", "z_bikes", "bike_starts_prof"),
        ("complaint_count", "z_complaints", "complaint_count_prof"),
    ]:
        mean = df[prof_col].fillna(0)
        std = mean.clip(lower=1) * 0.5
        z = ((df[metric].fillna(0) - mean).abs() / std).clip(upper=3.0)
        df[z_col] = z
    df["max_z"] = df[["z_pickups","z_subway","z_bikes","z_complaints"]].max(axis=1)
    df["avg_z"] = df[["z_pickups","z_subway","z_bikes","z_complaints"]].mean(axis=1)
    wp = 0.0
    for col_name, w in [("is_rain",0.03),("is_heavy_rain",0.06),("is_extreme_cold",0.03),
                         ("is_extreme_heat",0.03),("is_high_wind",0.05),("is_low_visibility",0.03)]:
        if col_name in df.columns: wp = wp + df[col_name].fillna(0) * w
    df["weather_penalty"] = wp
    df["complaint_signal"] = (df["z_complaints"] * 0.07).clip(upper=0.2)
    sev = df["disruption_severity"].fillna(0) if "disruption_severity" in df.columns else 0
    has_alert = (df["affected_line"].fillna("-") != "-") if "affected_line" in df.columns else False
    df["disruption_indicator"] = has_alert.astype(float) * (0.45 + sev * 0.45)
    df["anomaly_score"] = (df["max_z"]*0.04 + df["avg_z"]*0.03 +
        df["weather_penalty"] + df["complaint_signal"] + df["disruption_indicator"]).clip(0,1)
    df["is_disruption"] = (df["anomaly_score"] >= 0.80).astype(int)
    out_cols = ["zone_id","timestamp","anomaly_score","is_disruption","affected_line",
                "disruption_severity","traffic_speed_mph","z_subway","z_pickups",
                "taxi_pickups","subway_ridership","bike_starts"]
    for c in out_cols:
        if c not in df.columns: df[c] = 0
    return df[out_cols]

def run(interval):
    print("="*60)
    print("  NeuroTraffic - Live Processor (Python)")
    print("="*60)
    profile = load_profile()
    print(f"  Profile: {len(profile)} rows")
    seen = set()
    batch_num = 0
    while True:
        try:
            files = hdfs_list(INPUT_DIR)
            new_files = sorted(set(files) - seen)
            if new_files:
                records = []
                for fname in new_files[-1:]:
                    text = hdfs_read(f"{INPUT_DIR}/{fname}")
                    if text:
                        for line in text.strip().split(chr(10)):
                            try: records.append(json.loads(line))
                            except json.JSONDecodeError: pass
                    seen.add(fname)
                if records:
                    batch_num += 1
                    bdf = pd.DataFrame(records).drop_duplicates(subset=["zone_id"], keep="last")
                    scored = compute_scores(bdf, profile)
                    csv_text = scored.to_csv(index=False)
                    local_out = os.path.join(os.path.dirname(__file__), "..", "08_dashboard", "streaming", "output", "latest.csv")
                    history_out = os.path.join(os.path.dirname(__file__), "..", "08_dashboard", "streaming", "output", "live_history.csv")
                    os.makedirs(os.path.dirname(local_out), exist_ok=True)
                    with open(local_out, "w") as lf: lf.write(csv_text)
                    # Append to rolling history (keep last 24h of snapshots, one per minute)
                    append_header = not os.path.exists(history_out)
                    if batch_num % 4 == 1:  # every ~60s (4 x 15s intervals)
                        with open(history_out, "a") as hf:
                            if append_header: hf.write(csv_text)
                            else:
                                lines = csv_text.split(chr(10))
                                hf.write(chr(10).join(lines[1:]))
                    if hdfs_write(OUTPUT_CSV, csv_text):
                        nd = int(scored["is_disruption"].sum())
                        peak = scored["anomaly_score"].max()
                        now_s = datetime.now(NYC).strftime("%H:%M:%S")
                        flag = "[ALERT]" if nd > 0 else "[OK]   "
                        print(f"  {flag} Batch {batch_num:04d} | {now_s} | "
                              f"{len(scored)} zones | {nd} disruptions | peak={peak:.3f}")
                    else:
                        print("  [ERR]  HDFS write failed")
                for old in sorted(seen)[:-10]:
                    hdfs_delete(f"{INPUT_DIR}/{old}")
                    seen.discard(old)
            else:
                if batch_num == 0:
                    now_s = datetime.now(NYC).strftime("%H:%M:%S")
                    print(f"  [WAIT] {now_s} | Waiting for input in {INPUT_DIR}")
        except Exception as e:
            print(f"  [ERR]  {e}")
        time.sleep(interval)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=15)
    run(ap.parse_args().interval)
