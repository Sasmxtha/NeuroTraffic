"""
NeuroTraffic — Real-Time Data Simulator
========================================
Generates realistic NYC transport events in real-time and writes
JSON micro-batches to HDFS for Spark Structured Streaming to process.

Data is generated from real statistical baselines (zone capacities,
borough patterns) — not replayed from historical files.

Usage:
    python streaming_simulator.py
    python streaming_simulator.py --speed 3 --time-scale 120
"""

import requests
import json
import time
import random
import math
import sys
import os
import argparse
import io
import re
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# ── WebHDFS config ──
WEBHDFS_HOST = "localhost"
WEBHDFS_PORT = 9870
WEBHDFS_BASE = f"http://{WEBHDFS_HOST}:{WEBHDFS_PORT}/webhdfs/v1"

DATANODE_PORT_MAP = {
    "datanode1": 9864,
    "datanode2": 9865,
    "datanode3": 9866,
}

INPUT_DIR = "/data/streaming/input"
NODES_PATH = "/data/neurotraffic/dashboard/graph_nodes.csv"

# ── Borough baseline traffic multipliers (from real data analysis) ──
BOROUGH_MULTIPLIERS = {
    "Manhattan": 3.5,
    "Brooklyn": 1.8,
    "Queens": 1.2,
    "Bronx": 0.8,
    "Staten Island": 0.4,
    "EWR": 0.3,
}

# ── Subway lines and their boroughs ──
SUBWAY_LINES = {
    "L":       ["Manhattan", "Brooklyn"],
    "A/C/E":   ["Manhattan", "Brooklyn", "Queens"],
    "1/2/3":   ["Manhattan", "Bronx", "Brooklyn"],
    "4/5/6":   ["Manhattan", "Bronx", "Brooklyn"],
    "N/Q/R/W": ["Manhattan", "Brooklyn", "Queens"],
    "7":       ["Manhattan", "Queens"],
    "G":       ["Brooklyn", "Queens"],
    "B/D/F/M": ["Manhattan", "Brooklyn", "Queens"],
}

LINE_NAMES = list(SUBWAY_LINES.keys())

OUTPUT_CSV = '/data/streaming/output/latest.csv'


def compute_anomaly_score(record):
    weather_penalty = (
        record.get('is_rain', 0) * 0.03 +
        record.get('is_heavy_rain', 0) * 0.06 +
        record.get('is_extreme_cold', 0) * 0.03 +
        record.get('is_extreme_heat', 0) * 0.03 +
        record.get('is_high_wind', 0) * 0.05 +
        record.get('is_low_visibility', 0) * 0.03
    )
    complaint_z = min(3.0, max(0, record['complaint_count'] - 2) / max(1, 1.5))
    complaint_signal = min(0.2, complaint_z * 0.07)
    affected_line = record.get('affected_line', '-')
    disruption_severity = record.get('disruption_severity', 0.0)
    if affected_line != '-' and disruption_severity > 0:
        disruption_indicator = 0.45 + disruption_severity * 0.45
    else:
        disruption_indicator = 0.0
    base_z = random.gauss(0.3, 0.15)
    base_z = max(0, min(3.0, base_z))
    raw_score = base_z * 0.04 + base_z * 0.03 + weather_penalty + complaint_signal + disruption_indicator
    return max(0.0, min(1.0, raw_score))


def write_latest_csv(records):
    rows = []
    for r in records:
        score = compute_anomaly_score(r)
        rows.append({
            'zone_id': r['zone_id'], 'timestamp': r['timestamp'],
            'anomaly_score': round(score, 4),
            'is_disruption': 1 if score >= 0.80 else 0,
            'affected_line': r['affected_line'],
            'disruption_severity': round(r.get('disruption_severity', 0.0), 3),
            'traffic_speed_mph': '',
            'z_subway': round(random.gauss(0.3, 0.15), 4),
            'z_pickups': round(random.gauss(0.3, 0.15), 4),
            'taxi_pickups': r['taxi_pickups'],
            'subway_ridership': r['subway_ridership'],
            'bike_starts': r['bike_starts'],
        })
    df = pd.DataFrame(rows)
    csv_text = df.to_csv(index=False)
    return hdfs_write(OUTPUT_CSV, csv_text)



def fix_redirect(url):
    for node, port in DATANODE_PORT_MAP.items():
        if f"//{node}:" in url:
            url = re.sub(rf'//{node}:\d+', f'//{WEBHDFS_HOST}:{port}', url)
            break
    return url


def hdfs_read(path):
    url = f"{WEBHDFS_BASE}{path}?op=OPEN"
    r = requests.get(url, allow_redirects=False, timeout=10)
    if r.status_code == 307:
        r = requests.get(fix_redirect(r.headers["Location"]), timeout=10)
    return r


def hdfs_write(path, data):
    url = f"{WEBHDFS_BASE}{path}?op=CREATE&overwrite=true"
    r = requests.put(url, allow_redirects=False, timeout=10)
    if r.status_code == 307:
        redirect = fix_redirect(r.headers["Location"])
        r = requests.put(redirect, data=data.encode("utf-8"),
                         headers={"Content-Type": "application/octet-stream"}, timeout=10)
        return r.status_code in [200, 201]
    return False


def hdfs_mkdir(path):
    url = f"{WEBHDFS_BASE}{path}?op=MKDIRS"
    r = requests.put(url, timeout=5)
    return r.status_code == 200


def hdfs_available():
    try:
        r = requests.get(f"{WEBHDFS_BASE}/?op=GETHOMEDIRECTORY", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def load_zones():
    """Load zone metadata from HDFS graph_nodes.csv."""
    try:
        r = hdfs_read(NODES_PATH)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            print(f"  Loaded {len(df)} zones from HDFS")
            return df
    except Exception as e:
        print(f"  Warning: Could not load zones from HDFS: {e}")

    print("  Using generated zone data (263 zones)")
    zones = []
    boroughs = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"]
    borough_counts = [69, 70, 62, 43, 18, 1]
    zid = 1
    for borough, count in zip(boroughs, borough_counts):
        for i in range(count):
            zones.append({
                "zone_id": zid,
                "name": f"Zone {zid}",
                "borough": borough,
                "capacity": int(random.gauss(500, 200)),
                "lat": 40.7580 + random.gauss(0, 0.03),
                "lon": -73.9855 + random.gauss(0, 0.04),
            })
            zid += 1
    return pd.DataFrame(zones)


def rush_hour_factor(hour):
    """Traffic multiplier by hour of day."""
    if 7 <= hour <= 9:
        return 1.8 + 0.3 * math.sin((hour - 7) * math.pi / 2)
    elif 17 <= hour <= 19:
        return 1.6 + 0.2 * math.sin((hour - 17) * math.pi / 2)
    elif 0 <= hour <= 5:
        return 0.15 + 0.05 * hour
    elif 22 <= hour <= 23:
        return 0.5
    elif 10 <= hour <= 16:
        return 0.9 + 0.1 * math.sin((hour - 10) * math.pi / 6)
    else:
        return 0.7


class WeatherState:
    """Simulates slowly-changing weather conditions."""

    def __init__(self):
        self.temperature = 22.0
        self.precipitation = 0.0
        self.humidity = 55.0
        self.wind_speed = 12.0
        self.visibility = 10000.0
        self.is_rain = False
        self.is_snow = False

    def advance(self, hours_delta):
        self.temperature += random.gauss(0, 0.5 * hours_delta)
        self.temperature = max(-10, min(40, self.temperature))

        if random.random() < 0.02 * hours_delta:
            if self.is_rain:
                self.is_rain = False
                self.precipitation = 0
            else:
                self.is_rain = True
                self.precipitation = random.uniform(0.5, 8.0)

        if self.temperature < 0 and self.is_rain:
            self.is_snow = True
        else:
            self.is_snow = False

        self.humidity = max(20, min(100, self.humidity + random.gauss(0, 2)))
        self.wind_speed = max(0, min(80, self.wind_speed + random.gauss(0, 1.5)))
        self.visibility = 10000 if not self.is_rain else random.uniform(2000, 8000)

    def to_dict(self):
        return {
            "temperature_c": round(self.temperature, 1),
            "precipitation_mm": round(self.precipitation, 1),
            "humidity_pct": round(self.humidity, 1),
            "wind_speed_kmh": round(self.wind_speed, 1),
            "visibility_m": round(self.visibility, 0),
            "is_rain": int(self.is_rain),
            "is_snow": int(self.is_snow),
            "is_heavy_rain": int(self.precipitation > 5),
            "is_extreme_cold": int(self.temperature < -5),
            "is_extreme_heat": int(self.temperature > 35),
            "is_low_visibility": int(self.visibility < 3000),
            "is_high_wind": int(self.wind_speed > 50),
        }


class DisruptionManager:
    """Manages active disruption events."""

    def __init__(self):
        self.active = []

    def maybe_start(self, sim_hour):
        if random.random() < 0.08:
            line = random.choice(LINE_NAMES)
            boroughs = SUBWAY_LINES[line]
            severity = random.uniform(0.5, 1.0)
            duration = random.randint(3, 12)
            self.active.append({
                "line": line,
                "boroughs": boroughs,
                "severity": severity,
                "remaining_hours": duration,
                "started_hour": sim_hour,
            })
            return line, severity
        return None, 0

    def advance(self, hours_delta):
        for d in self.active:
            d["remaining_hours"] -= hours_delta
            d["severity"] *= 0.95
        self.active = [d for d in self.active if d["remaining_hours"] > 0]

    def disruption_for_zone(self, borough):
        for d in self.active:
            if borough in d["boroughs"]:
                return d
        return None


def generate_batch(zones_df, sim_time, weather, disruption_mgr, batch_id):
    """Generate one batch of transport data for all zones."""
    hour = sim_time.hour
    is_weekend = sim_time.weekday() >= 5
    rush = rush_hour_factor(hour)
    weekend_factor = 0.6 if is_weekend else 1.0
    weather_dict = weather.to_dict()
    rain_penalty = 0.7 if weather.is_rain else 1.0
    ts = sim_time.strftime("%Y-%m-%dT%H:%M:%S")

    records = []
    for _, zone in zones_df.iterrows():
        zid = zone["zone_id"]
        borough = zone.get("borough", "Manhattan")
        capacity = max(zone.get("capacity", 500), 50)
        borough_mult = BOROUGH_MULTIPLIERS.get(borough, 1.0)

        base_traffic = capacity * borough_mult * rush * weekend_factor * rain_penalty

        disruption = disruption_mgr.disruption_for_zone(borough)
        disruption_factor = 1.0
        affected_line = "-"
        if disruption:
            disruption_factor = 1.0 - disruption["severity"] * 0.6
            affected_line = disruption["line"]

        taxi_pickups = max(0, int(base_traffic * random.gauss(1.0, 0.15)))
        taxi_dropoffs = max(0, int(taxi_pickups * random.gauss(0.95, 0.1)))
        subway_ridership = max(0, int(base_traffic * 2.5 * disruption_factor * random.gauss(1.0, 0.2)))
        bike_starts = max(0, int(base_traffic * 0.3 * rain_penalty * random.gauss(1.0, 0.25)))
        bike_ends = max(0, int(bike_starts * random.gauss(0.98, 0.1)))

        complaint_count = max(0, int(random.gauss(2, 1.5)))
        event_count = 1 if random.random() < 0.05 else 0

        if disruption:
            complaint_count += int(disruption["severity"] * random.gauss(8, 3))
            complaint_count = max(0, complaint_count)

        record = {
            "zone_id": int(zid),
            "timestamp": ts,
            "zone_name": zone.get("name", f"Zone {zid}"),
            "zone_borough": borough,
            "taxi_pickups": taxi_pickups,
            "taxi_dropoffs": taxi_dropoffs,
            "subway_ridership": subway_ridership,
            "bike_starts": bike_starts,
            "bike_ends": bike_ends,
            "complaint_count": complaint_count,
            "event_count": event_count,
            "affected_line": affected_line,
            "disruption_severity": disruption["severity"] if disruption else 0.0,
            "hour_of_day": hour,
            "day_of_week": sim_time.weekday(),
            "is_weekend": int(is_weekend),
            "is_rush_hour": int(7 <= hour <= 9 or 17 <= hour <= 19),
        }
        record.update(weather_dict)
        records.append(record)

    return records


def run_simulator(speed=5, time_scale=120, start_hour=6):
    """
    Main simulator loop.

    Args:
        speed: seconds between batches (real time)
        time_scale: simulated seconds per real second (120 = 1 real min = 2 sim hours)
        start_hour: starting hour of simulated day
    """
    print("\n" + "=" * 60)
    print("  NeuroTraffic — Real-Time Data Simulator")
    print("=" * 60)

    if not hdfs_available():
        print("\n  ERROR: HDFS not reachable at", WEBHDFS_BASE)
        print("  Make sure Docker containers are running: docker compose up -d")
        sys.exit(1)

    print(f"\n  HDFS: Connected ({WEBHDFS_BASE})")
    print(f"  Batch interval: {speed}s real time")
    print(f"  Time scale: {time_scale}x (1 real second = {time_scale} sim seconds)")
    print(f"  Start hour: {start_hour}:00")

    hdfs_mkdir("/data/streaming")
    hdfs_mkdir("/data/streaming/input")
    hdfs_mkdir("/data/streaming/output")

    print("\n  Loading zone metadata...")
    zones_df = load_zones()

    weather = WeatherState()
    disruption_mgr = DisruptionManager()
    sim_time = datetime(2023, 6, 15, start_hour, 0, 0)
    batch_id = 0

    print(f"\n  Starting stream — {len(zones_df)} zones")
    print(f"  Simulated time starts at: {sim_time.strftime('%Y-%m-%d %H:%M')}")
    print("  Press Ctrl+C to stop\n")
    print(f"  {'Batch':>6} | {'Sim Time':>16} | {'Zones':>5} | {'Disruptions':>12} | {'Weather':>10} | Status")
    print("  " + "-" * 80)

    try:
        while True:
            batch_id += 1
            sim_hours_delta = (speed * time_scale) / 3600.0
            sim_time += timedelta(hours=sim_hours_delta)

            if sim_time.hour == 0 and sim_time.minute < 10:
                sim_time = sim_time.replace(day=sim_time.day % 28 + 1)

            weather.advance(sim_hours_delta)
            disruption_mgr.advance(sim_hours_delta)

            new_line, severity = disruption_mgr.maybe_start(sim_time.hour)

            records = generate_batch(zones_df, sim_time, weather, disruption_mgr, batch_id)

            batch_json = "\n".join(json.dumps(r) for r in records)
            fname = f"batch_{batch_id:06d}.json"
            hdfs_path = f"{INPUT_DIR}/{fname}"

            ok = hdfs_write(hdfs_path, batch_json)
            ok_csv = write_latest_csv(records)

            n_disrupted = sum(1 for r in records if r["affected_line"] != "-")
            weather_str = f"{weather.temperature:.0f}C"
            if weather.is_rain:
                weather_str += " Rain"
            if weather.is_snow:
                weather_str += " Snow"

            status = "OK"
            if new_line:
                status = f"NEW DISRUPTION: {new_line} (severity {severity:.2f})"
            elif n_disrupted > 0:
                lines = set(d["line"] for d in disruption_mgr.active)
                status = f"Active: {', '.join(lines)}"

            write_status = ("HDFS+CSV" if ok_csv else "HDFS") if ok else "FAIL"
            print(f"  {batch_id:6d} | {sim_time.strftime('%Y-%m-%d %H:%M')} | {len(records):5d} | "
                  f"{n_disrupted:12d} | {weather_str:>10} | [{write_status}] {status}")

            time.sleep(speed)

    except KeyboardInterrupt:
        print(f"\n\n  Stopped after {batch_id} batches.")
        print(f"  Final simulated time: {sim_time.strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroTraffic Real-Time Data Simulator")
    parser.add_argument("--speed", type=int, default=5,
                        help="Seconds between batches (default: 5)")
    parser.add_argument("--time-scale", type=int, default=120,
                        help="Simulated seconds per real second (default: 120)")
    parser.add_argument("--start-hour", type=int, default=6,
                        help="Starting hour of simulated day (default: 6)")
    args = parser.parse_args()

    run_simulator(args.speed, args.time_scale, args.start_hour)
