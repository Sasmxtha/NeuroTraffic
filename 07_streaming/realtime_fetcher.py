"""
NeuroTraffic - Live NYC Data Fetcher
====================================
Builds one record per taxi zone every --interval seconds and writes it to HDFS
(/data/streaming/input) for the Spark Structured Streaming pipeline. Also writes
/data/streaming/output/source_status.json so the dashboard can show how fresh
each source is. All times are New York time.

  Source            Provider                 Kind       How it is used
  ----------------  -----------------------  ---------  ------------------------------------------
  Weather           Open-Meteo               live       current conditions (model updates ~15 min)
  Subway alerts     MTA GTFS-RT alerts       live       unplanned, currently-active alerts only,
                                                        mapped to zones that have a station on the line
  Subway trains     MTA GTFS-RT train feeds  live       trains currently reporting a position, per line
  Road traffic      NYC DOT traffic speeds   live*      average speed of DOT road segments per zone
                                                        (*the city feed sometimes pauses; marked stale)
  Citi Bike        GBFS station_status      live       bike_starts = dock changes over the last 15 min
                                                        (a lower bound on real trips)
  311 complaints    NYC Open Data            delayed    published ~daily; context only, not used in scoring
  Events            NYC permitted events     daily      active permits per borough (context only)
  Taxi pickups      historical profile       estimated  zone x hour x day-type average, weather-adjusted
  Subway ridership  historical profile       estimated  zone x hour x day-type average

Prerequisites (once):
    python 07_streaming/build_live_lookups.py --upload
    docker exec spark-master /spark/bin/spark-shell -i /project/07_streaming/build_live_profiles.scala

Usage:
    python 07_streaming/realtime_fetcher.py
    python 07_streaming/realtime_fetcher.py --interval 10
"""

import argparse
import io
import json
import os
import re
import sys
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from google.transit import gtfs_realtime_pb2
from shapely.geometry import Point
from shapely.strtree import STRtree

from build_live_lookups import TO_STATEPLANE, load_zone_polygons

NYC = ZoneInfo("America/New_York")
HERE = os.path.dirname(os.path.abspath(__file__))
LOOKUP_DIR = os.path.join(HERE, "lookups")

WEBHDFS_HOST = "localhost"
WEBHDFS_BASE = f"http://{WEBHDFS_HOST}:9870/webhdfs/v1"
DATANODE_PORT_MAP = {"datanode1": 9864, "datanode2": 9865, "datanode3": 9866}

INPUT_DIR = "/data/streaming/input"
STATUS_PATH = "/data/streaming/output/source_status.json"
PROFILE_HDFS = "/data/neurotraffic/lookups/zone_hour_profile.csv"
PROFILE_LOCAL = os.path.join(LOOKUP_DIR, "zone_hour_profile.csv")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
MTA_ALERTS_URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/camsys%2Fsubway-alerts.json"
MTA_TRAINS_URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2F{}"
MTA_TRAIN_FEEDS = ["gtfs", "gtfs-ace", "gtfs-bdfm", "gtfs-g", "gtfs-jz", "gtfs-l", "gtfs-nqrw", "gtfs-si"]
DOT_SPEEDS_URL = "https://data.cityofnewyork.us/resource/i4gi-tjb9.json"
CITIBIKE_STATUS_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"
CITIBIKE_INFO_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
NYC_311_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
NYC_EVENTS_URL = "https://data.cityofnewyork.us/resource/tvpp-9vvx.json"

# Same complaint types clean_311.scala kept for the historical data
TRANSPORT_311_TYPES = [
    "Street Condition", "Traffic Signal Condition", "Street Light Condition",
    "Blocked Driveway", "Illegal Parking", "Noise - Street/Sidewalk",
    "Water Main Break", "Sewer", "Bus Stop Shelter Complaint",
    "Traffic", "Taxi Complaint", "Bike/Roller/Skate Condition",
]

MTA_IGNORED_TYPES = {"Station Notice", "Extra Service", "Special Schedule", "Boarding Change"}
ROUTE_ALIASES = {"GS": "S", "FS": "S", "H": "S", "SI": "SIR", "SS": "SIR"}
BIKE_WINDOW = timedelta(minutes=15)
DOT_STALE_AFTER = timedelta(minutes=30)


def nyc_now():
    return datetime.now(NYC)


def age_minutes(observed):
    return None if observed is None else round((nyc_now() - observed).total_seconds() / 60, 1)


# ── HDFS ────────────────────────────────────────────────────────────

def fix_redirect(url):
    for node, port in DATANODE_PORT_MAP.items():
        url = re.sub(rf"//{node}:\d+", f"//{WEBHDFS_HOST}:{port}", url)
    return url


def hdfs_write(path, text):
    r = requests.put(f"{WEBHDFS_BASE}{path}?op=CREATE&overwrite=true", allow_redirects=False, timeout=10)
    if r.status_code != 307:
        return False
    r = requests.put(fix_redirect(r.headers["Location"]), data=text.encode("utf-8"),
                     headers={"Content-Type": "application/octet-stream"}, timeout=10)
    return r.status_code in (200, 201)


def hdfs_read(path):
    r = requests.get(f"{WEBHDFS_BASE}{path}?op=OPEN", allow_redirects=False, timeout=10)
    if r.status_code == 307:
        r = requests.get(fix_redirect(r.headers["Location"]), timeout=30)
    return r.text if r.status_code == 200 else None


def hdfs_available():
    try:
        return requests.get(f"{WEBHDFS_BASE}/?op=GETHOMEDIRECTORY", timeout=2).status_code == 200
    except Exception:
        return False


# ── Static inputs ───────────────────────────────────────────────────

def load_zones():
    path = os.path.join(LOOKUP_DIR, "zone_centroids.csv")
    if not os.path.exists(path):
        sys.exit("Missing 07_streaming/lookups/zone_centroids.csv - run: python 07_streaming/build_live_lookups.py")
    return pd.read_csv(path)


def load_line_zones():
    path = os.path.join(LOOKUP_DIR, "subway_station_zones.csv")
    if not os.path.exists(path):
        sys.exit("Missing 07_streaming/lookups/subway_station_zones.csv - run: python 07_streaming/build_live_lookups.py")
    line_zones = {}
    for _, s in pd.read_csv(path).iterrows():
        for line in str(s["lines"]).split("|"):
            line_zones.setdefault(line, set()).add(int(s["zone_id"]))
    return line_zones


def load_profile():
    text = hdfs_read(PROFILE_HDFS)
    if text:
        with open(PROFILE_LOCAL, "w", encoding="utf-8") as f:
            f.write(text)
        df = pd.read_csv(io.StringIO(text))
    elif os.path.exists(PROFILE_LOCAL):
        df = pd.read_csv(PROFILE_LOCAL)
    else:
        sys.exit("Missing historical profile - run: docker exec spark-master /spark/bin/spark-shell "
                 "-i /project/07_streaming/build_live_profiles.scala")
    return {(int(r.zone_id), int(r.hour_of_day), int(r.is_weekend)): r for r in df.itertuples()}


# ── Live sources ────────────────────────────────────────────────────

class LiveWeather:
    REFRESH = timedelta(minutes=5)

    def __init__(self):
        self.data = None
        self.observed_at = None
        self.fetched_at = None
        self.error = None

    def fetch(self):
        if self.fetched_at and nyc_now() - self.fetched_at < self.REFRESH:
            return
        try:
            params = {
                "latitude": 40.7831, "longitude": -73.9712, "timezone": "America/New_York",
                "current": "temperature_2m,relative_humidity_2m,precipitation,rain,snowfall,wind_speed_10m,visibility",
            }
            j = requests.get(OPEN_METEO_URL, params=params, timeout=10).json()["current"]
            temp, precip = j["temperature_2m"], j["precipitation"]
            wind, vis = j["wind_speed_10m"], j.get("visibility") or 10000.0
            self.data = {
                "temperature_c": round(temp, 1),
                "precipitation_mm": round(precip, 1),
                "humidity_pct": float(j["relative_humidity_2m"]),
                "wind_speed_kmh": round(wind, 1),
                "visibility_m": float(vis),
                "is_rain": int(j.get("rain", 0) > 0 or precip > 0),
                "is_snow": int(j.get("snowfall", 0) > 0),
                "is_heavy_rain": int(precip > 5),
                "is_extreme_cold": int(temp < -5),
                "is_extreme_heat": int(temp > 35),
                "is_low_visibility": int(vis < 3000),
                "is_high_wind": int(wind > 50),
            }
            self.observed_at = datetime.fromisoformat(j["time"]).replace(tzinfo=NYC)
            self.error = None
        except Exception as e:
            self.error = str(e)[:120]
        self.fetched_at = nyc_now()

    def status(self):
        d = self.data or {}
        return {"name": "Weather", "provider": "Open-Meteo", "kind": "live",
                "observed_at": self.observed_at, "error": self.error,
                "detail": f"{d.get('temperature_c', '?')}°C, {d.get('precipitation_mm', '?')} mm precip, "
                          f"{d.get('wind_speed_kmh', '?')} km/h wind"}


def route_to_line(route_id):
    line = ROUTE_ALIASES.get(route_id, route_id)
    if len(line) == 2 and line.endswith("X"):
        line = line[0]
    return line


def alert_severity(alert_type):
    t = alert_type.lower()
    if "suspend" in t or "no trains" in t or "severe" in t:
        return 0.8
    if "delay" in t:
        return 0.6
    return 0.4


class LiveMTA:
    REFRESH = timedelta(seconds=60)

    def __init__(self, line_zones):
        self.line_zones = line_zones
        self.alerts = []
        self.zone_lines = {}
        self.n_total = 0
        self.n_planned = 0
        self.observed_at = None
        self.fetched_at = None
        self.error = None

    def fetch(self):
        if self.fetched_at and nyc_now() - self.fetched_at < self.REFRESH:
            return
        try:
            j = requests.get(MTA_ALERTS_URL, timeout=15).json()
            now_ts = time.time()
            alerts, n_planned = [], 0
            entities = j.get("entity", [])
            for ent in entities:
                a = ent.get("alert", {})
                atype = a.get("transit_realtime.mercury_alert", {}).get("alert_type", "")
                if atype.startswith("Planned"):
                    n_planned += 1
                    continue
                if atype in MTA_IGNORED_TYPES:
                    continue
                periods = a.get("active_period", [])
                if periods and not any(p.get("start", 0) <= now_ts and (not p.get("end") or now_ts <= p["end"])
                                       for p in periods):
                    continue
                lines = {route_to_line(ie["route_id"]) for ie in a.get("informed_entity", []) if ie.get("route_id")}
                lines &= self.line_zones.keys()
                if not lines:
                    continue
                header = next((t["text"] for t in a.get("header_text", {}).get("translation", [])
                               if t.get("language") == "en"), "")
                alerts.append({"lines": sorted(lines), "type": atype,
                               "severity": alert_severity(atype), "header": header[:160]})

            zone_lines = {}
            for al in alerts:
                for line in al["lines"]:
                    for z in self.line_zones[line]:
                        cur = zone_lines.setdefault(z, {})
                        cur[line] = max(cur.get(line, 0), al["severity"])

            self.alerts, self.zone_lines = alerts, zone_lines
            self.n_total, self.n_planned = len(entities), n_planned
            self.observed_at = datetime.fromtimestamp(int(j["header"]["timestamp"]), NYC)
            self.error = None
        except Exception as e:
            self.error = str(e)[:120]
        self.fetched_at = nyc_now()

    def for_zone(self, zone_id):
        lines = self.zone_lines.get(zone_id)
        if not lines:
            return "-", 0.0
        return ",".join(sorted(lines)), max(lines.values())

    def status(self):
        lines = sorted({l for a in self.alerts for l in a["lines"]})
        detail = (f"{len(self.alerts)} active unplanned alert(s) on {','.join(lines)}" if self.alerts
                  else "no active unplanned alerts")
        detail += f"; {self.n_planned} planned-work notices ignored"
        return {"name": "Subway alerts", "provider": "MTA GTFS-RT", "kind": "live",
                "observed_at": self.observed_at, "error": self.error, "detail": detail}


class ZoneLocator:
    """Point-in-polygon lookup against the TLC taxi-zone shapefile."""

    def __init__(self):
        polygons = load_zone_polygons()
        self.ids = list(polygons.keys())
        self.geoms = [polygons[i] for i in self.ids]
        self.tree = STRtree(self.geoms)

    def zone_of(self, lat, lon):
        x, y = TO_STATEPLANE.transform(lon, lat)
        pt = Point(x, y)
        for idx in self.tree.query(pt):
            if self.geoms[idx].contains(pt):
                return self.ids[idx]
        return None


class LiveSubwayTrains:
    REFRESH = timedelta(seconds=30)

    def __init__(self):
        self.by_line = {}
        self.observed_at = None
        self.fetched_at = None
        self.error = None

    def fetch(self):
        if self.fetched_at and nyc_now() - self.fetched_at < self.REFRESH:
            return
        by_line, oldest, failed = Counter(), None, []
        for feed in MTA_TRAIN_FEEDS:
            try:
                msg = gtfs_realtime_pb2.FeedMessage()
                msg.ParseFromString(requests.get(MTA_TRAINS_URL.format(feed), timeout=15).content)
                for ent in msg.entity:
                    if ent.HasField("vehicle") and ent.vehicle.trip.route_id:
                        by_line[route_to_line(ent.vehicle.trip.route_id)] += 1
                ts = datetime.fromtimestamp(msg.header.timestamp, NYC)
                oldest = ts if oldest is None or ts < oldest else oldest
            except Exception:
                failed.append(feed)
        if by_line:
            self.by_line, self.observed_at = dict(by_line), oldest
        self.error = f"feeds failed: {', '.join(failed)}" if failed else None
        self.fetched_at = nyc_now()

    def status(self):
        total = sum(self.by_line.values())
        return {"name": "Subway trains", "provider": "MTA GTFS-RT train feeds", "kind": "live",
                "observed_at": self.observed_at, "error": self.error,
                "detail": f"{total} trains reporting positions on {len(self.by_line)} lines"}


class LiveTrafficSpeeds:
    REFRESH = timedelta(minutes=2)
    SNAPSHOT = timedelta(minutes=15)

    def __init__(self, locator):
        self.locator = locator
        self.link_zone = {}
        self.links = []
        self.zone_speed = {}
        self.observed_at = None
        self.fetched_at = None
        self.error = None

    @property
    def stale(self):
        return self.observed_at is None or nyc_now() - self.observed_at > DOT_STALE_AFTER

    def _zone_for_link(self, link_id, link_points):
        if link_id not in self.link_zone:
            pts = [p.split(",") for p in str(link_points).split() if "," in p]
            zone = None
            if pts:
                try:
                    lat, lon = map(float, pts[len(pts) // 2][:2])
                    zone = self.locator.zone_of(lat, lon)
                except ValueError:
                    pass
            self.link_zone[link_id] = zone
        return self.link_zone[link_id]

    def fetch(self):
        if self.fetched_at and nyc_now() - self.fetched_at < self.REFRESH:
            return
        try:
            # Aggregates such as max() time out on this multi-million-row table; newest rows are fast.
            rows = requests.get(DOT_SPEEDS_URL, params={
                "$select": "link_id, link_name, borough, speed, link_points, data_as_of",
                "$order": "data_as_of DESC",
                "$limit": 1500,
            }, timeout=60).json()
            if not rows:
                raise ValueError("empty response from DOT API")
            def as_of(r):
                return datetime.fromisoformat(r["data_as_of"][:19]).replace(tzinfo=NYC)
            newest = as_of(rows[0])
            latest = {}
            for r in rows:
                if r["link_id"] not in latest and newest - as_of(r) <= self.SNAPSHOT:
                    latest[r["link_id"]] = r
            links, per_zone = [], {}
            for lid, r in latest.items():
                speed = float(r.get("speed") or 0)
                if speed <= 0:
                    continue
                zone = self._zone_for_link(lid, r.get("link_points", ""))
                links.append({"name": r.get("link_name", ""), "borough": r.get("borough", ""),
                              "speed_mph": round(speed, 1), "zone_id": zone})
                if zone is not None:
                    per_zone.setdefault(zone, []).append(speed)
            self.links = sorted(links, key=lambda l: l["speed_mph"])
            self.zone_speed = {z: round(sum(v) / len(v), 1) for z, v in per_zone.items()}
            self.observed_at, self.error = newest, None
        except Exception as e:
            self.error = str(e)[:120]
        self.fetched_at = nyc_now()

    def speed_for_zone(self, zone_id):
        return None if self.stale else self.zone_speed.get(zone_id)

    def summary(self):
        speeds = [l["speed_mph"] for l in self.links]
        return {"stale": self.stale, "segments": len(speeds),
                "avg_speed_mph": round(sum(speeds) / len(speeds), 1) if speeds else None,
                "slowest": self.links[:8]}

    def status(self):
        s = self.summary()
        detail = (f"{s['segments']} road segments, citywide avg {s['avg_speed_mph']} mph"
                  if s["segments"] else "no segments reporting")
        if self.stale:
            detail = "city feed paused - last data shown is not current; " + detail
        return {"name": "Road traffic", "provider": "NYC DOT traffic speeds",
                "kind": "stale" if self.stale else "live",
                "observed_at": self.observed_at, "error": self.error, "detail": detail}


class LiveCitiBike:
    REFRESH = timedelta(seconds=60)

    def __init__(self, locator):
        self.station_zone = {}
        self.prev = None
        self.prev_time = None
        self.windows = deque()
        self.observed_at = None
        self.fetched_at = None
        self.error = None
        self._map_stations(locator)

    def _map_stations(self, locator):
        try:
            stations = requests.get(CITIBIKE_INFO_URL, timeout=20).json()["data"]["stations"]
        except Exception as e:
            self.error = f"station_information: {e}"[:120]
            return
        for s in stations:
            zone = locator.zone_of(s["lat"], s["lon"])
            if zone is not None:
                self.station_zone[s["station_id"]] = zone
        print(f"    Citi Bike: {len(self.station_zone)} stations mapped to taxi zones")

    def fetch(self):
        if self.fetched_at and nyc_now() - self.fetched_at < self.REFRESH:
            return
        try:
            j = requests.get(CITIBIKE_STATUS_URL, timeout=15).json()
            now = nyc_now()
            snap = {s["station_id"]: s.get("num_bikes_available", 0) for s in j["data"]["stations"]}
            if self.prev is not None:
                moves = Counter()
                for sid, bikes in snap.items():
                    zone = self.station_zone.get(sid)
                    if zone is not None and sid in self.prev:
                        moves[zone] += abs(bikes - self.prev[sid])
                self.windows.append((self.prev_time, now, moves))
                while self.windows and now - self.windows[0][0] > BIKE_WINDOW:
                    self.windows.popleft()
            self.prev, self.prev_time = snap, now
            self.observed_at = datetime.fromtimestamp(j["last_updated"], NYC)
            self.error = None
        except Exception as e:
            self.error = str(e)[:120]
        self.fetched_at = nyc_now()

    def starts_per_hour(self):
        if not self.windows:
            return None
        hours = (self.windows[-1][1] - self.windows[0][0]).total_seconds() / 3600
        total = Counter()
        for _, _, moves in self.windows:
            total.update(moves)
        return {z: m / 2 / hours for z, m in total.items()}, hours * 60

    @property
    def zones_with_stations(self):
        return set(self.station_zone.values())

    def status(self):
        measured = self.starts_per_hour()
        detail = (f"bike_starts measured from dock changes over last {measured[1]:.0f} min (lower bound)"
                  if measured else "warming up - using historical profile until 2 snapshots exist")
        return {"name": "Citi Bike", "provider": "Citi Bike GBFS", "kind": "live",
                "observed_at": self.observed_at, "error": self.error, "detail": detail}


class Live311:
    REFRESH = timedelta(minutes=30)

    def __init__(self, hist_daily_by_borough):
        self.hist = hist_daily_by_borough
        self.factor = {}
        self.newest = None
        self.fetched_at = None
        self.error = None

    def fetch(self):
        if self.fetched_at and nyc_now() - self.fetched_at < self.REFRESH:
            return
        try:
            types = ",".join(f"'{t}'" for t in TRANSPORT_311_TYPES)
            type_filter = f"complaint_type in({types})"
            j = requests.get(NYC_311_URL, params={"$select": "max(created_date) as m", "$where": type_filter},
                             timeout=30).json()
            newest = datetime.fromisoformat(j[0]["m"][:19]).replace(tzinfo=NYC)
            since = (newest - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
            rows = requests.get(NYC_311_URL, params={
                "$select": "borough, count(*) as cnt",
                "$where": f"{type_filter} AND created_date > '{since}'",
                "$group": "borough",
            }, timeout=30).json()
            if not rows:
                raise ValueError("empty response from 311 API")
            day_type = int(newest.weekday() >= 5)
            factor = {}
            for r in rows:
                borough = r.get("borough", "").title()
                hist = self.hist.get((borough, day_type))
                if hist:
                    factor[borough] = min(2.0, max(0.5, int(r["cnt"]) / hist))
            self.factor, self.newest, self.error = factor, newest, None
            self.fetched_at = nyc_now()
        except Exception as e:
            self.error = str(e)[:120]
            self.fetched_at = nyc_now() - self.REFRESH + timedelta(minutes=2)

    def status(self):
        detail = ("published ~daily, context only; latest 24 h vs 2022-24 average: " +
                  ", ".join(f"{b} x{f:.2f}" for b, f in sorted(self.factor.items()))) if self.factor \
            else "no recent data"
        return {"name": "311 complaints", "provider": "NYC Open Data", "kind": "delayed",
                "observed_at": self.newest, "error": self.error, "detail": detail}


class LiveEvents:
    REFRESH = timedelta(minutes=30)

    def __init__(self):
        self.by_borough = {}
        self.fetched_at = None
        self.error = None

    def fetch(self):
        if self.fetched_at and nyc_now() - self.fetched_at < self.REFRESH:
            return
        try:
            now = nyc_now().strftime("%Y-%m-%dT%H:%M:%S")
            rows = requests.get(NYC_EVENTS_URL, params={
                "$select": "event_borough, count(*) as cnt",
                "$where": f"start_date_time <= '{now}' AND end_date_time >= '{now}'",
                "$group": "event_borough",
            }, timeout=30).json()
            self.by_borough = {r.get("event_borough", "").title(): int(r["cnt"]) for r in rows}
            self.error = None
        except Exception as e:
            self.error = str(e)[:120]
        self.fetched_at = nyc_now()

    def status(self):
        total = sum(self.by_borough.values())
        return {"name": "Events", "provider": "NYC permitted events", "kind": "daily",
                "observed_at": self.fetched_at, "error": self.error,
                "detail": f"{total} permitted events active now (borough level, not used in scoring)"}


# ── Batch assembly ──────────────────────────────────────────────────

def weather_factor(w):
    f = 1.3 if w["is_heavy_rain"] else 1.15 if w["is_rain"] else 1.0
    if w["is_snow"]:
        f *= 1.4
    if w["is_extreme_cold"]:
        f *= 1.1
    return f


def hist_daily_311(profile, zones):
    """Historical transport-311 count per borough per day (profile is borough-level, repeated per zone)."""
    out = {}
    for borough, group in zones.groupby("borough"):
        for day_type in (0, 1):
            per_zone = [sum(getattr(profile.get((z, h, day_type)), "complaint_count", 0) or 0 for h in range(24))
                        for z in group["zone_id"]]
            if per_zone:
                out[(borough, day_type)] = sum(per_zone) / len(per_zone)
    return out


def build_batch(zones, profile, weather, mta, bikes, events, traffic):
    now = nyc_now()
    hour, wknd = now.hour, int(now.weekday() >= 5)
    ts = now.strftime("%Y-%m-%dT%H:%M:%S")
    wf = weather_factor(weather.data)
    measured = bikes.starts_per_hour()
    bike_rates = measured[0] if measured else None
    bike_zones = bikes.zones_with_stations

    records = []
    for z in zones.itertuples():
        p = profile.get((z.zone_id, hour, wknd))
        get = (lambda k: float(getattr(p, k, 0) or 0)) if p is not None else (lambda k: 0.0)
        line, severity = mta.for_zone(z.zone_id)

        if bike_rates is not None and z.zone_id in bike_zones:
            bike_starts = round(bike_rates.get(z.zone_id, 0.0))
        else:
            bike_starts = round(get("bike_starts") * (0.6 if weather.data["is_rain"] else 1.0))

        rec = {
            "zone_id": int(z.zone_id),
            "timestamp": ts,
            "zone_name": z.name,
            "zone_borough": z.borough,
            "taxi_pickups": round(get("taxi_pickups") * wf),
            "taxi_dropoffs": round(get("taxi_dropoffs") * wf),
            "subway_ridership": round(get("subway_ridership")),
            "bike_starts": bike_starts,
            "bike_ends": bike_starts,
            "complaint_count": round(get("complaint_count")),
            "event_count": events.by_borough.get(z.borough, 0),
            "affected_line": line,
            "disruption_severity": severity,
            "traffic_speed_mph": traffic.speed_for_zone(z.zone_id),
            "hour_of_day": hour,
            "day_of_week": now.weekday(),
            "is_weekend": wknd,
            "is_rush_hour": int(7 <= hour <= 9 or 17 <= hour <= 19),
        }
        rec.update(weather.data)
        records.append(rec)
    return records


def write_status(sources, weather, mta, trains, traffic):
    def ser(s):
        s = dict(s)
        obs = s.get("observed_at")
        s["age_min"] = age_minutes(obs)
        s["observed_at"] = obs.isoformat() if obs else None
        return s

    status = {
        "generated_at": nyc_now().isoformat(),
        "timezone": "America/New_York",
        "sources": [ser(s.status()) for s in sources] + [
            {"name": "Taxi pickups", "provider": "historical profile", "kind": "estimated",
             "observed_at": None, "age_min": None, "error": None,
             "detail": "no public real-time feed; zone x hour x day-type average from combined_v3, weather-adjusted"},
            {"name": "Subway ridership", "provider": "historical profile", "kind": "estimated",
             "observed_at": None, "age_min": None, "error": None,
             "detail": "MTA publishes ridership with a delay; zone x hour x day-type average from combined_v3"},
        ],
        "mta_alerts": mta.alerts[:10],
        "subway_lines": [
            {"line": line, "trains": trains.by_line.get(line, 0),
             "alert_severity": max((a["severity"] for a in mta.alerts if line in a["lines"]), default=0.0)}
            for line in sorted(set(trains.by_line) | {l for a in mta.alerts for l in a["lines"]})
        ],
        "traffic": traffic.summary(),
        "weather": weather.data,
    }
    return hdfs_write(STATUS_PATH, json.dumps(status))


def run(interval):
    print("\n" + "=" * 64)
    print("  NeuroTraffic - Live NYC Data Fetcher (all times America/New_York)")
    print("=" * 64)
    for _ in range(30):
        if hdfs_available():
            break
        time.sleep(2)
    else:
        sys.exit(f"HDFS not reachable at {WEBHDFS_BASE} - run: docker compose up -d")

    for d in ("/data/streaming", INPUT_DIR, "/data/streaming/output"):
        requests.put(f"{WEBHDFS_BASE}{d}?op=MKDIRS", timeout=5)

    zones = load_zones()
    profile = load_profile()
    line_zones = load_line_zones()
    print(f"  {len(zones)} zones | {len(profile)} profile rows | {len(line_zones)} subway lines mapped")

    locator = ZoneLocator()
    weather = LiveWeather()
    mta = LiveMTA(line_zones)
    trains = LiveSubwayTrains()
    traffic = LiveTrafficSpeeds(locator)
    bikes = LiveCitiBike(locator)
    c311 = Live311(hist_daily_311(profile, zones))
    events = LiveEvents()
    sources = [weather, mta, trains, traffic, bikes, c311, events]

    batch_id = 0
    print(f"\n  {'Batch':>6} | {'NYC time':>8} | {'Weather':>8} | {'Subway':>11} | {'Traffic':>13} | Alerts")
    print("  " + "-" * 72)
    pool = ThreadPoolExecutor(max_workers=len(sources))
    try:
        while True:
            list(pool.map(lambda s: s.fetch(), sources))
            if weather.data is None:
                print(f"  waiting for weather data ({weather.error})")
                time.sleep(interval)
                continue

            batch_id += 1
            records = build_batch(zones, profile, weather, mta, bikes, events, traffic)
            ok = hdfs_write(f"{INPUT_DIR}/live_{nyc_now():%Y%m%d_%H%M%S}_{batch_id:06d}.json",
                            "\n".join(json.dumps(r) for r in records))
            write_status(sources, weather, mta, trains, traffic)

            w = weather.data
            wstr = f"{w['temperature_c']}C" + (" rain" if w["is_rain"] else "")
            tstr = "stale" if traffic.stale else f"{traffic.summary()['avg_speed_mph']}mph"
            sub = (f"{len(mta.alerts)} alert(s): " + ",".join(sorted({l for a in mta.alerts for l in a['lines']}))
                   if mta.alerts else "no unplanned alerts")
            print(f"  {batch_id:6d} | {nyc_now():%H:%M:%S} | {wstr:>8} | {sum(trains.by_line.values()):4d} trains | "
                  f"road {tstr:>8} | {sub} [{'HDFS' if ok else 'WRITE FAILED'}]")
            for s in sources:
                if s.error:
                    print(f"         ! {type(s).__name__}: {s.error}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n  Stopped after {batch_id} batches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroTraffic live NYC data fetcher")
    parser.add_argument("--interval", type=int, default=10, help="seconds between batches (default 10)")
    run(parser.parse_args().interval)
