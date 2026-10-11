"""
Builds the static lookup tables the live fetcher needs, from data already in the repo:

  07_streaming/lookups/zone_centroids.csv        real centroid of every TLC taxi zone (from the shapefile)
  07_streaming/lookups/subway_station_zones.csv  every subway station complex, its lines, and the taxi zone it sits in

With --upload (HDFS must be running) it also:
  - copies both files to HDFS /data/neurotraffic/lookups/
  - replaces the placeholder lat/lon in HDFS /data/neurotraffic/dashboard/graph_nodes.csv with real centroids

Usage:
    python 07_streaming/build_live_lookups.py
    python 07_streaming/build_live_lookups.py --upload
"""

import argparse
import io
import os
import re
import sys

import pandas as pd
import requests
import shapefile
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAPEFILE = os.path.join(ROOT, "neurotraffic_data", "taxi", "zones", "taxi_zones", "taxi_zones", "taxi_zones.shp")
ZONE_LOOKUP = os.path.join(ROOT, "neurotraffic_data", "taxi", "zones", "taxi_zone_lookup.csv")
SUBWAY_CSV = os.path.join(ROOT, "neurotraffic_data", "subway", "subway_hourly_2024.csv")
OUT_DIR = os.path.join(ROOT, "07_streaming", "lookups")

WEBHDFS = "http://localhost:9870/webhdfs/v1"
DATANODE_PORT_MAP = {"datanode1": 9864, "datanode2": 9865, "datanode3": 9866}
HDFS_LOOKUP_DIR = "/data/neurotraffic/lookups"
HDFS_GRAPH_NODES = "/data/neurotraffic/dashboard/graph_nodes.csv"

# Shapefile is NY State Plane Long Island (US feet)
TO_WGS84 = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
TO_STATEPLANE = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)

LINE_TOKEN = re.compile(r"^[A-Z0-9]{1,3}$")

# LocationIDs that share a polygon with another ID in the TLC shapefile
SHARED_SHAPES = {57: 56, 104: 103, 105: 103}


def load_zone_polygons():
    sf = shapefile.Reader(SHAPEFILE)
    fields = [f[0] for f in sf.fields[1:]]
    loc_idx = fields.index("LocationID")
    parts = {}
    for sr in sf.shapeRecords():
        zid = int(sr.record[loc_idx])
        parts.setdefault(zid, []).append(shape(sr.shape.__geo_interface__))
    return {zid: unary_union(polys) for zid, polys in parts.items()}


def build_zone_centroids(polygons):
    lookup = pd.read_csv(ZONE_LOOKUP)
    rows = []
    for _, z in lookup.iterrows():
        zid = int(z["LocationID"])
        poly = polygons.get(zid) or polygons.get(SHARED_SHAPES.get(zid, -1))
        if poly is None:
            continue
        c = poly.centroid
        lon, lat = TO_WGS84.transform(c.x, c.y)
        rows.append({"zone_id": zid, "name": z["Zone"], "borough": z["Borough"],
                     "lat": round(lat, 6), "lon": round(lon, 6)})
    return pd.DataFrame(rows).sort_values("zone_id")


def parse_lines(complex_name):
    lines = set()
    for group in re.findall(r"\(([^)]*)\)", complex_name):
        for tok in group.split(","):
            tok = tok.strip()
            if LINE_TOKEN.match(tok):
                lines.add(tok)
    return sorted(lines)


def build_station_zones(polygons):
    df = pd.read_csv(SUBWAY_CSV, nrows=400_000,
                     usecols=["transit_mode", "station_complex_id", "station_complex", "latitude", "longitude"])
    df = df.drop_duplicates("station_complex_id")

    ids = list(polygons.keys())
    geoms = [polygons[i] for i in ids]
    tree = STRtree(geoms)

    rows = []
    for _, s in df.iterrows():
        lines = parse_lines(str(s["station_complex"]))
        if not lines:
            continue
        x, y = TO_STATEPLANE.transform(float(s["longitude"]), float(s["latitude"]))
        pt = Point(x, y)
        zone_id = None
        for idx in tree.query(pt):
            if geoms[idx].contains(pt):
                zone_id = ids[idx]
                break
        if zone_id is None:
            zone_id = ids[int(tree.nearest(pt))]
        rows.append({"complex_id": s["station_complex_id"], "name": s["station_complex"],
                     "lines": "|".join(lines), "lat": s["latitude"], "lon": s["longitude"],
                     "zone_id": zone_id})
    return pd.DataFrame(rows).sort_values("complex_id")


def _fix_redirect(url):
    for node, port in DATANODE_PORT_MAP.items():
        url = re.sub(rf"//{node}:\d+", f"//localhost:{port}", url)
    return url


def hdfs_put(path, text):
    r = requests.put(f"{WEBHDFS}{path}?op=CREATE&overwrite=true", allow_redirects=False, timeout=10)
    if r.status_code != 307:
        return False
    r = requests.put(_fix_redirect(r.headers["Location"]), data=text.encode("utf-8"),
                     headers={"Content-Type": "application/octet-stream"}, timeout=30)
    return r.status_code in (200, 201)


def hdfs_get(path):
    r = requests.get(f"{WEBHDFS}{path}?op=OPEN", allow_redirects=False, timeout=10)
    if r.status_code == 307:
        r = requests.get(_fix_redirect(r.headers["Location"]), timeout=30)
    return r.text if r.status_code == 200 else None


def with_real_coords(nodes, centroids):
    nodes = nodes.drop(columns=[c for c in ("lat", "lon") if c in nodes.columns])
    return nodes.merge(centroids[["zone_id", "lat", "lon"]], on="zone_id", how="left")


def upload(centroids, stations):
    try:
        requests.get(f"{WEBHDFS}/?op=GETHOMEDIRECTORY", timeout=3).raise_for_status()
    except Exception:
        print("HDFS not reachable - start the cluster first (docker compose up -d)")
        sys.exit(1)

    requests.put(f"{WEBHDFS}{HDFS_LOOKUP_DIR}?op=MKDIRS", timeout=5)
    for name, df in [("zone_centroids.csv", centroids), ("subway_station_zones.csv", stations)]:
        ok = hdfs_put(f"{HDFS_LOOKUP_DIR}/{name}", df.to_csv(index=False))
        print(f"  HDFS {HDFS_LOOKUP_DIR}/{name}: {'OK' if ok else 'FAILED'}")

    text = hdfs_get(HDFS_GRAPH_NODES)
    if text:
        nodes = pd.read_csv(io.StringIO(text))
        order = list(nodes.columns)
        fixed = with_real_coords(nodes, centroids)[order]
        ok = hdfs_put(HDFS_GRAPH_NODES, fixed.to_csv(index=False))
        print(f"  HDFS {HDFS_GRAPH_NODES}: real coordinates {'written' if ok else 'FAILED'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="also push to HDFS and fix graph_nodes.csv")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading taxi zone shapefile...")
    polygons = load_zone_polygons()

    centroids = build_zone_centroids(polygons)
    centroids.to_csv(os.path.join(OUT_DIR, "zone_centroids.csv"), index=False)
    print(f"  {len(centroids)} zone centroids -> 07_streaming/lookups/zone_centroids.csv")

    print("Mapping subway stations to zones...")
    stations = build_station_zones(polygons)
    stations.to_csv(os.path.join(OUT_DIR, "subway_station_zones.csv"), index=False)
    print(f"  {len(stations)} stations in {stations['zone_id'].nunique()} zones "
          f"-> 07_streaming/lookups/subway_station_zones.csv")

    if args.upload:
        upload(centroids, stations)


if __name__ == "__main__":
    main()
