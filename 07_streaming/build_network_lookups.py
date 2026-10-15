"""
Builds the two static layers of the dashboard's Network Map (run once):

  07_streaming/lookups/taxi_zones.geojson     outline of every taxi zone (GPS coordinates, simplified)
  07_streaming/lookups/subway_segments.csv    approximate track of every subway line

The MTA station list has no stop order, so each line is drawn by linking its stations
with the shortest possible network (a minimum spanning tree): neighbouring stations get
connected, which follows the real track closely, including branches.

Usage:
    python 07_streaming/build_network_lookups.py
"""

import json
import math
import os

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import mapping

from build_live_lookups import TO_WGS84, load_zone_polygons

HERE = os.path.dirname(os.path.abspath(__file__))
LOOKUPS = os.path.join(HERE, "lookups")


def km_between(a, b):
    """Haversine distance in km between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def build_zone_geojson():
    zones = pd.read_csv(os.path.join(LOOKUPS, "zone_centroids.csv")).set_index("zone_id")
    features = []
    for zone_id, poly in load_zone_polygons().items():
        if zone_id not in zones.index:
            continue
        # Simplify in State Plane feet (100 ft tolerance), then convert to GPS lat/lon
        shape = shapely.transform(poly.simplify(100),
                                  lambda xy: np.column_stack(TO_WGS84.transform(xy[:, 0], xy[:, 1])))
        features.append({"type": "Feature", "geometry": mapping(shape),
                         "properties": {"zone_id": int(zone_id), "name": zones.at[zone_id, "name"],
                                        "borough": zones.at[zone_id, "borough"]}})
    path = os.path.join(LOOKUPS, "taxi_zones.geojson")
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"  {len(features)} zones -> {path}")


def build_subway_segments():
    stations = pd.read_csv(os.path.join(LOOKUPS, "subway_station_zones.csv"))
    rows = []
    for line in sorted({l for s in stations["lines"] for l in str(s).split("|")}):
        on_line = stations[stations["lines"].str.split("|").apply(lambda ls: line in ls)]
        pts = list(zip(on_line["lat"], on_line["lon"]))
        ids = list(on_line["complex_id"])
        # Prim's algorithm: grow the network by always adding the closest unconnected station
        connected, best = {0}, {i: (km_between(pts[0], pts[i]), 0) for i in range(1, len(pts))}
        while best:
            nxt = min(best, key=lambda i: best[i][0])
            dist, frm = best.pop(nxt)
            connected.add(nxt)
            # "S" is three separate shuttles (42 St, Franklin Av, Rockaway Park): don't draw links between them
            if not (line == "S" and dist > 3):
                rows.append({"line": line, "from_id": ids[frm], "to_id": ids[nxt],
                             "from_lat": pts[frm][0], "from_lon": pts[frm][1],
                             "to_lat": pts[nxt][0], "to_lon": pts[nxt][1]})
            for i in best:
                d = km_between(pts[nxt], pts[i])
                if d < best[i][0]:
                    best[i] = (d, nxt)
    path = os.path.join(LOOKUPS, "subway_segments.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  {len(rows)} track segments on {len({r['line'] for r in rows})} lines -> {path}")


if __name__ == "__main__":
    print("Building Network Map lookups...")
    build_zone_geojson()
    build_subway_segments()
