"""
Map Citi Bike stations to NYC taxi zones using the taxi zone shapefile.
Runs LOCALLY (not in Docker) — uses pyshp + shapely (no GDAL needed).
"""

import shapefile
from shapely.geometry import Point, shape
from pyproj import Transformer
import csv
import sys, os

SHAPEFILE = r"E:\Big_Data_Project\neurotraffic_data\taxi\zones\taxi_zones\taxi_zones\taxi_zones.shp"
STATION_CSV = r"E:\Big_Data_Project\bike_stations.csv"
OUTPUT_CSV = r"E:\Big_Data_Project\bike_station_zones.csv"

def main():
    if not os.path.exists(STATION_CSV):
        print(f"ERROR: {STATION_CSV} not found.")
        sys.exit(1)

    print("Loading taxi zone shapefile...")
    sf = shapefile.Reader(SHAPEFILE)
    fields = [f[0] for f in sf.fields[1:]]
    loc_idx = fields.index("LocationID")

    zones = []
    for sr in sf.shapeRecords():
        zone_id = sr.record[loc_idx]
        polygon = shape(sr.shape.__geo_interface__)
        zones.append((zone_id, polygon))
    print(f"  Loaded {len(zones)} taxi zones")

    print("Loading bike stations...")
    stations = []
    with open(STATION_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stations.append(row)
    print(f"  Loaded {len(stations)} unique stations")

    print("Mapping stations to taxi zones (point-in-polygon)...")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)

    mapped = 0
    unmapped = 0
    results = []

    for station in stations:
        lat = float(station["lat"])
        lng = float(station["lng"])
        x, y = transformer.transform(lng, lat)
        point = Point(x, y)

        matched_zone = None
        for zone_id, polygon in zones:
            if polygon.contains(point):
                matched_zone = zone_id
                break

        if matched_zone is not None:
            results.append({
                "station_id": station["station_id"],
                "station_name": station["station_name"],
                "lat": station["lat"],
                "lng": station["lng"],
                "zone_id": matched_zone,
            })
            mapped += 1
        else:
            unmapped += 1

    print(f"  Mapped: {mapped}/{mapped + unmapped} stations")
    if unmapped > 0:
        print(f"  Unmapped: {unmapped} stations (outside taxi zone boundaries)")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["station_id", "station_name", "lat", "lng", "zone_id"])
        writer.writeheader()
        writer.writerows(results)

    print(f"  Written to {OUTPUT_CSV}")
    print(f"  Final: {len(results)} stations mapped to taxi zones")
    print("DONE")

if __name__ == "__main__":
    main()
