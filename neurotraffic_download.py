"""
=============================================================================
NeuroTraffic — Dataset Downloader (2022–2024)
=============================================================================
Downloads all 6 datasets required for the NeuroTraffic project:
  1. NYC TLC Taxi (Yellow + Green + HVFHV)  — Parquet files
  2. Citi Bike Trip Data                     — CSV (zipped)
  3. MTA Subway Hourly Ridership             — CSV via Socrata API
  4. NOAA Weather (NYC Central Park)         — CSV via NOAA CDO API
  5. NYC 311 Service Requests                — CSV via Socrata API
  6. NYC Event Permits                       — CSV via Socrata API

Usage:
  pip install requests tqdm
  python neurotraffic_download.py

  # To download only a specific dataset:
  python neurotraffic_download.py --only taxi
  python neurotraffic_download.py --only bike
  python neurotraffic_download.py --only subway
  python neurotraffic_download.py --only weather
  python neurotraffic_download.py --only 311
  python neurotraffic_download.py --only events

  # For weather data, you need a free NOAA CDO API token:
  python neurotraffic_download.py --noaa-token YOUR_TOKEN_HERE

Notes:
  - Total download: ~50-65 GB
  - Ensure sufficient disk space before running
  - Some downloads (311, subway) use paginated API calls and may take time
  - Weather requires a free NOAA API token from:
    https://www.ncdc.noaa.gov/cdo-web/token
=============================================================================
"""

import os
import sys
import time
import argparse
import requests
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Try importing tqdm for progress bars; fall back to a simple print if missing
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[INFO] Install tqdm for progress bars: pip install tqdm")


# ===========================================================================
# CONFIGURATION
# ===========================================================================
BASE_DIR = Path("neurotraffic_data")
START_YEAR = 2022
END_YEAR = 2024

# Sub-directories for each dataset
DIRS = {
    "taxi_yellow":  BASE_DIR / "taxi" / "yellow",
    "taxi_green":   BASE_DIR / "taxi" / "green",
    "taxi_hvfhv":   BASE_DIR / "taxi" / "hvfhv",
    "taxi_zones":   BASE_DIR / "taxi" / "zones",
    "bike":         BASE_DIR / "citibike",
    "subway":       BASE_DIR / "subway",
    "weather":      BASE_DIR / "weather",
    "complaints":   BASE_DIR / "311",
    "events":       BASE_DIR / "events",
}


def create_dirs():
    """Create all output directories."""
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created directory structure under ./{BASE_DIR}/\n")


# ===========================================================================
# HELPER — download a single file with optional progress bar
# ===========================================================================
def download_file(url, dest_path, description=None, chunk_size=1024 * 1024,
                  max_retries=5):
    """Download a file from url to dest_path with retry + resume support."""
    if dest_path.exists():
        print(f"  [SKIP] Already exists: {dest_path.name}")
        return True

    desc = description or dest_path.name
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    for attempt in range(1, max_retries + 1):
        try:
            # Resume from where we left off if tmp file exists
            downloaded = tmp_path.stat().st_size if tmp_path.exists() else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

            resp = requests.get(url, stream=True, timeout=120, headers=headers)

            # 416 = range not satisfiable (file already complete)
            if resp.status_code == 416:
                tmp_path.rename(dest_path)
                print(f"  [OK] {dest_path.name} (resumed, already complete)")
                return True

            resp.raise_for_status()

            # If server supports Range, content-range header is present
            is_resume = resp.status_code == 206
            total = int(resp.headers.get("content-length", 0))
            if is_resume:
                total += downloaded
            elif downloaded > 0:
                # Server doesn't support Range — start fresh
                downloaded = 0

            mode = "ab" if is_resume else "wb"

            if HAS_TQDM:
                with open(tmp_path, mode) as f, tqdm(
                    total=total or None, initial=downloaded,
                    unit="B", unit_scale=True, desc=f"  {desc}"
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        bar.update(len(chunk))
            else:
                size_mb = f" ({total / 1e6:.1f} MB)" if total else ""
                resume_msg = f" (resuming from {downloaded / 1e6:.1f} MB)" if downloaded else ""
                print(f"  [DOWNLOADING] {desc}{size_mb}{resume_msg} ...")
                with open(tmp_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        f.write(chunk)

            tmp_path.rename(dest_path)
            print(f"  [OK] {dest_path.name}")
            return True

        except requests.exceptions.RequestException as e:
            wait = min(30, 5 * attempt)
            if attempt < max_retries:
                print(f"  [RETRY {attempt}/{max_retries}] {desc} — {type(e).__name__}. "
                      f"Resuming in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [FAIL] {desc} — gave up after {max_retries} attempts: {e}")
                tmp_path.unlink(missing_ok=True)
                return False


# ===========================================================================
# 1. NYC TLC TAXI DATA
# ===========================================================================
def download_taxi():
    """Download Yellow, Green, and HVFHV taxi trip Parquet files (2022-2024)."""
    print("=" * 70)
    print("1. NYC TLC TAXI DATA (Yellow + Green + HVFHV)")
    print("=" * 70)

    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"

    types = {
        "yellow": ("taxi_yellow", "yellow_tripdata"),
        "green":  ("taxi_green",  "green_tripdata"),
        "hvfhv":  ("taxi_hvfhv",  "fhvhv_tripdata"),
    }

    total_files = 0
    for taxi_type, (dir_key, prefix) in types.items():
        print(f"\n--- {taxi_type.upper()} TAXI ---")
        for year in range(START_YEAR, END_YEAR + 1):
            for month in range(1, 13):
                filename = f"{prefix}_{year}-{month:02d}.parquet"
                url = f"{base_url}/{filename}"
                dest = DIRS[dir_key] / filename
                download_file(url, dest, description=filename)
                total_files += 1

    # Also download taxi zone lookup table and shapefile
    print("\n--- TAXI ZONE REFERENCE FILES ---")
    zone_files = {
        "taxi_zone_lookup.csv": "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
        "taxi_zones.zip": "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip",
    }
    for fname, url in zone_files.items():
        download_file(url, DIRS["taxi_zones"] / fname, description=fname)

    print(f"\n[DONE] Taxi: {total_files} trip files + 2 zone reference files\n")


# ===========================================================================
# 2. CITI BIKE TRIP DATA
# ===========================================================================
def download_bike():
    """Download Citi Bike trip data (2022-2024).
    2022-2023: yearly zips; 2024: monthly zips."""
    print("=" * 70)
    print("2. CITI BIKE TRIP DATA")
    print("=" * 70)

    base_url = "https://s3.amazonaws.com/tripdata"

    # 2022 and 2023 are single yearly zip files on S3
    for year in [2022, 2023]:
        filename = f"{year}-citibike-tripdata.zip"
        url = f"{base_url}/{filename}"
        dest = DIRS["bike"] / filename
        download_file(url, dest, description=f"CitiBike {year} (full year)")

    # 2024 uses monthly zip files
    for month in range(1, 13):
        ym = f"2024{month:02d}"
        filename = f"{ym}-citibike-tripdata.zip"
        url = f"{base_url}/{filename}"
        dest = DIRS["bike"] / filename
        download_file(url, dest, description=f"CitiBike 2024-{month:02d}")

    print(f"\n[DONE] Citi Bike: 2 yearly (2022-2023) + 12 monthly (2024) files\n")


# ===========================================================================
# 3. MTA SUBWAY HOURLY RIDERSHIP
# ===========================================================================
def download_subway():
    """Download MTA Subway Hourly Ridership data (2022-2024) via Socrata API."""
    print("=" * 70)
    print("3. MTA SUBWAY HOURLY RIDERSHIP (2022-2024)")
    print("=" * 70)

    # Socrata dataset: MTA-Subway-Hourly-Ridership-2020-2024
    # Dataset ID: wujg-7c2s
    # We filter for 2022-01-01 to 2024-12-31
    base_url = "https://data.ny.gov/resource/wujg-7c2s.csv"

    # Socrata has a default limit of 1000 rows; max per request is 50000
    # The dataset is large, so we download year by year
    limit = 50000

    for year in range(START_YEAR, END_YEAR + 1):
        dest = DIRS["subway"] / f"subway_hourly_{year}.csv"
        if dest.exists():
            print(f"  [SKIP] Already exists: {dest.name}")
            continue

        print(f"\n  [DOWNLOADING] Subway ridership {year} (paginated, may take a while)...")

        start_date = f"{year}-01-01T00:00:00"
        end_date = f"{year}-12-31T23:59:59"
        offset = 0

        with open(dest, "w", newline="", encoding="utf-8") as f:
            while True:
                params = {
                    "$where": f"transit_timestamp >= '{start_date}' AND transit_timestamp <= '{end_date}'",
                    "$limit": limit,
                    "$offset": offset,
                    "$order": "transit_timestamp ASC",
                }
                try:
                    resp = requests.get(base_url, params=params, timeout=120)
                    resp.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"  [FAIL] Subway {year} at offset {offset} — {e}")
                    break

                text = resp.text
                lines = text.strip().split("\n")

                if offset == 0:
                    # Write header + all data
                    f.write(text)
                    if not text.endswith("\n"):
                        f.write("\n")
                    header_written = True
                else:
                    # Skip header line, write only data
                    if len(lines) > 1:
                        f.write("\n".join(lines[1:]))
                        f.write("\n")

                num_rows = len(lines) - 1  # minus header
                offset += limit
                print(f"    ... fetched {offset} rows so far for {year}")

                if num_rows < limit:
                    break

                # Be nice to the API
                time.sleep(1)

        print(f"  [OK] {dest.name}")

    print(f"\n[DONE] Subway: 3 yearly CSV files\n")
    print("[NOTE] This dataset is large (~50-80M rows total for 2022-2024).")
    print("       If download is slow, you can also manually export from:")
    print("       https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s\n")


# ===========================================================================
# 4. NOAA WEATHER DATA
# ===========================================================================
def download_weather(noaa_token=None):
    """Download hourly weather data for NYC from NOAA CDO API."""
    print("=" * 70)
    print("4. NOAA WEATHER DATA (NYC Central Park)")
    print("=" * 70)

    if not noaa_token:
        print("""
  [ACTION REQUIRED] You need a free NOAA CDO API token.

  Steps:
    1. Go to: https://www.ncdc.noaa.gov/cdo-web/token
    2. Enter your email address
    3. You'll receive a token via email (usually instant)
    4. Re-run this script with:
       python neurotraffic_download.py --noaa-token YOUR_TOKEN

  Alternatively, you can download weather data manually:
    1. Go to: https://www.ncei.noaa.gov/cdo-web/search
    2. Select "Daily Summaries" dataset
    3. Date range: 2022-01-01 to 2024-12-31
    4. Search for station: "NY CITY CENTRAL PARK" (GHCND:USW00094728)
    5. Add to cart → select all data types → order as CSV

  Another easy alternative (no token needed):
    Open-Meteo Historical Weather API (free, no auth):
    https://archive-api.open-meteo.com/v1/archive?
      latitude=40.7831&longitude=-73.9712
      &start_date=2022-01-01&end_date=2024-12-31
      &hourly=temperature_2m,precipitation,wind_speed_10m,
              visibility,weather_code
      &timezone=America/New_York
""")
        # Download from Open-Meteo as fallback (free, no token)
        print("  [AUTO] Downloading from Open-Meteo (free alternative)...\n")
        _download_weather_openmeteo()
        return

    # --- NOAA CDO API download ---
    # Station: Central Park, NYC
    station_id = "GHCND:USW00094728"
    base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": noaa_token}

    # NOAA API limits: 1 year per request for daily data, 1000 records per call
    for year in range(START_YEAR, END_YEAR + 1):
        dest = DIRS["weather"] / f"weather_nyc_{year}.csv"
        if dest.exists():
            print(f"  [SKIP] Already exists: {dest.name}")
            continue

        print(f"  [DOWNLOADING] Weather data {year}...")

        all_records = []
        offset = 1

        while True:
            params = {
                "datasetid": "GHCND",
                "stationid": station_id,
                "startdate": f"{year}-01-01",
                "enddate": f"{year}-12-31",
                "datatypeid": "TMAX,TMIN,PRCP,AWND,SNOW,SNWD",
                "units": "metric",
                "limit": 1000,
                "offset": offset,
            }

            try:
                resp = requests.get(base_url, headers=headers, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  [FAIL] Weather {year} at offset {offset} — {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            all_records.extend(results)
            offset += 1000
            print(f"    ... fetched {len(all_records)} records for {year}")

            if len(results) < 1000:
                break

            time.sleep(0.25)  # Rate limit: 5 req/sec

        # Write to CSV
        if all_records:
            import csv
            fieldnames = ["date", "datatype", "station", "value", "attributes"]
            with open(dest, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in all_records:
                    writer.writerow({
                        "date": rec.get("date", ""),
                        "datatype": rec.get("datatype", ""),
                        "station": rec.get("station", ""),
                        "value": rec.get("value", ""),
                        "attributes": rec.get("attributes", ""),
                    })
            print(f"  [OK] {dest.name} ({len(all_records)} records)")
        else:
            print(f"  [WARN] No records returned for {year}")

    print(f"\n[DONE] Weather data\n")


def _download_weather_openmeteo():
    """Fallback: download weather from Open-Meteo (free, no token)."""
    # Open-Meteo has a limit on date range per request, so download year by year
    base_url = "https://archive-api.open-meteo.com/v1/archive"

    for year in range(START_YEAR, END_YEAR + 1):
        dest = DIRS["weather"] / f"weather_openmeteo_nyc_{year}.csv"
        if dest.exists():
            print(f"  [SKIP] Already exists: {dest.name}")
            continue

        params = {
            "latitude": 40.7831,
            "longitude": -73.9712,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,"
                      "rain,snowfall,snow_depth,weather_code,"
                      "wind_speed_10m,wind_gusts_10m,visibility",
            "timezone": "America/New_York",
            "format": "csv",
        }

        try:
            resp = requests.get(base_url, params=params, timeout=120)
            resp.raise_for_status()
            with open(dest, "w", encoding="utf-8") as f:
                f.write(resp.text)
            print(f"  [OK] {dest.name}")
        except requests.exceptions.RequestException as e:
            print(f"  [FAIL] Weather {year} — {e}")
            print(f"         Try manual download from: {base_url}")

    print(f"\n[DONE] Weather (Open-Meteo fallback)\n")


# ===========================================================================
# 5. NYC 311 SERVICE REQUESTS
# ===========================================================================
def download_311():
    """Download NYC 311 complaints (2022-2024) via Socrata API, filtered for
    transport-relevant complaint types."""
    print("=" * 70)
    print("5. NYC 311 SERVICE REQUESTS (2022-2024)")
    print("=" * 70)

    # Dataset: 311 Service Requests from 2020 to Present
    # Dataset ID: erm2-nwe9
    base_url = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"

    # Transport-relevant complaint types for NeuroTraffic
    transport_types = [
        "Street Condition",
        "Traffic Signal Condition",
        "Street Light Condition",
        "Blocked Driveway",
        "Illegal Parking",
        "Noise - Street/Sidewalk",
        "Water Main Break",
        "Sewer",
        "Bus Stop Shelter Complaint",
    ]

    type_filter = " OR ".join([f"complaint_type='{t}'" for t in transport_types])
    limit = 50000

    for year in range(START_YEAR, END_YEAR + 1):
        dest = DIRS["complaints"] / f"311_transport_{year}.csv"
        if dest.exists():
            print(f"  [SKIP] Already exists: {dest.name}")
            continue

        print(f"\n  [DOWNLOADING] 311 requests {year} (paginated)...")

        start_date = f"{year}-01-01T00:00:00"
        end_date = f"{year}-12-31T23:59:59"
        offset = 0

        with open(dest, "w", newline="", encoding="utf-8") as f:
            while True:
                where_clause = (
                    f"created_date >= '{start_date}' AND "
                    f"created_date <= '{end_date}' AND "
                    f"({type_filter})"
                )
                params = {
                    "$where": where_clause,
                    "$limit": limit,
                    "$offset": offset,
                    "$order": "created_date ASC",
                }

                try:
                    resp = requests.get(base_url, params=params, timeout=120)
                    resp.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"  [FAIL] 311 {year} at offset {offset} — {e}")
                    break

                text = resp.text
                lines = text.strip().split("\n")

                if offset == 0:
                    f.write(text)
                    if not text.endswith("\n"):
                        f.write("\n")
                else:
                    if len(lines) > 1:
                        f.write("\n".join(lines[1:]))
                        f.write("\n")

                num_rows = len(lines) - 1
                offset += limit
                print(f"    ... fetched {offset} rows so far for {year}")

                if num_rows < limit:
                    break

                time.sleep(1)

        print(f"  [OK] {dest.name}")

    # Also download the FULL 311 dataset (all complaint types) if user wants
    print(f"\n[DONE] 311: transport-relevant complaints for 2022-2024\n")
    print("[NOTE] Only transport-relevant complaint types were downloaded.")
    print("       To download ALL complaint types, modify the 'transport_types'")
    print("       list in this script, or remove the type filter entirely.\n")
    print("       Full dataset: https://data.cityofnewyork.us/Social-Services/"
          "311-Service-Requests-from-2020-to-Present/erm2-nwe9\n")


# ===========================================================================
# 6. NYC EVENT PERMITS
# ===========================================================================
def download_events():
    """Download NYC event permits (2022-2024) via Socrata API."""
    print("=" * 70)
    print("6. NYC EVENT PERMITS (2022-2024)")
    print("=" * 70)

    # Historical events dataset ID: bkfu-528j
    base_url = "https://data.cityofnewyork.us/resource/bkfu-528j.csv"
    limit = 50000

    dest = DIRS["events"] / "event_permits_2022_2024.csv"
    if dest.exists():
        print(f"  [SKIP] Already exists: {dest.name}")
    else:
        print(f"  [DOWNLOADING] Event permits 2022-2024...")
        offset = 0

        with open(dest, "w", newline="", encoding="utf-8") as f:
            while True:
                params = {
                    "$where": (
                        "start_date_time >= '2022-01-01T00:00:00' AND "
                        "start_date_time <= '2024-12-31T23:59:59'"
                    ),
                    "$limit": limit,
                    "$offset": offset,
                    "$order": "start_date_time ASC",
                }

                try:
                    resp = requests.get(base_url, params=params, timeout=120)
                    resp.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"  [FAIL] Events at offset {offset} — {e}")
                    break

                text = resp.text
                lines = text.strip().split("\n")

                if offset == 0:
                    f.write(text)
                    if not text.endswith("\n"):
                        f.write("\n")
                else:
                    if len(lines) > 1:
                        f.write("\n".join(lines[1:]))
                        f.write("\n")

                num_rows = len(lines) - 1
                offset += limit
                print(f"    ... fetched {offset} rows so far")

                if num_rows < limit:
                    break

                time.sleep(1)

        print(f"  [OK] {dest.name}")

    # Also download film permits
    print(f"\n--- FILM PERMITS ---")
    film_url = "https://data.cityofnewyork.us/resource/tg4x-b46p.csv"
    film_dest = DIRS["events"] / "film_permits_2022_2024.csv"

    if film_dest.exists():
        print(f"  [SKIP] Already exists: {film_dest.name}")
    else:
        print(f"  [DOWNLOADING] Film permits 2022-2024...")
        offset = 0

        with open(film_dest, "w", newline="", encoding="utf-8") as f:
            while True:
                params = {
                    "$where": (
                        "startdatetime >= '2022-01-01T00:00:00' AND "
                        "startdatetime <= '2024-12-31T23:59:59'"
                    ),
                    "$limit": limit,
                    "$offset": offset,
                    "$order": "startdatetime ASC",
                }

                try:
                    resp = requests.get(film_url, params=params, timeout=120)
                    resp.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"  [FAIL] Film permits at offset {offset} — {e}")
                    break

                text = resp.text
                lines = text.strip().split("\n")

                if offset == 0:
                    f.write(text)
                    if not text.endswith("\n"):
                        f.write("\n")
                else:
                    if len(lines) > 1:
                        f.write("\n".join(lines[1:]))
                        f.write("\n")

                num_rows = len(lines) - 1
                offset += limit
                print(f"    ... fetched {offset} film permit rows so far")

                if num_rows < limit:
                    break

                time.sleep(1)

        print(f"  [OK] {film_dest.name}")

    print(f"\n[DONE] Event permits\n")


# ===========================================================================
# SUMMARY
# ===========================================================================
def print_summary():
    """Print a summary of what was downloaded."""
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    total_size = 0
    total_files = 0

    for name, directory in DIRS.items():
        if directory.exists():
            files = list(directory.iterdir())
            dir_size = sum(f.stat().st_size for f in files if f.is_file())
            total_size += dir_size
            total_files += len(files)
            size_str = _format_size(dir_size)
            print(f"  {name:20s} — {len(files):4d} files — {size_str}")

    print(f"\n  {'TOTAL':20s} — {total_files:4d} files — {_format_size(total_size)}")
    print()

    # Print directory structure
    print("Directory structure:")
    print(f"  {BASE_DIR}/")
    for name, directory in DIRS.items():
        rel = directory.relative_to(BASE_DIR)
        print(f"    {rel}/")

    print()
    print("Next steps:")
    print("  1. Verify all files downloaded correctly")
    print("  2. Upload to HDFS:  hdfs dfs -put neurotraffic_data/ /data/raw/")
    print("  3. Run PySpark cleaning scripts on each dataset")
    print("  4. Join all cleaned datasets by zone_id + hourly_timestamp")
    print()


def _format_size(size_bytes):
    """Format bytes into human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="NeuroTraffic Dataset Downloader (2022-2024)"
    )
    parser.add_argument(
        "--only",
        choices=["taxi", "bike", "subway", "weather", "311", "events"],
        help="Download only a specific dataset",
    )
    parser.add_argument(
        "--noaa-token",
        type=str,
        default=None,
        help="NOAA CDO API token (get free at https://www.ncdc.noaa.gov/cdo-web/token)",
    )
    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                 NeuroTraffic — Dataset Downloader                   ║
║                    Timeline: 2022 – 2024                            ║
║              Estimated total: ~50-65 GB, ~530-660M rows             ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)

    create_dirs()

    if args.only:
        dispatch = {
            "taxi": download_taxi,
            "bike": download_bike,
            "subway": download_subway,
            "weather": lambda: download_weather(args.noaa_token),
            "311": download_311,
            "events": download_events,
        }
        dispatch[args.only]()
    else:
        download_taxi()
        download_bike()
        download_subway()
        download_weather(args.noaa_token)
        download_311()
        download_events()

    print_summary()


if __name__ == "__main__":
    main()
