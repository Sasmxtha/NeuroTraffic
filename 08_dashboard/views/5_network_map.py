"""
NeuroTraffic — Page 5: Network Map
====================================

One map with every transport mode and the links between them:

  Layer              Source                                        Live?
  Taxi zones         latest.csv (anomaly score per zone)           LIVE score (taxi volume itself is estimated)
  Subway lines       subway_segments.csv + MTA alerts per line     LIVE alerts and train counts
  Subway stations    subway_station_zones.csv + MTA alerts         LIVE alerts
  Bike docks         bike_docks.json (Citi Bike per dock)          LIVE
  Station-dock links stations and docks less than 0.8 km apart     LIVE activity on fixed links

A link turns orange when its station's line has an alert AND the dock is busy:
that is the "riders switch from subway to bike" signal the project looks for.
"""

import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
import hdfs_helper
import ui

NYC = ZoneInfo("America/New_York")
LINK_KM = 0.8          # walking distance between a station and a bike dock
BUSY_DOCK_MOVES = 3    # bike changes in the last 15 min that make a dock "busy"

ui.setup_page("Network Map", "🗺",
              "Taxi zones, subway and Citi Bike on one live map, with the walking links between subway stations "
              "and nearby bike docks. When a subway line has a problem, watch its links: busy docks next to an "
              "affected station show riders switching from subway to bike.")
st.markdown("""<style>
.legend { display:flex; flex-wrap:wrap; gap:0.4rem 1.2rem; font-size:0.78rem; color:#cbd5e1; margin:0.4rem 0 0.8rem; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:middle; }
.bar { display:inline-block; width:18px; height:4px; margin-right:5px; vertical-align:middle; }
.tag { font-size:0.62rem; font-weight:700; padding:1px 6px; border-radius:8px; margin-left:4px; }
.live { background:rgba(34,197,94,0.15); color:#22c55e; }
.hist { background:rgba(148,163,184,0.15); color:#94a3b8; }
</style>""", unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────
STATIC_FILES = [config.ZONES_GEOJSON, config.SUBWAY_STATIONS_CSV, config.SUBWAY_SEGMENTS_CSV]

@st.cache_data
def load_static(file_times):   # file_times: reload automatically when a lookup file is rebuilt
    with open(config.ZONES_GEOJSON) as f:
        zones = json.load(f)
    stations = pd.read_csv(config.SUBWAY_STATIONS_CSV)
    segments = pd.read_csv(config.SUBWAY_SEGMENTS_CSV)
    return zones, stations, segments

@st.cache_data(ttl=10)
def load_live(use_hdfs):
    latest = hdfs_helper.load_dataframe(config.LATEST_CSV, config.HDFS_LATEST_CSV, use_hdfs=use_hdfs)
    status = hdfs_helper.load_json(config.SOURCE_STATUS_JSON, config.HDFS_SOURCE_STATUS_JSON, use_hdfs=use_hdfs)
    docks = hdfs_helper.load_json(config.BIKE_DOCKS_JSON, config.HDFS_BIKE_DOCKS_JSON, use_hdfs=use_hdfs)
    return latest, status or {}, docks or {}


def rgb(hex_color, alpha=255):
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [alpha]

def zone_fill(score):
    """Shared level colour; normal zones stay almost transparent so problem zones stand out."""
    name, colour = ui.level(score)
    return rgb(colour, {"Disruption": 130, "Anomaly": 100, "Elevated": 60, "Normal": 18}[name])

def alert_color(severity):
    if severity >= 0.8: return rgb("#ef4444")         # suspended / severe
    if severity >= 0.6: return rgb("#f97316")         # delays
    if severity > 0:    return rgb("#facc15")         # reduced service / reroute
    return rgb("#e2e8f0")                             # no alert

def dock_color(bikes, docks_free):
    total = bikes + docks_free
    share = bikes / total if total else 0
    if share < 0.10: return rgb("#ef4444", 220)       # almost empty
    if share < 0.30: return rgb("#f59e0b", 220)       # running low
    return rgb("#38bdf8", 200)                        # plenty of bikes

def age_text(iso):
    if not iso:
        return "unknown"
    secs = (datetime.now(NYC) - datetime.fromisoformat(iso).astimezone(NYC)).total_seconds()
    return f"{secs:.0f} s ago" if secs < 120 else f"{secs / 60:.0f} min ago"


use_hdfs = ui.use_hdfs()
zones_geo, stations, segments = load_static(tuple(os.path.getmtime(f) for f in STATIC_FILES))
latest, status, dock_data = load_live(use_hdfs)

# Alert severity and live train count per line, from source_status.json
line_info = {l["line"]: l for l in status.get("subway_lines", [])}
line_alert = {line: info.get("alert_severity", 0.0) for line, info in line_info.items()}
alerted_lines = sorted(l for l, sev in line_alert.items() if sev > 0)


# ── Sidebar controls ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗺 Network Map")
    st.caption("Switch layers on and off. Hover over anything on the map for details.")
    show_zones = st.checkbox("Taxi zones (anomaly score)", value=True)
    show_lines = st.checkbox("Subway lines", value=True)
    show_stations = st.checkbox("Subway stations", value=True)
    show_docks = st.checkbox("Citi Bike docks", value=True)
    link_mode = st.radio("Station ↔ bike dock links",
                         ["Only stations with an alert", "All stations", "Hide"], index=0)
    st.markdown("---")
    focus_options = ["Whole city"] + [f"Line {l} (alert)" for l in alerted_lines] + \
                    ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    focus = st.selectbox("Focus on", focus_options)
    st.markdown("---")
    if st.button("🔄 Refresh now", width="stretch"):
        load_live.clear()
        st.rerun()
    auto = st.toggle("Auto-refresh every 30 s", value=False,
                     help="Off by default: each refresh resets your zoom and position on the map.")


# ── Prepare layers ────────────────────────────────────────────────────
# Zones: fill colour = live anomaly score
scores = latest.set_index("zone_id")["anomaly_score"].to_dict() if latest is not None else {}
for feat in zones_geo["features"]:
    p = feat["properties"]
    score = float(scores.get(p["zone_id"], 0.0))
    p["fill"] = zone_fill(score)
    p["tip"] = f"<b>{p['name']}</b> ({p['borough']})<br>Anomaly score: {score:.3f}"

# Stations: colour = worst live alert on any of its lines
stations = stations.copy()
stations["severity"] = stations["lines"].apply(lambda ls: max(line_alert.get(l, 0.0) for l in str(ls).split("|")))
stations["color"] = stations["severity"].apply(alert_color)
stations["tip"] = stations.apply(lambda s: f"<b>🚇 {s['name']}</b><br>Lines: {s['lines'].replace('|', ', ')}<br>"
                                 + ("Active alert" if s["severity"] > 0 else "No alert"), axis=1)

# Line segments: official line colour, thicker when the line has an alert
segments = segments.copy()
segments["color"] = segments["line"].apply(lambda l: rgb(config.LINE_COLORS.get(l, "#94a3b8"), 200))
segments["width"] = segments["line"].apply(lambda l: 6 if line_alert.get(l, 0) > 0 else 2.5)
segments["tip"] = segments["line"].apply(
    lambda l: f"<b>Line {l}</b><br>Trains reporting now: {line_info.get(l, {}).get('trains', 0)}<br>"
              + ("⚠ Active alert" if line_alert.get(l, 0) > 0 else "No alert"))

# Bike docks: colour = share of bikes available, size = activity in the last 15 min
docks = pd.DataFrame(dock_data.get("docks", []))
if len(docks):
    docks["color"] = docks.apply(lambda d: dock_color(d["bikes"], d["docks_free"]), axis=1)
    docks["radius"] = 25 + docks["moves_15min"].clip(upper=15) * 6
    docks["tip"] = docks.apply(lambda d: f"<b>🚲 {d['name']}</b><br>Bikes: {d['bikes']} · Free docks: {d['docks_free']}"
                               f"<br>Bike changes in last 15 min: {d['moves_15min']}", axis=1)

# Links: every station-dock pair closer than LINK_KM (fast flat-earth distance, fine within NYC)
links = pd.DataFrame()
if len(docks) and link_mode != "Hide":
    src = stations if link_mode == "All stations" else stations[stations["severity"] > 0]
    if len(src):
        dy = (src["lat"].values[:, None] - docks["lat"].values[None, :]) * 110.57
        dx = (src["lon"].values[:, None] - docks["lon"].values[None, :]) * 111.32 * np.cos(np.radians(40.73))
        si, di = np.nonzero(np.hypot(dx, dy) < LINK_KM)
        s_rows, d_rows = src.iloc[si].reset_index(drop=True), docks.iloc[di].reset_index(drop=True)
        links = pd.DataFrame({"from_lon": s_rows["lon"], "from_lat": s_rows["lat"],
                              "to_lon": d_rows["lon"], "to_lat": d_rows["lat"],
                              "active": (s_rows["severity"] > 0) & (d_rows["moves_15min"] >= BUSY_DOCK_MOVES)})
        links["color"] = links["active"].apply(lambda a: rgb("#fb923c", 230) if a else rgb("#94a3b8", 70))
        links["width"] = links["active"].apply(lambda a: 2.5 if a else 1)
        links["tip"] = s_rows["name"] + " ↔ " + d_rows["name"] + links["active"].map(
            {True: "<br><b>Active:</b> line alert + busy dock", False: ""})

# Where the map looks
view = pdk.ViewState(latitude=config.MAP_CENTER["lat"], longitude=config.MAP_CENTER["lon"], zoom=10.5)
if focus.startswith("Line "):
    on_line = stations[stations["lines"].str.split("|").apply(lambda ls: focus.split()[1] in ls)]
    view = pdk.ViewState(latitude=on_line["lat"].mean(), longitude=on_line["lon"].mean(), zoom=11.5)
elif focus != "Whole city":
    in_boro = [f for f in zones_geo["features"] if f["properties"]["borough"] == focus]
    ids = {f["properties"]["zone_id"] for f in in_boro}
    on_boro = stations[stations["zone_id"].isin(ids)]
    if len(on_boro):
        view = pdk.ViewState(latitude=on_boro["lat"].mean(), longitude=on_boro["lon"].mean(), zoom=11.8)


# ── Layers (drawn bottom to top) ──────────────────────────────────────
layers = []
if show_zones:
    layers.append(pdk.Layer("GeoJsonLayer", zones_geo, pickable=True, stroked=True, filled=True,
                            get_fill_color="properties.fill", get_line_color=[148, 163, 184, 60],
                            line_width_min_pixels=0.5))
if show_lines:
    layers.append(pdk.Layer("LineLayer", segments, pickable=True, get_source_position=["from_lon", "from_lat"],
                            get_target_position=["to_lon", "to_lat"], get_color="color", get_width="width"))
if len(links):
    layers.append(pdk.Layer("LineLayer", links, pickable=True, get_source_position=["from_lon", "from_lat"],
                            get_target_position=["to_lon", "to_lat"], get_color="color", get_width="width"))
if show_docks and len(docks):
    layers.append(pdk.Layer("ScatterplotLayer", docks, pickable=True, get_position=["lon", "lat"],
                            get_fill_color="color", get_radius="radius", radius_min_pixels=1.5,
                            radius_max_pixels=9))
if show_stations:
    layers.append(pdk.Layer("ScatterplotLayer", stations, pickable=True, get_position=["lon", "lat"],
                            get_fill_color="color", get_radius=70, radius_min_pixels=2.5, radius_max_pixels=7,
                            stroked=True, get_line_color=[15, 23, 42, 255], line_width_min_pixels=1))


# ── Page ──────────────────────────────────────────────────────────────

zone_time = latest["timestamp"].max() if latest is not None and "timestamp" in latest else None
st.markdown(
    f"<div style='font-size:0.75rem;color:#94a3b8;'>Zone scores: <b>{zone_time or 'none'}</b> (NYC time) · "
    f"Subway alerts and trains: <b>{age_text(status.get('generated_at'))}</b> · "
    f"Bike docks: <b>{age_text(dock_data.get('observed_at'))}</b></div>", unsafe_allow_html=True)

if not dock_data:
    st.warning("No live bike dock data yet. Start (or restart) the fetcher: "
               "`python 07_streaming/realtime_fetcher.py --interval 10`")

n_links_active = int(links["active"].sum()) if len(links) else 0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Subway lines with an alert", len(alerted_lines), help=", ".join(alerted_lines) or "none")
c2.metric("Stations on those lines", int((stations["severity"] > 0).sum()))
c3.metric("Bike docks reporting", f"{len(docks):,}")
c4.metric("Active station ↔ dock links", n_links_active,
          help=f"Station has a line alert and the dock had ≥ {BUSY_DOCK_MOVES} bike changes in 15 min")

st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto", map_style="dark",
                         tooltip={"html": "{tip}", "style": {"backgroundColor": "#0f172a", "color": "#e2e8f0",
                                                             "fontSize": "12px"}}),
                height=640)

# Legend: what every colour means and whether it is live
st.markdown("""
<div class='legend'>
 <span><b>Taxi zones</b><span class='tag live'>LIVE score</span></span>
 <span><span class='dot' style='background:#22c55e'></span>normal &lt;0.25</span>
 <span><span class='dot' style='background:#facc15'></span>elevated</span>
 <span><span class='dot' style='background:#f97316'></span>anomaly ≥0.5</span>
 <span><span class='dot' style='background:#ef4444'></span>disruption ≥0.8</span>
</div>
<div class='legend'>
 <span><b>Subway stations</b><span class='tag live'>LIVE alerts</span></span>
 <span><span class='dot' style='background:#e2e8f0'></span>no alert</span>
 <span><span class='dot' style='background:#facc15'></span>reduced service / reroute</span>
 <span><span class='dot' style='background:#f97316'></span>delays</span>
 <span><span class='dot' style='background:#ef4444'></span>suspended</span>
 <span><span class='bar' style='background:#EE352E'></span>lines in official MTA colours, thick = alert
 (paths are approximate)</span>
</div>
<div class='legend'>
 <span><b>Citi Bike docks</b><span class='tag live'>LIVE</span></span>
 <span><span class='dot' style='background:#38bdf8'></span>plenty of bikes</span>
 <span><span class='dot' style='background:#f59e0b'></span>running low (&lt;30%)</span>
 <span><span class='dot' style='background:#ef4444'></span>almost empty (&lt;10%)</span>
 <span>bigger dot = more bikes taken/returned in the last 15 min</span>
</div>
<div class='legend'>
 <span><b>Links</b> (station ↔ dock within 0.8 km)</span>
 <span><span class='bar' style='background:#fb923c'></span>active: line alert + busy dock</span>
 <span><span class='bar' style='background:#64748b'></span>walking link</span>
 <span><span class='tag hist'>ESTIMATED</span> taxi and subway ridership volumes come from the 2022-24 profile;
 NYC publishes no live feed for them.</span>
</div>""", unsafe_allow_html=True)

# Active alerts in words
alerts = status.get("mta_alerts", [])
if alerts:
    st.markdown("**🚇 Active unplanned subway alerts**")
    st.dataframe(pd.DataFrame([{"Lines": ", ".join(a["lines"]), "Type": a["type"], "Message": a["header"]}
                               for a in alerts]), width="stretch", hide_index=True)
else:
    st.info("No unplanned subway alerts right now, so every station is white and no link is highlighted.")

if auto:
    time.sleep(30)
    load_live.clear()
    st.rerun()
