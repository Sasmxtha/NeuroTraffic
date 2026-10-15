"""
NeuroTraffic — Live Monitor
============================
The live anomaly score of every taxi zone (latest.csv, written by live_pipeline.scala),
the zones that need attention, and how fresh every live data source is (source_status.json).
"""

import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
import hdfs_helper
import ui

NYC = ZoneInfo("America/New_York")

ui.setup_page("Live Monitor", "📡",
              "The live anomaly score of all 263 taxi zones. A score rises when the zone's subway line has an "
              "MTA alert, when it rains, or when Citi Bike activity is far from normal for that hour.")
use_hdfs = ui.use_hdfs()

with st.sidebar:
    auto_refresh = st.toggle("Auto-refresh", value=True)
    interval = st.select_slider("Refresh every (seconds)", options=[10, 15, 30, 60], value=15)


@st.cache_data(ttl=5)
def load_live(use_hdfs):
    latest = hdfs_helper.load_dataframe(config.LATEST_CSV, config.HDFS_LATEST_CSV, use_hdfs=use_hdfs)
    status = hdfs_helper.load_json(config.SOURCE_STATUS_JSON, config.HDFS_SOURCE_STATUS_JSON, use_hdfs=use_hdfs)
    return latest, status or {}

@st.cache_data(ttl=600)
def load_nodes(use_hdfs):
    return hdfs_helper.load_dataframe(config.GRAPH_NODES_CSV, config.HDFS_GRAPH_NODES_CSV, use_hdfs=use_hdfs)


latest, status = load_live(use_hdfs)
nodes = load_nodes(use_hdfs)
if latest is None or nodes is None:
    st.warning("No live data found. Start `07_streaming/realtime_fetcher.py` and `07_streaming/live_pipeline.scala`.")
    st.stop()

zones = latest.merge(nodes[["zone_id", "name", "borough", "lat", "lon"]], on="zone_id", how="left")
zones["level"] = zones["anomaly_score"].apply(lambda s: ui.level(s)[0])

# ── Freshness ─────────────────────────────────────────────────────────
snapshot = pd.to_datetime(zones["timestamp"].max())
age_s = (datetime.now(NYC) - snapshot.tz_localize(NYC)).total_seconds()
st.markdown(f"<div class='nt-caption'>Snapshot <b>{snapshot:%H:%M:%S}</b> NYC time · {age_s:.0f} s ago"
            f"{' · auto-refresh every ' + str(interval) + ' s' if auto_refresh else ''}</div>", unsafe_allow_html=True)
if age_s > 600:
    st.warning(f"This snapshot is {age_s / 60:.0f} minutes old: the live pipeline is not running.")

counts = zones["level"].value_counts()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🔴 Disruption", int(counts.get("Disruption", 0)), help="Score ≥ 0.80")
c2.metric("🟠 Anomaly", int(counts.get("Anomaly", 0)), help="Score 0.50 – 0.80")
c3.metric("🟡 Elevated", int(counts.get("Elevated", 0)), help="Score 0.25 – 0.50")
c4.metric("🟢 Normal", int(counts.get("Normal", 0)), help="Score below 0.25")
c5.metric("Average score", f"{zones['anomaly_score'].mean():.3f}")

# ── Map ───────────────────────────────────────────────────────────────
ui.section("Zone map", "One dot per taxi zone, at its centre. Colour = level; bigger = higher score. "
           "Hover for details.", "live")
zones["hover"] = ("<b>" + zones["name"].fillna("Zone " + zones["zone_id"].astype(str)) + "</b> (" +
                  zones["borough"].fillna("?") + ")<br>Score: " + zones["anomaly_score"].map("{:.3f}".format) +
                  " · " + zones["level"] + "<br>Subway line with alert: " + zones["affected_line"].fillna("-"))
fig = go.Figure()
for _, name, colour in reversed(ui.LEVELS):          # draw Normal first so red stays on top
    sub = zones[zones["level"] == name]
    if len(sub):
        fig.add_trace(go.Scattermap(lat=sub["lat"], lon=sub["lon"], mode="markers", name=f"{name} ({len(sub)})",
                                    marker=dict(size=6 + sub["anomaly_score"] * 14, color=colour, opacity=0.9),
                                    text=sub["hover"], hovertemplate="%{text}<extra></extra>"))
st.plotly_chart(ui.zone_map(fig, height=500), width="stretch")
ui.level_legend()

# ── Zones needing attention + borough breakdown ───────────────────────
left, right = st.columns([3, 2])
with left:
    ui.section("Zones needing attention", "Every zone with a score of 0.50 or more, highest first.")
    attention = zones[zones["anomaly_score"] >= 0.5].sort_values("anomaly_score", ascending=False)
    if len(attention):
        table = attention[["name", "borough", "anomaly_score", "level", "affected_line"]].rename(columns={
            "name": "Zone", "borough": "Borough", "anomaly_score": "Score", "level": "Level",
            "affected_line": "Subway line with alert"})
        st.dataframe(table, hide_index=True, width="stretch", height=320,
                     column_config={"Score": st.column_config.NumberColumn(format="%.3f")})
    else:
        st.success("No zone is above 0.50 right now.")
with right:
    ui.section("Zones per level, by borough", "How the 263 zones split across the four levels.")
    by_borough = zones.groupby(["borough", "level"]).size().reset_index(name="zones")
    fig_b = px.bar(by_borough, x="zones", y="borough", color="level", orientation="h",
                   color_discrete_map={n: c for _, n, c in ui.LEVELS},
                   category_orders={"level": [n for _, n, _ in reversed(ui.LEVELS)]})
    st.plotly_chart(ui.style_chart(fig_b, height=320, xaxis_title="Zones", yaxis_title=None,
                                   legend_title_text=None), width="stretch")

# ── Live sources ──────────────────────────────────────────────────────
ui.section("Data sources", "What is measured live, what is estimated, and how old each source is.")
if status.get("sources"):
    now = datetime.now(NYC)
    rows = []
    for s in status["sources"]:
        obs = s.get("observed_at")
        minutes = (now - datetime.fromisoformat(obs)).total_seconds() / 60 if obs else None
        rows.append({"Source": s["name"], "Type": s["kind"].upper(),
                     "Age": "—" if minutes is None else ("< 1 min" if minutes < 1 else f"{minutes:.0f} min"),
                     "Provider": s["provider"],
                     "Detail": f"ERROR: {s['error']}" if s.get("error") else s["detail"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
else:
    st.info("source_status.json not found: the live fetcher has not run yet.")

alerts = status.get("mta_alerts", [])
if alerts:
    ui.section("Active unplanned subway alerts", kind="live")
    st.dataframe(pd.DataFrame([{"Lines": ", ".join(a["lines"]), "Type": a["type"], "Message": a["header"]}
                               for a in alerts]), hide_index=True, width="stretch")

if auto_refresh:
    time.sleep(interval)
    load_live.clear()
    st.rerun()
