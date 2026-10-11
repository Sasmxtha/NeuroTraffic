"""
NeuroTraffic — Home Page
=========================
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
import time

# ── Path setup ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import hdfs_helper
import requests

if "use_hdfs" not in st.session_state:
    st.session_state.use_hdfs = config.USE_HDFS

config.USE_HDFS = st.session_state.use_hdfs

# ── Page config ──
st.set_page_config(
    page_title="NeuroTraffic — NYC Transport Intelligence",
    page_icon="🚊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & Base ── */
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #090e1a !important;
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1425 0%, #111827 100%) !important;
    border-right: 1px solid rgba(0, 212, 255, 0.12) !important;
}

[data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0, 212, 255, 0.15) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    backdrop-filter: blur(10px);
}
[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.75rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 1.8rem !important; font-weight: 700 !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0ea5e9) !important;
    color: #090e1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.03em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0, 212, 255, 0.35) !important;
}

/* ── Selects & sliders ── */
.stSelectbox > div > div, .stSlider { color: #e2e8f0 !important; }
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── DataFrames ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.03) !important; border-radius: 10px !important; padding: 4px !important; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #94a3b8 !important; border-radius: 8px !important; padding: 0.5rem 1rem !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background: rgba(0, 212, 255, 0.15) !important; color: #00d4ff !important; }

/* ── Alert boxes ── */
.stAlert { border-radius: 10px !important; border: none !important; }

/* ── Custom card ── */
.nt-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.nt-card:hover {
    border-color: rgba(0, 212, 255, 0.35);
    box-shadow: 0 4px 32px rgba(0, 212, 255, 0.08);
}

/* ── Status badge ── */
.status-ok   { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-block; }
.status-warn { background: rgba(250,204,21,0.15); color: #facc15; border: 1px solid rgba(250,204,21,0.3); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-block; }
.status-crit { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-block; }

/* ── Page nav cards ── */
.nav-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s ease;
    height: 140px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}
.nav-card:hover {
    border-color: rgba(0, 212, 255, 0.4);
    background: rgba(0, 212, 255, 0.06);
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0, 212, 255, 0.12);
}
.nav-card .icon { font-size: 2rem; }
.nav-card .label { font-size: 0.9rem; font-weight: 600; color: #e2e8f0; }
.nav-card .desc  { font-size: 0.72rem; color: #64748b; }

/* ── Hero section ── */
.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-top: 0.75rem;
    font-weight: 400;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    color: #00d4ff;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Section header ── */
.section-header {
    font-size: 0.7rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

/* ── Stat highlight ── */
.stat-highlight {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.stat-highlight .value {
    font-size: 2rem;
    font-weight: 800;
    color: #00d4ff;
}
.stat-highlight .label {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 4px;
}

/* ── Plotly chart background fix ── */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1425; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.5); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar (minimal)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.5rem;">
        <div style="font-size:2.5rem;">🚊</div>
        <div style="font-size:1.1rem; font-weight:800; color:#00d4ff; letter-spacing:-0.02em;">NeuroTraffic</div>
        <div style="font-size:0.65rem; color:#475569; margin-top:4px;">NYC Transport Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.page_link("app.py",                  label="\U0001f3e0  Home")
    st.page_link("pages/1_live_monitor.py", label="\U0001f4e1  Live Monitor")
    st.page_link("pages/2_simulator.py",    label="\u26a1  Simulator")
    st.page_link("pages/3_explorer.py",     label="\U0001f4ca  Explorer")
    st.page_link("pages/4_analytics.py",    label="\U0001f9e0  Analytics")

use_hdfs = st.session_state.use_hdfs


# ─────────────────────────────────────────────
# Main: Hero
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 2.5rem 0 1.5rem;">
    <h1 class="hero-title">NeuroTraffic</h1>
    <p class="hero-sub">
        Multi-modal transport disruption intelligence for New York City.<br>
        Spark Structured Streaming · Graph Analytics · ML Anomaly Detection
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# System Status Bar
# ─────────────────────────────────────────────
status_c1, status_c2, status_c3 = st.columns([2, 1, 1])
with status_c1:
    latest = hdfs_helper.load_dataframe(config.LATEST_CSV, config.HDFS_LATEST_CSV, use_hdfs=use_hdfs)
    if latest is not None:
        try:
            disruptions_live = latest[latest["is_disruption"] == 1]
            n_disrupt = len(disruptions_live)
            if n_disrupt > 0:
                top_line = disruptions_live.iloc[0]["affected_line"]
                st.markdown(f'<span class="status-crit">⚠ DISRUPTION — {top_line} line · {n_disrupt} zones</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-ok">● All Systems Normal</span>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<span class="status-ok">● Data loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">○ No live data yet</span>', unsafe_allow_html=True)
with status_c2:
    use_hdfs_new = st.toggle("HDFS Mode", value=st.session_state.use_hdfs, key="hdfs_toggle")
    if use_hdfs_new != st.session_state.use_hdfs:
        st.session_state.use_hdfs = use_hdfs_new
        config.USE_HDFS = use_hdfs_new
        st.rerun()
with status_c3:
    if use_hdfs:
        hdfs_ok = hdfs_helper.is_hdfs_available()
        if hdfs_ok:
            st.markdown('<span class="status-ok">● HDFS Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-crit">○ HDFS Offline</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="font-size:0.75rem;color:#64748b;">Local mode</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Live stats from real data
# ─────────────────────────────────────────────
nodes_df = hdfs_helper.load_dataframe(config.GRAPH_NODES_CSV, config.HDFS_GRAPH_NODES_CSV, use_hdfs=use_hdfs)
hist_df = hdfs_helper.load_dataframe(config.HISTORICAL_PARQUET, config.HDFS_HISTORICAL_PARQUET, use_hdfs=use_hdfs)

total_zones = len(nodes_df) if nodes_df is not None else 0
total_days = 0
total_disruptions = 0
peak_score = 0.0
total_records = 0

if hist_df is not None:
    if "date" not in hist_df.columns and "hourly_timestamp" in hist_df.columns:
        hist_df["date"] = pd.to_datetime(hist_df["hourly_timestamp"]).dt.date
    if "date" in hist_df.columns:
        total_days = hist_df["date"].nunique()
    total_records = len(hist_df)
    if "is_anomaly" in hist_df.columns:
        total_disruptions = int(hist_df["is_anomaly"].sum())
    if "anomaly_score" in hist_df.columns:
        peak_score = float(hist_df["anomaly_score"].max())

col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Taxi Zones", f"{total_zones}")
with col2: st.metric("Days Analyzed", f"{total_days}")
with col3: st.metric("Disruptions Found", f"{total_disruptions:,}")
with col4: st.metric("Peak Anomaly", f"{peak_score:.3f}")
with col5: st.metric("Data Sources", "6", help="Taxi, Subway, Bike, Weather, 311, Events")

# ─────────────────────────────────────────────
# Navigation Cards
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Explore</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    <a href="/live_monitor" style="text-decoration:none;">
    <div class="nav-card">
        <div class="icon">📡</div>
        <div class="label">Live Monitor</div>
        <div class="desc">Real-time NYC map · Auto-refreshes every 5 s</div>
    </div></a>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <a href="/simulator" style="text-decoration:none;">
    <div class="nav-card">
        <div class="icon">⚡</div>
        <div class="label">Disruption Simulator</div>
        <div class="desc">Pick a subway line · See cascade predictions</div>
    </div></a>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <a href="/explorer" style="text-decoration:none;">
    <div class="nav-card">
        <div class="icon">📊</div>
        <div class="label">Historical Explorer</div>
        <div class="desc">6-month trend analysis · Disruption timeline</div>
    </div></a>""", unsafe_allow_html=True)
with c4:
    st.markdown("""
    <a href="/analytics" style="text-decoration:none;">
    <div class="nav-card">
        <div class="icon">🧠</div>
        <div class="label">Analytics Dashboard</div>
        <div class="desc">Model metrics · Graph stats · Feature importance</div>
    </div></a>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Borough breakdown from real data
# ─────────────────────────────────────────────
if hist_df is not None and "borough" in hist_df.columns:
    st.markdown('<div class="section-header">Disruptions by Borough</div>', unsafe_allow_html=True)

    borough_stats = hist_df.groupby("borough").agg(
        total_days_active=("date", "nunique") if "date" in hist_df.columns else ("borough", "count"),
        avg_score=("anomaly_score", "mean"),
        peak=("anomaly_score", "max"),
        disruptions=("is_anomaly", "sum") if "is_anomaly" in hist_df.columns else ("borough", "count"),
    ).sort_values("disruptions", ascending=False)

    borough_colors = {
        "Manhattan": "#00d4ff", "Brooklyn": "#a78bfa", "Queens": "#34d399",
        "Bronx": "#fb923c", "Staten Island": "#f472b6", "EWR": "#94a3b8",
    }

    cols = st.columns(min(len(borough_stats), 5))
    for i, (borough, row) in enumerate(borough_stats.head(5).iterrows()):
        color = borough_colors.get(borough, "#94a3b8")
        with cols[i]:
            st.markdown(f"""
            <div class="stat-highlight" style="border-color: {color}33;">
                <div class="value" style="color: {color};">{int(row['disruptions']):,}</div>
                <div class="label">{borough}</div>
                <div style="font-size:0.68rem; color:#475569; margin-top:6px;">
                    Avg Score: {row['avg_score']:.3f} · Peak: {row['peak']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Pipeline Architecture
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Pipeline Architecture</div>', unsafe_allow_html=True)
st.markdown("""
<div class="nt-card">
<div style="font-family: 'Courier New', monospace; font-size: 0.78rem; color: #94a3b8; line-height: 2; overflow-x: auto;">
<span style="color:#00d4ff; font-weight:700;">NYC Open Data</span>
&nbsp;──►&nbsp;
<span style="color:#a78bfa; font-weight:700;">Spark ETL</span>&nbsp;(6 sources → cleaned Parquet)
&nbsp;──►&nbsp;
<span style="color:#34d399; font-weight:700;">GraphX</span>&nbsp;(PageRank · Communities)
&nbsp;──►&nbsp;
<span style="color:#fb923c; font-weight:700;">ML Pipeline</span>&nbsp;(Isolation Forest + GBT)
&nbsp;──►&nbsp;
<span style="color:#f472b6; font-weight:700;">Dashboard</span>&nbsp;(Streaming + Streamlit)
<br>
<span style="color:#475569;">Taxi · Subway · Bike · Weather · 311 · Events</span>
&nbsp;&nbsp;→&nbsp;&nbsp;
<span style="color:#475569;">6.9M rows HDFS</span>
&nbsp;&nbsp;→&nbsp;&nbsp;
<span style="color:#475569;">263 zones scored</span>
&nbsp;&nbsp;→&nbsp;&nbsp;
<span style="color:#475569;">Anomaly detection</span>
&nbsp;&nbsp;→&nbsp;&nbsp;
<span style="color:#475569;">Live monitoring</span>
</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data Status
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Data Status</div>', unsafe_allow_html=True)

files_to_check = [
    (config.GRAPH_NODES_CSV,         config.HDFS_GRAPH_NODES_CSV,         "graph_nodes.csv",               "Zone metadata"),
    (config.HISTORICAL_PARQUET,       config.HDFS_HISTORICAL_PARQUET,       "historical_predictions.parquet", "Historical analysis"),
    (config.EVALUATION_METRICS_JSON,  config.HDFS_EVALUATION_METRICS_JSON,  "evaluation_metrics.json",        "Model evaluation"),
    (config.FEATURE_IMPORTANCE_CSV,   config.HDFS_FEATURE_IMPORTANCE_CSV,   "feature_importance.csv",         "Feature weights"),
    (config.CONFUSION_MATRIX_CSV,     config.HDFS_CONFUSION_MATRIX_CSV,     "confusion_matrix.csv",           "Classification matrix"),
    (config.LATEST_CSV,               config.HDFS_LATEST_CSV,               "latest.csv",                     "Live predictions"),
]

hdfs_active = st.session_state.use_hdfs and hdfs_helper.is_hdfs_available()

for lpath, hpath, fname, source in files_to_check:
    if hdfs_active:
        url = f"{config.HDFS_WEB_PREFIX}{hpath}?op=GETFILESTATUS"
        try:
            r = requests.get(url, timeout=1.0)
            exists = r.status_code == 200
            if exists:
                file_status = r.json().get("FileStatus", {})
                size_bytes = file_status.get("length", 0)
                if size_bytes > 1048576:
                    size = f"{size_bytes/1048576:.1f} MB"
                else:
                    size = f"{size_bytes/1024:.1f} KB"
            else:
                size = "missing"
        except Exception:
            exists = False
            size = "conn error"
        loc = "HDFS"
    else:
        exists = os.path.exists(lpath)
        size = f"{os.path.getsize(lpath)/1024:.0f} KB" if exists else "missing"
        loc = "Local"

    icon = "✅" if exists else "❌"

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
        <span style="font-size:1rem;">{icon}</span>
        <span style="color:#e2e8f0;font-size:0.8rem;font-weight:500;flex:1;">{fname}</span>
        <span style="color:#475569;font-size:0.72rem;">{source} ({loc})</span>
        <span style="color:#64748b;font-size:0.72rem;">{size}</span>
    </div>
    """, unsafe_allow_html=True)
