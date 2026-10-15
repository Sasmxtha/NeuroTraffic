"""
NeuroTraffic — Page 1: Live Monitor
=====================================

Real-time NYC map showing anomaly scores per zone.
Reads streaming/output/latest.csv every 5 seconds.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys, time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np

NYC = ZoneInfo("America/New_York")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ── Inherit global CSS from app.py via a common include ──
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    html, body, [data-testid="stAppViewContainer"] { background: #090e1a !important; color: #e2e8f0 !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1425 0%, #111827 100%) !important; border-right: 1px solid rgba(0,212,255,0.12) !important; }
    #MainMenu, footer, header { visibility: hidden; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"] { visibility: visible !important; }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: "Material Symbols Rounded" !important; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="metric-container"] { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(0,212,255,0.15) !important; border-radius: 12px !important; padding: 1rem !important; }
    [data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.75rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 1.6rem !important; font-weight: 700 !important; }
    .stButton > button { background: linear-gradient(135deg,#00d4ff,#0ea5e9) !important; color: #090e1a !important; border: none !important; border-radius: 8px !important; font-weight: 700 !important; }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(0,212,255,0.35) !important; }
    .stSelectbox > div > div { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(0,212,255,0.2) !important; border-radius: 8px !important; color: #e2e8f0 !important; }
    .stDataFrame { border-radius: 10px !important; }
    .status-ok   { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-block; }
    .status-crit { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-block; }
    .status-warn { background: rgba(250,204,21,0.15); color: #facc15; border: 1px solid rgba(250,204,21,0.3); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-block; }
    .nt-card { background: rgba(255,255,255,0.035); border: 1px solid rgba(0,212,255,0.12); border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem; }
    ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0d1425; } ::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(
    page_title="Live Monitor — NeuroTraffic",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
import config
import hdfs_helper
inject_css()


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
@st.cache_data(ttl=3)
def load_latest(use_hdfs):
    return hdfs_helper.load_dataframe(config.LATEST_CSV, config.HDFS_LATEST_CSV, use_hdfs=use_hdfs)

def load_status(use_hdfs):
    return hdfs_helper.load_json(config.SOURCE_STATUS_JSON, config.HDFS_SOURCE_STATUS_JSON, use_hdfs=use_hdfs)

def format_age(minutes):
    if minutes is None:
        return "—"
    if minutes < 1:
        return "< 1 min"
    if minutes < 120:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.0f} h"

@st.cache_data(ttl=60)
def load_nodes(use_hdfs):
    return hdfs_helper.load_dataframe(config.GRAPH_NODES_CSV, config.HDFS_GRAPH_NODES_CSV, use_hdfs=use_hdfs)

def get_color_hex(score):
    if score >= config.ANOMALY_THRESHOLD_RED:    return "#ef4444"
    if score >= config.ANOMALY_THRESHOLD_YELLOW: return "#facc15"
    return "#22c55e"

def get_status_label(score):
    if score >= config.ANOMALY_THRESHOLD_RED:    return "CRITICAL"
    if score >= config.ANOMALY_THRESHOLD_YELLOW: return "ANOMALY"
    return "NORMAL"

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1.5rem;">
        <div style="font-size:2.5rem;">📡</div>
        <div style="font-size:1rem;font-weight:800;color:#00d4ff;">Live Monitor</div>
        <div style="font-size:0.65rem;color:#475569;margin-top:4px;">Real-time anomaly detection</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.page_link("app.py",                  label="\U0001f3e0  Home")
    st.page_link("pages/1_live_monitor.py", label="\U0001f4e1  Live Monitor")
    st.page_link("pages/2_simulator.py",    label="\u26a1  Simulator")
    st.page_link("pages/3_explorer.py",     label="\U0001f4ca  Explorer")
    st.page_link("pages/4_analytics.py",    label="\U0001f9e0  Analytics")

# ── Data loading (before main content) ──
use_hdfs = st.session_state.get("use_hdfs", config.USE_HDFS)
latest_df = load_latest(use_hdfs)
nodes_df  = load_nodes(use_hdfs)
status    = load_status(use_hdfs)
if status and isinstance(status, dict) and "generated_at" in status and datetime.now(NYC) - datetime.fromisoformat(status["generated_at"]) > timedelta(hours=24):
    status = None

if latest_df is None or nodes_df is None:
    st.warning("No live data found. Start 07_streaming/realtime_fetcher.py and 07_streaming/streaming_pipeline.scala.")
    st.stop()

node_cols = ["zone_id","lat","lon","name","borough","page_rank"]
if "primary_line" in nodes_df.columns:
    node_cols.append("primary_line")
merged = latest_df.merge(nodes_df[node_cols], on="zone_id", how="left")
if "primary_line" not in merged.columns:
    merged["primary_line"] = "-"
disruptions = merged[merged["is_disruption"] == 1].sort_values("anomaly_score", ascending=False)
anomalies   = merged[merged["anomaly_score"] >= config.ANOMALY_THRESHOLD_YELLOW].sort_values("anomaly_score", ascending=False)
n_disrupt = len(disruptions)


# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────
# Header with inline controls
col_title, col_ctrl1, col_ctrl2, col_time = st.columns([3, 0.8, 0.8, 1.2])
with col_title:
    if n_disrupt > 0:
        top_line = disruptions.iloc[0]["affected_line"]
        st.markdown(f"""
        <h1 style="font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#ef4444,#f97316);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0;">
        📡 Live Monitor <span style="font-size:0.7rem;background:rgba(239,68,68,0.15);color:#ef4444;padding:3px 10px;border-radius:20px;border:1px solid rgba(239,68,68,0.3);vertical-align:middle;-webkit-text-fill-color:#ef4444;">⚠ DISRUPTION — {top_line}</span>
        </h1>
        <p style="color:#94a3b8;font-size:0.8rem;margin:0.3rem 0 0;">
        {n_disrupt} disruption zones · {len(anomalies)} anomalies detected across 263 NYC taxi zones
        </p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <h1 style="font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#00d4ff,#a78bfa);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0;">
        📡 Live Monitor <span style="font-size:0.7rem;background:rgba(34,197,94,0.15);color:#22c55e;padding:3px 10px;border-radius:20px;border:1px solid rgba(34,197,94,0.3);vertical-align:middle;-webkit-text-fill-color:#22c55e;">● Normal</span>
        </h1>
        <p style="color:#64748b;font-size:0.8rem;margin:0.3rem 0 0;">
        Real-time anomaly detection across 263 NYC taxi zones
        </p>
        """, unsafe_allow_html=True)
with col_ctrl1:
    auto_refresh = st.toggle("Auto Refresh", value=True)
with col_ctrl2:
    refresh_interval = st.select_slider(
        "Interval (s)", options=[3, 5, 10, 15, 30], value=5
    )
with col_time:
    data_ts = pd.to_datetime(latest_df["timestamp"].max())
    if status:
        lag = (datetime.now(NYC) - data_ts.tz_localize(NYC)).total_seconds()
        caption = f"NYC time of latest data · {lag:.0f}s ago"
    else:
        caption = "Timestamp of latest data (live fetcher not running)"
    st.markdown(f"""
    <div style="text-align:right;padding-top:0.5rem;">
        <div style="font-size:1.4rem;font-weight:700;color:#00d4ff;font-variant-numeric:tabular-nums;">{data_ts:%H:%M:%S}</div>
        <div style="font-size:0.65rem;color:#475569;">{caption}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Quick stats row ──
if latest_df is not None and nodes_df is not None:
    total = len(merged)
    n_red    = int((merged["anomaly_score"] >= 0.80).sum())
    n_yellow = int(((merged["anomaly_score"] >= 0.50) & (merged["anomaly_score"] < 0.80)).sum())
    n_green  = total - n_red - n_yellow
    avg_score = merged["anomaly_score"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Zones",   f"{total}")
    with c2: st.metric("🟢 Normal",     f"{n_green}",  delta=None)
    with c3: st.metric("🟡 Anomaly",    f"{n_yellow}", delta=None)
    with c4: st.metric("🔴 Disruption", f"{n_red}",    delta=None)
    with c5: st.metric("Avg Score",     f"{avg_score:.3f}")

    # ── Map ──
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    merged["color_label"] = merged["anomaly_score"].apply(get_status_label)
    merged["size_scaled"] = merged["anomaly_score"].clip(0.05, 1.0) * 14 + 4
    merged["hover_text"]  = (
        "<b>" + merged["name"].fillna("Zone " + merged["zone_id"].astype(str)) + "</b><br>" +
        "Anomaly Score: " + merged["anomaly_score"].map("{:.3f}".format) + "<br>" +
        "Borough: " + merged["borough"].fillna("Unknown") + "<br>" +
        "Subway Line: " + merged["affected_line"].fillna("-") + "<br>" +
        "MTA alert severity: " + merged["disruption_severity"].fillna(0).map("{:.1f}".format) + "<br>" +
        "PageRank: " + merged["page_rank"].map("{:.4f}".format)
    )

    color_map = {"NORMAL": "#22c55e", "ANOMALY": "#facc15", "CRITICAL": "#ef4444"}

    fig_map = go.Figure()

    # Draw by status level (green first so red appears on top)
    for level in ["NORMAL", "ANOMALY", "CRITICAL"]:
        subset = merged[merged["color_label"] == level]
        if len(subset) == 0:
            continue
        fig_map.add_trace(go.Scattermap(
            lat=subset["lat"],
            lon=subset["lon"],
            mode="markers",
            name=level,
            marker=dict(
                size=subset["size_scaled"],
                color=color_map[level],
                opacity=0.85 if level != "CRITICAL" else 1.0,
            ),
            text=subset["hover_text"],
            hoverinfo="text",
            hovertemplate="%{text}<extra></extra>",
        ))

    # Pulsing ring for disruption zones
    crit = merged[merged["color_label"] == "CRITICAL"]
    if len(crit) > 0:
        fig_map.add_trace(go.Scattermap(
            lat=crit["lat"],
            lon=crit["lon"],
            mode="markers",
            name="Disruption Ring",
            marker=dict(size=crit["size_scaled"] * 2.2, color="rgba(239,68,68,0.18)"),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig_map.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=config.MAP_CENTER,
            zoom=config.MAP_ZOOM,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=0.01,
            xanchor="right",  x=0.99,
            bgcolor="rgba(9,14,26,0.8)",
            bordercolor="rgba(0,212,255,0.2)",
            borderwidth=1,
            font=dict(color="#e2e8f0", size=11),
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Bottom row: line chart + disruption table ──
    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        st.markdown("**📈 Anomaly Score Distribution (last 60 zones by zone_id)**")
        sample = merged.sort_values("zone_id").tail(60).copy()
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=sample["zone_id"],
            y=sample["anomaly_score"],
            mode="lines+markers",
            line=dict(color="#00d4ff", width=1.5),
            marker=dict(
                color=sample["anomaly_score"].apply(
                    lambda s: "#ef4444" if s >= 0.8 else ("#facc15" if s >= 0.5 else "#22c55e")
                ),
                size=5,
            ),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.06)",
            hovertemplate="Zone %{x}<br>Score: %{y:.3f}<extra></extra>",
        ))
        fig_line.add_hline(y=0.8, line_dash="dash", line_color="rgba(239,68,68,0.5)", annotation_text="Critical (0.8)")
        fig_line.add_hline(y=0.5, line_dash="dot",  line_color="rgba(250,204,21,0.5)", annotation_text="Anomaly (0.5)")
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=10, t=10, b=0), height=220,
            xaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)", title="Zone ID"),
            yaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)", range=[0,1], title="Anomaly Score"),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col_table:
        st.markdown("**🚨 Active Disruptions**")
        if n_disrupt > 0:
            show_cols = ["name","anomaly_score","affected_line","disruption_severity"]
            labels    = {"name":"Zone","anomaly_score":"Score","affected_line":"Line",
                         "disruption_severity":"MTA severity"}
            disp = disruptions[show_cols].rename(columns=labels).head(10)
            disp["Score"] = disp["Score"].map("{:.3f}".format)
            st.dataframe(disp, use_container_width=True, hide_index=True, height=210)
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#22c55e;">
                <div style="font-size:2rem;">✅</div>
                <div style="font-size:0.9rem;margin-top:0.5rem;">All zones operating normally</div>
            </div>""", unsafe_allow_html=True)

    # Top anomalies (moved from sidebar)
    if len(anomalies) > 0:
        st.markdown("**🔴 Top Anomaly Zones**")
        top_anom = anomalies.head(8)[["name","anomaly_score","affected_line","borough","disruption_severity"]].copy()
        top_anom.columns = ["Zone", "Score", "Line", "Borough", "MTA Severity"]
        top_anom["Score"] = top_anom["Score"].map("{:.3f}".format)
        st.dataframe(top_anom, use_container_width=True, hide_index=True)


    if status and isinstance(status, dict) and "sources" in status:
        st.markdown("**📶 Data sources — what is live and how old it is**")
        now_nyc = datetime.now(NYC)
        rows = []
        for s in status["sources"]:
            obs = s.get("observed_at")
            age = (now_nyc - datetime.fromisoformat(obs)).total_seconds() / 60 if obs else None
            rows.append({
                "Source": s["name"],
                "Type": s["kind"].upper(),
                "Age": format_age(age),
                "Provider": s["provider"],
                "Detail": f"ERROR: {s['error']}" if s.get("error") else s["detail"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        alerts = status.get("mta_alerts") or []
        if alerts:
            st.markdown("**🚇 Active unplanned MTA alerts**")
            for a in alerts:
                st.markdown(f"- **{', '.join(a['lines'])}** · {a['type']} — {a['header']}")

# ── Auto-refresh ──
if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()
