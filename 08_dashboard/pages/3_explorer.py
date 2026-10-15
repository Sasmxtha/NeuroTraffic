"""
NeuroTraffic — Page 3: Explorer
================================

Live streaming insights (primary) + historical analysis (secondary)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, sys
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

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
    .stDateInput > div { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(0,212,255,0.2) !important; border-radius: 8px !important; }
    .stDataFrame { border-radius: 10px !important; }
    .stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.03) !important; border-radius: 10px !important; padding: 4px !important; }
    .stTabs [data-baseweb="tab"] { background: transparent !important; color: #94a3b8 !important; border-radius: 8px !important; }
    .stTabs [aria-selected="true"] { background: rgba(0,212,255,0.15) !important; color: #00d4ff !important; }
    .nt-card { background: rgba(255,255,255,0.035); border: 1px solid rgba(0,212,255,0.12); border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem; }
    ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0d1425; } ::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }
    </style>""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Explorer — NeuroTraffic",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)
import config
import hdfs_helper
inject_css()

# ── Minimal sidebar (no filters) ──
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1.5rem;">
        <div style="font-size:2.5rem;">\U0001f4ca</div>
        <div style="font-size:1rem;font-weight:800;color:#00d4ff;">Explorer</div>
        <div style="font-size:0.65rem;color:#475569;margin-top:4px;">Live + Historical \u00b7 263 zones</div>
    </div>""", unsafe_allow_html=True)
    st.page_link("app.py", label="\u2190 Back to Home")

# ── Data loading ──
@st.cache_resource(ttl=300)
def load_historical(use_hdfs):
    df = hdfs_helper.load_dataframe(config.HISTORICAL_PARQUET, config.HDFS_HISTORICAL_PARQUET, use_hdfs=use_hdfs)
    if df is not None and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=300)
def load_nodes(use_hdfs):
    return hdfs_helper.load_dataframe(config.GRAPH_NODES_CSV, config.HDFS_GRAPH_NODES_CSV, use_hdfs=use_hdfs)

@st.cache_data(ttl=60)
def load_live_history(use_hdfs):
    hist_path = os.path.join(config.STREAMING_DIR, "output", "live_history.csv")
    df = hdfs_helper.load_dataframe(hist_path, config.HDFS_LIVE_HISTORY_CSV, use_hdfs=use_hdfs)
    if df is not None and "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_data(ttl=10)
def load_live_data(use_hdfs):
    df = hdfs_helper.load_dataframe(config.LATEST_CSV, config.HDFS_LATEST_CSV, use_hdfs=use_hdfs)
    if df is not None and "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"])
    return df

DISRUPTION_EVENTS = {
    "2023-01-17": "L train \u2014 Full suspension",
    "2023-02-03": "A/C \u2014 Signal failure",
    "2023-02-28": "4/5 \u2014 Track fire",
    "2023-03-14": "L train \u2014 Power outage",
    "2023-04-07": "N/Q \u2014 Signal malfunction",
    "2023-04-22": "1 train \u2014 Suspension",
    "2023-05-11": "7 train \u2014 Derailment",
    "2023-06-01": "J/Z \u2014 Service disruption",
    "2023-06-15": "G train \u2014 Track work",
}

use_hdfs = st.session_state.get("use_hdfs", config.USE_HDFS)
hist_df  = load_historical(use_hdfs)
nodes_df = load_nodes(use_hdfs)
if nodes_df is not None and "primary_line" not in nodes_df.columns:
    nodes_df["primary_line"] = "-"

# ── Page header ──
st.markdown("""
<h1 style="font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#00d4ff,#a78bfa);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 0.2rem;">
\U0001f4ca Explorer
</h1>
<p style="color:#64748b;font-size:0.8rem;margin:0 0 1rem;">
Live streaming insights \u00b7 real-time anomaly detection \u00b7 263 NYC zones
</p>""", unsafe_allow_html=True)


# =============================================
# SECTION 1: LIVE INSIGHTS (PRIMARY)
# =============================================
st.markdown("""
<div style="background:linear-gradient(135deg,rgba(239,68,68,0.08),rgba(0,212,255,0.05));
border:1px solid rgba(239,68,68,0.25);border-radius:14px;padding:1rem 1.2rem;margin-bottom:1rem;">
<span style="font-size:0.7rem;background:rgba(239,68,68,0.2);color:#ef4444;padding:3px 10px;
border-radius:20px;font-weight:700;letter-spacing:0.5px;">LIVE</span>
<span style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-left:0.6rem;">
Streaming Data \u2014 Current NYC Anomaly Snapshot</span>
</div>""", unsafe_allow_html=True)

live_df = load_live_data(use_hdfs)
if live_df is None or len(live_df) == 0:
    st.warning("No live data available. Make sure the realtime_fetcher and spark_live_processor are running.")
    st.code("python 07_streaming/realtime_fetcher.py --interval 30\npython 07_streaming/spark_live_processor.py --interval 15", language="bash")
else:
    if nodes_df is not None:
        live_merged = live_df.merge(nodes_df[["zone_id","name","borough","lat","lon","primary_line"]], on="zone_id", how="left")
    else:
        live_merged = live_df.copy()

    ts_str = live_df["timestamp"].iloc[0] if "timestamp" in live_df.columns else "Unknown"
    n_live = len(live_merged)
    n_disruptions = int((live_merged["anomaly_score"] >= 0.80).sum())
    n_anomalies = int(((live_merged["anomaly_score"] >= 0.50) & (live_merged["anomaly_score"] < 0.80)).sum())
    peak_live = live_merged["anomaly_score"].max()
    avg_live = live_merged["anomaly_score"].mean()

    lc1, lc2, lc3, lc4, lc5 = st.columns(5)
    with lc1: st.metric("Zones", f"{n_live}")
    with lc2: st.metric("Disruptions", f"{n_disruptions}", delta=f"{n_disruptions}" if n_disruptions > 0 else None, delta_color="inverse")
    with lc3: st.metric("Anomalies", f"{n_anomalies}")
    with lc4: st.metric("Peak Score", f"{peak_live:.3f}")
    with lc5: st.metric("Avg Score", f"{avg_live:.3f}")

    st.markdown(f'<div style="color:#64748b;font-size:0.7rem;margin:-0.5rem 0 1rem;">Snapshot: {ts_str} \u00b7 Auto-refreshes every 10 seconds</div>', unsafe_allow_html=True)

    col_live_map, col_live_detail = st.columns([3, 2])

    with col_live_map:
        st.markdown("**Live Anomaly Map**")
        if "lat" in live_merged.columns:
            def score_color(s):
                if s >= 0.80: return "#ef4444"
                elif s >= 0.50: return "#f97316"
                elif s >= 0.25: return "#facc15"
                return "#22c55e"
            live_merged["color"] = live_merged["anomaly_score"].apply(score_color)
            live_merged["size"] = (live_merged["anomaly_score"] * 15 + 4).clip(4, 20)

            fig_live = go.Figure()
            for label, color, filt in [("Normal","#22c55e", live_merged["anomaly_score"] < 0.25),
                                       ("Elevated","#facc15", (live_merged["anomaly_score"] >= 0.25) & (live_merged["anomaly_score"] < 0.50)),
                                       ("Anomaly","#f97316", (live_merged["anomaly_score"] >= 0.50) & (live_merged["anomaly_score"] < 0.80)),
                                       ("Disruption","#ef4444", live_merged["anomaly_score"] >= 0.80)]:
                sub = live_merged[filt]
                if len(sub) == 0: continue
                fig_live.add_trace(go.Scattermap(
                    lat=sub["lat"], lon=sub["lon"], mode="markers", name=f"{label} ({len(sub)})",
                    marker=dict(size=sub["size"], color=color, opacity=0.85),
                    text=sub["name"].fillna("Zone " + sub["zone_id"].astype(str)) + "<br>Score: " + sub["anomaly_score"].map("{:.3f}".format),
                    hovertemplate="%{text}<extra></extra>",
                ))
            fig_live.update_layout(
                map=dict(style="carto-darkmatter", center=config.MAP_CENTER, zoom=config.MAP_ZOOM),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=0,t=0,b=0), height=420,
                legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=0.99,
                            bgcolor="rgba(9,14,26,0.8)", bordercolor="rgba(0,212,255,0.2)", borderwidth=1,
                            font=dict(color="#e2e8f0", size=11)),
            )
            st.plotly_chart(fig_live, use_container_width=True)

    with col_live_detail:
        st.markdown("**Top Disrupted Zones (live)**")
        top_live = live_merged.nlargest(15, "anomaly_score")
        show_cols = ["name","anomaly_score","affected_line"]
        if "borough" in top_live.columns: show_cols.insert(1, "borough")
        display = top_live[show_cols].copy()
        display.columns = ["Zone","Borough","Score","Line"] if "borough" in show_cols else ["Zone","Score","Line"]
        display["Score"] = display["Score"].map("{:.3f}".format)
        display["Line"] = display["Line"].replace("0", "-").replace(0, "-").fillna("-")
        st.dataframe(display, use_container_width=True, hide_index=True, height=300)

        st.markdown("**Borough Breakdown (live)**")
        if "borough" in live_merged.columns:
            bor_live = live_merged.groupby("borough").agg(
                avg_score=("anomaly_score","mean"),
                disruptions=("anomaly_score", lambda x: int((x >= 0.80).sum())),
            ).reset_index().sort_values("avg_score", ascending=False)
            st.dataframe(bor_live.rename(columns={"avg_score":"Avg Score","disruptions":"Disruptions"}),
                         use_container_width=True, hide_index=True)

    # Rolling history timeline
    live_hist = load_live_history(use_hdfs)
    if live_hist is not None and len(live_hist) > 10:
        st.markdown("---")
        st.markdown("**Live Anomaly Timeline (rolling history)**")
        lh = live_hist.copy()
        lh["date"] = pd.to_datetime(lh["timestamp"])

        range_col1, range_col2 = st.columns([2, 3])
        with range_col1:
            time_range = st.selectbox("View Range", ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours"], index=2, key="live_range")
        range_map = {"Last 1 Hour": 1/24, "Last 6 Hours": 6/24, "Last 24 Hours": 1}
        days_back = range_map[time_range]
        # History timestamps are NYC local time, so compare against NYC "now"
        nyc_now = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
        cutoff = nyc_now - pd.Timedelta(days=days_back)
        hist_min, hist_max = lh["date"].min(), lh["date"].max()
        lh = lh[lh["date"] >= cutoff]

        with range_col2:
            st.metric("Data Points", f"{len(lh):,}")

        if len(lh) == 0:
            st.info(f"No live data in this range. History covers {hist_min:%b %d %H:%M} - {hist_max:%b %d %H:%M} (NYC time). Run the streaming pipeline to collect more.")
        if len(lh) > 0:
            freq = "1min" if days_back <= 1/24 else "5min"

            timeline = lh.groupby(lh["date"].dt.floor(freq)).agg(
                avg_score=("anomaly_score", "mean"),
                max_score=("anomaly_score", "max"),
                n_disruptions=("is_disruption", "sum") if "is_disruption" in lh.columns else ("anomaly_score", lambda x: int((x >= 0.80).sum())),
                n_anomalies=("anomaly_score", lambda x: int((x >= 0.50).sum())),
                n_zones=("zone_id", "nunique"),
            ).reset_index()

            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(x=timeline["date"], y=timeline["avg_score"],
                name="Avg Score", line=dict(color="#00d4ff", width=2),
                fill="tozeroy", fillcolor="rgba(0,212,255,0.06)"))
            fig_tl.add_trace(go.Scatter(x=timeline["date"], y=timeline["max_score"],
                name="Peak Score", line=dict(color="#f97316", width=1.5, dash="dot")))
            if "n_disruptions" in timeline.columns:
                fig_tl.add_trace(go.Bar(x=timeline["date"], y=timeline["n_disruptions"],
                    name="Disruptions", marker_color="rgba(239,68,68,0.4)",
                    yaxis="y2", width=60000))
            fig_tl.add_hline(y=0.8, line_dash="dash", line_color="rgba(239,68,68,0.4)", annotation_text="Critical (0.8)")
            fig_tl.add_hline(y=0.5, line_dash="dot", line_color="rgba(250,204,21,0.35)", annotation_text="Anomaly (0.5)")
            fig_tl.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=0,t=10,b=0), height=300,
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                            bgcolor="rgba(9,14,26,0.8)", font=dict(color="#e2e8f0",size=11)),
                xaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)", range=[0,1.05], title="Score"),
                yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Count", color="#ef4444"),
            )
            st.plotly_chart(fig_tl, use_container_width=True)

            st.markdown(f"**{time_range} Summary**")
            sum1, sum2, sum3, sum4 = st.columns(4)
            total_disruptions = int(timeline["n_disruptions"].sum()) if "n_disruptions" in timeline.columns else 0
            total_anomalies = int(timeline["n_anomalies"].sum()) if "n_anomalies" in timeline.columns else 0
            sum1.metric("Avg Score", f"{lh['anomaly_score'].mean():.3f}")
            sum2.metric("Peak Score", f"{lh['anomaly_score'].max():.3f}")
            sum3.metric("Disruption Events", f"{total_disruptions:,}")
            sum4.metric("Anomaly Events", f"{total_anomalies:,}")

            st.caption(f"Showing {len(timeline)} snapshots ({freq} resolution) \u00b7 History keeps the last 24 hours")

    # Live vs Historical
    if hist_df is not None:
        st.markdown("---")
        st.markdown("**Live vs Historical Comparison**")
        hist_zone_avg = hist_df.groupby("zone_id")["anomaly_score"].mean().reset_index()
        hist_zone_avg.columns = ["zone_id", "hist_avg"]
        compare = live_merged[["zone_id","anomaly_score"]].merge(hist_zone_avg, on="zone_id", how="left")
        compare["diff"] = compare["anomaly_score"] - compare["hist_avg"].fillna(0)
        if nodes_df is not None:
            compare = compare.merge(nodes_df[["zone_id","name","borough"]], on="zone_id", how="left")
        biggest_change = compare.nlargest(10, "diff")
        if len(biggest_change) > 0:
            fig_comp = go.Figure()
            x_labels = biggest_change["name"].fillna("Zone " + biggest_change["zone_id"].astype(str))
            fig_comp.add_trace(go.Bar(x=x_labels, y=biggest_change["hist_avg"].fillna(0),
                name="Historical Avg", marker_color="#0ea5e9"))
            fig_comp.add_trace(go.Bar(x=x_labels, y=biggest_change["anomaly_score"],
                name="Live Now", marker_color="#ef4444"))
            fig_comp.update_layout(
                barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=0,t=10,b=0), height=280,
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                            bgcolor="rgba(9,14,26,0.8)", font=dict(color="#e2e8f0",size=11)),
                xaxis=dict(color="#475569", tickangle=-45),
                yaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)", title="Anomaly Score"),
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            st.caption("Zones with the biggest increase vs their 2023 historical average")


# =============================================
# SECTION 2: HISTORICAL ANALYSIS (SECONDARY)
# =============================================
st.markdown("---")
st.markdown("""
<div style="margin:1.5rem 0 0.8rem;">
<span style="font-size:0.65rem;background:rgba(0,212,255,0.12);color:#0ea5e9;padding:3px 10px;
border-radius:20px;font-weight:600;letter-spacing:0.5px;">HISTORICAL</span>
<span style="font-size:1rem;font-weight:600;color:#94a3b8;margin-left:0.6rem;">
2023 Data Analysis</span>
</div>""", unsafe_allow_html=True)

if hist_df is None:
    st.info("Historical data not loaded. Run `spark-shell -i /project/08_dashboard/generate_dashboard_data.scala` on the cluster.")
else:
    # Use fixed date range (full dataset)
    date_start = date(2023, 1, 1)
    date_end = date(2023, 6, 30)
    mask = (hist_df["date"] >= pd.Timestamp(date_start)) & (hist_df["date"] <= pd.Timestamp(date_end))
    filtered = hist_df[mask].copy()
    agg_key = "date"
    daily = filtered.groupby(agg_key).agg(
        avg_anomaly=("anomaly_score", "mean"),
        max_anomaly=("anomaly_score", "max"),
        n_disruptions=("is_anomaly", "sum"),
        avg_taxi=("taxi_pickups", "mean"),
        avg_subway=("subway_ridership", "mean"),
        avg_bike=("bike_starts", "mean"),
    ).reset_index()
    daily.rename(columns={agg_key: "period"}, inplace=True)

    n_days = (pd.Timestamp(date_end) - pd.Timestamp(date_start)).days + 1
    n_zones = filtered["zone_id"].nunique()
    total_anomalies = int(filtered["is_anomaly"].sum())
    peak_score = filtered["anomaly_score"].max()

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Days", f"{n_days}")
    with c2: st.metric("Zones", f"{n_zones}")
    with c3: st.metric("Disruptions", f"{total_anomalies:,}")
    with c4: st.metric("Peak Score", f"{peak_score:.3f}")

    tab1, tab2, tab3, tab4 = st.tabs(["\U0001f4c8 Anomaly Timeline", "\U0001f525 Zone Heatmap", "\U0001f4cb Disruption Log", "\U0001f504 Predicted vs Actual"])

    with tab1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.65, 0.35], vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=daily["period"], y=daily["avg_anomaly"],
            name="Avg Anomaly Score", line=dict(color="#00d4ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
            hovertemplate="%{x|%b %d}<br>Avg Score: %{y:.3f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=daily["period"], y=daily["max_anomaly"],
            name="Peak Score", line=dict(color="#f97316", width=1.5, dash="dot"),
            hovertemplate="%{x|%b %d}<br>Peak: %{y:.3f}<extra></extra>"), row=1, col=1)

        # Trend
        if len(daily) > 3:
            xs = np.arange(len(daily))
            z = np.polyfit(xs, daily["avg_anomaly"].values, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(x=daily["period"], y=p(xs), name="Trend",
                line=dict(color="rgba(167,139,250,0.6)", width=1.5, dash="dash"),
                hoverinfo="skip"), row=1, col=1)

        for event_date, event_desc in DISRUPTION_EVENTS.items():
            evt = pd.Timestamp(event_date)
            if pd.Timestamp(date_start) <= evt <= pd.Timestamp(date_end):
                fig.add_vline(x=evt, line_dash="dash", line_color="rgba(239,68,68,0.4)", line_width=1.5, row=1, col=1)
                fig.add_annotation(x=evt, y=0.95, yref="y",
                    text=event_desc.split("\u2014")[0].strip(),
                    font=dict(size=9, color="#ef4444"), showarrow=False, textangle=-90, xshift=6, row=1, col=1)

        fig.add_hrect(y0=0.8, y1=1.05, fillcolor="rgba(239,68,68,0.05)", line_width=0, row=1, col=1)
        fig.add_hline(y=0.8, line_dash="dash", line_color="rgba(239,68,68,0.4)", annotation_text="Critical (0.8)", row=1, col=1)
        fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(250,204,21,0.35)", annotation_text="Anomaly (0.5)", row=1, col=1)

        fig.add_trace(go.Bar(x=daily["period"], y=daily["n_disruptions"], name="Disruption Count",
            marker_color="#ef4444", marker_opacity=0.7,
            hovertemplate="%{x|%b %d}<br>Disruptions: %{y:,}<extra></extra>"), row=2, col=1)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=0), height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                        bgcolor="rgba(9,14,26,0.8)", font=dict(color="#e2e8f0",size=11)),
            xaxis2=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)", range=[0,1.05], title="Anomaly Score"),
            yaxis2=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)", title="# Disruptions"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("**Disruption Heatmap by Zone (top 40)**")
        zone_summary = filtered.groupby("zone_id").agg(
            avg_score=("anomaly_score","mean"), n_disruptions=("is_anomaly","sum")).reset_index()
        if "zone_name" in filtered.columns:
            zone_names = filtered[["zone_id","zone_name"]].drop_duplicates().rename(columns={"zone_name":"name"})
            zone_summary = zone_summary.merge(zone_names, on="zone_id", how="left")
        if "name" not in zone_summary.columns and nodes_df is not None:
            zone_summary = zone_summary.merge(nodes_df[["zone_id","name"]], on="zone_id", how="left")
        if "name" not in zone_summary.columns:
            zone_summary["name"] = "Zone " + zone_summary["zone_id"].astype(str)
        if "borough" not in zone_summary.columns:
            if "borough" in filtered.columns:
                bor = filtered[["zone_id","borough"]].drop_duplicates()
                zone_summary = zone_summary.merge(bor, on="zone_id", how="left")
            elif nodes_df is not None:
                zone_summary = zone_summary.merge(nodes_df[["zone_id","borough"]], on="zone_id", how="left")
            else:
                zone_summary["borough"] = "Unknown"
        top40 = zone_summary.nlargest(40, "avg_score")
        fig_heat = px.bar(top40.sort_values("avg_score", ascending=True),
            x="avg_score", y="name", color="n_disruptions",
            color_continuous_scale=[[0,"#1e293b"],[0.5,"#f97316"],[1,"#ef4444"]],
            labels={"avg_score":"Avg Anomaly Score","name":"Zone","n_disruptions":"# Disruptions"},
            orientation="h", hover_data=["borough","n_disruptions"])
        fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=0), height=600,
            xaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(color="#e2e8f0", tickfont=dict(size=10)),
            coloraxis_colorbar=dict(title="# Disruptions", tickfont=dict(color="#94a3b8")))
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.markdown("**Disruption Event Log**")
        disrupt_log = []
        for event_date, event_desc in DISRUPTION_EVENTS.items():
            evt = pd.Timestamp(event_date)
            day_data = filtered[filtered["date"] == evt]
            n_aff = int(day_data["is_anomaly"].sum())
            peak = float(day_data["anomaly_score"].max()) if len(day_data) > 0 else 0
            disrupt_log.append({"Date": event_date, "Event": event_desc,
                "Zones Affected": n_aff, "Peak Score": f"{peak:.3f}", "Day": evt.strftime("%A")})
        if disrupt_log:
            st.dataframe(pd.DataFrame(disrupt_log), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.markdown("**Export filtered data**")
        csv_data = filtered[["zone_id","date","anomaly_score","is_anomaly","taxi_pickups",
                              "subway_ridership","bike_starts","affected_line"]].to_csv(index=False)
        st.download_button("\u2b07 Download CSV", csv_data, file_name="neurotraffic_export.csv", mime="text/csv")

    with tab4:
        st.markdown("**Model Accuracy: Predicted vs Actual Disruptions**")
        comparison = filtered[filtered["is_anomaly"].isin([0,1])].copy()
        if comparison["prediction"].dtype == object:
            comparison["_pred"] = comparison["prediction"].isin(["ANOMALY", "CRITICAL"]).astype(int)
        else:
            comparison["_pred"] = comparison["prediction"].astype(int)
        tp = int(((comparison["_pred"] == 1) & (comparison["is_anomaly"] == 1)).sum())
        fp = int(((comparison["_pred"] == 1) & (comparison["is_anomaly"] == 0)).sum())
        fn = int(((comparison["_pred"] == 0) & (comparison["is_anomaly"] == 1)).sum())
        tn = int(((comparison["_pred"] == 0) & (comparison["is_anomaly"] == 0)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)
        acc = (tp + tn) / max(1, tp + fp + fn + tn)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Precision", f"{precision:.3f}")
        with c2: st.metric("Recall", f"{recall:.3f}")
        with c3: st.metric("F1 Score", f"{f1:.3f}")
        with c4: st.metric("Accuracy", f"{acc:.3f}")

        scatter_data = comparison[comparison["predicted_surge_pct"] > 0].copy()
        if len(scatter_data) > 0:
            fig_scatter = px.scatter(
                scatter_data.sample(min(2000, len(scatter_data))),
                x="predicted_surge_pct", y="anomaly_score", color="is_anomaly",
                color_discrete_map={0:"#22c55e", 1:"#ef4444"},
                labels={"predicted_surge_pct":"Predicted Surge %","anomaly_score":"Actual Anomaly Score","is_anomaly":"Ground Truth"},
                title="Predicted Surge % vs Actual Anomaly Score", opacity=0.6)
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=0,t=40,b=0), height=380,
                xaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)"),
                font=dict(color="#94a3b8"))
            st.plotly_chart(fig_scatter, use_container_width=True)
