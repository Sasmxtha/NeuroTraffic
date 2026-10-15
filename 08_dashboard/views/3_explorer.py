"""
NeuroTraffic — Explorer
========================
Two views over time:
  1. Last 24 hours  — the live scores kept by live_pipeline.scala (live_history.csv, one snapshot per minute)
  2. 2022 – 2024    — the batch results per zone per day (historical_predictions.parquet)
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
import hdfs_helper
import ui

ui.setup_page("Explorer", "📊",
              "How anomaly scores change over time: the last 24 hours of live data, and three years of history "
              "(2022 – 2024) by day, zone and borough.")
use_hdfs = ui.use_hdfs()


@st.cache_data(ttl=60)
def load_live_history(use_hdfs):
    local = os.path.join(config.STREAMING_OUTPUT_DIR, "live_history.csv")
    df = hdfs_helper.load_dataframe(local, config.HDFS_LIVE_HISTORY_CSV, use_hdfs=use_hdfs)
    if df is not None:
        df["time"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_data(ttl=600)
def load_history(use_hdfs):
    df = hdfs_helper.load_dataframe(config.HISTORICAL_PARQUET, config.HDFS_HISTORICAL_PARQUET, use_hdfs=use_hdfs)
    if df is not None:
        df["date"] = pd.to_datetime(df["date"])
    return df


tab_live, tab_hist = st.tabs(["🕒 Last 24 hours (live)", "📅 History 2022 – 2024"])

# ── 1. Last 24 hours ──────────────────────────────────────────────────
with tab_live:
    live = load_live_history(use_hdfs)
    ui.section("Live scores over time",
               "Every minute the live pipeline stores all 263 zone scores; anything older than 24 hours is "
               "deleted. Times are New York time.", "live")
    if live is None or len(live) == 0:
        st.info("No live history yet. It fills up while the fetcher and live_pipeline.scala are running.")
    else:
        hours = st.radio("Show the last", [1, 6, 24], index=2, horizontal=True, format_func=lambda h: f"{h} h")
        nyc_now = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
        window = live[live["time"] >= nyc_now - pd.Timedelta(hours=hours)]
        if len(window) == 0:
            st.info(f"No snapshots in the last {hours} h. History covers {live['time'].min():%d %b %H:%M} – "
                    f"{live['time'].max():%d %b %H:%M} (NYC time).")
        else:
            bucket = "1min" if hours == 1 else "5min"
            per_time = window.groupby(window["time"].dt.floor(bucket)).agg(
                avg_score=("anomaly_score", "mean"), peak_score=("anomaly_score", "max"),
                zones_flagged=("anomaly_score", lambda s: int((s >= 0.5).sum() / max(1, s.size / 263))),
            )
            # Empty buckets (pipeline not running) become gaps in the lines instead of being bridged
            per_time = per_time.reindex(pd.date_range(per_time.index.min(), per_time.index.max(), freq=bucket))
            per_time = per_time.rename_axis("time").reset_index()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Snapshots", window["time"].nunique())
            c2.metric("Average score", f"{window['anomaly_score'].mean():.3f}")
            c3.metric("Peak score", f"{window['anomaly_score'].max():.3f}")
            c4.metric("Zones ≥ 0.50 (typical snapshot)", int(per_time["zones_flagged"].median(skipna=True)))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=per_time["time"], y=per_time["avg_score"], name="Average score",
                                     line=dict(color="#00d4ff", width=2)))
            fig.add_trace(go.Scatter(x=per_time["time"], y=per_time["peak_score"], name="Highest zone score",
                                     line=dict(color="#f97316", width=1.5, dash="dot")))
            fig.add_trace(go.Bar(x=per_time["time"], y=per_time["zones_flagged"], name="Zones ≥ 0.50",
                                 marker_color="rgba(239,68,68,0.35)", yaxis="y2"))
            fig.add_hline(y=0.8, line_dash="dash", line_color="rgba(239,68,68,0.5)", annotation_text="Disruption 0.80")
            fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(249,115,22,0.5)", annotation_text="Anomaly 0.50")
            ui.style_chart(fig, height=340, yaxis=dict(range=[0, 1.05], title="Score"),
                           yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Zones"),
                           legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"{bucket.replace('min', '-minute')} averages · gaps = the live pipeline was not running · "
                       f"history covers {live['time'].min():%d %b %H:%M} – {live['time'].max():%d %b %H:%M} NYC time")

# ── 2. History 2022 – 2024 ────────────────────────────────────────────
with tab_hist:
    hist = load_history(use_hdfs)
    if hist is None:
        st.info("historical_predictions.parquet not found (run 08_dashboard/generate_dashboard_data.scala).")
        st.stop()
    ui.section("Disruptions by day",
               "A zone-day counts as disrupted when the zone had at least one hour flagged by the batch "
               "detector (z-scores + KMeans, 06_ml/detect_anomalies.scala).", "historical")
    first, last = hist["date"].min().date(), hist["date"].max().date()
    c1, c2 = st.columns([3, 2])
    start, end = c1.slider("Date range", first, last, (first, last), format="DD MMM YYYY")
    boroughs = c2.multiselect("Boroughs", sorted(hist["borough"].dropna().unique()), placeholder="All boroughs")
    sel = hist[(hist["date"].dt.date >= start) & (hist["date"].dt.date <= end)]
    if boroughs:
        sel = sel[sel["borough"].isin(boroughs)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days", f"{sel['date'].nunique():,}")
    c2.metric("Zones", sel["zone_id"].nunique())
    c3.metric("Disrupted zone-days", f"{int(sel['is_anomaly'].sum()):,}")
    c4.metric("Share of zone-days", f"{sel['is_anomaly'].mean():.1%}" if len(sel) else "—")

    daily = sel.groupby("date")["is_anomaly"].sum().rename("zones").reset_index()
    daily["week_avg"] = daily["zones"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["date"], y=daily["zones"], name="Disrupted zones that day",
                         marker_color="rgba(0,212,255,0.35)"))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["week_avg"], name="7-day average",
                             line=dict(color="#f97316", width=2)))
    st.plotly_chart(ui.style_chart(fig, height=320, yaxis_title="Zones",
                                   legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)")),
                    width="stretch")

    left, right = st.columns(2)
    with left:
        ui.section("Most disrupted zones", "Zones with the most disrupted days in the selected range. Quiet "
                   "zones (parks, islands) rank high because a small change in very low traffic "
                   "gives a large z-score.")
        top_zones = (sel.groupby(["name", "borough"])["is_anomaly"].sum().rename("days").reset_index()
                     .nlargest(15, "days").sort_values("days"))
        fig_z = px.bar(top_zones, x="days", y="name", color="borough", orientation="h",
                       color_discrete_map=config.BOROUGH_COLORS)
        st.plotly_chart(ui.style_chart(fig_z, height=430, xaxis_title="Disrupted days", yaxis_title=None,
                                       legend_title_text=None), width="stretch")
    with right:
        ui.section("By day of the week", "Average share of zones disrupted on each weekday.")
        sel = sel.assign(weekday=sel["date"].dt.day_name())
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_day = sel.groupby("weekday")["is_anomaly"].mean().reindex(order).reset_index()
        fig_d = go.Figure(go.Bar(x=by_day["weekday"], y=by_day["is_anomaly"] * 100, marker_color="#a78bfa"))
        st.plotly_chart(ui.style_chart(fig_d, height=430, yaxis_title="% of zones disrupted"), width="stretch")

    ui.section("Most disrupted days", "The 15 days with the most disrupted zones in the selected range.")
    worst = (sel.groupby("date").agg(zones=("is_anomaly", "sum"), hours=("disruption_hours", "sum"),
                                     rain_mm=("precipitation_mm", "mean"), events=("event_count", "mean"))
             .nlargest(15, "zones").reset_index())
    worst["date"] = worst["date"].dt.strftime("%a %d %b %Y")
    st.dataframe(worst.rename(columns={"date": "Date", "zones": "Disrupted zones", "hours": "Disrupted zone-hours",
                                       "rain_mm": "Avg rain (mm)", "events": "Avg events per zone"}),
                 hide_index=True, width="stretch",
                 column_config={"Avg rain (mm)": st.column_config.NumberColumn(format="%.2f"),
                                "Avg events per zone": st.column_config.NumberColumn(format="%.2f")})

    export = sel[["date", "zone_id", "name", "borough", "is_anomaly", "disruption_hours", "anomaly_score",
                  "taxi_pickups", "subway_ridership", "bike_starts", "precipitation_mm", "event_count"]]
    st.download_button("⬇ Download selection as CSV", export.to_csv(index=False),
                       file_name=f"neurotraffic_{start}_{end}.csv", mime="text/csv")
