"""
NeuroTraffic — Disruption Simulator (what-if)
==============================================
Pick a subway line, an hour and a day: the page shows how a disruption on that line
COULD spread. It is an illustration built from simple rules plus the trained GBT model,
not a forecast. Same inputs always give the same result.
"""

import os
import sys
import warnings
import zlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
import hdfs_helper
import ui

ui.setup_page("Disruption Simulator", "⚡",
              "What-if tool: choose a subway line, an hour and a day to see how a disruption on that line could "
              "spread across nearby zones. This is a scenario illustration, not a prediction of what will happen.")
use_hdfs = ui.use_hdfs()


@st.cache_resource(show_spinner="Loading the GBT model (first time only, about 10 s)…")
def load_model():
    """The scikit-learn GBT saved as models/gbt_model.pkl (None if missing)."""
    try:
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # model was saved with a newer scikit-learn version
            return joblib.load(config.GBT_MODEL_PKL) if os.path.exists(config.GBT_MODEL_PKL) else None
    except ImportError:
        return None

@st.cache_data(ttl=600)
def load_nodes(use_hdfs):
    return hdfs_helper.load_dataframe(config.GRAPH_NODES_CSV, config.HDFS_GRAPH_NODES_CSV, use_hdfs=use_hdfs)


# ── Scenario rules ────────────────────────────────────────────────────
def time_multiplier(hour, weekend):
    """How busy the city is: scales volumes, number of affected zones and severity."""
    rush = 7 <= hour <= 9 or 17 <= hour <= 19
    if rush:
        return (2.2, "Weekday rush hour") if not weekend else (1.4, "Weekend rush hour")
    if 10 <= hour <= 16:
        return (1.3 if not weekend else 1.0), "Midday"
    if 20 <= hour <= 23:
        return 0.7, "Evening"
    return 0.5, "Late night"


def feature_row(zone, hour, weekend, role, mult, rng):
    """Made-up transport volumes for one zone in this scenario (the model's input)."""
    pr = zone.get("page_rank", 0.01)
    taxi, sub, bike = pr * 50000 * mult, pr * 80000 * mult, pr * 20000 * mult
    if role == "Epicenter":      # subway nearly stops, riders switch to taxi and bike
        taxi, sub, bike = taxi * rng.uniform(2.5, 4.0), sub * rng.uniform(0.05, 0.25), bike * rng.uniform(1.8, 3.0)
    elif role == "Cascade":      # milder spill-over
        taxi, sub, bike = taxi * rng.uniform(1.3, 2.0), sub * rng.uniform(0.4, 0.7), bike * rng.uniform(1.2, 1.8)
    taxi, sub, bike = int(taxi), max(1, int(sub)), int(bike)
    complaints = int(rng.poisson(max(1, pr * 2000)))
    return {"taxi_pickups": taxi, "taxi_dropoffs": int(taxi * 0.9), "subway_ridership": sub,
            "bike_starts": bike, "bike_ends": int(bike * 0.95),
            "taxi_to_subway_ratio": taxi / sub, "bike_to_subway_ratio": bike / sub,
            "total_surface_transport": taxi + bike, "complaint_density": complaints / max(1, taxi),
            "temperature_c": 15.0, "humidity_pct": 65.0, "precipitation_mm": 0.0, "wind_speed_kmh": 12.0,
            "is_rain": 0, "is_snow": 0, "is_extreme_cold": 0, "is_extreme_heat": 0,
            "hour_of_day": hour, "is_rush_hour": int(7 <= hour <= 9 or 17 <= hour <= 19),
            "is_weekend": int(weekend), "complaint_count": complaints, "max_complaint_severity": 2,
            "event_count": int(role == "Epicenter"), "page_rank": pr,
            "community_id": zone.get("community_id", 0), "in_degree": zone.get("in_degree", 10),
            "out_degree": zone.get("out_degree", 10)}


def simulate(nodes, line, hour, day, n_epicenter, n_cascade):
    rng = np.random.RandomState(zlib.crc32(f"{line}|{hour}|{day}".encode()))   # same inputs → same result
    weekend = day >= 5
    mult, period = time_multiplier(hour, weekend)

    # 1. Zones with a station on the chosen line (fallback: the line's boroughs)
    letters = {s.strip() for s in line.replace(",", "/").split("/") if s.strip()}
    on_line = nodes[nodes["all_lines"].fillna("-").apply(lambda s: bool(letters & set(s.split("|"))))]
    if len(on_line) < 3:
        on_line = nodes[nodes["borough"].isin(config.SUBWAY_LINES.get(line, {}).get("boroughs", []))]

    # 2. Epicenters: busy zones (high PageRank) are more likely, with some randomness
    count = min(len(on_line), max(3, int(n_epicenter * mult / 1.3)))
    pr = on_line["page_rank"].fillna(0.001).values
    noise = rng.exponential(0.5, len(pr))
    weight = pr * 0.3 + noise * 0.7 if (hour >= 17 or hour <= 6) else pr * 0.5 + noise * 0.5
    epicenters = on_line.iloc[rng.choice(len(on_line), size=count, replace=False, p=weight / weight.sum())]

    # 3. Cascade: the zones closest to the epicenters
    rest = nodes[~nodes["zone_id"].isin(epicenters["zone_id"])].copy()
    rest["dist"] = np.hypot(rest["lat"] - epicenters["lat"].mean(), rest["lon"] - epicenters["lon"].mean())
    cascade = rest.nsmallest(max(5, int(n_cascade * mult)), "dist")
    normal = rest[~rest["zone_id"].isin(cascade["zone_id"])]

    result = pd.concat([epicenters.assign(role="Epicenter"), cascade.assign(role="Cascade"),
                        normal.assign(role="Normal")], ignore_index=True)

    # 4. Score: base value by role, plus up to 0.15 / 0.30 from the model's disruption probability
    model = load_model()
    features = pd.DataFrame([feature_row(z, hour, weekend, z["role"], mult, rng) for _, z in result.iterrows()])
    if model is not None:
        X = features.reindex(columns=model["feature_cols"], fill_value=0).replace([np.inf, -np.inf], 0).fillna(0)
        p = model["model"].predict_proba(model["scaler"].transform(X.values))[:, 1]
    else:
        p = np.zeros(len(result))
    base = result["role"].map({"Epicenter": 0.70 + mult * 0.08, "Cascade": 0.25 + mult * 0.10, "Normal": 0.0})
    from_model = result["role"].map({"Epicenter": 0.15, "Cascade": 0.30, "Normal": 0.25 * mult / 1.3}) * p
    result["score"] = np.clip(base + from_model + rng.uniform(0, 0.06, len(result)), 0, 1)

    # 5. Scenario estimates derived from the score
    result["taxi_surge_pct"] = (result["score"] * 280 * mult / 1.3).astype(int)
    result["bike_surge_pct"] = (result["score"] * 190 * mult / 1.3).astype(int)
    result["subway_drop_pct"] = (result["score"] * 75 * mult / 1.3).clip(upper=95).astype(int)   # ridership can't fall below zero
    peak_max = {"Epicenter": max(1, int(15 / mult)), "Cascade": max(6, int(35 / mult))}
    result["minutes_to_peak"] = [rng.randint(0 if r == "Epicenter" else 5, peak_max[r]) if r != "Normal" else 0
                                 for r in result["role"]]
    return result, mult, period, model is not None


# ── Controls ──────────────────────────────────────────────────────────
nodes = load_nodes(use_hdfs)
if nodes is None:
    st.error("graph_nodes.csv not found.")
    st.stop()

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
c1, c2, c3, c4, c5 = st.columns([1.3, 1.6, 1.2, 1.2, 1.2])
line = c1.selectbox("Subway line", list(config.SUBWAY_LINES))
hour = c2.slider("Hour of day", 0, 23, 8, format="%d:00")
day = DAYS.index(c3.selectbox("Day", DAYS, index=1))
n_epicenter = c4.slider("Epicenter zones", 5, 30, 15, help="Before the time-of-day multiplier")
n_cascade = c5.slider("Cascade zones", 10, 60, 25, help="Before the time-of-day multiplier")

with st.expander("How the simulator works"):
    st.markdown("""
1. **Candidate zones** — every taxi zone with a station on the chosen line.
2. **Time of day** sets a multiplier: weekday rush 2.2 · weekend rush 1.4 · midday 1.3 (weekend 1.0) · evening 0.7 · night 0.5.
   It scales how many zones are hit and how strongly.
3. **Epicenters (red)** — picked from the candidate zones; busier zones (higher PageRank) are more likely.
4. **Cascade (orange)** — the zones closest to the epicenters (straight-line distance, not along the track).
5. **Volumes** — epicenters get taxi × 2.5–4, subway × 0.05–0.25, bike × 1.8–3; cascade zones a milder version.
6. **Score** — a base value per role, plus up to 0.15 (epicenter) / 0.30 (cascade) from the trained GBT model's
   disruption probability for those volumes.
7. **Surge %, subway drop % and minutes to peak** are calculated from the score with fixed factors: scenario
   estimates, not measurements. The subway drop is capped at 95% (a full suspension).
""")

result, mult, period, used_model = simulate(nodes, line, hour, day, n_epicenter, n_cascade)
epi, cas = result[result["role"] == "Epicenter"], result[result["role"] == "Cascade"]

# ── Results ───────────────────────────────────────────────────────────
ui.section(f"Scenario: {line} line disrupted · {DAYS[day]} {hour:02d}:00 ({period}, multiplier {mult})",
           None if used_model else "GBT model file not found: scores use the role rules only.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Epicenter zones", len(epi))
c2.metric("Cascade zones", len(cas))
c3.metric("Highest score", f"{result['score'].max():.3f}")
c4.metric("Avg taxi surge at epicenters", f"+{int(epi['taxi_surge_pct'].mean())}%", help="Scenario estimate")

ROLE_STYLE = {"Normal": ("#22c55e", 5), "Cascade": ("#f97316", 9), "Epicenter": ("#ef4444", 14)}
result["hover"] = ("<b>" + result["name"].fillna("Zone " + result["zone_id"].astype(str)) + "</b><br>" +
                   result["role"] + " · score " + result["score"].map("{:.3f}".format) +
                   "<br>Taxi +" + result["taxi_surge_pct"].astype(str) + "% · peak in " +
                   result["minutes_to_peak"].astype(str) + " min")
fig = go.Figure()
for role, (colour, size) in ROLE_STYLE.items():
    sub = result[result["role"] == role]
    fig.add_trace(go.Scattermap(lat=sub["lat"], lon=sub["lon"], mode="markers", name=f"{role} ({len(sub)})",
                                marker=dict(size=size, color=colour, opacity=0.9),
                                text=sub["hover"], hovertemplate="%{text}<extra></extra>"))
left, right = st.columns([3, 2])
with left:
    ui.section("Where the disruption spreads", "Red = epicenter, orange = cascade, green = not affected.")
    st.plotly_chart(ui.zone_map(fig, height=460), width="stretch")
with right:
    ui.section("Epicenter zones", "Highest score first.")
    table = epi.sort_values("score", ascending=False)[["name", "score", "taxi_surge_pct", "bike_surge_pct",
                                                       "minutes_to_peak"]]
    st.dataframe(table.rename(columns={"name": "Zone", "score": "Score", "taxi_surge_pct": "Taxi %",
                                       "bike_surge_pct": "Bike %", "minutes_to_peak": "Peak (min)"}),
                 hide_index=True, width="stretch", height=240,
                 column_config={"Score": st.column_config.NumberColumn(format="%.3f")})
    ui.section("How fast it spreads", "Zones reaching their peak, by minutes after the disruption starts.")
    affected = pd.concat([epi, cas])
    waves = pd.cut(affected["minutes_to_peak"], bins=[-1, 5, 10, 15, 20, 30, 100],
                   labels=["0–5", "5–10", "10–15", "15–20", "20–30", "30+"]).value_counts().sort_index()
    fig_w = go.Figure(go.Bar(x=waves.index.astype(str), y=waves.values, marker_color="#f97316"))
    st.plotly_chart(ui.style_chart(fig_w, height=180, xaxis_title="Minutes to peak", yaxis_title="Zones"),
                    width="stretch")

ui.section("Modal shift at the epicenters", "Scenario estimates: how riders would move away from the subway.")
c1, c2, c3 = st.columns(3)
c1.metric("🚕 Taxi demand", f"+{int(epi['taxi_surge_pct'].mean())}%")
c2.metric("🚲 Citi Bike demand", f"+{int(epi['bike_surge_pct'].mean())}%")
c3.metric("🚇 Subway ridership", f"-{int(epi['subway_drop_pct'].mean())}%")
