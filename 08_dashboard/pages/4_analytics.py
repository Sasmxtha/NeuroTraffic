"""
NeuroTraffic — Page 4: Analytics Dashboard
============================================

Model evaluation metrics + confusion matrix + feature importance + graph analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, sys, json

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
    [data-testid="stDecoration"] { display: none; }
    [data-testid="metric-container"] { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(0,212,255,0.15) !important; border-radius: 12px !important; padding: 1rem !important; }
    [data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.75rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 1.6rem !important; font-weight: 700 !important; }
    .stButton > button { background: linear-gradient(135deg,#00d4ff,#0ea5e9) !important; color: #090e1a !important; border: none !important; border-radius: 8px !important; font-weight: 700 !important; }
    .stSelectbox > div > div { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(0,212,255,0.2) !important; border-radius: 8px !important; color: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.03) !important; border-radius: 10px !important; padding: 4px !important; }
    .stTabs [data-baseweb="tab"] { background: transparent !important; color: #94a3b8 !important; border-radius: 8px !important; }
    .stTabs [aria-selected="true"] { background: rgba(0,212,255,0.15) !important; color: #00d4ff !important; }
    .stDataFrame { border-radius: 10px !important; }
    .nt-card { background: rgba(255,255,255,0.035); border: 1px solid rgba(0,212,255,0.12); border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem; }
    .big-metric { background: rgba(255,255,255,0.03); border: 1px solid rgba(0,212,255,0.15); border-radius: 16px; padding: 1.5rem; text-align: center; }
    .big-metric .val { font-size: 2.4rem; font-weight: 800; color: #00d4ff; line-height: 1; }
    .big-metric .lbl { font-size: 0.72rem; color: #64748b; margin-top: 0.4rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; }
    .big-metric .delta { font-size: 0.78rem; margin-top: 0.3rem; }
    ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0d1425; } ::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }
    </style>""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Analytics — NeuroTraffic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)
import hdfs_helper
inject_css()

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_metrics():
    use_hdfs = st.session_state.get("use_hdfs", config.USE_HDFS)
    return hdfs_helper.load_json(config.EVALUATION_METRICS_JSON, config.HDFS_EVALUATION_METRICS_JSON, use_hdfs=use_hdfs)

@st.cache_data(ttl=300)
def load_feature_importance():
    use_hdfs = st.session_state.get("use_hdfs", config.USE_HDFS)
    return hdfs_helper.load_dataframe(config.FEATURE_IMPORTANCE_CSV, config.HDFS_FEATURE_IMPORTANCE_CSV, use_hdfs=use_hdfs)

@st.cache_data(ttl=300)
def load_confusion_matrix():
    use_hdfs = st.session_state.get("use_hdfs", config.USE_HDFS)
    return hdfs_helper.load_dataframe(config.CONFUSION_MATRIX_CSV, config.HDFS_CONFUSION_MATRIX_CSV, use_hdfs=use_hdfs)

@st.cache_data(ttl=300)
def load_nodes():
    use_hdfs = st.session_state.get("use_hdfs", config.USE_HDFS)
    return hdfs_helper.load_dataframe(config.GRAPH_NODES_CSV, config.HDFS_GRAPH_NODES_CSV, use_hdfs=use_hdfs)

# ─────────────────────────────────────────────
# Sidebar (minimal)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1.5rem;">
        <div style="font-size:2.5rem;">🧠</div>
        <div style="font-size:1rem;font-weight:800;color:#00d4ff;">Analytics</div>
        <div style="font-size:0.65rem;color:#475569;margin-top:4px;">Model metrics · Graph analytics</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.page_link("app.py",                  label="\U0001f3e0  Home")
    st.page_link("pages/1_live_monitor.py", label="\U0001f4e1  Live Monitor")
    st.page_link("pages/2_simulator.py",    label="\u26a1  Simulator")
    st.page_link("pages/3_explorer.py",     label="\U0001f4ca  Explorer")
    st.page_link("pages/4_analytics.py",    label="\U0001f9e0  Analytics")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#00d4ff,#a78bfa);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 0.2rem;">
🧠 Analytics Dashboard
</h1>
<p style="color:#64748b;font-size:0.8rem;margin:0 0 0.8rem;">
Model evaluation · Feature importance · Graph structure analytics
</p>""", unsafe_allow_html=True)

# ── Inline Controls ──
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.5, 1, 1, 1.2])
with ctrl1:
    model_choice = st.selectbox("Model", ["GBT Classifier", "Isolation Forest"], index=0)
    model_key = "gbt_classifier" if model_choice == "GBT Classifier" else "isolation_forest"
with ctrl2:
    n_features = st.slider("Top N features", 5, 30, 15)
with ctrl3:
    n_top_nodes = st.slider("Top N nodes (PageRank)", 5, 30, 10)
with ctrl4:
    color_by = st.selectbox("Color nodes by", ["community_id","borough","node_type"])

metrics = load_metrics()
fi_df   = load_feature_importance()
cm_df   = load_confusion_matrix()
nodes_df = load_nodes()

if metrics is None:
    st.error("evaluation_metrics.json not found. Run `spark-shell -i /project/08_dashboard/generate_dashboard_data.scala` on the cluster first.")
    st.stop()

m = metrics.get(model_key, {})

# ─────────────────────────────────────────────
# Big metric cards
# ─────────────────────────────────────────────
st.markdown("### 📊 Model Performance")
c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (c1, f"{m.get('accuracy',0)*100:.1f}%",  "Accuracy",  "Overall correct predictions",   "#00d4ff"),
    (c2, f"{m.get('precision',0)*100:.1f}%", "Precision", "Of predicted disruptions, true", "#34d399"),
    (c3, f"{m.get('recall',0)*100:.1f}%",    "Recall",    "Disruptions actually caught",    "#a78bfa"),
    (c4, f"{m.get('f1_score',0)*100:.1f}%",  "F1 Score",  "Harmonic mean P & R",            "#fb923c"),
    (c5, f"{m.get('auc_roc',0):.4f}",        "AUC-ROC",   "Area under ROC curve",           "#f472b6"),
]
for col, val, lbl, tip, color in cards:
    with col:
        st.markdown(f"""
        <div class="big-metric" style="border-color:{color}22;">
            <div class="val" style="color:{color};">{val}</div>
            <div class="lbl">{lbl}</div>
            <div class="delta" style="color:{color}88;">{tip}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# Sample summary
ds = metrics.get("data_summary", {})
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Total Zone-Hours",  f"{ds.get('total_zone_hours',0):,}")
with c2: st.metric("Zones",            f"{ds.get('zones',0)}")
with c3: st.metric("True Positives",   f"{m.get('true_positives',0):,}")
with c4: st.metric("False Positives",  f"{m.get('false_positives',0):,}")


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔲 Confusion Matrix", "📊 Feature Importance", "🌐 Graph Analytics", "🔬 Model Details"])

# ── Tab 1: Confusion Matrix ──
with tab1:
    st.markdown("**Confusion Matrix — Disruption Detection**")
    if cm_df is not None:
        model_cm = cm_df[cm_df["model"] == ("GBT" if model_key == "gbt_classifier" else "IsolationForest")]
        if len(model_cm) == 0:
            model_cm = cm_df  # fallback

        # Pivot to 2x2
        pivot = model_cm.pivot_table(index="actual", columns="predicted", values="count", aggfunc="sum").fillna(0)
        labels = sorted(pivot.index.tolist())

        z = [[int(pivot.loc[r, c]) if r in pivot.index and c in pivot.columns else 0
              for c in labels] for r in labels]
        z_text = [[f"{v:,}" for v in row] for row in z]

        fig_cm = go.Figure(go.Heatmap(
            z=z, x=labels, y=labels,
            text=z_text, texttemplate="%{text}",
            textfont=dict(size=18, color="white"),
            colorscale=[[0,"#0d1425"],[0.4,"#1e3a5f"],[0.7,"#0ea5e9"],[1,"#00d4ff"]],
            showscale=True,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
        ))
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20,r=20,t=20,b=20), height=340,
            xaxis=dict(color="#94a3b8", title="Predicted", side="bottom"),
            yaxis=dict(color="#94a3b8", title="Actual"),
            font=dict(color="#94a3b8"),
        )
        col_cm, col_cm_stats = st.columns([2, 1])
        with col_cm:
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_cm_stats:
            st.markdown("""
            <div style="padding:1rem;">
                <div style="font-size:0.8rem;color:#94a3b8;line-height:2;">
                    <b style="color:#22c55e;">True Positive (TP)</b><br>
                    Disruption correctly predicted<br><br>
                    <b style="color:#ef4444;">False Positive (FP)</b><br>
                    Normal flagged as disruption<br><br>
                    <b style="color:#facc15;">False Negative (FN)</b><br>
                    Disruption missed by model<br><br>
                    <b style="color:#00d4ff;">True Negative (TN)</b><br>
                    Normal correctly identified
                </div>
            </div>""", unsafe_allow_html=True)
            st.metric("TP", f"{m.get('true_positives',0):,}")
            st.metric("FP", f"{m.get('false_positives',0):,}")
            st.metric("FN", f"{m.get('true_positives',0) - int(m.get('true_positives',0) * m.get('recall',1)):,}")


# ── Tab 2: Feature Importance ──
with tab2:
    if fi_df is not None:
        st.markdown(f"**Top {n_features} Features by GBT Importance Score**")
        top_fi = fi_df.nlargest(n_features, "importance").sort_values("importance")

        category_colors = {
            "Anomaly":     "#ef4444",
            "Cross-modal": "#f97316",
            "Graph":       "#00d4ff",
            "Transport":   "#34d399",
            "Temporal":    "#a78bfa",
            "Weather":     "#64748b",
            "External":    "#f472b6",
        }

        fig_fi = go.Figure(go.Bar(
            x=top_fi["importance"],
            y=top_fi["feature_name"],
            orientation="h",
            marker=dict(
                color=[category_colors.get(c, "#94a3b8") for c in top_fi["category"]],
                line=dict(color="rgba(0,0,0,0.3)", width=0.5),
            ),
            text=top_fi["importance"].map("{:.4f}".format),
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=60,t=10,b=0),
            height=max(300, n_features * 28),
            xaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)", title="Importance Score"),
            yaxis=dict(color="#e2e8f0", tickfont=dict(size=11)),
        )

        col_fi, col_legend = st.columns([4, 1])
        with col_fi:
            st.plotly_chart(fig_fi, use_container_width=True)
        with col_legend:
            st.markdown("**Categories**")
            for cat, col in category_colors.items():
                st.markdown(f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{col};margin-right:6px;"></span>'
                           f'<span style="font-size:0.75rem;color:#94a3b8;">{cat}</span>', unsafe_allow_html=True)

        # Category-level aggregation
        st.markdown("**Feature Importance by Category**")
        cat_agg = fi_df.groupby("category")["importance"].sum().reset_index().sort_values("importance", ascending=False)
        fig_cat = px.pie(
            cat_agg, values="importance", names="category",
            color="category",
            color_discrete_map=category_colors,
            hole=0.5,
        )
        fig_cat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=0), height=300,
            legend=dict(font=dict(color="#94a3b8",size=11), bgcolor="rgba(0,0,0,0)"),
            font=dict(color="#94a3b8"),
        )
        fig_cat.update_traces(textinfo="label+percent", textfont_color="#e2e8f0")
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.warning("feature_importance.csv not found")


# ── Tab 3: Graph Analytics ──
with tab3:
    if nodes_df is not None:
        st.markdown(f"**Top {n_top_nodes} Nodes by PageRank (most structurally critical)**")

        top_nodes = nodes_df.nlargest(n_top_nodes, "page_rank")

        col_bar, col_map = st.columns([2, 3])
        with col_bar:
            fig_pr = go.Figure(go.Bar(
                x=top_nodes["page_rank"],
                y=top_nodes["name"].str[:25],
                orientation="h",
                marker=dict(
                    color=top_nodes["page_rank"],
                    colorscale=[[0,"#1e3a5f"],[0.5,"#0ea5e9"],[1,"#00d4ff"]],
                    showscale=False,
                ),
                text=top_nodes["page_rank"].map("{:.5f}".format),
                textposition="outside",
                textfont=dict(color="#94a3b8",size=10),
                hovertemplate="<b>%{y}</b><br>PageRank: %{x:.5f}<extra></extra>",
            ))
            fig_pr.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=60,t=10,b=0),
                height=360,
                xaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)", title="PageRank Score"),
                yaxis=dict(color="#e2e8f0", tickfont=dict(size=10)),
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        with col_map:
            # Community / borough map
            color_col = color_by
            fig_community = go.Figure()

            color_vals = nodes_df[color_col].astype(str) if color_col != "page_rank" else nodes_df[color_col]

            if color_col in ["community_id", "borough"]:
                unique_vals = nodes_df[color_col].unique()
                palette = ["#00d4ff","#a78bfa","#34d399","#fb923c","#f472b6","#facc15","#ef4444","#64748b"]
                val_color = {v: palette[i % len(palette)] for i, v in enumerate(unique_vals)}
                for val in unique_vals:
                    sub = nodes_df[nodes_df[color_col] == val]
                    fig_community.add_trace(go.Scattermap(
                        lat=sub["lat"], lon=sub["lon"],
                        mode="markers",
                        name=str(val),
                        marker=dict(size=6, color=val_color[val], opacity=0.8),
                        text=sub["name"],
                        hovertemplate="<b>%{text}</b><br>Zone: %{customdata}<extra></extra>",
                        customdata=sub["zone_id"],
                    ))
            else:
                fig_community.add_trace(go.Scattermap(
                    lat=nodes_df["lat"], lon=nodes_df["lon"],
                    mode="markers",
                    marker=dict(size=6, color=nodes_df["page_rank"],
                                colorscale=[[0,"#1e3a5f"],[0.5,"#0ea5e9"],[1,"#00d4ff"]],
                                showscale=True, opacity=0.8),
                    text=nodes_df["name"],
                    hovertemplate="<b>%{text}</b><extra></extra>",
                ))

            fig_community.update_layout(
                map=dict(style="carto-darkmatter", center=config.MAP_CENTER, zoom=9),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=0,t=0,b=0), height=360,
                legend=dict(bgcolor="rgba(9,14,26,0.8)", font=dict(color="#e2e8f0",size=10),
                            bordercolor="rgba(0,212,255,0.2)", borderwidth=1),
                showlegend=(color_col != "page_rank"),
            )
            st.plotly_chart(fig_community, use_container_width=True)

        # Network statistics
        st.markdown("**📐 Network Statistics**")
        c1, c2, c3, c4, c5 = st.columns(5)
        avg_degree = (nodes_df["in_degree"] + nodes_df["out_degree"]).mean() / 2
        max_pr     = nodes_df["page_rank"].max()
        n_communities = nodes_df["community_id"].nunique()
        with c1: st.metric("Total Nodes",     f"{len(nodes_df):,}")
        with c2: st.metric("Communities",     f"{n_communities}")
        with c3: st.metric("Avg Degree",      f"{avg_degree:.1f}")
        with c4: st.metric("Max PageRank",    f"{max_pr:.5f}")
        with c5: st.metric("Boroughs",        f"{nodes_df['borough'].nunique()}")

        # Degree distribution
        st.markdown("**📈 Degree Distribution**")
        nodes_df["total_degree"] = nodes_df["in_degree"] + nodes_df["out_degree"]
        fig_deg = px.histogram(
            nodes_df, x="total_degree", nbins=30,
            color_discrete_sequence=["#00d4ff"],
            labels={"total_degree":"Total Degree (in + out)","count":"Nodes"},
        )
        fig_deg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=0), height=220,
            xaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)"),
            bargap=0.1,
        )
        fig_deg.update_traces(marker_line_color="rgba(0,212,255,0.3)", marker_line_width=0.5)
        st.plotly_chart(fig_deg, use_container_width=True)
    else:
        st.warning("graph_nodes.csv not found")


# ── Tab 4: Model Details ──
with tab4:
    st.markdown("**🔬 Model Architecture Details**")
    if model_key == "gbt_classifier":
        st.markdown("""
        <div class="nt-card">
        <h4 style="color:#00d4ff;margin:0 0 1rem;">Gradient Boosted Trees (GBT) — PySpark MLlib</h4>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;font-size:0.82rem;color:#94a3b8;line-height:1.8;">
            <div>
                <b style="color:#e2e8f0;">Architecture</b><br>
                • Ensemble of 50 decision trees<br>
                • Max depth: 5 levels<br>
                • Step size: 0.1 (learning rate)<br>
                • Subsampling: 1.0<br>
                • Seed: 42 (reproducible)
            </div>
            <div>
                <b style="color:#e2e8f0;">Features (30 total)</b><br>
                • Transport: taxi, subway, bike metrics<br>
                • Cross-modal ratios (taxi/subway)<br>
                • Graph: PageRank, community, degree<br>
                • Weather: temp, rain, snow, wind<br>
                • Temporal: hour, rush hour, weekend
            </div>
            <div>
                <b style="color:#e2e8f0;">Preprocessing</b><br>
                • VectorAssembler → feature vector<br>
                • StandardScaler (withMean=False)<br>
                • handleInvalid="skip" for missing data
            </div>
            <div>
                <b style="color:#e2e8f0;">Training</b><br>
                • 80/20 train/test split<br>
                • Labels: z-score anomaly flags<br>
                • Imbalanced class handling built-in<br>
                • Run on PySpark cluster (Docker)
            </div>
        </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="nt-card">
        <h4 style="color:#00d4ff;margin:0 0 1rem;">Isolation Forest — Z-Score Implementation (PySpark)</h4>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;font-size:0.82rem;color:#94a3b8;line-height:1.8;">
            <div>
                <b style="color:#e2e8f0;">Algorithm</b><br>
                • Statistical z-score per zone × hour<br>
                • Baselines: mean + stddev per (zone_id, hour_of_day, is_weekend)<br>
                • Disruption signature: z_subway &lt; -2 AND (z_pickups &gt; 2 OR z_bikes &gt; 2)
            </div>
            <div>
                <b style="color:#e2e8f0;">Detection Logic</b><br>
                • Subway ridership drops 2+ std devs<br>
                • AND taxi/bike surges 2+ std devs<br>
                • Cross-modal signature = disruption<br>
                • No labels required (unsupervised)
            </div>
        </div>
        </div>""", unsafe_allow_html=True)

