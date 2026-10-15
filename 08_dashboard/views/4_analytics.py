"""
NeuroTraffic — Analytics
=========================
How well the GBT disruption model works (06_ml/train_gbt.scala), which inputs it relies on,
and the structure of the transport graph (05_graph/run_graph_pipeline.scala).
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
import hdfs_helper
import ui

ui.setup_page("Analytics", "🧠",
              "How well the disruption model works, which inputs it relies on, and the structure of the "
              "transport graph behind the project.")
use_hdfs = ui.use_hdfs()


@st.cache_data(ttl=600)
def load_all(use_hdfs):
    metrics = hdfs_helper.load_json(config.EVALUATION_METRICS_JSON, config.HDFS_EVALUATION_METRICS_JSON, use_hdfs=use_hdfs)
    features = hdfs_helper.load_dataframe(config.FEATURE_IMPORTANCE_CSV, config.HDFS_FEATURE_IMPORTANCE_CSV, use_hdfs=use_hdfs)
    nodes = hdfs_helper.load_dataframe(config.GRAPH_NODES_CSV, config.HDFS_GRAPH_NODES_CSV, use_hdfs=use_hdfs)
    return metrics, features, nodes


metrics, features, nodes = load_all(use_hdfs)
if metrics is None:
    st.error("evaluation_metrics.json not found (written by 06_ml/train_gbt.scala).")
    st.stop()

m, data = metrics["gbt_classifier"], metrics.get("data_summary", {})
tp, fp, fn, tn = m["true_positives"], m["false_positives"], m["false_negatives"], m["true_negatives"]

st.markdown("""<div class='nt-card' style='font-size:0.85rem;color:#cbd5e1;line-height:1.7;'>
<b>What the model does.</b> A Gradient-Boosted Trees classifier (Spark MLlib) looks at one zone-hour —
taxi, subway and bike volumes, weather, complaints, events, time of day and the zone's graph position —
and answers: <i>is this a disruption?</i> The <b>correct answers it learned from</b> are the hours that the batch
detector (z-scores + KMeans, <code>06_ml/detect_anomalies.scala</code>) flagged. So the scores below show how well
the model reproduces that detector; they are not measured against a list of real-world incidents.
</div>""", unsafe_allow_html=True)

# ── Scores ────────────────────────────────────────────────────────────
ui.section("How well it works", f"Measured on {tp + fp + fn + tn:,} test zone-hours the model never saw during training.")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{(tp + tn) / (tp + fp + fn + tn):.1%}", help="Share of all test hours classified correctly")
c2.metric("Precision", f"{tp / (tp + fp):.1%}", help="Of the hours it called a disruption, how many really were")
c3.metric("Recall", f"{tp / (tp + fn):.1%}", help="Of the real disruption hours, how many it caught")
c4.metric("F1 score", f"{2 * tp / (2 * tp + fp + fn):.1%}", help="Balance of precision and recall")
c5.metric("AUC-ROC", f"{m['auc_roc']:.3f}", help="1.0 = perfect ranking of disruptions above normal hours, 0.5 = random")
st.caption(f"Training data: {data.get('training_rows', m.get('n_train_samples', 0)):,} zone-hours "
           f"(all {data.get('positives_total', 0):,} disrupted hours + a 5% sample of normal hours, so the rare "
           f"disruptions are not drowned out) · test: {data.get('test_rows', m.get('n_test_samples', 0)):,} · "
           f"80 / 20 random split.")

tab_err, tab_feat, tab_graph, tab_settings = st.tabs(
    ["Errors", "What the model relies on", "Transport graph", "Model settings"])

# ── Errors (confusion matrix) ─────────────────────────────────────────
with tab_err:
    left, right = st.columns([3, 2])
    with left:
        z = [[tn, fp], [fn, tp]]
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted normal", "Predicted disruption"],
                                   y=["Actually normal", "Actually disruption"], text=[[f"{v:,}" for v in r] for r in z],
                                   texttemplate="%{text}", textfont=dict(size=20, color="white"), showscale=False,
                                   colorscale=[[0, "#0d1425"], [0.5, "#0ea5e9"], [1, "#00d4ff"]]))
        st.plotly_chart(ui.style_chart(fig, height=340, yaxis=dict(autorange="reversed")), width="stretch")
    with right:
        st.markdown(f"""
- ✅ **{tn:,} correct "normal"** — normal hours recognised as normal
- ✅ **{tp:,} correct "disruption"** — disruptions caught
- ⚠️ **{fp:,} false alarms** — normal hours called a disruption
- ❌ **{fn:,} missed disruptions** — disruptions called normal

The two kinds of mistakes are about equally common, so the model is balanced: it neither cries wolf
nor misses much.""")

# ── Feature importance ────────────────────────────────────────────────
with tab_feat:
    if features is None:
        st.info("feature_importance.csv not found.")
    else:
        n = st.slider("Number of inputs to show", 5, len(features), min(15, len(features)))
        top = features.nlargest(n, "importance").sort_values("importance")
        palette = {"External": "#f472b6", "Weather": "#64748b", "Graph": "#00d4ff", "Transport": "#34d399",
                   "Cross-modal": "#f97316", "Temporal": "#a78bfa", "Anomaly": "#ef4444"}
        fig = px.bar(top, x="importance", y="feature_name", color="category", orientation="h",
                     color_discrete_map=palette)
        st.plotly_chart(ui.style_chart(fig, height=max(300, n * 26), xaxis_title="Share of the model's decisions",
                                       yaxis_title=None, legend_title_text="Input type"), width="stretch")
        first, second = features.nlargest(2, "importance")["feature_name"].tolist()
        st.caption(f"Importance = how much each input contributes to the trees' decisions (all add up to 1). "
                   f"**{first}** and **{second}** dominate because the detector that produced the labels adds "
                   f"points for events and bad weather, so the model learned to look at them first.")

# ── Transport graph ───────────────────────────────────────────────────
with tab_graph:
    if nodes is None:
        st.info("graph_nodes.csv not found.")
    else:
        st.caption("The graph links taxi zones, subway stations and bike docks by real trips and walking "
                   "distance (Spark GraphX). Shown here: the 263 taxi zones and their graph measures.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Taxi zones", len(nodes))
        c2.metric("Communities", nodes["community_id"].nunique(), help="Groups of zones that trade many trips (Label Propagation)")
        c3.metric("Average links per zone", f"{(nodes['in_degree'] + nodes['out_degree']).mean() / 2:.1f}")
        c4.metric("Highest PageRank", f"{nodes['page_rank'].max():.4f}")
        left, right = st.columns([2, 3])
        with left:
            k = st.slider("Top zones by PageRank", 5, 30, 12)
            top = nodes.nlargest(k, "page_rank").sort_values("page_rank")
            fig = go.Figure(go.Bar(x=top["page_rank"], y=top["name"].str[:28], orientation="h", marker_color="#00d4ff"))
            st.plotly_chart(ui.style_chart(fig, height=max(320, k * 24), xaxis_title="PageRank (how central)"),
                            width="stretch")
        with right:
            colour_by = st.radio("Colour zones by", ["community", "borough"], horizontal=True)
            col = "community_id" if colour_by == "community" else "borough"
            fig = go.Figure()
            for value, sub in nodes.groupby(col):
                fig.add_trace(go.Scattermap(lat=sub["lat"], lon=sub["lon"], mode="markers", name=str(value),
                                            marker=dict(size=4 + sub["page_rank"] / nodes["page_rank"].max() * 14),
                                            text=sub["name"], hovertemplate="%{text}<extra></extra>"))
            st.plotly_chart(ui.zone_map(fig, height=max(320, k * 24), zoom=9.3), width="stretch")
            st.caption("Bigger dot = higher PageRank.")

# ── Model settings ────────────────────────────────────────────────────
with tab_settings:
    st.markdown("""
| Setting | Value | Meaning |
|---|---|---|
| Algorithm | GBTClassifier (Spark MLlib) | Many small decision trees; each one corrects the previous trees' mistakes |
| Trees (`maxIter`) | 150 | Number of trees |
| Depth (`maxDepth`) | 4 | Questions each tree may ask |
| Learning rate (`stepSize`) | 0.1 | How much each new tree is trusted |
| Row sample (`subsamplingRate`) | 0.8 | Each tree sees a random 80% of the rows |
| Inputs | 23 | Taxi, subway, bike, cross-modal ratios, weather, time, complaints, events, PageRank, community, degrees |
| Preparation | VectorAssembler → StandardScaler (mean 0, std 1) | Packs the inputs into one vector on a common scale |
| Class weights | disruptions × (normal ÷ disrupted) | Makes the rare disruptions count as much as all normal hours |
| Split | 80% train / 20% test, seed 42 | Test rows are never used for training |
| Labels | `06_ml/detect_anomalies.scala` | z-scores + KMeans, disruption score > 3.0 |

The simulator page uses a scikit-learn copy of this model (`08_dashboard/models/gbt_model.pkl`).
""")
