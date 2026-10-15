"""
NeuroTraffic — shared look and helpers for every dashboard page
================================================================

One place for: the dark theme, the page header, the four anomaly levels
(same thresholds and colours on every page), LIVE / ESTIMATED / HISTORICAL
tags, section titles and a common Plotly chart style.
"""

import streamlit as st

import config

# Anomaly levels, highest first: (minimum score, name, colour)
LEVELS = [
    (0.80, "Disruption", "#ef4444"),
    (0.50, "Anomaly",    "#f97316"),
    (0.25, "Elevated",   "#facc15"),
    (0.00, "Normal",     "#22c55e"),
]


def level(score):
    """Return (name, colour) of the anomaly level for a score between 0 and 1."""
    for minimum, name, colour in LEVELS:
        if score >= minimum:
            return name, colour
    return LEVELS[-1][1], LEVELS[-1][2]


def use_hdfs():
    """True when the dashboard should read from HDFS (set by the toggle in the sidebar)."""
    return st.session_state.get("use_hdfs", config.USE_HDFS)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }
html, body, [data-testid="stAppViewContainer"] { background: #090e1a !important; color: #e2e8f0 !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1425 0%, #111827 100%) !important;
                            border-right: 1px solid rgba(0,212,255,0.12) !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"] { visibility: visible !important; }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: "Material Symbols Rounded" !important; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stMetricValue"] { color: #00d4ff !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
.stTabs [aria-selected="true"] { color: #00d4ff !important; }
.nt-title { font-size: 1.9rem; font-weight: 800; margin: 0; color: #e2e8f0; }
.nt-sub   { color: #94a3b8; font-size: 0.88rem; margin: 0.25rem 0 1.2rem; max-width: 60rem; }
.nt-section { font-size: 1.05rem; font-weight: 700; color: #e2e8f0; margin: 1.6rem 0 0.2rem; }
.nt-caption { color: #94a3b8; font-size: 0.8rem; margin: 0 0 0.6rem; }
.nt-tag { font-size: 0.62rem; font-weight: 700; padding: 2px 7px; border-radius: 8px; margin-left: 6px;
          vertical-align: middle; letter-spacing: 0.04em; }
.nt-live { background: rgba(34,197,94,0.15); color: #22c55e; }
.nt-est  { background: rgba(250,204,21,0.15); color: #facc15; }
.nt-hist { background: rgba(148,163,184,0.15); color: #94a3b8; }
.nt-card { background: rgba(255,255,255,0.035); border: 1px solid rgba(0,212,255,0.12);
           border-radius: 14px; padding: 1.1rem 1.2rem; margin-bottom: 0.8rem; }
.nt-legend { display: flex; flex-wrap: wrap; gap: 0.3rem 1.1rem; font-size: 0.78rem; color: #cbd5e1; margin: 0.3rem 0 0.8rem; }
.nt-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
</style>
"""

TAGS = {"live": ("LIVE", "nt-live"), "estimated": ("ESTIMATED", "nt-est"), "historical": ("HISTORICAL", "nt-hist")}


def tag(kind):
    text, css = TAGS[kind]
    return f"<span class='nt-tag {css}'>{text}</span>"


def setup_page(title, icon, subtitle):
    """Page config, shared style and the page header. Call once at the top of every page."""
    page_title = title if title == "NeuroTraffic" else f"{title} — NeuroTraffic"
    st.set_page_config(page_title=page_title, page_icon=icon, layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"<div class='nt-title'>{icon} {title}</div><div class='nt-sub'>{subtitle}</div>", unsafe_allow_html=True)


def section(title, caption=None, kind=None):
    """Section title with an optional one-line explanation and LIVE/ESTIMATED/HISTORICAL tag."""
    st.markdown(f"<div class='nt-section'>{title}{tag(kind) if kind else ''}</div>"
                + (f"<div class='nt-caption'>{caption}</div>" if caption else ""), unsafe_allow_html=True)


def level_legend():
    """Colour key for the four anomaly levels."""
    bounds = {"Disruption": "≥ 0.80", "Anomaly": "0.50 – 0.80", "Elevated": "0.25 – 0.50", "Normal": "< 0.25"}
    items = "".join(f"<span><span class='nt-dot' style='background:{c}'></span>{n} {bounds[n]}</span>"
                    for _, n, c in reversed(LEVELS))
    st.markdown(f"<div class='nt-legend'>{items}</div>", unsafe_allow_html=True)


def style_chart(fig, height=300, **layout):
    """Transparent dark Plotly style shared by every chart."""
    style = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
                 margin=dict(l=0, r=10, t=10, b=0), height=height,
                 legend=dict(bgcolor="rgba(9,14,26,0.8)", font=dict(color="#e2e8f0", size=11)))
    style.update(layout)                     # a page's own settings win over the defaults
    fig.update_layout(**style)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", color="#94a3b8")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", color="#94a3b8")
    return fig


def zone_map(fig, height=480, zoom=None):
    """Dark NYC basemap for Plotly Scattermap figures."""
    fig.update_layout(map=dict(style="carto-darkmatter", center=config.MAP_CENTER, zoom=zoom or config.MAP_ZOOM),
                      paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0), height=height,
                      legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=0.99,
                                  bgcolor="rgba(9,14,26,0.85)", font=dict(color="#e2e8f0", size=11)))
    return fig
