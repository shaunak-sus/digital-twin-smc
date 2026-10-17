"""

"""

import json
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ═══════════════════════════ CONFIG ════════════════════════════════════════
# Update these paths to match your machine
CSV_PATH         = r"cycle_log.csv"
LIVE_STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "live_machine_status.json")
REFRESH_INTERVAL = 1  # seconds

# ═══════════════════════════ PAGE SETUP ════════════════════════════════════
st.set_page_config(
    page_title  = "Digital Twin | SMC Machining Center",
    page_icon   = "⚙️",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ═══════════════════════════ DESIGN TOKENS & CSS ═══════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #0A0E17; color: #E2E8F0; }
  section[data-testid="stSidebar"] { background: #0D1220; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1.2rem 1.8rem 2rem; max-width: 100%; }

  /* ── Header bar ── */
  .dt-header {
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(90deg, #111827 0%, #0D1829 100%);
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 14px 24px 14px 20px;
    margin-bottom: 18px;
  }
  .dt-header-left { display: flex; align-items: center; gap: 14px; }
  .dt-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem; font-weight: 700;
    color: #00D4FF; letter-spacing: 0.04em;
  }
  .dt-subtitle {
    font-size: 0.75rem; color: #64748B;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.06em; text-transform: uppercase;
  }
  .dt-live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #0F2A1A; border: 1px solid #166534;
    border-radius: 20px; padding: 4px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: #10B981; letter-spacing: 0.08em;
  }
  .dt-live-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #10B981;
    animation: pulse-dot 1.4s ease-in-out infinite;
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
  }

  /* ── Section headers ── */
  .section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 500;
    color: #475569; letter-spacing: 0.12em;
    text-transform: uppercase;
    border-left: 3px solid #1E40AF;
    padding-left: 10px; margin: 22px 0 12px;
  }

  /* ── Status cards (Section 1) ── */
  .status-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 16px 18px 14px;
    height: 100%;
  }
  .status-card-label {
    font-size: 0.62rem; font-weight: 600;
    color: #475569; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 8px;
  }
  .status-card-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem; font-weight: 700;
    line-height: 1.2;
  }
  .status-dot {
    display: inline-block; width: 9px; height: 9px;
    border-radius: 50%; margin-right: 7px; flex-shrink: 0;
  }
  .status-row { display: flex; align-items: center; }

  /* colours */
  .c-cyan    { color: #00D4FF; }
  .c-green   { color: #10B981; }
  .c-amber   { color: #F59E0B; }
  .c-red     { color: #EF4444; }
  .c-gray    { color: #94A3B8; }
  .c-white   { color: #F1F5F9; }
  .bg-cyan   { background: #00D4FF; }
  .bg-green  { background: #10B981; }
  .bg-amber  { background: #F59E0B; }
  .bg-red    { background: #EF4444; }
  .bg-gray   { background: #475569; }

  /* ── KPI cards (Section 2) ── */
  .kpi-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 14px 16px 12px;
    position: relative; overflow: hidden;
  }
  .kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
  }
  .kpi-accent-cyan::before   { background: #00D4FF; }
  .kpi-accent-green::before  { background: #10B981; }
  .kpi-accent-amber::before  { background: #F59E0B; }
  .kpi-accent-red::before    { background: #EF4444; }
  .kpi-accent-indigo::before { background: #6366F1; }
  .kpi-accent-violet::before { background: #8B5CF6; }

  .kpi-label {
    font-size: 0.6rem; font-weight: 600;
    color: #475569; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 6px;
  }
  .kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem; font-weight: 700;
    color: #F1F5F9; line-height: 1;
  }
  .kpi-unit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; color: #64748B; margin-left: 4px;
  }
  .kpi-sub {
    font-size: 0.62rem; color: #475569; margin-top: 5px;
  }

  /* ── FSM diagram ── */
  .fsm-wrapper {
    background: #0D1220;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 22px 18px;
    overflow-x: auto;
  }

  /* ── Component metric rows ── */
  .comp-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 16px 18px;
  }
  .comp-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 14px;
  }
  .metric-row {
    display: flex; justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px solid #1E293B;
    padding: 7px 0;
    font-size: 0.78rem;
  }
  .metric-row:last-child { border-bottom: none; }
  .metric-name { color: #64748B; }
  .metric-val  {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem; font-weight: 600; color: #E2E8F0;
  }

  /* ── Event log ── */
  .event-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.75rem;
  }
  .event-table th {
    background: #0D1220;
    color: #475569;
    font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 8px 12px; text-align: left;
    border-bottom: 1px solid #1E293B;
    position: sticky; top: 0;
  }
  .event-table td {
    padding: 7px 12px;
    border-bottom: 1px solid #1A2236;
    color: #CBD5E1;
    font-family: 'JetBrains Mono', monospace;
  }
  .event-table tr:hover td { background: #131F35; }
  .badge {
    display: inline-block;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.06em;
  }
  .badge-complete  { background: #064E3B; color: #34D399; }
  .badge-emergency { background: #450A0A; color: #F87171; }
  .badge-default   { background: #1E293B; color: #94A3B8; }

  /* ── Plotly chart container ── */
  .chart-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 4px 6px;
  }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════ DATA LOADERS ══════════════════════════════════
def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        # Ensure required columns
        for col in ["cycle_number", "total_cycle_time",
                    "piston_active_time", "actuator_active_time",
                    "emergency_events", "cycle_start_time", "cycle_end_time"]:
            if col not in df.columns:
                df[col] = None
        # Keep only CYCLE_COMPLETE rows for analytics
        df["_is_cycle"] = df["event"] == "CYCLE_COMPLETE"
        return df
    except Exception:
        return pd.DataFrame()


def load_live(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return round(s.mean(), 2) if len(s) else 0.0

def safe_max(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return round(s.max(), 2) if len(s) else 0.0

def safe_min(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return round(s.min(), 2) if len(s) else 0.0

def safe_sum(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return round(s.sum(), 2) if len(s) else 0.0


# ═══════════════════════════ CHART THEME ═══════════════════════════════════
CHART_LAYOUT = dict(
    paper_bgcolor = "#111827",
    plot_bgcolor  = "#0D1220",
    font          = dict(family="JetBrains Mono, monospace", color="#64748B", size=11),
    margin        = dict(l=10, r=10, t=30, b=10),
    xaxis         = dict(gridcolor="#1E293B", zerolinecolor="#1E293B",
                         tickfont=dict(size=10)),
    yaxis         = dict(gridcolor="#1E293B", zerolinecolor="#1E293B",
                         tickfont=dict(size=10)),
    legend        = dict(bgcolor="#111827", bordercolor="#1E293B", borderwidth=1),
    hoverlabel    = dict(bgcolor="#0D1220", bordercolor="#1E293B",
                         font_size=11, font_family="JetBrains Mono"),
)


# ═══════════════════════════ FSM DIAGRAM ═══════════════════════════════════
FSM_STATES = [
    "IDLE", "OBJECT_PRESENT", "PISTON_EXTENDING",
    "PISTON_RETRACTING", "ACTUATOR_ACTIVE", "CYCLE_COMPLETE",
]
FSM_LABELS = [
    "IDLE", "OBJECT\nPRESENT", "PISTON\nEXTENDING",
    "PISTON\nRETRACTING", "ACTUATOR\nACTIVE", "CYCLE\nCOMPLETE",
]
FSM_COLORS = {
    "IDLE":               "#1E293B",
    "OBJECT_PRESENT":     "#1A2F4A",
    "PISTON_EXTENDING":   "#0F2A2A",
    "PISTON_RETRACTING":  "#18A279",
    "ACTUATOR_ACTIVE":    "#2117AF",
    "CYCLE_COMPLETE":     "#5AA925",
    "EMERGENCY":          "#3A0A0A",
}
FSM_TEXT_COLORS = {
    "IDLE":               "#94A3B8",
    "OBJECT_PRESENT":     "#60A5FA",
    "PISTON_EXTENDING":   "#22D3EE",
    "PISTON_RETRACTING":  "#FCD34D",
    "ACTUATOR_ACTIVE":    "#34D399",
    "CYCLE_COMPLETE":     "#86EFAC",
    "EMERGENCY":          "#F87171",
}

def build_fsm_figure(active_state: str) -> go.Figure:
    n      = len(FSM_STATES)
    xs     = list(range(n))
    y      = [0] * n

    fig = go.Figure()

    # Arrow lines between states
    for i in range(n - 1):
        fig.add_shape(type="line",
                      x0=xs[i] + 0.38, y0=0,
                      x1=xs[i + 1] - 0.38, y1=0,
                      line=dict(color="#1E3A5F", width=2))
        # Arrowhead (triangle approximation)
        ax = xs[i + 1] - 0.38
        fig.add_annotation(
            x=xs[i + 1] - 0.37, y=0,
            ax=xs[i] + 0.38, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowcolor="#1E3A5F", arrowwidth=2,
        )

    # State nodes
    for i, (state, label) in enumerate(zip(FSM_STATES, FSM_LABELS)):
        is_active = (state == active_state)
        fill_col  = "#00304A" if is_active else FSM_COLORS.get(state, "#1E293B")
        text_col  = "#00D4FF" if is_active else FSM_TEXT_COLORS.get(state, "#94A3B8")
        border    = "#00D4FF" if is_active else "#1E3A5F"
        bw        = 2.5 if is_active else 1

        fig.add_shape(
            type="rect",
            x0=xs[i] - 0.37, y0=-0.38,
            x1=xs[i] + 0.37, y1=0.38,
            line=dict(color=border, width=bw),
            fillcolor=fill_col,
            layer="below",
        )

        # Glow effect for active state
        if is_active:
            fig.add_shape(
                type="rect",
                x0=xs[i] - 0.41, y0=-0.42,
                x1=xs[i] + 0.41, y1=0.42,
                line=dict(color="#003A5A", width=4),
                fillcolor="rgba(0,0,0,0)",
                layer="below",
            )

        fig.add_annotation(
            x=xs[i], y=0,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(
                family="JetBrains Mono, monospace",
                size=10.5 if is_active else 9.5,
                color=text_col,
            ),
            align="center",
        )

    # Emergency annotation
    if active_state == "EMERGENCY":
        fig.add_annotation(
            x=n / 2 - 0.5, y=-0.68,
            text="⚠  EMERGENCY — GREEN LIGHT OFF",
            showarrow=False,
            font=dict(family="JetBrains Mono", size=11, color="#EF4444"),
        )

    fig.update_layout(
        **CHART_LAYOUT,
        height      = 160,
        xaxis       = dict(visible=False, range=[-0.6, n - 0.4]),
        yaxis       = dict(visible=False, range=[-0.8, 0.7]),
        margin      = dict(l=10, r=10, t=10, b=10),
        showlegend  = False,
    )
    return fig


# ═══════════════════════════ TIMELINE BUILDER ══════════════════════════════
def build_timeline(df: pd.DataFrame) -> go.Figure:
    cycles = df[df["_is_cycle"]].copy()
    if cycles.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, height=180,
                          title=dict(text="No cycle data yet", font=dict(color="#475569")))
        return fig

    cycles = cycles.reset_index(drop=True)
    cycles["cycle_number"] = pd.to_numeric(cycles["cycle_number"], errors="coerce")
    cycles["total_cycle_time"]     = pd.to_numeric(cycles["total_cycle_time"],     errors="coerce").fillna(0)
    cycles["piston_active_time"]   = pd.to_numeric(cycles["piston_active_time"],   errors="coerce").fillna(0)
    cycles["actuator_active_time"] = pd.to_numeric(cycles["actuator_active_time"], errors="coerce").fillna(0)

    fig = go.Figure()
    for _, row in cycles.iterrows():
        c    = row["cycle_number"]
        tot  = row["total_cycle_time"]
        pist = row["piston_active_time"]
        act  = row["actuator_active_time"]
        idle = max(0, tot - pist - act)

        # Idle segment
        if idle > 0:
            fig.add_trace(go.Bar(
                x=[idle], y=[c], orientation="h",
                name="IDLE" if c == cycles["cycle_number"].iloc[0] else "",
                marker_color="#1E293B",
                showlegend=(c == cycles["cycle_number"].iloc[0]),
                legendgroup="idle",
                hovertemplate=f"Cycle {c:.0f} — Idle: {idle:.2f}s<extra></extra>",
            ))
        # Piston segment
        if pist > 0:
            fig.add_trace(go.Bar(
                x=[pist], y=[c], orientation="h",
                name="PISTON" if c == cycles["cycle_number"].iloc[0] else "",
                marker_color="#0891B2",
                showlegend=(c == cycles["cycle_number"].iloc[0]),
                legendgroup="piston",
                hovertemplate=f"Cycle {c:.0f} — Piston: {pist:.2f}s<extra></extra>",
            ))
        # Actuator segment
        if act > 0:
            fig.add_trace(go.Bar(
                x=[act], y=[c], orientation="h",
                name="ACTUATOR" if c == cycles["cycle_number"].iloc[0] else "",
                marker_color="#D97706",
                showlegend=(c == cycles["cycle_number"].iloc[0]),
                legendgroup="actuator",
                hovertemplate=f"Cycle {c:.0f} — Actuator: {act:.2f}s<extra></extra>",
            ))

    fig.update_layout(
        **CHART_LAYOUT,
        barmode  = "stack",
        height   = 220,
        xaxis    = dict(**CHART_LAYOUT["xaxis"], title="Time (s)"),
        yaxis    = dict(**CHART_LAYOUT["yaxis"], title="Cycle #",
                        tickmode="linear", dtick=1),
        legend   = dict(**CHART_LAYOUT["legend"], orientation="h",
                        yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ═══════════════════════════ TREND CHARTS ══════════════════════════════════
def trend_chart(df: pd.DataFrame, y_col: str, title: str,
                color: str, unit: str = "s") -> go.Figure:
    cycles = df[df["_is_cycle"]].copy()
    if cycles.empty or y_col not in cycles.columns:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, height=200,
                          title=dict(text="No data", font=dict(color="#475569")))
        return fig

    cycles[y_col]          = pd.to_numeric(cycles[y_col], errors="coerce")
    cycles["cycle_number"] = pd.to_numeric(cycles["cycle_number"], errors="coerce")
    cycles = cycles.dropna(subset=[y_col, "cycle_number"]).sort_values("cycle_number")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cycles["cycle_number"], y=cycles[y_col],
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(color=color, size=5),
        fill="tozeroy",
        fillcolor=color.replace(")", ", 0.08)").replace("rgb", "rgba")
                   if color.startswith("rgb") else color + "14",
        hovertemplate=f"Cycle %{{x}}: %{{y:.2f}}{unit}<extra></extra>",
    ))
    # Average line
    avg = cycles[y_col].mean()
    fig.add_hline(y=avg, line_dash="dot", line_color="#475569", line_width=1,
                  annotation_text=f"avg {avg:.2f}{unit}",
                  annotation_font_size=9, annotation_font_color="#475569")

    fig.update_layout(**{
        **CHART_LAYOUT,
        "height": 180,
        "title": dict(text=title, font=dict(size=11, color="#64748B"), x=0.01),
        "showlegend": False,
    })
    return fig


def throughput_chart(df: pd.DataFrame) -> go.Figure:
    cycles = df[df["_is_cycle"]].copy()
    if cycles.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, height=200)
        return fig

    cycles["cycle_number"] = pd.to_numeric(cycles["cycle_number"], errors="coerce")
    cycles["total_cycle_time"] = pd.to_numeric(cycles["total_cycle_time"], errors="coerce").fillna(1)
    # Rolling throughput: cycles per hour using last 5 cycle average
    cycles = cycles.sort_values("cycle_number").dropna(subset=["cycle_number"])
    cycles["rolling_avg_ct"] = cycles["total_cycle_time"].rolling(5, min_periods=1).mean()
    cycles["throughput_cph"] = (3600 / cycles["rolling_avg_ct"]).round(1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cycles["cycle_number"], y=cycles["throughput_cph"],
        mode="lines+markers",
        line=dict(color="#6366F1", width=2),
        marker=dict(color="#6366F1", size=5),
        fill="tozeroy", fillcolor="#6366F114",
        hovertemplate="Cycle %{x}: %{y:.1f} cycles/hr<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=200,
        title=dict(text="Throughput Trend (cycles/hr)", font=dict(size=11, color="#64748B"), x=0.01),
        xaxis=dict(**CHART_LAYOUT["xaxis"], title="Cycle #"),
        yaxis=dict(**CHART_LAYOUT["yaxis"], title="cycles/hr"),
        showlegend=False,
    )
    return fig


# ═══════════════════════════ HELPER: STATUS CARD HTML ══════════════════════
def status_card(label: str, value: str, color_class: str, dot: bool = True) -> str:
    dot_html = (f'<span class="status-dot bg-{color_class.split("-")[1]}"></span>'
                if dot and color_class.startswith("c-") else "")
    return f"""
    <div class="status-card">
      <div class="status-card-label">{label}</div>
      <div class="status-row">
        {dot_html}
        <div class="status-card-value {color_class}">{value}</div>
      </div>
    </div>"""


def kpi_card(label: str, value, unit: str = "",
             sub: str = "", accent: str = "cyan") -> str:
    return f"""
    <div class="kpi-card kpi-accent-{accent}">
      <div class="kpi-label">{label}</div>
      <div>
        <span class="kpi-value">{value}</span>
        <span class="kpi-unit">{unit}</span>
      </div>
      {f'<div class="kpi-sub">{sub}</div>' if sub else ""}
    </div>"""


def comp_metrics(title: str, color: str, metrics: list) -> str:
    rows = "".join(
        f'<div class="metric-row"><span class="metric-name">{n}</span>'
        f'<span class="metric-val">{v}</span></div>'
        for n, v in metrics
    )
    return f"""
    <div class="comp-card">
      <div class="comp-title" style="color:{color}">{title}</div>
      {rows}
    </div>"""


# ═══════════════════════════ MAIN RENDER ═══════════════════════════════════
def render():
    st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="datarefresh")
    df   = load_csv(CSV_PATH)
    live = load_live(LIVE_STATUS_PATH)

    # ── HEADER ───────────────────────────────────────────────────────────────
    now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    st.markdown(f"""
    <div class="dt-header">
      <div class="dt-header-left">
        <div>
          <div class="dt-logo">⚙ DIGITAL TWIN</div>
          <div class="dt-subtitle">SMC Virtual Machining Center · IIT Indore</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:16px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#334155;">{now_str}</div>
        <div class="dt-live-badge">
          <div class="dt-live-dot"></div>LIVE
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Derive KPIs ───────────────────────────────────────────────────────────
    cycles_df = df[df["_is_cycle"]] if not df.empty else pd.DataFrame()
    ct_series  = pd.to_numeric(cycles_df.get("total_cycle_time",     pd.Series()), errors="coerce").dropna()
    pt_series  = pd.to_numeric(cycles_df.get("piston_active_time",   pd.Series()), errors="coerce").dropna()
    at_series  = pd.to_numeric(cycles_df.get("actuator_active_time", pd.Series()), errors="coerce").dropna()

    total_cycles    = len(ct_series)
    avg_ct          = safe_mean(ct_series)
    last_ct         = round(ct_series.iloc[-1], 2) if len(ct_series) else 0.0
    best_ct         = safe_min(ct_series)
    worst_ct        = safe_max(ct_series)
    total_run_time  = safe_sum(ct_series)
    throughput_cph  = round(3600 / avg_ct, 1) if avg_ct > 0 else 0.0
    emerg_count     = int(df["event"].eq("EMERGENCY_STOP").sum()) if not df.empty else 0

    # Utilization: piston+actuator time / total cycle time
    utilization = 0.0
    if total_run_time > 0:
        util_num = safe_sum(pt_series) + safe_sum(at_series)
        utilization = min(round(util_num / total_run_time * 100, 1), 100.0)

    # Current cycle elapsed from live status
    current_ct   = live.get("cycle_elapsed_s", 0.0)
    fsm_state    = live.get("state", "IDLE")
    obj_pres     = live.get("object_present", False)
    pist_st      = live.get("piston_state", "IDLE")
    act_st       = live.get("actuator_state", "IDLE")
    gl_st        = live.get("green_light_state", "OFF")
    vest_status  = live.get("vest_status", "NO_WORKER")
    vest_total   = live.get("vest_workers_total", 0)
    vest_ok      = live.get("vest_workers_compliant", 0)
    vest_viol    = live.get("vest_workers_violation", 0)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: Live Machine Status
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">01 — Live Machine Status</div>',
                unsafe_allow_html=True)

    # FSM state colour mapping
    state_color = {
        "IDLE": "c-gray", "OBJECT_PRESENT": "c-cyan",
        "PISTON_EXTENDING": "c-cyan", "PISTON_RETRACTING": "c-amber",
        "ACTUATOR_ACTIVE": "c-green", "CYCLE_COMPLETE": "c-green",
        "EMERGENCY": "c-red",
    }
    machine_status = ("EMERGENCY" if fsm_state == "EMERGENCY"
                      else "IDLE" if fsm_state == "IDLE" else "RUNNING")
    machine_color  = {"RUNNING": "c-green", "IDLE": "c-gray",
                      "EMERGENCY": "c-red"}[machine_status]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.markdown(status_card("FSM State", fsm_state,
                                     state_color.get(fsm_state, "c-gray"), dot=False),
                         unsafe_allow_html=True)
    with c2: st.markdown(status_card("Machine Status", machine_status,
                                     machine_color), unsafe_allow_html=True)
    with c3: st.markdown(status_card("Green Light",
                                     "ON" if gl_st == "ON" else "OFF",
                                     "c-green" if gl_st == "ON" else "c-red"),
                         unsafe_allow_html=True)
    with c4: st.markdown(status_card("Object",
                                     "PRESENT" if obj_pres else "ABSENT",
                                     "c-green" if obj_pres else "c-gray"),
                         unsafe_allow_html=True)
    with c5: st.markdown(status_card("Piston", pist_st,
                                     "c-cyan" if pist_st == "ACTIVE" else "c-gray"),
                         unsafe_allow_html=True)
    with c6: st.markdown(status_card("Actuator", act_st,
                                     "c-amber" if act_st == "ACTIVE" else "c-gray"),
                         unsafe_allow_html=True)

    # Vest violation alert banner
    if vest_status == "VIOLATION":
        st.markdown(f"""
        <div style="background:#450A0A;border:1.5px solid #EF4444;border-radius:8px;
                    padding:12px 20px;margin:10px 0;display:flex;align-items:center;gap:14px;">
          <span style="font-size:1.6rem;">⚠️</span>
          <div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                        font-weight:700;color:#F87171;letter-spacing:0.06em;">
              SAFETY VIOLATION — WORKER WITHOUT SAFETY VEST
            </div>
            <div style="font-size:0.72rem;color:#FCA5A5;margin-top:3px;">
              {vest_viol} worker(s) detected without vest · {vest_ok} compliant · {vest_total} total
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: Production KPIs
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">02 — Production KPIs</div>',
                unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k5, k6, k7, k8 = st.columns(4)

    with k1: st.markdown(kpi_card("Total Cycles", total_cycles, "", accent="cyan"),
                         unsafe_allow_html=True)
    with k2: st.markdown(kpi_card("Current Cycle", f"{current_ct:.1f}", "s",
                                  sub="elapsed this cycle", accent="cyan"),
                         unsafe_allow_html=True)
    with k3: st.markdown(kpi_card("Last Cycle Time", f"{last_ct:.2f}", "s",
                                  accent="green"), unsafe_allow_html=True)
    with k4: st.markdown(kpi_card("Avg Cycle Time", f"{avg_ct:.2f}", "s",
                                  accent="green"), unsafe_allow_html=True)

    with k5: st.markdown(kpi_card("Best Cycle", f"{best_ct:.2f}", "s",
                                  accent="green"), unsafe_allow_html=True)
    with k6: st.markdown(kpi_card("Worst Cycle", f"{worst_ct:.2f}", "s",
                                  accent="amber"), unsafe_allow_html=True)
    with k7: st.markdown(kpi_card("Machine Utilization", f"{utilization:.1f}", "%",
                                  sub="piston + actuator / total", accent="indigo"),
                         unsafe_allow_html=True)
    with k8: st.markdown(kpi_card("Throughput", f"{throughput_cph:.1f}", "c/hr",
                                  sub=f"Emergencies: {emerg_count}", accent="violet"),
                         unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: Component Analytics
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">03 — Component Analytics</div>',
                unsafe_allow_html=True)

    cur_piston   = live.get("current_cycle_piston_time",   0.0)
    cur_actuator = live.get("current_cycle_actuator_time", 0.0)

    ca1, ca2 = st.columns(2)
    with ca1:
        st.markdown(comp_metrics("⬡ Piston", "#00D4FF", [
            ("Current Active Time",  f"{cur_piston:.2f} s"),
            ("Average Active Time",  f"{safe_mean(pt_series):.2f} s"),
            ("Maximum Active Time",  f"{safe_max(pt_series):.2f} s"),
            ("Minimum Active Time",  f"{safe_min(pt_series):.2f} s"),
            ("Total Active Time",    f"{safe_sum(pt_series):.2f} s"),
        ]), unsafe_allow_html=True)

    with ca2:
        st.markdown(comp_metrics("⬡ Actuator", "#F59E0B", [
            ("Current Active Time",  f"{cur_actuator:.2f} s"),
            ("Average Active Time",  f"{safe_mean(at_series):.2f} s"),
            ("Maximum Active Time",  f"{safe_max(at_series):.2f} s"),
            ("Minimum Active Time",  f"{safe_min(at_series):.2f} s"),
            ("Total Active Time",    f"{safe_sum(at_series):.2f} s"),
        ]), unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3b: Safety / PPE Status
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">03b — Safety &amp; PPE Compliance</div>',
                unsafe_allow_html=True)

    vest_badge_col = (
        "#EF4444" if vest_status == "VIOLATION" else
        "#10B981" if vest_status == "COMPLIANT" else "#475569"
    )
    vest_bg_col = (
        "#450A0A" if vest_status == "VIOLATION" else
        "#064E3B" if vest_status == "COMPLIANT" else "#1E293B"
    )
    sv1, sv2, sv3, sv4 = st.columns(4)
    with sv1:
        st.markdown(f"""
        <div class="kpi-card" style="border-top:3px solid {vest_badge_col};">
          <div class="kpi-label">Vest Status</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                      font-weight:700;color:{vest_badge_col};">{vest_status}</div>
        </div>""", unsafe_allow_html=True)
    with sv2:
        st.markdown(kpi_card("Workers Detected", vest_total, "", accent="cyan"), unsafe_allow_html=True)
    with sv3:
        st.markdown(kpi_card("Compliant", vest_ok, "", accent="green"), unsafe_allow_html=True)
    with sv4:
        st.markdown(kpi_card("Violations", vest_viol, "", accent="red"), unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4: FSM Diagram
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">04 — Finite State Machine</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="fsm-wrapper">', unsafe_allow_html=True)
    st.plotly_chart(build_fsm_figure(fsm_state),
                    use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5: Trend Analysis
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">05 — Trend Analysis</div>',
                unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(
            trend_chart(df, "total_cycle_time",
                        "Cycle Time Trend", "#00D4FF"),
            use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with t2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(
            trend_chart(df, "piston_active_time",
                        "Piston Active Time Trend", "#0891B2"),
            use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    t3, t4 = st.columns(2)
    with t3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(
            trend_chart(df, "actuator_active_time",
                        "Actuator Active Time Trend", "#D97706"),
            use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with t4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(
            throughput_chart(df),
            use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6: Machine Timeline
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">06 — Machine Timeline</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(build_timeline(df), use_container_width=True,
                    config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 7: Event Log
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">07 — Event Log</div>',
                unsafe_allow_html=True)

    if df.empty:
        st.markdown('<div style="color:#475569;font-size:0.8rem;padding:20px;">'
                    'No events logged yet. Start the Digital Twin to record cycles.'
                    '</div>', unsafe_allow_html=True)
    else:
        log_df = df.copy().iloc[::-1]  # newest first
        log_df["cycle_number"]    = pd.to_numeric(log_df.get("cycle_number",    0), errors="coerce").fillna(0).astype(int)
        log_df["total_cycle_time"] = pd.to_numeric(log_df.get("total_cycle_time", 0), errors="coerce").fillna(0)

        rows_html = ""
        for _, row in log_df.head(50).iterrows():
            ev    = str(row.get("event", ""))
            badge = ("badge-complete"  if ev == "CYCLE_COMPLETE"  else
                     "badge-emergency" if ev == "EMERGENCY_STOP"  else "badge-default")
            cyc  = int(row.get("cycle_number", 0)) or "—"
            dur  = row.get("total_cycle_time", 0)
            dur_s = f"{dur:.2f}s" if dur else "—"
            ts   = str(row.get("timestamp", ""))
            state = str(row.get("machine_state", fsm_state))

            rows_html += f"""
            <tr>
              <td>{ts}</td>
              <td><span class="badge {badge}">{ev}</span></td>
              <td style="color:#64748B">{state}</td>
              <td>{cyc}</td>
              <td>{dur_s}</td>
              <td><span class="badge badge-default">OK</span></td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1E293B;border-radius:10px;
                    overflow:hidden;max-height:320px;overflow-y:auto;">
          <table class="event-table">
            <thead>
              <tr>
                <th>Timestamp</th><th>Event</th><th>State</th>
                <th>Cycle #</th><th>Duration</th><th>Status</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

    # ── FOOTER ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:32px;padding:16px 0;
                border-top:1px solid #1E293B;
                display:flex;justify-content:space-between;align-items:center;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#1E293B;">
        DIGITAL TWIN v3 · IIT Indore · SMC Machining Center
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#1E293B;">
        FUTURE: PLC · OPC-UA · MQTT · TwinCAT · Multi-Station
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════ ENTRY POINT ═══════════════════════════════════
if __name__ == "__main__":
    render()
else:
    render()
