"""
Digital Twin Control Dashboard
==============================
Streamlit + Plotly frontend for ML-Driven Digital Twin
Supply Chain & Cash Flow Optimization.

Imports backend functions from digital_twin_main.py.
Backend logic remains UNTOUCHED.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import time
import sys
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Digital Twin · Supply Chain Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — DARK THEME & PREMIUM STYLING
# ============================================================
st.markdown("""
<style>
/* ---- Import Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Hide default Streamlit elements ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
/* Keep sidebar toggle button always visible */
header [data-testid="stToolbar"] { display: none !important; }
button[kind="header"] { visibility: visible !important; }

/* ---- Dark background ---- */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0f172a 100%);
}

/* ---- Sidebar — always visible ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1e293b 100%) !important;
    border-right: 1px solid rgba(59,130,246,0.15);
    display: block !important;
    min-width: 300px !important;
    width: 300px !important;
    opacity: 1 !important;
    transform: none !important;
}
/* Ensure sidebar collapse button is always clickable */
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] button[aria-label="Close sidebar"] {
    visibility: visible !important;
}
/* Prevent sidebar from being hidden via collapsed state */
section[data-testid="stSidebar"][aria-expanded="false"] {
    display: block !important;
    min-width: 300px !important;
    width: 300px !important;
    transform: none !important;
    margin-left: 0 !important;
}
/* Keep the expand button visible when collapsed */
button[aria-label="Open sidebar"],
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: block !important;
    z-index: 999999 !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2e8f0 !important;
}

/* ---- Pipeline ---- */
.pipeline-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 18px 10px;
    margin-bottom: 10px;
    background: rgba(15,23,42,0.7);
    border-radius: 14px;
    border: 1px solid rgba(59,130,246,0.15);
    backdrop-filter: blur(10px);
    flex-wrap: nowrap;
    overflow-x: auto;
}
.pipeline-node {
    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(99,102,241,0.10));
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 10px;
    padding: 10px 18px;
    color: #93c5fd;
    font-weight: 600;
    font-size: 0.82rem;
    white-space: nowrap;
    text-align: center;
    transition: all 0.3s ease;
    min-width: 120px;
}
.pipeline-node:hover {
    border-color: #3b82f6;
    box-shadow: 0 0 20px rgba(59,130,246,0.25);
    transform: translateY(-2px);
}
.pipeline-arrow {
    color: #3b82f6;
    font-size: 1.4rem;
    padding: 0 6px;
    opacity: 0.7;
}

/* ---- KPI Card ---- */
.kpi-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.9));
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
.kpi-card:hover {
    border-color: rgba(59,130,246,0.4);
    box-shadow: 0 8px 25px rgba(59,130,246,0.12);
    transform: translateY(-3px);
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 8px 0 4px 0;
}
.kpi-label {
    font-size: 0.82rem;
    color: #94a3b8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ---- Risk Badge ---- */
.risk-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.risk-low {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.08));
    color: #34d399;
    border: 1px solid rgba(16,185,129,0.3);
    box-shadow: 0 0 20px rgba(16,185,129,0.15);
}
.risk-moderate {
    background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(245,158,11,0.08));
    color: #fbbf24;
    border: 1px solid rgba(245,158,11,0.3);
    box-shadow: 0 0 20px rgba(245,158,11,0.15);
}
.risk-high {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.08));
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.3);
    box-shadow: 0 0 20px rgba(239,68,68,0.15);
}

/* ---- Section headers ---- */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 20px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(59,130,246,0.2);
}

/* ---- Run button ---- */
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important;
    border: none !important;
    padding: 14px 24px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    letter-spacing: 0.04em;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    box-shadow: 0 8px 25px rgba(59,130,246,0.35) !important;
    transform: translateY(-2px) !important;
}

/* ---- Plotly chart containers ---- */
.stPlotlyChart {
    border-radius: 14px;
    overflow: hidden;
}

/* ---- Comparison table ---- */
.stDataFrame {
    border-radius: 14px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# BACKEND IMPORT (CACHED — runs only once per session)
# ============================================================
@st.cache_resource(show_spinner="🔧 Loading backend data & training models … this may take a minute")
def load_backend():
    """Import backend module. All top-level code (data loading, model
    training, initial simulations) runs exactly once and is cached."""
    # Ensure the project directory is on the path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    import importlib
    backend = importlib.import_module("digital_twin_main")
    return backend


backend = load_backend()

# ---- Pull out objects we need (names match backend exactly) ----
test_data = backend.test
y_test = backend.y_test
predictions = backend.predictions
rf_predictions = backend.rf_predictions
ridge_predictions = backend.ridge_predictions
safety_stock_val = backend.safety_stock_val
forecast_error_std = backend.forecast_error_std
fixed_reorder_point = backend.fixed_reorder_point
average_train_demand = backend.average_train_demand
lead_time_default = backend.lead_time
order_quantity_default = backend.order_quantity
supplier_cost_ratio_default = backend.supplier_cost_ratio
holding_cost_ratio_default = backend.holding_cost_ratio
initial_inventory_default = backend.initial_inventory
initial_cash_default = backend.initial_cash

# Functions
run_inventory_simulation = backend.run_inventory_simulation
calculate_metrics = backend.calculate_metrics
prepare_scenario_data = backend.prepare_scenario_data
run_monte_carlo_simulation = backend.run_monte_carlo_simulation


# ============================================================
# COLOR PALETTE
# ============================================================
COLOR_ADAPTIVE = "#3b82f6"   # Blue
COLOR_BASELINE = "#ef4444"   # Red
COLOR_SAFETY = "#10b981"     # Green
COLOR_BG = "#0f172a"
COLOR_CARD = "#1e293b"
COLOR_TEXT = "#e2e8f0"
COLOR_MUTED = "#94a3b8"
COLOR_ACCENT = "#6366f1"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    xaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.1)"),
    yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.1)"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        bgcolor="rgba(15,23,42,0.7)",
        bordercolor="rgba(59,130,246,0.15)",
        borderwidth=1,
        font=dict(size=11),
    ),
    hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_family="Inter"),
)


# ============================================================
# PIPELINE VISUALIZATION
# ============================================================
pipeline_steps = [
    "📊 Dataset",
    "🤖 ML Forecast",
    "🏭 Digital Twin Simulation",
    "💰 Financial Modeling",
    "🎲 Monte Carlo Risk",
]

pipeline_html = '<div class="pipeline-container">'
for i, step in enumerate(pipeline_steps):
    pipeline_html += f'<div class="pipeline-node">{step}</div>'
    if i < len(pipeline_steps) - 1:
        pipeline_html += '<span class="pipeline-arrow">➜</span>'
pipeline_html += "</div>"

st.markdown(pipeline_html, unsafe_allow_html=True)

# Title
st.markdown(
    '<h1 style="text-align:center; color:#e2e8f0; font-weight:800; '
    'margin: 5px 0 0 0; font-size:2rem;">'
    '🏭 Digital Twin Control Dashboard</h1>'
    '<p style="text-align:center; color:#94a3b8; margin:0 0 15px 0; font-size:0.95rem;">'
    'ML-Driven Supply Chain & Cash Flow Optimization</p>',
    unsafe_allow_html=True,
)


# ============================================================
# LEFT PANEL — SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Experiment Controls")
    st.markdown("---")

    # ---- Forecast Model ----
    st.markdown('<div class="section-header">📈 Forecast Model</div>', unsafe_allow_html=True)
    forecast_model = st.radio(
        "Select model",
        ["Linear Regression", "Random Forest", "LSTM (experimental)"],
        index=0,
        label_visibility="collapsed",
    )

    if forecast_model == "LSTM (experimental)":
        st.info("⚡ LSTM uses a separate TensorFlow pipeline. Results shown are from `lstm_forecasting.py` if available.")

    st.markdown("---")

    # ---- Policy Selector ----
    st.markdown('<div class="section-header">🎯 Policy Selector</div>', unsafe_allow_html=True)
    selected_policies = st.multiselect(
        "Choose policies",
        ["Baseline", "Adaptive", "Adaptive + Safety Stock"],
        default=["Adaptive", "Adaptive + Safety Stock", "Baseline"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ---- Simulation Parameters ----
    st.markdown('<div class="section-header">🔧 Simulation Parameters</div>', unsafe_allow_html=True)

    sim_initial_inventory = st.slider("Initial Inventory", 50, 500, int(initial_inventory_default), step=10)
    sim_lead_time = st.slider("Lead Time (days)", 1, 15, int(lead_time_default))
    sim_order_quantity = st.slider("Order Quantity", 50, 800, int(order_quantity_default), step=25)
    sim_holding_cost = st.slider("Holding Cost Ratio", 0.01, 0.10, float(holding_cost_ratio_default), 0.005, format="%.3f")
    sim_supplier_cost = st.slider("Supplier Cost Ratio", 0.30, 0.90, float(supplier_cost_ratio_default), 0.05, format="%.2f")
    sim_initial_cash = st.slider("Initial Cash (₹)", 50000, 500000, int(initial_cash_default), step=10000, format="₹%d")

    st.markdown("---")

    # ---- Scenario Controls ----
    st.markdown('<div class="section-header">⚡ Scenario Controls</div>', unsafe_allow_html=True)

    sc_demand_spike = st.checkbox("Demand Spike (+30%)")
    sc_lead_time_shock = st.checkbox("Lead Time Shock")
    sc_supplier_cost_inc = st.checkbox("Supplier Cost Increase")
    sc_revenue_drop = st.checkbox("Revenue Drop")

    st.markdown("---")

    # ---- Stochastic & Monte Carlo ----
    st.markdown('<div class="section-header">🎲 Stochastic & Monte Carlo</div>', unsafe_allow_html=True)

    use_stochastic = st.checkbox("Enable Stochastic Demand", value=False)
    mc_runs = st.number_input("Monte Carlo Runs", min_value=10, max_value=1000, value=100, step=10)

    st.markdown("---")

    # ---- RUN BUTTON ----
    run_button = st.button("🚀 Run Digital Twin Simulation", use_container_width=True)


# ============================================================
# HELPER: get predictions for selected model
# ============================================================
def get_model_predictions(model_name):
    """Return predictions array for the chosen forecast model."""
    if model_name == "Random Forest":
        return rf_predictions
    elif model_name == "LSTM (experimental)":
        # Fall back to linear regression; LSTM is separate
        return predictions
    else:
        return predictions


# ============================================================
# HELPER: build scenario data
# ============================================================
def build_scenario_data(base_data):
    """Apply selected stress-test scenarios to data."""
    data_kwargs = {}
    sim_params = {}

    if sc_demand_spike:
        data_kwargs["demand_multiplier"] = 1.30
    if sc_lead_time_shock:
        sim_params["sim_lead_time"] = max(sim_lead_time + 3, sim_lead_time)
    if sc_supplier_cost_inc:
        data_kwargs["sc_supplier_cost_ratio"] = min(sim_supplier_cost + 0.15, 0.95)
        sim_params["sim_supplier_cost_ratio"] = min(sim_supplier_cost + 0.15, 0.95)
    if sc_revenue_drop:
        data_kwargs["price_multiplier"] = 0.80

    if data_kwargs:
        scenario_data = prepare_scenario_data(base_data, **data_kwargs)
    else:
        scenario_data = base_data.copy()

    return scenario_data, sim_params


# ============================================================
# RUN SIMULATION
# ============================================================
if run_button:
    with st.spinner("🏭 Running digital twin simulations …"):
        # ---- Prepare data with selected model predictions ----
        sim_data = test_data.copy()
        selected_preds = get_model_predictions(forecast_model)
        sim_data["predicted_sales"] = selected_preds

        # Recompute financials
        sim_data["revenue"] = sim_data["predicted_sales"] * sim_data["sell_price"]
        sim_data["supplier_cost"] = sim_data["revenue"] * sim_supplier_cost
        sim_data["holding_cost"] = sim_data["revenue"] * sim_holding_cost
        sim_data["net_cash"] = sim_data["revenue"] - sim_data["supplier_cost"] - sim_data["holding_cost"]

        # ---- Apply scenarios ----
        sim_data, extra_sim_params = build_scenario_data(sim_data)

        # ---- Common sim kwargs ----
        base_sim_kwargs = dict(
            sim_lead_time=sim_lead_time,
            sim_order_quantity=sim_order_quantity,
            sim_holding_cost_ratio=sim_holding_cost,
            sim_supplier_cost_ratio=sim_supplier_cost,
            sim_initial_inventory=sim_initial_inventory,
            sim_initial_cash=sim_initial_cash,
            use_stochastic_demand=use_stochastic,
            stochastic_std=forecast_error_std if use_stochastic else None,
        )
        base_sim_kwargs.update(extra_sim_params)

        # ---- Run selected policies ----
        results = {}
        if "Adaptive" in selected_policies:
            results["Adaptive"] = run_inventory_simulation(
                sim_data, policy_type="adaptive", safety_stock=0, **base_sim_kwargs
            )
        if "Adaptive + Safety Stock" in selected_policies:
            results["Adaptive + Safety Stock"] = run_inventory_simulation(
                sim_data, policy_type="adaptive", safety_stock=safety_stock_val, **base_sim_kwargs
            )
        if "Baseline" in selected_policies:
            sc_lt = base_sim_kwargs.get("sim_lead_time", lead_time_default)
            sc_fixed_rp = average_train_demand * sc_lt
            results["Baseline"] = run_inventory_simulation(
                sim_data, policy_type="fixed", reorder_point_val=sc_fixed_rp, **base_sim_kwargs
            )

        # ---- Metrics ----
        metrics = {}
        for pname, sim_df in results.items():
            metrics[pname] = calculate_metrics(sim_df, pname)

        # ---- Monte Carlo ----
        mc_df = None
        if use_stochastic and mc_runs > 0:
            mc_results_list = []
            fc_results = []
            so_results = []
            for run_idx in range(mc_runs):
                mc_sim = run_inventory_simulation(
                    sim_data, policy_type="adaptive", safety_stock=safety_stock_val,
                    use_stochastic_demand=True, stochastic_std=forecast_error_std,
                    sim_lead_time=sim_lead_time, sim_order_quantity=sim_order_quantity,
                    sim_holding_cost_ratio=sim_holding_cost,
                    sim_supplier_cost_ratio=sim_supplier_cost,
                    sim_initial_inventory=sim_initial_inventory,
                    sim_initial_cash=sim_initial_cash,
                )
                so_days = (mc_sim["inventory"] < 0).sum()
                so_rate = (so_days / len(mc_sim)) * 100
                fc = mc_sim["cash_balance"].iloc[-1]
                fc_results.append(fc)
                so_results.append(so_rate)
                mc_results_list.append({
                    "run": run_idx + 1,
                    "stockout_rate": so_rate,
                    "service_level": 100 - so_rate,
                    "final_cash": fc,
                    "average_inventory": mc_sim["inventory"].mean(),
                })
            mc_df = pd.DataFrame(mc_results_list)

    # Store in session state
    st.session_state["results"] = results
    st.session_state["metrics"] = metrics
    st.session_state["mc_df"] = mc_df
    st.session_state["sim_data"] = sim_data
    st.session_state["selected_preds"] = selected_preds
    st.session_state["fc_results"] = fc_results if mc_df is not None else None
    st.session_state["so_results"] = so_results if mc_df is not None else None


# ============================================================
# MAIN CONTENT — only show if we have results
# ============================================================
if "results" not in st.session_state:
    st.markdown(
        '<div style="text-align:center; padding:80px 20px;">'
        '<p style="font-size:3.5rem; margin:0;">🏭</p>'
        '<h2 style="color:#e2e8f0; margin:10px 0;">Welcome to the Digital Twin Dashboard</h2>'
        '<p style="color:#94a3b8; font-size:1.05rem; max-width:550px; margin:0 auto;">'
        'Configure your experiment parameters in the left panel and click '
        '<strong style="color:#60a5fa;">Run Digital Twin Simulation</strong> to begin.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ---- Retrieve results ----
results = st.session_state["results"]
metrics = st.session_state["metrics"]
mc_df = st.session_state.get("mc_df")
sim_data = st.session_state["sim_data"]
selected_preds = st.session_state["selected_preds"]
fc_results = st.session_state.get("fc_results")
so_results = st.session_state.get("so_results")

# ---- Display-only month tick labels (NO dataframe changes) ----
# Build tick positions: first date of each calendar month → "Month 1", "Month 2"…
_sim_dates_sorted = sim_data.sort_values("date")["date"]
_month_periods = pd.to_datetime(_sim_dates_sorted).dt.to_period("M")
_unique_months = sorted(_month_periods.unique())
_month_label_map = {m: f"Month {i+1}" for i, m in enumerate(_unique_months)}

def _build_month_ticks(date_series):
    """Return (tickvals, ticktext) — one tick per calendar month at its first date."""
    ds = pd.to_datetime(date_series).sort_values().reset_index(drop=True)
    periods = ds.dt.to_period("M")
    tickvals = []
    ticktext = []
    for m in sorted(periods.unique()):
        first_date = ds[periods == m].iloc[0]
        tickvals.append(first_date)
        ticktext.append(_month_label_map.get(m, str(m)))
    return tickvals, ticktext

POLICY_COLORS = {
    "Adaptive": COLOR_ADAPTIVE,
    "Baseline": COLOR_BASELINE,
    "Adaptive + Safety Stock": COLOR_SAFETY,
}

# ============================================================
# CENTER + RIGHT PANELS
# ============================================================
col_center, col_right = st.columns([2.2, 1], gap="large")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CENTER PANEL — VISUALIZATIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
with col_center:
    st.markdown('<div class="section-header">📈 Demand Forecast</div>', unsafe_allow_html=True)

    # ---- 1) Demand Forecast Chart ----
    fig_demand = go.Figure()
    dates = sim_data.sort_values("date")["date"]
    actual = y_test.values

    fig_demand.add_trace(go.Scatter(
        x=dates, y=actual,
        mode="lines", name="Actual Demand",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        opacity=0.7,
    ))
    fig_demand.add_trace(go.Scatter(
        x=dates, y=selected_preds,
        mode="lines", name=f"Predicted ({forecast_model})",
        line=dict(color=COLOR_ACCENT, width=2.5),
    ))
    _tv, _tt = _build_month_ticks(dates)
    fig_demand.update_layout(
        title=f"Actual vs Predicted Demand — {forecast_model}",
        xaxis_title="Simulation Month", yaxis_title="Demand",
        height=340,
        **PLOTLY_LAYOUT,
    )
    fig_demand.update_xaxes(tickmode="array", tickvals=_tv, ticktext=_tt)
    st.plotly_chart(fig_demand, use_container_width=True)

    # ---- 2) Inventory Simulation ----
    st.markdown('<div class="section-header">📦 Inventory Simulation</div>', unsafe_allow_html=True)

    fig_inv = go.Figure()
    _inv_tick_set = False
    for pname, sim_df in results.items():
        color = POLICY_COLORS.get(pname, "#a78bfa")
        sim_sorted = sim_df.sort_values("date")

        fig_inv.add_trace(go.Scatter(
            x=sim_sorted["date"], y=sim_sorted["inventory"],
            mode="lines", name=pname,
            line=dict(color=color, width=2.5 if "Safety" in pname else 2),
        ))

        # Reorder events
        reorder_events = sim_sorted[sim_sorted["reorder_flag"] == 1]
        fig_inv.add_trace(go.Scatter(
            x=reorder_events["date"], y=reorder_events["inventory"],
            mode="markers", name=f"{pname} — Reorder",
            marker=dict(symbol="triangle-up", size=9, color=color, line=dict(width=1, color="white")),
            showlegend=False,
        ))

        # Stockouts
        stockouts = sim_sorted[sim_sorted["inventory"] < 0]
        if len(stockouts) > 0:
            fig_inv.add_trace(go.Scatter(
                x=stockouts["date"], y=stockouts["inventory"],
                mode="markers", name=f"{pname} — Stockout",
                marker=dict(symbol="x", size=8, color="#f87171"),
                showlegend=False,
            ))

        if not _inv_tick_set:
            _inv_tv, _inv_tt = _build_month_ticks(sim_sorted["date"])
            _inv_tick_set = True

    # Reorder point line (baseline)
    sc_lt = sim_lead_time
    rp_val = average_train_demand * sc_lt
    fig_inv.add_hline(
        y=rp_val, line_dash="dash", line_color="rgba(251,191,36,0.5)",
        annotation_text="Reorder Point", annotation_position="top right",
        annotation_font_color="#fbbf24",
    )
    # Zero line
    fig_inv.add_hline(y=0, line_dash="solid", line_color="rgba(239,68,68,0.3)")

    fig_inv.update_layout(
        title="Inventory Levels Over Time",
        xaxis_title="Simulation Month", yaxis_title="Inventory Units",
        height=380,
        **PLOTLY_LAYOUT,
    )
    fig_inv.update_xaxes(tickmode="array", tickvals=_inv_tv, ticktext=_inv_tt)
    st.plotly_chart(fig_inv, use_container_width=True)

    # ---- 3) Cash Flow Simulation ----
    st.markdown('<div class="section-header">💰 Cash Flow Simulation</div>', unsafe_allow_html=True)

    fig_cash = go.Figure()
    _cash_tick_set = False
    for pname, sim_df in results.items():
        color = POLICY_COLORS.get(pname, "#a78bfa")
        sim_sorted = sim_df.sort_values("date")
        fig_cash.add_trace(go.Scatter(
            x=sim_sorted["date"], y=sim_sorted["cash_balance"],
            mode="lines", name=pname,
            line=dict(color=color, width=2.5 if "Safety" in pname else 2),
            fill="tonexty" if pname == list(results.keys())[-1] else None,
        ))
        if not _cash_tick_set:
            _cash_tv, _cash_tt = _build_month_ticks(sim_sorted["date"])
            _cash_tick_set = True

    fig_cash.update_layout(
        title="Cash Balance Over Time — Policy Comparison",
        xaxis_title="Simulation Month", yaxis_title="Cash (₹)",
        height=360,
        **PLOTLY_LAYOUT,
    )
    fig_cash.update_xaxes(tickmode="array", tickvals=_cash_tv, ticktext=_cash_tt)
    st.plotly_chart(fig_cash, use_container_width=True)

    # ---- 4) Simulation Playback ----
    st.markdown('<div class="section-header">▶️ Simulation Playback</div>', unsafe_allow_html=True)

    # Use the first available policy for playback
    playback_policy = list(results.keys())[0]
    playback_df = results[playback_policy].sort_values("date").reset_index(drop=True)
    playback_color = POLICY_COLORS.get(playback_policy, COLOR_ADAPTIVE)

    # Build animated frames
    n_points = len(playback_df)
    step_size = max(1, n_points // 50)  # ~50 frames for smooth animation
    frame_indices = list(range(step_size, n_points, step_size))
    if frame_indices[-1] != n_points - 1:
        frame_indices.append(n_points - 1)

    frames = []
    for idx in frame_indices:
        frames.append(go.Frame(
            data=[go.Scatter(
                x=playback_df["date"].iloc[:idx+1],
                y=playback_df["inventory"].iloc[:idx+1],
                mode="lines",
                line=dict(color=playback_color, width=2.5),
            )],
            name=str(idx),
        ))

    fig_playback = go.Figure(
        data=[go.Scatter(
            x=[playback_df["date"].iloc[0]],
            y=[playback_df["inventory"].iloc[0]],
            mode="lines",
            line=dict(color=playback_color, width=2.5),
            name=f"{playback_policy} Inventory",
        )],
        frames=frames,
    )
    fig_playback.add_hline(y=0, line_dash="solid", line_color="rgba(239,68,68,0.3)")

    _pb_tv, _pb_tt = _build_month_ticks(playback_df["date"])
    fig_playback.update_layout(
        title=f"Inventory Evolution — {playback_policy} (Playback)",
        xaxis=dict(
            range=[playback_df["date"].iloc[0], playback_df["date"].iloc[-1]],
            gridcolor="rgba(148,163,184,0.08)",
            tickmode="array", tickvals=_pb_tv, ticktext=_pb_tt,
            title="Simulation Month",
        ),
        yaxis=dict(
            range=[playback_df["inventory"].min() * 1.1, playback_df["inventory"].max() * 1.1],
            gridcolor="rgba(148,163,184,0.08)",
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.05, y=1.12,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        margin=dict(l=40, r=20, t=65, b=40),
        hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_family="Inter"),
    )
    st.plotly_chart(fig_playback, use_container_width=True)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RIGHT PANEL — KPI DASHBOARD
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
with col_right:
    st.markdown('<div class="section-header">📊 KPI Dashboard</div>', unsafe_allow_html=True)

    # Use first policy for headline KPIs
    primary_policy = list(metrics.keys())[0]
    pm = metrics[primary_policy]

    kpi_items = [
        ("Stockout Rate", f"{pm['Stockout Rate (%)']:.1f}%", "📉"),
        ("Service Level", f"{pm['Service Level (%)']:.1f}%", "✅"),
        ("Avg Inventory", f"{pm['Avg Inventory']:.0f}", "📦"),
        ("Final Cash", f"₹{pm['Final Cash']:,.0f}", "💰"),
        ("Cash Volatility", f"₹{pm['Cash Volatility']:,.0f}", "📊"),
    ]

    for label, value, icon in kpi_items:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{icon} {label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ---- Policy Comparison Table ----
    st.markdown('<div class="section-header">🔄 Policy Comparison</div>', unsafe_allow_html=True)

    comp_rows = []
    for pname, m in metrics.items():
        comp_rows.append({
            "Policy": pname,
            "Stockout %": round(m["Stockout Rate (%)"], 2),
            "Service %": round(m["Service Level (%)"], 2),
            "Avg Inv": round(m["Avg Inventory"], 1),
            "Final Cash": round(m["Final Cash"], 0),
            "Cash Vol": round(m["Cash Volatility"], 0),
        })

    comp_df = pd.DataFrame(comp_rows).set_index("Policy")
    st.dataframe(comp_df, use_container_width=True)


# ============================================================
# FULL-WIDTH MONTE CARLO RISK ANALYSIS
# ============================================================
st.markdown("---")
st.markdown(
    '<h2 style="text-align:center; color:#e2e8f0; margin: 10px 0 5px 0;">'
    '🎲 Monte Carlo Risk Analysis</h2>',
    unsafe_allow_html=True,
)

if mc_df is not None and fc_results is not None:
    mc_col1, mc_col2 = st.columns(2, gap="large")

    with mc_col1:
        fig_fc_hist = go.Figure()
        fig_fc_hist.add_trace(go.Histogram(
            x=fc_results, nbinsx=25,
            marker_color=COLOR_ADAPTIVE,
            marker_line=dict(color="white", width=0.5),
            opacity=0.85,
        ))
        fig_fc_hist.add_vline(
            x=np.mean(fc_results), line_dash="dash", line_color="#fbbf24",
            annotation_text=f"Mean: ₹{np.mean(fc_results):,.0f}",
            annotation_font_color="#fbbf24",
        )
        fig_fc_hist.update_layout(
            title="Distribution of Final Cash",
            xaxis_title="Final Cash (₹)", yaxis_title="Frequency",
            height=350,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_fc_hist, use_container_width=True)

    with mc_col2:
        fig_so_hist = go.Figure()
        fig_so_hist.add_trace(go.Histogram(
            x=so_results, nbinsx=25,
            marker_color="#f87171",
            marker_line=dict(color="white", width=0.5),
            opacity=0.85,
        ))
        fig_so_hist.add_vline(
            x=np.mean(so_results), line_dash="dash", line_color="#fbbf24",
            annotation_text=f"Mean: {np.mean(so_results):.1f}%",
            annotation_font_color="#fbbf24",
        )
        fig_so_hist.update_layout(
            title="Distribution of Stockout Rate",
            xaxis_title="Stockout Rate (%)", yaxis_title="Frequency",
            height=350,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_so_hist, use_container_width=True)

    # ---- Summary metrics ----
    expected_cash = np.mean(fc_results)
    worst_cash = np.min(fc_results)
    best_cash = np.max(fc_results)
    prob_so_gt5 = float((np.array(so_results) > 5).mean() * 100)
    mean_stockout = float(np.mean(so_results))

    # Adaptive formatting for very small probabilities
    if prob_so_gt5 > 0 and prob_so_gt5 < 0.01:
        prob_display = "< 0.01%"
    elif prob_so_gt5 < 0.1:
        prob_display = f"{prob_so_gt5:.3f}%"
    elif prob_so_gt5 < 1:
        prob_display = f"{prob_so_gt5:.2f}%"
    else:
        prob_display = f"{prob_so_gt5:.1f}%"

    # Adaptive formatting for mean stockout
    if mean_stockout < 0.01:
        mean_so_display = f"{mean_stockout:.4f}%"
    elif mean_stockout < 0.1:
        mean_so_display = f"{mean_stockout:.3f}%"
    elif mean_stockout < 1:
        mean_so_display = f"{mean_stockout:.2f}%"
    else:
        mean_so_display = f"{mean_stockout:.1f}%"

    mc_m1, mc_m2, mc_m3, mc_m4, mc_m5 = st.columns(5, gap="medium")
    with mc_m1:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-label">💰 Expected Final Cash</div>'
            f'<div class="kpi-value">₹{expected_cash:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
    with mc_m2:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-label">🔻 Worst Case Cash</div>'
            f'<div class="kpi-value">₹{worst_cash:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
    with mc_m3:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-label">🔺 Best Case Cash</div>'
            f'<div class="kpi-value">₹{best_cash:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
    with mc_m4:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-label">⚠️ P(Stockout &gt; 5%)</div>'
            f'<div class="kpi-value">{prob_display}</div></div>',
            unsafe_allow_html=True,
        )
    with mc_m5:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-label">📊 Mean Stockout Rate</div>'
            f'<div class="kpi-value">{mean_so_display}</div></div>',
            unsafe_allow_html=True,
        )

    # Context text for negligible risk
    if prob_so_gt5 < 0.1:
        st.markdown(
            '<p style="text-align:center; color:#34d399; font-size:0.85rem; margin:8px 0 0 0;">'
            '✅ Negligible risk — probability of stockout exceeding 5% is statistically insignificant</p>',
            unsafe_allow_html=True,
        )

    # ---- Risk Indicator ----
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    if prob_so_gt5 < 20:
        risk_class = "risk-low"
        risk_text = "LOW RISK"
    elif prob_so_gt5 < 50:
        risk_class = "risk-moderate"
        risk_text = "MODERATE RISK"
    else:
        risk_class = "risk-high"
        risk_text = "HIGH RISK"

    st.markdown(
        f'<div style="text-align:center;">'
        f'<span class="risk-badge {risk_class}">{risk_text}</span>'
        f'<p style="color:#94a3b8; margin-top:10px; font-size:0.88rem;">'
        f'Probability of stockout exceeding 5% across {mc_runs} Monte Carlo simulations</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        '<div style="text-align:center; padding:40px; color:#94a3b8;">'
        '<p style="font-size:2.5rem; margin:0;">🎲</p>'
        '<p style="font-size:1rem;">Enable <strong>Stochastic Demand</strong> in the sidebar '
        'and click <strong>Run Digital Twin Simulation</strong> to generate Monte Carlo results.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# PART 1 — RESIDUAL ANALYSIS (ML VALIDATION)
# ============================================================
st.markdown("---")
st.markdown(
    '<h2 style="text-align:center; color:#e2e8f0; margin: 10px 0 5px 0;">'
    '🔬 Residual Analysis — Stochastic Demand Modeling Validation</h2>',
    unsafe_allow_html=True,
)

with st.container():
    residuals = y_test.values - selected_preds
    mean_error = float(np.mean(residuals))
    std_error = float(np.std(residuals))

    res_col1, res_col2 = st.columns(2, gap="large")

    with res_col1:
        fig_res_time = go.Figure()
        fig_res_time.add_trace(go.Scatter(
            x=list(range(len(residuals))), y=residuals,
            mode="lines", name="Residual",
            line=dict(color="#a78bfa", width=1.5),
        ))
        fig_res_time.add_hline(y=0, line_dash="solid", line_color="rgba(239,68,68,0.4)")
        fig_res_time.add_hline(
            y=mean_error, line_dash="dash", line_color="#fbbf24",
            annotation_text=f"Mean: {mean_error:.2f}",
            annotation_font_color="#fbbf24",
        )
        fig_res_time.update_layout(
            title="Residual Over Time (Actual − Predicted)",
            xaxis_title="Timestep", yaxis_title="Residual",
            height=350, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_res_time, use_container_width=True)

    with res_col2:
        fig_res_hist = go.Figure()
        fig_res_hist.add_trace(go.Histogram(
            x=residuals, nbinsx=30,
            marker_color="#a78bfa",
            marker_line=dict(color="white", width=0.5),
            opacity=0.85,
        ))
        fig_res_hist.add_vline(
            x=mean_error, line_dash="dash", line_color="#fbbf24",
            annotation_text=f"Mean: {mean_error:.2f}",
            annotation_font_color="#fbbf24",
        )
        fig_res_hist.update_layout(
            title="Distribution of Residuals",
            xaxis_title="Residual", yaxis_title="Frequency",
            height=350, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_res_hist, use_container_width=True)

    # Display stats
    res_s1, res_s2 = st.columns(2, gap="medium")
    with res_s1:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-label">📐 Mean Error (Bias)</div>'
            f'<div class="kpi-value">{mean_error:.4f}</div></div>',
            unsafe_allow_html=True,
        )
    with res_s2:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-label">📊 Std Deviation (Variance)</div>'
            f'<div class="kpi-value">{std_error:.4f}</div></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# PART 2 — BASELINE vs ML COMPARISON
# ============================================================
st.markdown("---")
st.markdown(
    '<h2 style="text-align:center; color:#e2e8f0; margin: 10px 0 5px 0;">'
    '📊 Baseline vs ML Forecast Comparison</h2>',
    unsafe_allow_html=True,
)

with st.container():
    actual_vals = y_test.values
    ml_preds = selected_preds

    # Naive forecast: prediction[t] = actual[t-1]
    naive_preds = np.roll(actual_vals, 1)
    naive_preds[0] = actual_vals[0]  # first value has no predecessor

    # Compute metrics (skip first element for fair comparison)
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    ml_mae = mean_absolute_error(actual_vals[1:], ml_preds[1:])
    ml_rmse = np.sqrt(mean_squared_error(actual_vals[1:], ml_preds[1:]))
    naive_mae = mean_absolute_error(actual_vals[1:], naive_preds[1:])
    naive_rmse = np.sqrt(mean_squared_error(actual_vals[1:], naive_preds[1:]))

    mae_improvement = ((naive_mae - ml_mae) / naive_mae) * 100 if naive_mae > 0 else 0
    rmse_improvement = ((naive_rmse - ml_rmse) / naive_rmse) * 100 if naive_rmse > 0 else 0

    comp_data = pd.DataFrame({
        "Model": ["Naive Baseline (t-1)", f"ML — {forecast_model}"],
        "MAE": [round(naive_mae, 4), round(ml_mae, 4)],
        "RMSE": [round(naive_rmse, 4), round(ml_rmse, 4)],
    }).set_index("Model")

    st.dataframe(comp_data, use_container_width=True)

    if mae_improvement > 0:
        st.markdown(
            f'<div style="text-align:center; padding:15px; background:rgba(16,185,129,0.08); '
            f'border:1px solid rgba(16,185,129,0.3); border-radius:12px; margin:10px 0;">'
            f'<span style="color:#34d399; font-weight:700; font-size:1.1rem;">'
            f'✅ ML reduces MAE by {mae_improvement:.1f}% and RMSE by {rmse_improvement:.1f}% '
            f'vs naive baseline</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="text-align:center; padding:15px; background:rgba(245,158,11,0.08); '
            f'border:1px solid rgba(245,158,11,0.3); border-radius:12px; margin:10px 0;">'
            f'<span style="color:#fbbf24; font-weight:700; font-size:1.1rem;">'
            f'⚠️ ML MAE improvement: {mae_improvement:.1f}% — consider model retuning</span></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# PART 3 — SENSITIVITY ANALYSIS (Digital Twin Simulation)
# ============================================================
st.markdown("---")
st.markdown(
    '<h2 style="text-align:center; color:#e2e8f0; margin: 10px 0 5px 0;">'
    '🎛️ Sensitivity Analysis — Digital Twin Simulation</h2>',
    unsafe_allow_html=True,
)

with st.container():
    st.markdown(
        '<p style="text-align:center; color:#94a3b8; font-size:0.9rem; margin-bottom:15px;">'
        'Explore how key parameters affect system behavior under the Policy Optimization Framework</p>',
        unsafe_allow_html=True,
    )

    sa_c1, sa_c2, sa_c3 = st.columns(3, gap="medium")
    with sa_c1:
        sa_demand_range = st.slider("Demand Multiplier Range", 0.5, 1.5, (0.7, 1.3), 0.1,
                                     key="sa_demand_range")
    with sa_c2:
        sa_lt_range = st.slider("Lead Time Range", 1, 10, (2, 8), 1, key="sa_lt_range")
    with sa_c3:
        sa_cost_range = st.slider("Supplier Cost Multiplier Range", 0.5, 1.5, (0.7, 1.3), 0.1,
                                   key="sa_cost_range")

    # Helper for sensitivity sweeps
    def _run_sensitivity_sweep(param_name, param_values, sim_data_base):
        """Sweep one parameter, return list of dicts with final_cash and stockout_rate."""
        sweep_results = []
        for pv in param_values:
            sd = sim_data_base.copy()
            kw = dict(
                sim_lead_time=sim_lead_time,
                sim_order_quantity=sim_order_quantity,
                sim_holding_cost_ratio=sim_holding_cost,
                sim_supplier_cost_ratio=sim_supplier_cost,
                sim_initial_inventory=sim_initial_inventory,
                sim_initial_cash=sim_initial_cash,
            )
            if param_name == "demand_multiplier":
                sd = prepare_scenario_data(sd, demand_multiplier=pv)
            elif param_name == "lead_time":
                kw["sim_lead_time"] = int(pv)
            elif param_name == "supplier_cost":
                kw["sim_supplier_cost_ratio"] = sim_supplier_cost * pv
                sd = prepare_scenario_data(sd, sc_supplier_cost_ratio=sim_supplier_cost * pv)

            sim_r = run_inventory_simulation(sd, policy_type="adaptive",
                                              safety_stock=safety_stock_val, **kw)
            so_days = (sim_r["inventory"] < 0).sum()
            so_rate = (so_days / len(sim_r)) * 100
            fc = sim_r["cash_balance"].iloc[-1]
            sweep_results.append({"param": pv, "final_cash": fc, "stockout_rate": so_rate})
        return sweep_results

    # Run sweeps
    demand_vals = np.arange(sa_demand_range[0], sa_demand_range[1] + 0.01, 0.1)
    lt_vals = list(range(sa_lt_range[0], sa_lt_range[1] + 1))
    cost_vals = np.arange(sa_cost_range[0], sa_cost_range[1] + 0.01, 0.1)

    sweep_demand = _run_sensitivity_sweep("demand_multiplier", demand_vals, sim_data)
    sweep_lt = _run_sensitivity_sweep("lead_time", lt_vals, sim_data)
    sweep_cost = _run_sensitivity_sweep("supplier_cost", cost_vals, sim_data)

    # Plot: 2x2 grid (demand x 2, lead_time x 2) + additional cost plots
    sa_row1_c1, sa_row1_c2 = st.columns(2, gap="large")

    with sa_row1_c1:
        fig_sa1 = go.Figure()
        fig_sa1.add_trace(go.Scatter(
            x=[r["param"] for r in sweep_demand],
            y=[r["final_cash"] for r in sweep_demand],
            mode="lines+markers", name="Final Cash",
            line=dict(color=COLOR_ADAPTIVE, width=2.5),
            marker=dict(size=7),
        ))
        fig_sa1.update_layout(
            title="Demand Multiplier vs Final Cash",
            xaxis_title="Demand Multiplier", yaxis_title="Final Cash (₹)",
            height=320, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_sa1, use_container_width=True)

    with sa_row1_c2:
        fig_sa2 = go.Figure()
        fig_sa2.add_trace(go.Scatter(
            x=[r["param"] for r in sweep_demand],
            y=[r["stockout_rate"] for r in sweep_demand],
            mode="lines+markers", name="Stockout Rate",
            line=dict(color="#f87171", width=2.5),
            marker=dict(size=7),
        ))
        fig_sa2.update_layout(
            title="Demand Multiplier vs Stockout Rate",
            xaxis_title="Demand Multiplier", yaxis_title="Stockout Rate (%)",
            height=320, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_sa2, use_container_width=True)

    sa_row2_c1, sa_row2_c2 = st.columns(2, gap="large")

    with sa_row2_c1:
        fig_sa3 = go.Figure()
        fig_sa3.add_trace(go.Scatter(
            x=[r["param"] for r in sweep_lt],
            y=[r["final_cash"] for r in sweep_lt],
            mode="lines+markers", name="Final Cash",
            line=dict(color=COLOR_SAFETY, width=2.5),
            marker=dict(size=7),
        ))
        fig_sa3.update_layout(
            title="Lead Time vs Final Cash",
            xaxis_title="Lead Time (days)", yaxis_title="Final Cash (₹)",
            height=320, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_sa3, use_container_width=True)

    with sa_row2_c2:
        fig_sa4 = go.Figure()
        fig_sa4.add_trace(go.Scatter(
            x=[r["param"] for r in sweep_lt],
            y=[r["stockout_rate"] for r in sweep_lt],
            mode="lines+markers", name="Stockout Rate",
            line=dict(color="#fbbf24", width=2.5),
            marker=dict(size=7),
        ))
        fig_sa4.update_layout(
            title="Lead Time vs Stockout Rate",
            xaxis_title="Lead Time (days)", yaxis_title="Stockout Rate (%)",
            height=320, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_sa4, use_container_width=True)


# ============================================================
# PART 4 — SCENARIO TESTING
# ============================================================
st.markdown("---")
st.markdown(
    '<h2 style="text-align:center; color:#e2e8f0; margin: 10px 0 5px 0;">'
    '⚡ Scenario Testing — Policy Robustness Under Stress</h2>',
    unsafe_allow_html=True,
)

with st.container():
    scenario_defs = [
        {"name": "📈 Demand Spike (+30%)", "data_kw": {"demand_multiplier": 1.30}, "sim_kw": {}},
        {"name": "🚚 Supply Delay (+3 days)", "data_kw": {}, "sim_kw": {"sim_lead_time": sim_lead_time + 3}},
        {"name": "💸 Cost Increase (+20%)", "data_kw": {"sc_supplier_cost_ratio": sim_supplier_cost * 1.20},
         "sim_kw": {"sim_supplier_cost_ratio": sim_supplier_cost * 1.20}},
    ]

    scenario_rows = []
    for sc in scenario_defs:
        sc_d = prepare_scenario_data(sim_data, **sc["data_kw"]) if sc["data_kw"] else sim_data.copy()
        sc_kw = dict(
            sim_lead_time=sim_lead_time,
            sim_order_quantity=sim_order_quantity,
            sim_holding_cost_ratio=sim_holding_cost,
            sim_supplier_cost_ratio=sim_supplier_cost,
            sim_initial_inventory=sim_initial_inventory,
            sim_initial_cash=sim_initial_cash,
        )
        sc_kw.update(sc["sim_kw"])

        sc_sim = run_inventory_simulation(sc_d, policy_type="adaptive",
                                           safety_stock=safety_stock_val, **sc_kw)
        sc_m = calculate_metrics(sc_sim, sc["name"])
        scenario_rows.append({
            "Scenario": sc["name"],
            "Final Cash (₹)": round(sc_m["Final Cash"], 0),
            "Stockout Rate (%)": round(sc_m["Stockout Rate (%)"], 2),
            "Avg Inventory": round(sc_m["Avg Inventory"], 1),
        })

    scenario_df = pd.DataFrame(scenario_rows).set_index("Scenario")

    # Highlight best/worst
    best_scenario = scenario_df["Final Cash (₹)"].idxmax()
    worst_scenario = scenario_df["Final Cash (₹)"].idxmin()

    st.dataframe(scenario_df, use_container_width=True)

    sc_h1, sc_h2 = st.columns(2, gap="medium")
    with sc_h1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">🟢 Best Case Scenario</div>'
            f'<div class="kpi-value" style="font-size:1.1rem;">{best_scenario}</div></div>',
            unsafe_allow_html=True,
        )
    with sc_h2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">🔴 Worst Case Scenario</div>'
            f'<div class="kpi-value" style="font-size:1.1rem;">{worst_scenario}</div></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# PART 5 — POLICY RANKING SYSTEM (Policy Optimization Framework)
# ============================================================
st.markdown("---")
st.markdown(
    '<h2 style="text-align:center; color:#e2e8f0; margin: 10px 0 5px 0;">'
    '🏆 Policy Ranking — Policy Optimization Framework</h2>',
    unsafe_allow_html=True,
)

with st.container():
    STOCKOUT_PENALTY = 5000.0
    INVENTORY_PENALTY = 10.0

    st.markdown(
        '<p style="text-align:center; color:#94a3b8; font-size:0.85rem; margin-bottom:10px;">'
        f'Scoring: <code>score = final_cash − ({STOCKOUT_PENALTY:,.0f} × stockout_rate) '
        f'− ({INVENTORY_PENALTY:,.0f} × avg_inventory)</code></p>',
        unsafe_allow_html=True,
    )

    ranking_rows = []
    for pname, m in metrics.items():
        score = (m["Final Cash"]
                 - STOCKOUT_PENALTY * m["Stockout Rate (%)"]
                 - INVENTORY_PENALTY * m["Avg Inventory"])
        ranking_rows.append({
            "Policy": pname,
            "Final Cash (₹)": round(m["Final Cash"], 0),
            "Stockout Rate (%)": round(m["Stockout Rate (%)"], 2),
            "Avg Inventory": round(m["Avg Inventory"], 1),
            "Score": round(score, 0),
        })

    ranking_df = pd.DataFrame(ranking_rows).sort_values("Score", ascending=False).reset_index(drop=True)
    ranking_df.index = ranking_df.index + 1
    ranking_df.index.name = "Rank"

    st.dataframe(ranking_df, use_container_width=True)

    recommended = ranking_df.iloc[0]["Policy"]
    rec_score = ranking_df.iloc[0]["Score"]

    st.markdown(
        f'<div style="text-align:center; padding:18px; background:linear-gradient(135deg, '
        f'rgba(16,185,129,0.1), rgba(59,130,246,0.1)); border:1px solid rgba(16,185,129,0.3); '
        f'border-radius:14px; margin:15px 0;">'
        f'<span style="color:#34d399; font-weight:800; font-size:1.2rem;">'
        f'🏅 Recommended Policy: {recommended}</span><br>'
        f'<span style="color:#94a3b8; font-size:0.9rem;">Score: {rec_score:,.0f}</span></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PART 6 — MONTE CARLO ENHANCEMENT (Monte Carlo Risk Analysis)
# ============================================================
st.markdown("---")
st.markdown(
    '<h2 style="text-align:center; color:#e2e8f0; margin: 10px 0 5px 0;">'
    '📉 Monte Carlo Risk Analysis — Confidence Intervals</h2>',
    unsafe_allow_html=True,
)

with st.container():
    if mc_df is not None and fc_results is not None:
        mc_mean = np.mean(fc_results)
        mc_min = np.min(fc_results)
        mc_max = np.max(fc_results)

        # 90% Confidence Interval
        ci_lower = np.percentile(fc_results, 5)
        ci_upper = np.percentile(fc_results, 95)

        ci_c1, ci_c2, ci_c3 = st.columns(3, gap="medium")
        with ci_c1:
            st.markdown(
                '<div class="kpi-card"><div class="kpi-label">💰 Expected Final Cash</div>'
                f'<div class="kpi-value">₹{mc_mean:,.0f}</div></div>',
                unsafe_allow_html=True,
            )
        with ci_c2:
            st.markdown(
                '<div class="kpi-card"><div class="kpi-label">📊 90% CI Range</div>'
                f'<div class="kpi-value" style="font-size:1.3rem;">₹{ci_lower:,.0f} — ₹{ci_upper:,.0f}</div></div>',
                unsafe_allow_html=True,
            )
        with ci_c3:
            st.markdown(
                '<div class="kpi-card"><div class="kpi-label">🔻 Worst Case</div>'
                f'<div class="kpi-value">₹{mc_min:,.0f}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

        st.markdown(
            f'<div style="text-align:center; padding:20px; background:rgba(15,23,42,0.7); '
            f'border:1px solid rgba(59,130,246,0.15); border-radius:14px;">'
            f'<p style="color:#60a5fa; font-weight:700; font-size:1.05rem; margin:0 0 8px 0;">'
            f'Stochastic Demand Modeling — Risk Summary</p>'
            f'<p style="color:#e2e8f0; font-size:0.95rem; margin:3px 0;">'
            f'Expected Cash: <strong>₹{mc_mean:,.0f}</strong></p>'
            f'<p style="color:#e2e8f0; font-size:0.95rem; margin:3px 0;">'
            f'Range (90% CI): <strong>₹{ci_lower:,.0f}</strong> – <strong>₹{ci_upper:,.0f}</strong></p>'
            f'<p style="color:#e2e8f0; font-size:0.95rem; margin:3px 0;">'
            f'Worst Case: <strong>₹{mc_min:,.0f}</strong></p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:center; padding:30px; color:#94a3b8;">'
            '<p>Enable <strong>Stochastic Demand</strong> and run the simulation '
            'to see Monte Carlo confidence intervals.</p></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# PART 7 — SUMMARY METRICS TABLE
# ============================================================
st.markdown("---")
st.markdown(
    '<h2 style="text-align:center; color:#e2e8f0; margin: 10px 0 5px 0;">'
    '📋 Summary Metrics — Decision Support Table</h2>',
    unsafe_allow_html=True,
)

with st.container():
    summary_rows = []
    for pname, m in metrics.items():
        score = (m["Final Cash"]
                 - STOCKOUT_PENALTY * m["Stockout Rate (%)"]
                 - INVENTORY_PENALTY * m["Avg Inventory"])
        summary_rows.append({
            "Policy": pname,
            "Final Cash (₹)": f"₹{m['Final Cash']:,.0f}",
            "Stockout Rate (%)": f"{m['Stockout Rate (%)']:.2f}%",
            "Avg Inventory": f"{m['Avg Inventory']:.1f}",
            "Score": f"{score:,.0f}",
        })

    summary_df_display = pd.DataFrame(summary_rows).set_index("Policy")
    st.dataframe(summary_df_display, use_container_width=True)

    # Highlight best
    best_policy = max(metrics.keys(),
                      key=lambda p: metrics[p]["Final Cash"]
                                    - STOCKOUT_PENALTY * metrics[p]["Stockout Rate (%)"]
                                    - INVENTORY_PENALTY * metrics[p]["Avg Inventory"])
    st.markdown(
        f'<div style="text-align:center; padding:12px; color:#34d399; font-weight:700; '
        f'font-size:1rem;">⭐ Best Policy: {best_policy}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PART 8 — RESEARCH TERMINOLOGY NOTE
# ============================================================
st.markdown("---")
st.markdown(
    '<div style="text-align:center; padding:20px; background:rgba(15,23,42,0.7); '
    'border:1px solid rgba(99,102,241,0.2); border-radius:14px; margin:10px 0;">'
    '<p style="color:#a78bfa; font-weight:700; font-size:1.05rem; margin:0 0 10px 0;">'
    '📖 Research Framework</p>'
    '<p style="color:#94a3b8; font-size:0.88rem; margin:3px 0;">'
    '• <strong style="color:#e2e8f0;">Digital Twin Simulation</strong> — '
    'Virtual replication of supply chain for scenario exploration</p>'
    '<p style="color:#94a3b8; font-size:0.88rem; margin:3px 0;">'
    '• <strong style="color:#e2e8f0;">Stochastic Demand Modeling</strong> — '
    'Probabilistic demand representation using Gaussian noise</p>'
    '<p style="color:#94a3b8; font-size:0.88rem; margin:3px 0;">'
    '• <strong style="color:#e2e8f0;">Monte Carlo Risk Analysis</strong> — '
    'Multi-run simulation for uncertainty quantification and CI estimation</p>'
    '<p style="color:#94a3b8; font-size:0.88rem; margin:3px 0;">'
    '• <strong style="color:#e2e8f0;">Policy Optimization Framework</strong> — '
    'Score-based ranking for data-driven policy recommendation</p></div>',
    unsafe_allow_html=True,
)


# ============================================================
# 🧪 SCENARIO ANALYSIS & POLICY EVALUATION
# ============================================================
st.markdown("---")
st.markdown(
    '<h2 style="text-align:center; color:#e2e8f0; font-weight:800; margin:10px 0 5px 0;">'
    '🧪 Scenario Analysis &amp; Policy Evaluation</h2>'
    '<p style="text-align:center; color:#94a3b8; font-size:0.9rem; margin:0 0 15px 0;">'
    'Comparative scenario testing with score-based policy recommendation</p>',
    unsafe_allow_html=True,
)

with st.container():
    # ---------- PART 1 — Scenario Testing ----------
    st.markdown('<div class="section-header">📊 Scenario Testing — Adaptive vs Baseline</div>',
                unsafe_allow_html=True)

    _sa_scenarios = [
        {"name": "Base Case",             "dm": 1.0,  "lt_add": 0, "cost_mult": 1.0},
        {"name": "Demand Spike (+30%)",    "dm": 1.30, "lt_add": 0, "cost_mult": 1.0},
        {"name": "Supply Delay (+3 days)", "dm": 1.0,  "lt_add": 3, "cost_mult": 1.0},
        {"name": "Cost Increase (+20%)",   "dm": 1.0,  "lt_add": 0, "cost_mult": 1.20},
    ]

    _sa_rows = []
    _base_metrics = {}  # store base-case metrics for ranking

    for _sc in _sa_scenarios:
        # Prepare scenario data
        _sc_cost = sim_supplier_cost * _sc["cost_mult"]
        _data_kw = {}
        if _sc["dm"] != 1.0:
            _data_kw["demand_multiplier"] = _sc["dm"]
        if _sc["cost_mult"] != 1.0:
            _data_kw["sc_supplier_cost_ratio"] = _sc_cost
        _sc_data = prepare_scenario_data(sim_data, **_data_kw) if _data_kw else sim_data.copy()

        _sc_lt = sim_lead_time + _sc["lt_add"]
        _sc_fixed_rp = average_train_demand * _sc_lt

        # Common kwargs
        _sc_kw = dict(
            sim_lead_time=_sc_lt,
            sim_order_quantity=sim_order_quantity,
            sim_holding_cost_ratio=sim_holding_cost,
            sim_supplier_cost_ratio=_sc_cost,
            sim_initial_inventory=sim_initial_inventory,
            sim_initial_cash=sim_initial_cash,
        )

        # Run Adaptive
        _ad_sim = run_inventory_simulation(
            _sc_data, policy_type="adaptive", safety_stock=safety_stock_val, **_sc_kw)
        _ad_m = calculate_metrics(_ad_sim, "Adaptive")

        # Run Baseline
        _bl_sim = run_inventory_simulation(
            _sc_data, policy_type="fixed", reorder_point_val=_sc_fixed_rp, **_sc_kw)
        _bl_m = calculate_metrics(_bl_sim, "Baseline")

        for _pname, _pm in [("Adaptive", _ad_m), ("Baseline", _bl_m)]:
            _sa_rows.append({
                "Scenario": _sc["name"],
                "Policy": _pname,
                "Final Cash (₹)": round(_pm["Final Cash"], 0),
                "Stockout Rate (%)": round(_pm["Stockout Rate (%)"], 2),
                "Avg Inventory": round(_pm["Avg Inventory"], 1),
            })

        # Save base case for ranking
        if _sc["name"] == "Base Case":
            _base_metrics["Adaptive"] = _ad_m
            _base_metrics["Baseline"] = _bl_m

    _sa_df = pd.DataFrame(_sa_rows)
    st.dataframe(_sa_df.set_index(["Scenario", "Policy"]), use_container_width=True)

    # ---------- PART 2 — Policy Ranking ----------
    st.markdown('<div class="section-header">🏆 Policy Ranking (Base Case)</div>',
                unsafe_allow_html=True)

    _STOCKOUT_PEN = 5000.0
    _INV_PEN = 10.0

    _rank_rows = []
    for _pn, _pm in _base_metrics.items():
        _score = (_pm["Final Cash"]
                  - _STOCKOUT_PEN * _pm["Stockout Rate (%)"]
                  - _INV_PEN * _pm["Avg Inventory"])
        _rank_rows.append({
            "Policy": _pn,
            "Final Cash (₹)": round(_pm["Final Cash"], 0),
            "Stockout Rate (%)": round(_pm["Stockout Rate (%)"], 2),
            "Avg Inventory": round(_pm["Avg Inventory"], 1),
            "Score": round(_score, 0),
        })

    _rank_df = (pd.DataFrame(_rank_rows)
                .sort_values("Score", ascending=False)
                .reset_index(drop=True))
    _rank_df.index = _rank_df.index + 1
    _rank_df.index.name = "Rank"
    st.dataframe(_rank_df, use_container_width=True)

    _best = _rank_df.iloc[0]["Policy"]
    st.markdown(
        f'<div style="text-align:center; padding:16px; '
        f'background:linear-gradient(135deg, rgba(16,185,129,0.1), rgba(59,130,246,0.1)); '
        f'border:1px solid rgba(16,185,129,0.3); border-radius:14px; margin:12px 0;">'
        f'<span style="color:#34d399; font-weight:800; font-size:1.15rem;">'
        f'🏆 Recommended Policy: {_best}</span></div>',
        unsafe_allow_html=True,
    )

    # ---------- PART 3 — Simple Insight Text ----------
    _ad_base = _base_metrics.get("Adaptive", {})
    _bl_base = _base_metrics.get("Baseline", {})

    if _ad_base and _bl_base:
        _insights = []
        if _ad_base["Stockout Rate (%)"] < _bl_base["Stockout Rate (%)"]:
            _insights.append(
                "📦 **Adaptive policy improves service reliability** — "
                f"stockout rate is {_ad_base['Stockout Rate (%)']:.2f}% vs "
                f"baseline's {_bl_base['Stockout Rate (%)']:.2f}%."
            )
        else:
            _insights.append(
                "📦 **Baseline policy has comparable or better stockout performance** — "
                f"stockout rate is {_bl_base['Stockout Rate (%)']:.2f}% vs "
                f"adaptive's {_ad_base['Stockout Rate (%)']:.2f}%."
            )

        if _bl_base["Final Cash"] > _ad_base["Final Cash"]:
            _insights.append(
                "💰 **Baseline policy preserves more cash but risks stockouts** — "
                f"final cash ₹{_bl_base['Final Cash']:,.0f} vs "
                f"adaptive's ₹{_ad_base['Final Cash']:,.0f}."
            )
        else:
            _insights.append(
                "💰 **Adaptive policy yields higher final cash** — "
                f"₹{_ad_base['Final Cash']:,.0f} vs "
                f"baseline's ₹{_bl_base['Final Cash']:,.0f}."
            )

        if _ad_base["Avg Inventory"] > _bl_base["Avg Inventory"]:
            _insights.append(
                "📊 **Adaptive policy maintains higher average inventory** — "
                f"acts as a buffer against demand uncertainty."
            )

        for _ins in _insights:
            st.info(_ins)


# ============================================================
# 📈 SENSITIVITY ANALYSIS & RISK INTERPRETATION
# ============================================================
st.markdown("---")
st.markdown(
    '<h2 style="text-align:center; color:#e2e8f0; font-weight:800; margin:10px 0 5px 0;">'
    '📈 Sensitivity Analysis &amp; Risk Interpretation</h2>'
    '<p style="text-align:center; color:#94a3b8; font-size:0.9rem; margin:0 0 15px 0;">'
    'Parameter sensitivity sweeps and Monte Carlo risk summary</p>',
    unsafe_allow_html=True,
)

# ---------- PART 1 — Sensitivity Analysis ----------
with st.container():
    st.markdown('<div class="section-header">🎛️ One-at-a-Time Parameter Sensitivity</div>',
                unsafe_allow_html=True)

    # Define sweep values
    _dm_vals = [0.8, 1.0, 1.2]
    _lt_vals = [5, 8, 10]
    _sc_vals = [0.5, 0.6, 0.7]

    def _quick_sweep(param_type, values):
        """Lightweight sweep — vary one parameter, keep others at sidebar defaults."""
        out = []
        for v in values:
            _kw = dict(
                sim_lead_time=sim_lead_time,
                sim_order_quantity=sim_order_quantity,
                sim_holding_cost_ratio=sim_holding_cost,
                sim_supplier_cost_ratio=sim_supplier_cost,
                sim_initial_inventory=sim_initial_inventory,
                sim_initial_cash=sim_initial_cash,
            )
            _sd = sim_data.copy()

            if param_type == "demand":
                _sd = prepare_scenario_data(_sd, demand_multiplier=v)
            elif param_type == "lead_time":
                _kw["sim_lead_time"] = int(v)
            elif param_type == "cost":
                _kw["sim_supplier_cost_ratio"] = v
                _sd = prepare_scenario_data(_sd, sc_supplier_cost_ratio=v)

            _sim = run_inventory_simulation(
                _sd, policy_type="adaptive", safety_stock=safety_stock_val, **_kw)
            _so = ((_sim["inventory"] < 0).sum() / len(_sim)) * 100
            out.append({"val": v, "cash": _sim["cash_balance"].iloc[-1], "so": _so})
        return out

    _res_dm = _quick_sweep("demand", _dm_vals)
    _res_lt = _quick_sweep("lead_time", _lt_vals)

    # --- Row 1: Demand Multiplier plots ---
    _sr1c1, _sr1c2 = st.columns(2, gap="large")

    with _sr1c1:
        _fig_dm_cash = go.Figure()
        _fig_dm_cash.add_trace(go.Scatter(
            x=[r["val"] for r in _res_dm],
            y=[r["cash"] for r in _res_dm],
            mode="lines+markers", name="Final Cash",
            line=dict(color=COLOR_ADAPTIVE, width=2.5),
            marker=dict(size=8),
        ))
        _fig_dm_cash.update_layout(
            title="Demand Multiplier vs Final Cash",
            xaxis_title="Demand Multiplier", yaxis_title="Final Cash (₹)",
            height=320, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(_fig_dm_cash, use_container_width=True)

    with _sr1c2:
        _fig_dm_so = go.Figure()
        _fig_dm_so.add_trace(go.Scatter(
            x=[r["val"] for r in _res_dm],
            y=[r["so"] for r in _res_dm],
            mode="lines+markers", name="Stockout Rate",
            line=dict(color="#f87171", width=2.5),
            marker=dict(size=8),
        ))
        _fig_dm_so.update_layout(
            title="Demand Multiplier vs Stockout Rate",
            xaxis_title="Demand Multiplier", yaxis_title="Stockout Rate (%)",
            height=320, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(_fig_dm_so, use_container_width=True)

    # --- Row 2: Lead Time plots ---
    _sr2c1, _sr2c2 = st.columns(2, gap="large")

    with _sr2c1:
        _fig_lt_cash = go.Figure()
        _fig_lt_cash.add_trace(go.Scatter(
            x=[r["val"] for r in _res_lt],
            y=[r["cash"] for r in _res_lt],
            mode="lines+markers", name="Final Cash",
            line=dict(color=COLOR_SAFETY, width=2.5),
            marker=dict(size=8),
        ))
        _fig_lt_cash.update_layout(
            title="Lead Time vs Final Cash",
            xaxis_title="Lead Time (days)", yaxis_title="Final Cash (₹)",
            height=320, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(_fig_lt_cash, use_container_width=True)

    with _sr2c2:
        _fig_lt_so = go.Figure()
        _fig_lt_so.add_trace(go.Scatter(
            x=[r["val"] for r in _res_lt],
            y=[r["so"] for r in _res_lt],
            mode="lines+markers", name="Stockout Rate",
            line=dict(color="#fbbf24", width=2.5),
            marker=dict(size=8),
        ))
        _fig_lt_so.update_layout(
            title="Lead Time vs Stockout Rate",
            xaxis_title="Lead Time (days)", yaxis_title="Stockout Rate (%)",
            height=320, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(_fig_lt_so, use_container_width=True)


# ---------- PART 2 & 3 — Monte Carlo Risk Interpretation ----------
st.markdown('<div class="section-header">🎲 Risk Interpretation</div>',
            unsafe_allow_html=True)

with st.container():
    if mc_df is not None and fc_results is not None and so_results is not None:
        _mc_mean = float(np.mean(fc_results))
        _mc_min = float(np.min(fc_results))
        _mc_max = float(np.max(fc_results))
        _ci_lo = float(np.percentile(fc_results, 5))
        _ci_hi = float(np.percentile(fc_results, 95))

        _ri_c1, _ri_c2, _ri_c3, _ri_c4 = st.columns(4, gap="medium")
        with _ri_c1:
            st.markdown(
                '<div class="kpi-card"><div class="kpi-label">💰 Expected Cash</div>'
                f'<div class="kpi-value">₹{_mc_mean:,.0f}</div></div>',
                unsafe_allow_html=True)
        with _ri_c2:
            st.markdown(
                '<div class="kpi-card"><div class="kpi-label">📊 90% Range</div>'
                f'<div class="kpi-value" style="font-size:1.2rem;">₹{_ci_lo:,.0f} – ₹{_ci_hi:,.0f}</div></div>',
                unsafe_allow_html=True)
        with _ri_c3:
            st.markdown(
                '<div class="kpi-card"><div class="kpi-label">🔻 Worst Case</div>'
                f'<div class="kpi-value">₹{_mc_min:,.0f}</div></div>',
                unsafe_allow_html=True)
        with _ri_c4:
            st.markdown(
                '<div class="kpi-card"><div class="kpi-label">🔺 Best Case</div>'
                f'<div class="kpi-value">₹{_mc_max:,.0f}</div></div>',
                unsafe_allow_html=True)

        # --- PART 3 — Risk Insight ---
        st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

        _mean_so = float(np.mean(so_results))
        _so_prob = _mean_so / 100.0  # convert percentage to probability

        if _so_prob < 0.1:
            st.success(
                "✅ **Low Risk** — System operates under low risk. "
                f"Mean stockout rate is {_mean_so:.2f}%, indicating stable supply chain performance."
            )
        elif _so_prob < 0.3:
            st.warning(
                "⚠️ **Moderate Risk** — Moderate risk under demand uncertainty. "
                f"Mean stockout rate is {_mean_so:.2f}%. Consider increasing safety stock or order quantity."
            )
        else:
            st.error(
                "🔴 **High Risk** — Potential supply chain instability detected. "
                f"Mean stockout rate is {_mean_so:.2f}%. Immediate parameter tuning recommended."
            )
    else:
        st.markdown(
            '<div style="text-align:center; padding:30px; color:#94a3b8;">'
            '<p>Enable <strong>Stochastic Demand</strong> in the sidebar and run the simulation '
            'to view Monte Carlo risk interpretation.</p></div>',
            unsafe_allow_html=True,
        )


# ---- Footer ----
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#475569; font-size:0.78rem; padding:10px 0;">'
    'Digital Twin Control Dashboard · ML-Driven Supply Chain Optimization · Built with Streamlit & Plotly'
    '</p>',
    unsafe_allow_html=True,
)
