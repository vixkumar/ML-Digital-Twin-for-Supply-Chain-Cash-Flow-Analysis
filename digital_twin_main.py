import pandas as pd

# =========================
# STEP 1: Load Sales Data
# =========================
sales = pd.read_csv("sales_train_evaluation.csv")

# Keep only one store (reduce size)
sales = sales[sales['store_id'] == 'CA_1']

# Melt wide format to long
sales_long = sales.melt(
    id_vars=['item_id', 'store_id'],
    var_name='d',
    value_name='sales'
)

# =========================
# STEP 2: Merge Calendar
# =========================
calendar = pd.read_csv("calendar.csv")
sales_long = sales_long.merge(calendar, on='d', how='left')

# =========================
# STEP 3: Merge Prices
# =========================
prices = pd.read_csv("sell_prices.csv")
sales_long = sales_long.merge(
    prices,
    on=['store_id', 'item_id', 'wm_yr_wk'],
    how='left'
)

# =========================
# STEP 4: Reduce to Top 3 Items
# =========================
top_items = sales_long['item_id'].value_counts().head(3).index
sales_long = sales_long[sales_long['item_id'].isin(top_items)]

# =========================
# STEP 5: Convert Types
# =========================
sales_long['sales'] = pd.to_numeric(sales_long['sales'], errors='coerce')
sales_long['date'] = pd.to_datetime(sales_long['date'])

# Sort properly for time series
sales_long = sales_long.sort_values(['item_id', 'date'])

# Forward fill missing values
sales_long = sales_long.ffill()

# =========================
# STEP 6: Feature Engineering
# =========================

# Date features
sales_long['dayofweek'] = sales_long['date'].dt.dayofweek
sales_long['month'] = sales_long['date'].dt.month

# Lag feature
sales_long['lag_7'] = (
    sales_long
    .groupby('item_id')['sales']
    .shift(7)
)

# Rolling mean feature
sales_long['rolling_7'] = (
    sales_long
    .groupby('item_id')['sales']
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
)

# =========================
# STEP 7: Drop Missing (after lag creation)
# =========================
sales_long = sales_long.dropna()

# =========================
# STEP 8: Define Features
# =========================
features = [
    'sell_price',   # Correct column name
    'dayofweek',
    'month',
    'lag_7',
    'rolling_7'
]

X = sales_long[features]
y = sales_long['sales']

# =========================
# STEP 9: Time-Based Split
# =========================
split_date = '2016-01-01'

train = sales_long[sales_long['date'] < split_date]
test = sales_long[sales_long['date'] >= split_date]

X_train = train[features]
y_train = train['sales']

X_test = test[features]
y_test = test['sales']

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE:", mae)
print("RMSE:", rmse)

# =====================================================
# ADDITIONAL MODELS: Random Forest & Ridge Regression
# =====================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# --- Random Forest Regressor ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

# --- Ridge Regression (alpha=1.0) ---
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)

ridge_mae = mean_absolute_error(y_test, ridge_predictions)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_predictions))

# --- Model Comparison Table ---
print("\n" + "=" * 55)
print("         MODEL COMPARISON TABLE")
print("=" * 55)
print(f"{'Model':<25} | {'MAE':>10} | {'RMSE':>10}")
print("-" * 55)
print(f"{'Linear Regression':<25} | {mae:>10.4f} | {rmse:>10.4f}")
print(f"{'Random Forest':<25} | {rf_mae:>10.4f} | {rf_rmse:>10.4f}")
print(f"{'Ridge Regression':<25} | {ridge_mae:>10.4f} | {ridge_rmse:>10.4f}")
print("=" * 55)
print("Note: Simulation uses Linear Regression predictions.\n")

# Use best model (Linear Regression in your case)
test = test.copy()
test['predicted_sales'] = predictions  # from linear regression

# Revenue
test['revenue'] = test['predicted_sales'] * test['sell_price']

# Costs
supplier_cost_ratio = 0.6
holding_cost_ratio = 0.02

test['supplier_cost'] = test['revenue'] * supplier_cost_ratio
test['holding_cost'] = test['revenue'] * holding_cost_ratio

# Net cash
test['net_cash'] = test['revenue'] - test['supplier_cost'] - test['holding_cost']


# =========================
# SIMULATION PARAMETERS
# =========================

initial_inventory = 150
lead_time = 8
order_quantity = 200
supplier_cost_ratio = 0.6
holding_cost_ratio = 0.02
initial_cash = 150000

# Calculate Fixed Reorder Point for Baseline
# Using average sales from training data
average_train_demand = y_train.mean()
fixed_reorder_point = average_train_demand * lead_time

# =========================
# SAFETY STOCK CALCULATION
# =========================

# Calculate forecast errors on test set
# errors = Actual - Predicted
residuals = y_test - predictions
forecast_error_std = residuals.std()

# Service Level factor (Z) for 95%
Z = 1.65
safety_stock_val = Z * forecast_error_std * np.sqrt(lead_time)

print(f"Forecast Error Std: {forecast_error_std:.2f}")
print(f"Calculated Safety Stock: {safety_stock_val:.2f}")

# =========================
# SIMULATION FUNCTION
# =========================

def run_inventory_simulation(data, policy_type, reorder_point_val=None, safety_stock=0,
                             sim_lead_time=None, sim_supplier_cost_ratio=None,
                             sim_holding_cost_ratio=None, sim_order_quantity=None,
                             sim_initial_inventory=None, sim_initial_cash=None,
                             use_stochastic_demand=False, stochastic_std=None):
    # Use overrides if provided, else fall back to globals
    _lead_time = sim_lead_time if sim_lead_time is not None else lead_time
    _supplier_cost_ratio = sim_supplier_cost_ratio if sim_supplier_cost_ratio is not None else supplier_cost_ratio
    _holding_cost_ratio = sim_holding_cost_ratio if sim_holding_cost_ratio is not None else holding_cost_ratio
    _order_quantity = sim_order_quantity if sim_order_quantity is not None else order_quantity
    _initial_inventory = sim_initial_inventory if sim_initial_inventory is not None else initial_inventory
    _initial_cash = sim_initial_cash if sim_initial_cash is not None else initial_cash

    sim = data.copy()
    sim = sim.sort_values('date').reset_index(drop=True)
    
    inventory = _initial_inventory
    cash = _initial_cash
    
    inventory_list = []
    cash_list = []
    reorder_list = []
    incoming_orders = []
    
    for i in range(len(sim)):
        predicted_demand = sim.loc[i, 'predicted_sales']

        # --- STOCHASTIC DEMAND (optional) ---
        if use_stochastic_demand:
            _stochastic_std = stochastic_std if stochastic_std is not None else forecast_error_std
            predicted_demand = max(0, np.random.normal(loc=predicted_demand, scale=_stochastic_std))
        # --- END STOCHASTIC DEMAND ---
        
        # Reduce inventory by predicted demand (approximating sales)
        inventory -= predicted_demand
        
        # Define reorder point based on policy
        if policy_type == 'adaptive':
            # Adaptive: reorder_point changes daily based on prediction + safety stock
            current_reorder_point = (predicted_demand * _lead_time) + safety_stock
        elif policy_type == 'fixed':
            # Baseline: constant reorder point
            current_reorder_point = reorder_point_val
        else:
            raise ValueError("Unknown policy type")
            
        # Check if any pending orders already exist
        pending_orders = any(day > i for day, qty in incoming_orders)
        
        if inventory < current_reorder_point and not pending_orders:
            incoming_orders.append((i + _lead_time, _order_quantity))
            reorder_list.append(1)
        else:
            reorder_list.append(0)
        
        # Handle arrivals
        arrivals_today = [qty for day, qty in incoming_orders if day == i]
        for qty in arrivals_today:
            inventory += qty
            # Cost of new order
            supplier_payment = qty * sim.loc[i, 'sell_price'] * _supplier_cost_ratio
            cash -= supplier_payment
        
        # Remove processed arrivals
        incoming_orders = [(day, qty) for day, qty in incoming_orders if day > i]
        
        # Holding cost
        holding_cost = inventory * _holding_cost_ratio
        
        # Update cash
        # Note: net_cash in data includes revenue - (revenue-based costs)
        cash += sim.loc[i, 'net_cash']
        cash -= holding_cost
        
        inventory_list.append(inventory)
        cash_list.append(cash)
        
    sim['inventory'] = inventory_list
    sim['cash_balance'] = cash_list
    sim['reorder_flag'] = reorder_list
    return sim

# =========================
# RUN SIMULATIONS
# =========================

# 1. Adaptive Policy (Original - No Safety Stock)
adaptive_simulation = run_inventory_simulation(test, policy_type='adaptive', safety_stock=0)

# 2. Adaptive Policy (Enhanced - With Safety Stock)
adaptive_ss_simulation = run_inventory_simulation(test, policy_type='adaptive', safety_stock=safety_stock_val)

# 3. Baseline Fixed Policy
baseline_simulation = run_inventory_simulation(test, policy_type='fixed', reorder_point_val=fixed_reorder_point)

# =========================
# EVALUATION METRICS
# =========================

def calculate_metrics(sim_df, name):
    # Stockout Rate: % days with inventory < 0
    stockout_days = (sim_df['inventory'] < 0).sum()
    stockout_rate = (stockout_days / len(sim_df)) * 100
    
    # Service Level: % days with inventory >= 0
    service_level = 100 - stockout_rate
    
    # Average Inventory
    avg_inventory = sim_df['inventory'].mean()
    
    # Total Reorders
    total_reorders = sim_df['reorder_flag'].sum()
    
    # Final Cash Balance
    final_cash = sim_df['cash_balance'].iloc[-1]
    
    # Cash Volatility (Std Dev)
    cash_volatility = sim_df['cash_balance'].std()
    
    return {
        'Policy': name,
        'Stockout Rate (%)': stockout_rate,
        'Service Level (%)': service_level,
        'Avg Inventory': avg_inventory,
        'Total Reorders': total_reorders,
        'Final Cash': final_cash,
        'Cash Volatility': cash_volatility
    }

metrics_adaptive = calculate_metrics(adaptive_simulation, 'Adaptive (Original)')
metrics_adaptive_ss = calculate_metrics(adaptive_ss_simulation, 'Adaptive + Safety Stock')
metrics_baseline = calculate_metrics(baseline_simulation, 'Baseline (Fixed)')

# Comparison
comparison = pd.DataFrame([metrics_adaptive, metrics_adaptive_ss, metrics_baseline])
comparison.set_index('Policy', inplace=True)

print("\n=== SIMULATION RESULTS ===")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(comparison.T) # Transpose for better readability

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(adaptive_simulation['date'], adaptive_simulation['inventory'], label='Adaptive (Orig)', linestyle=':', alpha=0.7)
plt.plot(adaptive_ss_simulation['date'], adaptive_ss_simulation['inventory'], label='Adaptive + SS', linewidth=2)
plt.plot(baseline_simulation['date'], baseline_simulation['inventory'], label='Baseline', linestyle='--', alpha=0.7)
plt.title("Inventory Level Comparison")
plt.legend()
plt.xticks(rotation=45)
# plt.show()

plt.figure(figsize=(12,5))
plt.plot(adaptive_simulation['date'], adaptive_simulation['cash_balance'], label='Adaptive (Orig)', linestyle=':', alpha=0.7)
plt.plot(adaptive_ss_simulation['date'], adaptive_ss_simulation['cash_balance'], label='Adaptive + SS', linewidth=2)
plt.plot(baseline_simulation['date'], baseline_simulation['cash_balance'], label='Baseline', linestyle='--', alpha=0.7)
plt.title("Cash Balance Comparison")
plt.legend()
plt.xticks(rotation=45)
# plt.show()

# ============================================================
# SCENARIO-BASED STRESS TESTING
# ============================================================

def prepare_scenario_data(base_data, demand_multiplier=1.5, price_multiplier=1.0,
                          sc_supplier_cost_ratio=None):
    """
    Create a modified copy of the test DataFrame for a stress scenario.
    Recomputes revenue, supplier_cost, holding_cost, and net_cash.
    """
    _scr = sc_supplier_cost_ratio if sc_supplier_cost_ratio is not None else supplier_cost_ratio

    scenario_data = base_data.copy()
    scenario_data['predicted_sales'] = scenario_data['predicted_sales'] * demand_multiplier
    scenario_data['sell_price']      = scenario_data['sell_price'] * price_multiplier

    # Recompute financials
    scenario_data['revenue']       = scenario_data['predicted_sales'] * scenario_data['sell_price']
    scenario_data['supplier_cost'] = scenario_data['revenue'] * _scr
    scenario_data['holding_cost']  = scenario_data['revenue'] * holding_cost_ratio
    scenario_data['net_cash']      = (scenario_data['revenue']
                                      - scenario_data['supplier_cost']
                                      - scenario_data['holding_cost'])
    return scenario_data


def run_stress_scenario(scenario_data, sim_params, fixed_rp):
    """
    Run both Adaptive and Baseline policies on scenario_data.
    Returns a dict with metrics for each policy.
    """
    adaptive_result = run_inventory_simulation(
        scenario_data, policy_type='adaptive', safety_stock=safety_stock_val, **sim_params
    )
    baseline_result = run_inventory_simulation(
        scenario_data, policy_type='fixed', reorder_point_val=fixed_rp, **sim_params
    )
    return {
        'Adaptive': calculate_metrics(adaptive_result, 'Adaptive'),
        'Baseline': calculate_metrics(baseline_result, 'Baseline'),
    }


# ---- Define Scenarios ----
scenarios = [
    {
        'name': 'Base Case',
        'data_kwargs': {},                               # no data modifications
        'sim_params': {},                                # no param overrides
    },
    {
        'name': 'Demand Spike (+30%)',
        'data_kwargs': {'demand_multiplier': 1.30},
        'sim_params': {},
    },
    {
        'name': 'Lead Time Shock (5 → 8)',
        'data_kwargs': {},
        'sim_params': {'sim_lead_time': 8},
    },
    {
        'name': 'Supplier Cost ↑ (0.6 → 0.75)',
        'data_kwargs': {'sc_supplier_cost_ratio': 0.75},
        'sim_params': {'sim_supplier_cost_ratio': 0.75},
    },
    {
        'name': 'Revenue Drop (−20%)',
        'data_kwargs': {'price_multiplier': 0.80},
        'sim_params': {},
    },
]

# ---- Run All Scenarios ----
stress_results = {}

for scenario in scenarios:
    sc_data = prepare_scenario_data(test, **scenario['data_kwargs'])

    # For fixed policy, recompute reorder point if lead time changes
    sc_lt = scenario['sim_params'].get('sim_lead_time', lead_time)
    sc_fixed_rp = average_train_demand * sc_lt

    results = run_stress_scenario(sc_data, scenario['sim_params'], sc_fixed_rp)
    stress_results[scenario['name']] = results

# ---- Build Comparison Table ----
metric_keys = ['Stockout Rate (%)', 'Service Level (%)', 'Avg Inventory',
               'Total Reorders', 'Final Cash', 'Cash Volatility']

rows = []
for sc_name, policies in stress_results.items():
    for policy_name, metrics in policies.items():
        row = {'Scenario': sc_name, 'Policy': policy_name}
        row.update({k: metrics[k] for k in metric_keys})
        rows.append(row)

stress_df = pd.DataFrame(rows)

# ---- Print Results ----
print("\n" + "=" * 80)
print("               STRESS TEST RESULTS")
print("=" * 80)

pd.set_option('display.float_format', '{:,.2f}'.format)

for sc_name in stress_results:
    print(f"\n--- {sc_name} ---")
    sc_slice = stress_df[stress_df['Scenario'] == sc_name].set_index('Policy')[metric_keys]
    print(sc_slice.T.to_string())

# ---- Summary: Key Metric Comparison Across Scenarios ----
print("\n" + "=" * 80)
print("          SUMMARY: Adaptive Policy Across All Scenarios")
print("=" * 80)

summary_rows = []
for sc_name, policies in stress_results.items():
    m = policies['Adaptive']
    summary_rows.append({
        'Scenario': sc_name,
        'Stockout %': m['Stockout Rate (%)'],
        'Service Level %': m['Service Level (%)'],
        'Final Cash': m['Final Cash'],
        'Cash Volatility': m['Cash Volatility'],
    })

summary_df = pd.DataFrame(summary_rows).set_index('Scenario')
print(summary_df.to_string())
print()

# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================
# Tests combinations of lead_time and order_quantity to evaluate
# how the Adaptive policy responds under different configurations.
# Uses LOCAL variables only — global parameters are NOT modified.
# ============================================================

lead_time_values = [3, 5, 8]
order_quantity_values = [200, 400, 600]

sensitivity_rows = []

for lt_val in lead_time_values:
    for oq_val in order_quantity_values:
        sa_sim = run_inventory_simulation(
            test,
            policy_type='adaptive',
            safety_stock=safety_stock_val,
            sim_lead_time=lt_val,
            sim_order_quantity=oq_val
        )
        sa_metrics = calculate_metrics(sa_sim, f"LT={lt_val}, OQ={oq_val}")

        sensitivity_rows.append({
            'Lead Time': lt_val,
            'Order Qty': oq_val,
            'Stockout Rate (%)': sa_metrics['Stockout Rate (%)'],
            'Avg Inventory': sa_metrics['Avg Inventory'],
            'Final Cash': sa_metrics['Final Cash'],
        })

sensitivity_df = pd.DataFrame(sensitivity_rows)

print("\n" + "=" * 80)
print("                SENSITIVITY ANALYSIS RESULTS")
print("       (Adaptive Policy — Lead Time × Order Quantity)")
print("=" * 80)
print(sensitivity_df.to_string(index=False))
print("=" * 80)
print()

# ============================================================
# STOCHASTIC DEMAND & MONTE CARLO SIMULATION
# ============================================================
# This section adds OPTIONAL stochastic demand modeling and
# Monte Carlo simulation. All existing deterministic logic
# above remains unchanged.
# ============================================================

# --- Stochastic Demand Flag ---
# Set to True to enable stochastic (random) demand in simulations;
# set to False to keep the original deterministic behavior.
use_stochastic_demand = True

# ============================================================
# MONTE CARLO SIMULATION FUNCTION
# ============================================================

def run_monte_carlo_simulation(n_runs=100):
    """
    Run the inventory simulation n_runs times with stochastic demand.
    Collects stockout_rate, service_level, final_cash, and average_inventory
    for each run, then prints summary statistics and risk metrics,
    and optionally displays histograms.
    """
    mc_results = []

    # --- Store per-run metrics in explicit lists ---
    final_cash_results = []
    stockout_rate_results = []
    service_level_results = []
    inventory_results = []

    for run_idx in range(n_runs):
        # Run simulation with stochastic demand enabled
        sim_result = run_inventory_simulation(
            test,
            policy_type='adaptive',
            safety_stock=safety_stock_val,
            use_stochastic_demand=True,
            stochastic_std=forecast_error_std
        )

        # Calculate metrics for this run
        stockout_days = (sim_result['inventory'] < 0).sum()
        stockout_rate = (stockout_days / len(sim_result)) * 100
        service_level = 100 - stockout_rate
        final_cash = sim_result['cash_balance'].iloc[-1]
        avg_inventory = sim_result['inventory'].mean()

        # Append to per-run lists
        final_cash_results.append(final_cash)
        stockout_rate_results.append(stockout_rate)
        service_level_results.append(service_level)
        inventory_results.append(avg_inventory)

        mc_results.append({
            'run': run_idx + 1,
            'stockout_rate': stockout_rate,
            'service_level': service_level,
            'final_cash': final_cash,
            'average_inventory': avg_inventory
        })

    mc_df = pd.DataFrame(mc_results)

    # ============================================================
    # MONTE CARLO RESULTS SUMMARY
    # ============================================================

    print("\n" + "=" * 70)
    print("            MONTE CARLO SIMULATION RESULTS")
    print(f"                   ({n_runs} runs)")
    print("=" * 70)

    # --- Average Results ---
    print("\n--- Average Results ---")
    print(f"  Mean Stockout Rate:     {mc_df['stockout_rate'].mean():.2f} %")
    print(f"  Mean Service Level:     {mc_df['service_level'].mean():.2f} %")
    print(f"  Mean Final Cash:        {mc_df['final_cash'].mean():,.2f}")
    print(f"  Mean Avg Inventory:     {mc_df['average_inventory'].mean():.2f}")

    # --- Risk Metrics ---
    cash_std = mc_df['final_cash'].std()
    prob_stockout_gt5 = (mc_df['stockout_rate'] > 5).mean() * 100
    min_cash = mc_df['final_cash'].min()
    max_cash = mc_df['final_cash'].max()

    print("\n--- Risk Metrics ---")
    print(f"  Std Dev of Final Cash:       {cash_std:,.2f}")
    print(f"  P(Stockout Rate > 5%):       {prob_stockout_gt5:.1f} %")
    print(f"  Minimum Final Cash:          {min_cash:,.2f}")
    print(f"  Maximum Final Cash:          {max_cash:,.2f}")
    print("=" * 70)

    # ============================================================
    # MONTE CARLO VISUALIZATION
    # ============================================================

    # --- Histogram: Final Cash Distribution ---
    plt.figure()
    plt.hist(final_cash_results, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
    plt.title("Monte Carlo Distribution of Final Cash")
    plt.xlabel("Final Cash")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # --- Histogram: Stockout Rate Distribution ---
    plt.figure()
    plt.hist(stockout_rate_results, bins=20, color='salmon', edgecolor='black', alpha=0.8)
    plt.title("Monte Carlo Distribution of Stockout Rate")
    plt.xlabel("Stockout Rate")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # ============================================================
    # OPTIONAL SUMMARY LINE
    # ============================================================
    print("\n--- Summary Statistics ---")
    print(f"  Mean Final Cash:          {np.mean(final_cash_results):,.2f}")
    print(f"  Std Dev of Final Cash:    {np.std(final_cash_results):,.2f}")
    print(f"  Mean Stockout Rate:       {np.mean(stockout_rate_results):.2f} %")
    print("=" * 70)

    return mc_df


# ============================================================
# INTEGRATION: Monte Carlo Toggle
# ============================================================
# Set run_monte_carlo = True  → run Monte Carlo simulation (100 runs)
# Set run_monte_carlo = False → keep existing deterministic output only
# ============================================================

run_monte_carlo = True

if run_monte_carlo:
    mc_results_df = run_monte_carlo_simulation(100)
else:
    # Deterministic simulation already ran above — nothing extra to do.
    print("\n[INFO] Monte Carlo disabled. Using deterministic simulation results above.")
