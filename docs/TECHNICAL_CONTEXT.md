# Technical Context Document

## ML-Driven Digital Twin for Supply Chain and Cash Flow Optimization

**Version:** 1.0  
**Date:** February 2026  
**Type:** Formal Technical Documentation

---

## 1. Project Overview

This project implements an ML-driven digital twin framework that unifies demand forecasting, day-level inventory simulation, cash flow modeling, and scenario-based stress testing into a single end-to-end pipeline for retail supply chain optimization. A Linear Regression model is trained on the M5 (Walmart) dataset to predict daily unit demand. These predictions are consumed by a discrete-event simulation engine that models inventory depletion, replenishment ordering with configurable lead times, and financial transactions (revenue, supplier costs, holding costs). The system compares three inventory policies — a fixed baseline reorder policy, an adaptive ML-driven reorder policy, and an adaptive policy augmented with statistically computed safety stock — across six operational and financial metrics, and further stress-tests each policy under five adverse scenarios (demand surge, lead time shock, supplier cost increase, revenue decline) to evaluate resilience. An LSTM-based forecasting module is also provided as a reference implementation demonstrating extensibility to deep learning models.

---

## 2. Problem Statement

### 2.1 Real-World Business Problem

Retail supply chains must balance two competing objectives: maintaining sufficient inventory to meet customer demand (maximizing service level) while minimizing the capital tied up in unsold stock and the associated holding costs. Poor inventory decisions directly translate to financial losses — stockouts result in missed revenue and customer attrition, while excess inventory inflates holding costs and erodes working capital. Cash flow — the net movement of money through the business — depends critically on the timing and magnitude of these inventory decisions, yet is rarely modeled alongside them.

### 2.2 Why Traditional Inventory Systems Fail

Traditional inventory management relies on static reorder policies, where a fixed reorder point is computed once from historical average demand and remains constant:

```
Reorder_Point_fixed = average_demand × lead_time
```

This approach fails because:

- **No demand adaptivity.** It cannot respond to weekly, monthly, or seasonal demand fluctuations.
- **No forecast error awareness.** Safety buffers, when used, are typically arbitrary (e.g., "add 10%") rather than derived from the statistical distribution of forecast errors.
- **No financial integration.** Inventory policies are evaluated on operational metrics (fill rate, stockout rate) without considering the financial consequences on cash balance and cash volatility.
- **No stress testing.** Traditional systems provide no mechanism to evaluate how a policy performs under hypothetical disruptions.

### 2.3 Why Integrating ML with Simulation Is Important

Machine learning captures demand patterns that static statistics miss — day-of-week effects, price elasticity, lagged autocorrelation, and rolling trends. However, a forecast alone does not prescribe inventory action. The simulation layer is necessary to:

1. Translate daily forecasts into dynamic reorder decisions.
2. Model the temporal mechanics of lead times, order arrivals, and inventory depletion.
3. Track the cumulative financial impact of thousands of sequential decisions.
4. Enable controlled experimentation — comparing policies under identical demand conditions.

---

## 3. System Architecture

### 3.1 End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA PIPELINE                                     │
│                                                                                 │
│  sales_train_evaluation.csv ──┐                                                 │
│  calendar.csv ────────────────┼──▶ Merge & Filter ──▶ Feature Engineering       │
│  sell_prices.csv ─────────────┘    (CA_1 store,       (dayofweek, month,        │
│                                     top 3 items)       lag_7, rolling_7)        │
└───────────────────────────────────────────────┬─────────────────────────────────┘
                                                │
                                    Train/Test Split (2016-01-01)
                                                │
                                    ┌───────────┴───────────┐
                                    │                       │
                                    ▼                       ▼
                            ┌──────────────┐      ┌──────────────┐
                            │   Training   │      │   Test Set   │
                            │   Set        │      │              │
                            └──────┬───────┘      └──────┬───────┘
                                   │                     │
                                   ▼                     │
                          ┌────────────────┐             │
                          │ Linear         │             │
                          │ Regression     │─── predict ─┤
                          │ Model          │             │
                          └────────────────┘             │
                                                         ▼
                                               ┌────────────────────┐
                                               │  predicted_sales   │
                                               │  added to test set │
                                               └────────┬───────────┘
                                                         │
                    ┌────────────────────────────────────┼──────────────────────┐
                    │                                    │                      │
                    ▼                                    ▼                      ▼
          ┌──────────────────┐             ┌──────────────────┐     ┌────────────────────┐
          │ Safety Stock     │             │ Cash Flow        │     │ Simulation         │
          │ Computation      │             │ Pre-computation  │     │ Parameters         │
          │ Z × σ_e × √L    │             │ revenue, costs,  │     │ inventory, lead    │
          │                  │             │ net_cash         │     │ time, order qty    │
          └────────┬─────────┘             └────────┬─────────┘     └─────────┬──────────┘
                   │                                │                         │
                   └────────────────┬───────────────┘                         │
                                    │                                         │
                                    ▼                                         │
                          ┌────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌──────────────────────────────┐
              │   DIGITAL TWIN SIMULATION    │
              │   run_inventory_simulation() │
              │                              │
              │   Executes for each policy:  │
              │   • Adaptive (no SS)         │
              │   • Adaptive (with SS)       │
              │   • Baseline (fixed)         │
              └──────────────┬───────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
      ┌──────────┐   ┌──────────┐   ┌──────────────────┐
      │ Inventory │   │ Cash     │   │ Reorder          │
      │ Time      │   │ Balance  │   │ Events           │
      │ Series    │   │ Series   │   │ Series           │
      └─────┬─────┘   └────┬─────┘   └────┬─────────────┘
            │              │              │
            └──────────┬───┘──────────────┘
                       │
                       ▼
              ┌──────────────────┐        ┌──────────────────────────┐
              │ calculate_       │        │ STRESS TESTING MODULE    │
              │ metrics()        │        │ 5 scenarios ×            │
              │ Per-policy       │        │ 2 policies each          │
              │ evaluation       │        │ prepare_scenario_data()  │
              └────────┬─────────┘        │ run_stress_scenario()    │
                       │                  └────────────┬─────────────┘
                       │                               │
                       └───────────┬───────────────────┘
                                   │
                                   ▼
                         ┌──────────────────┐
                         │ FINAL OUTPUT     │
                         │ • Comparison     │
                         │   Table          │
                         │ • Stress Test    │
                         │   Results Table  │
                         │ • Plots          │
                         └──────────────────┘
```

### 3.2 Component Interaction

| Component | Implemented In | Consumes | Produces |
|-----------|---------------|----------|----------|
| Data ingestion & preprocessing | `digital_twin_main.py` lines 1–77 | Raw CSV files | Clean, feature-enriched `sales_long` DataFrame |
| Feature engineering | `digital_twin_main.py` lines 56–88 | `sales_long` | Feature columns: `sell_price`, `dayofweek`, `month`, `lag_7`, `rolling_7` |
| ML training & prediction | `digital_twin_main.py` lines 90–130 | Feature matrix `X_train`, `X_test` | `predictions` array, MAE, RMSE |
| Financial pre-computation | `digital_twin_main.py` lines 132–143 | `predicted_sales`, `sell_price` | `revenue`, `supplier_cost`, `holding_cost`, `net_cash` columns |
| Safety stock computation | `digital_twin_main.py` lines 162–176 | `y_test`, `predictions`, `lead_time` | `safety_stock_val` (scalar) |
| Inventory simulation engine | `run_inventory_simulation()` lines 182–255 | Test DataFrame with predictions, policy config, parameters | Simulation DataFrame with `inventory`, `cash_balance`, `reorder_flag` |
| Metrics computation | `calculate_metrics()` lines 274–302 | Simulation DataFrame | Dict of 6 metrics |
| Stress testing | `prepare_scenario_data()`, `run_stress_scenario()` lines 342–421 | Test data, scenario configs | Per-scenario, per-policy metrics |
| LSTM forecasting (reference) | `lstm_forecasting.py` lines 1–207 | Raw sales data | LSTM predictions, MAE, RMSE, plots |

---

## 4. Dataset Description

### 4.1 Dataset: M5 Walmart Forecasting Competition

The M5 dataset is a publicly available, large-scale hierarchical retail sales dataset released by Walmart for the M5 Forecasting Competition on Kaggle. It contains daily unit sales for 3,049 items across 10 stores in 3 US states, spanning approximately 5.4 years (January 2011 – June 2016).

### 4.2 Files Used

#### 4.2.1 `sales_train_evaluation.csv`

- **Format:** Wide format — each row is one item–store combination; columns `d_1` through `d_1941` contain daily unit sales.
- **Key columns:** `item_id`, `store_id`, `d_1` … `d_1941`
- **Size:** ~30,490 rows × 1,947+ columns (before filtering)

#### 4.2.2 `calendar.csv`

- **Purpose:** Maps day identifiers (`d_1`, `d_2`, …) to calendar dates and temporal attributes.
- **Key columns used:** `d`, `date`, `wm_yr_wk` (Walmart year–week, used for price join)
- **Additional columns available:** `weekday`, `month`, `year`, `event_name_1`, `snap_CA`, etc.

#### 4.2.3 `sell_prices.csv`

- **Purpose:** Provides weekly selling prices at the store–item level.
- **Key columns:** `store_id`, `item_id`, `wm_yr_wk`, `sell_price`
- **Join key:** Merged with sales data on `(store_id, item_id, wm_yr_wk)`

### 4.3 Key Attributes (Post-Processing)

| Attribute     | Type      | Source                          | Description |
|---------------|-----------|---------------------------------|-------------|
| `item_id`     | String    | `sales_train_evaluation.csv`    | Unique product identifier |
| `store_id`    | String    | `sales_train_evaluation.csv`    | Store identifier (filtered to `CA_1`) |
| `date`        | Datetime  | `calendar.csv`                  | Calendar date |
| `sales`       | Float     | `sales_train_evaluation.csv`    | Daily unit sales (target variable) |
| `sell_price`  | Float     | `sell_prices.csv`               | Weekly selling price per unit |
| `dayofweek`   | Int (0–6) | Derived from `date`             | Day of week (Monday=0, Sunday=6) |
| `month`       | Int (1–12)| Derived from `date`             | Calendar month |
| `lag_7`       | Float     | Derived: `sales.shift(7)` per item | Sales value from 7 days ago |
| `rolling_7`   | Float     | Derived: `sales.rolling(7).mean()` per item | 7-day rolling average of sales |

### 4.4 Preprocessing Steps (Executed Sequentially)

```python
# Step 1: Load sales data and filter to store CA_1
sales = pd.read_csv("sales_train_evaluation.csv")
sales = sales[sales['store_id'] == 'CA_1']

# Step 2: Melt wide format (d_1..d_1941 columns) to long format
sales_long = sales.melt(id_vars=['item_id', 'store_id'], var_name='d', value_name='sales')

# Step 3: Merge with calendar.csv on 'd' to get dates and wm_yr_wk
sales_long = sales_long.merge(calendar, on='d', how='left')

# Step 4: Merge with sell_prices.csv on (store_id, item_id, wm_yr_wk)
sales_long = sales_long.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

# Step 5: Filter to top 3 items by frequency
top_items = sales_long['item_id'].value_counts().head(3).index
sales_long = sales_long[sales_long['item_id'].isin(top_items)]

# Step 6: Type conversion
sales_long['sales'] = pd.to_numeric(sales_long['sales'], errors='coerce')
sales_long['date']  = pd.to_datetime(sales_long['date'])

# Step 7: Sort by (item_id, date) and forward-fill missing values
sales_long = sales_long.sort_values(['item_id', 'date'])
sales_long = sales_long.ffill()

# Step 8: Feature engineering (dayofweek, month, lag_7, rolling_7)
# Step 9: Drop rows with NaN (from lag creation)
# Step 10: Time-based train/test split at 2016-01-01
```

### 4.5 Filtering Logic

| Filter                  | Rationale |
|-------------------------|-----------|
| Store = `CA_1`          | Reduces to a single-store scenario for computational tractability while preserving realistic demand patterns |
| Top 3 items by count    | Focuses on high-volume SKUs where demand patterns are most pronounced and statistically reliable |
| Date split at 2016-01-01| Ensures temporal integrity — no data leakage from future to past |

---

## 5. Demand Forecasting Model

### 5.1 Features Used

```python
features = ['sell_price', 'dayofweek', 'month', 'lag_7', 'rolling_7']
```

| Feature       | Type        | Rationale |
|---------------|-------------|-----------|
| `sell_price`  | Continuous  | Captures price sensitivity / demand elasticity |
| `dayofweek`   | Ordinal     | Captures weekday vs. weekend demand patterns |
| `month`       | Ordinal     | Captures seasonal and monthly demand cycles |
| `lag_7`       | Continuous  | 7-day autoregressive signal (weekly periodicity) |
| `rolling_7`   | Continuous  | Smoothed short-term demand trend |

### 5.2 Target Variable

```python
y = sales_long['sales']  # Daily unit sales (continuous, non-negative)
```

### 5.3 Train/Test Split

- **Split criterion:** Temporal split at `2016-01-01`
- **Training set:** All observations where `date < 2016-01-01`
- **Test set:** All observations where `date >= 2016-01-01`
- **Rationale:** Time-based split prevents data leakage and simulates a realistic deployment scenario where the model is trained on past data and deployed on future data.

```python
split_date = '2016-01-01'
train = sales_long[sales_long['date'] < split_date]
test  = sales_long[sales_long['date'] >= split_date]
```

### 5.4 Primary Model: Linear Regression

- **Library:** `sklearn.linear_model.LinearRegression`
- **Training:** `model.fit(X_train, y_train)`
- **Inference:** `predictions = model.predict(X_test)`
- **Strengths:** Fast, interpretable, provides a solid baseline.
- **Limitation:** Assumes a linear relationship between features and demand; cannot capture complex nonlinear interactions.

### 5.5 LSTM Attempt (Reference Implementation)

File: `lstm_forecasting.py`

| Parameter      | Value            |
|----------------|------------------|
| Sequence length| 14 days          |
| LSTM units     | 32               |
| Architecture   | LSTM(32) → Dense(1) |
| Optimizer      | Adam             |
| Loss function  | MSE              |
| Epochs         | 20               |
| Batch size     | 16               |
| Validation     | 10% of training data |
| Normalization  | MinMaxScaler [0, 1] |

**Data preparation for LSTM:**
1. Sales are normalized to [0, 1] range using `MinMaxScaler`.
2. Sliding windows of length 14 are created: input = 14 consecutive days, target = day 15.
3. Sequences are created per item, then concatenated across items.
4. Input is reshaped to 3D: `[samples, 14, 1]` for LSTM consumption.
5. Predictions are inverse-transformed back to the original scale for evaluation.

The LSTM module is a **standalone reference implementation** — its predictions are not currently integrated into the simulation engine. It demonstrates the framework's extensibility to deep learning methods.

### 5.6 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | `mean(|y_actual − y_predicted|)` | Average magnitude of prediction errors in sales units |
| **RMSE** | `sqrt(mean((y_actual − y_predicted)²))` | Root mean squared error; penalizes large deviations more than MAE |

Both are computed on the test set:
```python
mae  = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
```

---

## 6. Digital Twin Simulation Logic

### 6.1 What Is Simulated Daily

The function `run_inventory_simulation()` iterates through each day `i` in the test set (in chronological order) and simulates the following events in sequence:

```
For each day i = 0, 1, 2, ..., N-1:

    (a) DEMAND DEPLETION
        inventory ← inventory − predicted_sales[i]

    (b) REORDER POINT COMPUTATION
        If policy == 'adaptive':
            reorder_point ← predicted_sales[i] × lead_time + safety_stock
        If policy == 'fixed':
            reorder_point ← fixed_reorder_point  (constant)

    (c) REORDER DECISION
        If inventory < reorder_point AND no pending orders exist:
            Schedule order: (arrival_day = i + lead_time, quantity = order_quantity)
            Record reorder_flag = 1
        Else:
            Record reorder_flag = 0

    (d) ORDER ARRIVAL PROCESSING
        For each pending order with arrival_day == i:
            inventory ← inventory + order_quantity
            cash ← cash − (order_quantity × sell_price[i] × supplier_cost_ratio)
        Remove processed orders from queue

    (e) HOLDING COST
        holding_cost ← inventory × holding_cost_ratio

    (f) CASH UPDATE
        cash ← cash + net_cash[i] − holding_cost

    (g) RECORD STATE
        Append inventory, cash to time series
```

### 6.2 Inventory Update Formula

Per day, inventory changes as follows:

```
inventory[i] = inventory[i-1]
             − predicted_sales[i]                     (demand depletion)
             + Σ(arrivals at day i)                    (replenishment)
```

Inventory can go **negative**, representing a stockout condition. Negative inventory days are counted in the stockout rate metric.

### 6.3 Reorder Logic

- **Trigger condition:** `inventory < reorder_point AND NOT any_pending_orders`
- **Pending order check:** `any(day > i for day, qty in incoming_orders)` — prevents placing a new order while one is already in transit.
- **Order placed:** `incoming_orders.append((i + lead_time, order_quantity))`
- **This is a (s, Q) policy variant** — reorder when inventory drops below point `s`, order a fixed quantity `Q`.

### 6.4 Lead Time Modeling

- Lead time is modeled as a **deterministic delay**: an order placed on day `i` arrives exactly on day `i + lead_time`.
- Default `lead_time = 5` days.
- In stress testing, lead time can be overridden via `sim_lead_time` parameter (e.g., increased to 8 days for lead time shock scenario).

### 6.5 Safety Stock Formula

```
safety_stock = Z × σ_e × √L
```

Where:
- `Z = 1.65` — z-score corresponding to a 95% service level (one-tailed normal distribution)
- `σ_e` — standard deviation of forecast residuals: `std(y_test − predictions)`
- `L = 5` — lead time in days
- `√L` — accounts for demand uncertainty accumulating over the lead time window

Implementation:
```python
residuals = y_test - predictions
forecast_error_std = residuals.std()
Z = 1.65
safety_stock_val = Z * forecast_error_std * np.sqrt(lead_time)
```

### 6.6 How Stress Testing Modifies Parameters

The `prepare_scenario_data()` function creates a modified copy of the test data:

```python
scenario_data['predicted_sales'] *= demand_multiplier     # Scale demand
scenario_data['sell_price']      *= price_multiplier       # Scale price

# Recompute all financial columns:
scenario_data['revenue']       = predicted_sales × sell_price
scenario_data['supplier_cost'] = revenue × supplier_cost_ratio
scenario_data['holding_cost']  = revenue × holding_cost_ratio
scenario_data['net_cash']      = revenue − supplier_cost − holding_cost
```

The `run_stress_scenario()` function then runs both adaptive (with safety stock) and baseline policies on this modified data, optionally with overridden simulation parameters (e.g., `sim_lead_time=8`).

**Defined scenarios:**

| # | Scenario Name              | Data Modification                 | Simulation Override     |
|---|----------------------------|-----------------------------------|-------------------------|
| 1 | Base Case                  | None                              | None                    |
| 2 | Demand Spike (+30%)        | `demand_multiplier = 1.30`        | None                    |
| 3 | Lead Time Shock (5 → 8)   | None                              | `sim_lead_time = 8`     |
| 4 | Supplier Cost ↑ (0.6→0.75)| `sc_supplier_cost_ratio = 0.75`   | `sim_supplier_cost_ratio = 0.75` |
| 5 | Revenue Drop (−20%)       | `price_multiplier = 0.80`         | None                    |

---

## 7. Cash Flow Modeling

### 7.1 Revenue Calculation

Revenue is computed per day based on predicted demand and current selling price:

```
revenue[i] = predicted_sales[i] × sell_price[i]
```

**Note:** Revenue is based on *predicted* (not actual) demand for consistency with the simulation's decision-making framework — the system operates on its forecasts.

### 7.2 Supplier Cost

Two types of supplier costs exist:

**(a) Revenue-proportional supplier cost** (pre-computed in data):
```
supplier_cost[i] = revenue[i] × supplier_cost_ratio    (default ratio = 0.60)
```

**(b) Order arrival supplier payment** (computed during simulation):
```
supplier_payment = order_quantity × sell_price[i] × supplier_cost_ratio
```
Deducted from cash when a replenishment order physically arrives.

### 7.3 Holding Cost

Two holding cost formulations are used:

**(a) Revenue-proportional** (pre-computed, included in `net_cash`):
```
holding_cost_data[i] = revenue[i] × holding_cost_ratio   (default ratio = 0.02)
```

**(b) Inventory-based** (computed during simulation):
```
holding_cost_sim[i] = inventory[i] × holding_cost_ratio
```
Deducted separately from cash during simulation.

### 7.4 Net Cash

Pre-computed per day in the test DataFrame:
```
net_cash[i] = revenue[i] − supplier_cost[i] − holding_cost_data[i]
```

This represents the net cash inflow from daily sales, before inventory-level holding costs and replenishment payments.

### 7.5 Cash Balance Tracking

The running cash balance is updated each simulation day:

```
cash_balance[i] = cash_balance[i-1]
                + net_cash[i]                    (revenue net of revenue-based costs)
                − holding_cost_sim[i]            (inventory-level holding cost)
                − supplier_payment[i]            (payment for arriving orders, if any)
```

Starting value: `initial_cash = 100,000`.

### 7.6 Cash Volatility Calculation

```
cash_volatility = std(cash_balance[0], cash_balance[1], ..., cash_balance[N-1])
```

Standard deviation of the daily cash balance series. Lower volatility indicates more predictable, stable cash flow — a desirable property for operational planning and financial risk management.

---

## 8. Policies Implemented

### 8.1 Baseline Fixed Reorder Policy

```python
reorder_point = average_train_demand × lead_time   # Constant, computed once
```

- Reorder point does not change over the simulation.
- Computed from training-set average demand: `y_train.mean() × lead_time`.
- Represents traditional static inventory management.
- Does not use safety stock.

### 8.2 Adaptive ML-Based Reorder Policy (Without Safety Stock)

```python
reorder_point[i] = predicted_sales[i] × lead_time   # Changes daily
```

- Reorder point is recomputed every day using the ML prediction for that day.
- More responsive to demand fluctuations than the fixed policy.
- Safety stock = 0 — no buffer against forecast inaccuracy.

### 8.3 Adaptive ML-Based Reorder Policy with Safety Stock

```python
reorder_point[i] = predicted_sales[i] × lead_time + safety_stock   # Changes daily
```

- Same adaptive logic as 8.2, plus a constant safety stock buffer.
- Safety stock is computed statistically (see Section 6.5).
- Most robust policy — adapts to demand while maintaining a risk buffer.

### 8.4 Stress Scenarios

Each stress scenario modifies the test data and/or simulation parameters (detailed in Section 6.6). Both the adaptive (with safety stock) and baseline policies are run on each scenario, yielding 10 simulation runs total (5 scenarios × 2 policies).

---

## 9. Evaluation Framework

### 9.1 Metrics Used

The `calculate_metrics()` function computes six metrics for each simulation run:

| # | Metric                | Formula / Computation | What It Represents |
|---|----------------------|----------------------|-------------------|
| 1 | **Stockout Rate (%)** | `100 × count(inventory < 0) / total_days` | Percentage of days the system failed to meet demand |
| 2 | **Service Level (%)** | `100 − Stockout Rate` | Percentage of days demand was fully satisfiable |
| 3 | **Average Inventory** | `mean(inventory)` across all simulation days | Mean units held in stock; high = capital tied up, low = stockout risk |
| 4 | **Total Reorders** | `sum(reorder_flag)` | Number of replenishment orders placed; proxy for logistics burden |
| 5 | **Final Cash** | `cash_balance[last_day]` | End-of-horizon cash position; overall financial outcome |
| 6 | **Cash Volatility** | `std(cash_balance)` | Stability/predictability of cash flow over time |

### 9.2 How Results Are Compared

**Policy comparison (normal conditions):**
```python
comparison = pd.DataFrame([metrics_adaptive, metrics_adaptive_ss, metrics_baseline])
comparison.set_index('Policy', inplace=True)
print(comparison.T)  # Transposed for readability
```
All three policies are evaluated on the same test data under identical conditions, isolating the effect of the policy itself.

**Stress test comparison:**
For each of the 5 scenarios, adaptive and baseline policy metrics are placed side by side. A summary table shows how the adaptive policy's key metrics (stockout %, service level, final cash, cash volatility) change across scenarios.

**Visualization:**
Two comparison plots are generated:
1. **Inventory Level Comparison** — time series of daily inventory for all three policies.
2. **Cash Balance Comparison** — time series of daily cash balance for all three policies.

---

## 10. Current System Capabilities

### 10.1 What Works

| Capability | Status | Details |
|-----------|--------|---------|
| Data ingestion from M5 CSV files | ✅ Complete | Loads, merges, and filters all three files |
| Feature engineering | ✅ Complete | 5 features: sell_price, dayofweek, month, lag_7, rolling_7 |
| Linear Regression forecasting | ✅ Complete | Trained, evaluated (MAE/RMSE), predictions used in simulation |
| LSTM forecasting (standalone) | ✅ Complete | Separate module with training, eval, and plots; not integrated into simulation |
| Day-level inventory simulation | ✅ Complete | Supports adaptive and fixed policies, configurable parameters |
| Safety stock computation | ✅ Complete | Statistical method: Z × σ_e × √L |
| Cash flow modeling | ✅ Complete | Revenue, supplier cost, holding cost, net cash, cumulative tracking |
| Multi-policy comparison | ✅ Complete | 3 policies compared on 6 metrics with tabular and visual output |
| Scenario-based stress testing | ✅ Complete | 5 scenarios, 2 policies each, with comparison tables |
| Visualization | ✅ Complete | Inventory and cash balance time series plots |

### 10.2 What Has Been Stress Tested

- Demand surge (+30%)
- Lead time disruption (5 → 8 days)
- Supplier cost escalation (60% → 75%)
- Revenue decline (−20%)
- Base case (control)

### 10.3 Limitations

| Limitation | Description |
|-----------|-------------|
| Single store | Only `CA_1` is modeled; multi-store / multi-echelon supply chains are not covered |
| 3 items only | Filtered to top 3 items; full product catalog not simulated |
| Deterministic lead time | Lead time is fixed per scenario; stochastic lead time variation is not modeled |
| No demand uncertainty in simulation | Simulation uses point forecasts, not probabilistic demand distributions |
| Linear Regression only in pipeline | LSTM is standalone; more advanced models (XGBoost, etc.) are not integrated |
| No backorder / lost sales modeling | Negative inventory is tracked but not penalized (no backorder cost or lost sale penalty) |
| Revenue based on predicted demand | Cash flow uses forecasted sales, not actual; overestimation inflates revenue figures |
| No multi-objective optimization | Policies are compared empirically; no formal optimization objective (e.g., minimize cost subject to service level ≥ 95%) |
| No real-time / streaming | Batch-mode only; all data is available upfront |

---

## 11. Assumptions Made

### 11.1 Cost Assumptions

| Assumption | Value | Rationale |
|-----------|-------|-----------|
| Supplier cost = 60% of revenue | `supplier_cost_ratio = 0.60` | Typical retail COGS margin |
| Holding cost = 2% of inventory value per day | `holding_cost_ratio = 0.02` | Standard warehousing cost approximation |
| Initial cash = 100,000 | `initial_cash = 100000` | Arbitrary starting capital |
| Order quantity is fixed at 500 units | `order_quantity = 500` | Simplified; no EOQ optimization |

### 11.2 Demand Assumptions

| Assumption | Justification |
|-----------|---------------|
| Predicted sales = actual consumption | Simulation uses forecast as the actual demand drawn from inventory |
| Demand for different items is independent | No cross-item cannibalization or substitution effects |
| Past patterns continue into the future | The ML model assumes stationarity in learned relationships |
| Top 3 items are representative | High-volume items exhibit clearer patterns and are more forecastable |

### 11.3 Simplifications in Simulation

| Simplification | Impact |
|----------------|--------|
| At most one pending order at a time | `not pending_orders` check prevents overlapping orders; real systems may allow multiple |
| No perishability or shelf life | Inventory does not expire |
| No lead time variability | Orders always arrive exactly `lead_time` days after placement |
| No shortage cost or penalty | Stockouts reduce service level metric but do not incur an explicit cost |
| Single supplier | No multi-sourcing or supplier selection logic |
| No capacity constraints | Orders of 500 units are always fulfillable by the supplier |
| Safety stock is constant | Computed once from test-set residuals; not updated dynamically |

---

## 12. Final Contribution

### 12.1 What Makes This Project Meaningful

1. **End-to-end integration.** The project connects raw retail data through ML forecasting, inventory simulation, cash flow modeling, and stress testing in a single executable pipeline. Each component's output is a direct input to the next, eliminating the information silos that plague traditional supply chain tools.

2. **Quantifiable policy comparison.** Three inventory policies are compared under identical conditions using six rigorously defined metrics. This provides empirical evidence for the value of ML-driven adaptive decision-making over static rules.

3. **Financial-aware inventory management.** By incorporating cash balance tracking and cash volatility as first-class metrics, the system evaluates policies not just on operational performance but on financial sustainability — a critical dimension often missing in academic inventory models.

4. **Stress testing and resilience analysis.** Five structured adverse scenarios test policy robustness against realistic disruptions (demand spikes, lead time shocks, cost increases, revenue drops). This transforms the system from a point-estimate optimizer to a risk-aware decision support tool.

5. **Statistical safety stock.** The safety stock buffer is derived from forecast error statistics at a target service level (95%), not from arbitrary rules of thumb. This provides a principled trade-off between inventory investment and stockout risk.

### 12.2 Why It Qualifies as a Digital Twin

A digital twin is a virtual representation of a real-world system that:

1. **Mirrors physical behavior** — The simulation replicates daily inventory operations: demand depletion, reorder triggers, order processing, lead time delays, and financial flows.

2. **Uses real data** — The M5 Walmart dataset provides authentic retail demand patterns, prices, and temporal structure.

3. **Enables what-if analysis** — The stress testing module allows "what if demand increases 30%?" or "what if lead time doubles?" experiments without any physical risk.

4. **Supports decision-making** — Policy comparison metrics directly inform which inventory strategy should be adopted under specific operating conditions.

5. **Can be updated** — The modular architecture allows the forecasting model, simulation parameters, and scenarios to be independently modified and re-run.

The system therefore satisfies the core definition of a digital twin: a **data-driven virtual replica of a supply chain** that enables simulation, analysis, and optimization of operating policies without disrupting real operations.

---

*This document serves as a complete technical reference for the ML-Driven Digital Twin for Supply Chain and Cash Flow Optimization project. It is intended for use by evaluators, collaborators, and AI systems that require full context to understand, assess, or extend the system.*
