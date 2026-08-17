# ML-Driven Digital Twin for Supply Chain and Cash Flow Optimization

## Project Explanation Document

---

## 1. Problem Statement

### 1.1 Real-World Problem Addressed

Modern supply chains operate in environments characterized by demand volatility, lead-time uncertainty, fluctuating supplier costs, and unpredictable revenue streams. Enterprises must simultaneously manage **inventory levels** to prevent stockouts (lost sales) and overstocking (increased holding costs), while maintaining **healthy cash flows** to sustain day-to-day operations. Poor inventory decisions directly cascade into financial instability — excess stock ties up working capital, while insufficient stock leads to missed revenue opportunities and eroded customer trust.

This project addresses the fundamental challenge of **jointly optimizing inventory replenishment and cash flow** in a retail supply chain setting, using machine learning-driven demand forecasting coupled with a digital twin simulation framework.

### 1.2 Why Supply Chain and Cash Flow Management Is Difficult

1. **Demand is inherently stochastic.** Consumer purchasing behavior varies by day of week, season, promotional activity, and external events. Historical averages fail to capture these temporal dynamics.
2. **Lead times are non-trivial.** Replenishment orders placed today do not arrive immediately; the gap between ordering and receiving stock introduces risk of either stockout or excess.
3. **Cost structures are multi-dimensional.** Supplier costs, holding costs, and revenue interact in complex ways. A policy that minimizes stockouts may simultaneously erode margins through excessive holding costs or frequent reorder expenses.
4. **Cash flow is a lagging indicator.** Financial impact of inventory decisions manifests over time, making it difficult to evaluate policies without forward-looking simulation.

### 1.3 Gaps in Traditional Inventory Management Systems

Traditional inventory management systems rely on:

- **Static reorder points** computed from historical average demand, ignoring temporal trends and demand volatility.
- **Rule-of-thumb safety stocks** (e.g., fixed percentage buffers) without statistical grounding in forecast error distributions.
- **Disconnected financial models** that evaluate inventory costs and cash flow independently, preventing holistic policy evaluation.
- **No scenario analysis capability** — traditional systems cannot stress-test policies under hypothetical disruptions such as demand spikes, lead-time shocks, or supplier cost escalation.

This project fills these gaps by integrating **ML-based forecasting**, **day-level simulation**, **statistically computed safety stock**, and **scenario-based stress testing** into a unified digital twin framework.

---

## 2. Research Gap / Motivation

### 2.1 Insufficiency of Static Reorder Policies

A static (fixed) reorder policy sets a constant reorder point, typically computed as:

$$\text{Reorder Point}_{\text{fixed}} = \bar{d} \times L$$

where $\bar{d}$ is the average demand from historical data and $L$ is the lead time. This approach has critical limitations:

- It **does not adapt** to changing demand patterns (e.g., seasonal peaks, weekday vs. weekend effects).
- It **overestimates** reorder urgency during low-demand periods and **underestimates** it during demand surges.
- In this project, the baseline fixed reorder point is computed from training-period average demand (`average_train_demand * lead_time`), and its performance is benchmarked against adaptive alternatives.

### 2.2 Value of Integrating ML with Simulation

Machine learning models can learn temporal demand patterns that static statistics cannot capture — such as day-of-week effects, monthly seasonality, lagged autocorrelation, and price sensitivity. However, a forecast alone does not prescribe *action*. By embedding ML predictions into a **simulation engine** (digital twin), the system can:

- Translate forecasts into **dynamic reorder decisions** that adjust daily.
- Evaluate the **downstream financial consequences** (revenue, costs, cash balance) of these decisions over an extended horizon.
- Compare multiple policies under **identical demand conditions**, isolating the effect of the policy itself.

### 2.3 Why Cash Flow Is Often Ignored in Inventory Optimization

Most academic and industry inventory optimization models focus exclusively on operational metrics (fill rate, service level, average inventory). Cash flow is either assumed infinite or modeled separately in financial planning tools. This creates a disconnect:

- A policy with excellent service levels may be financially unsustainable due to aggressive reordering and high holding costs.
- Conversely, a cost-minimizing policy may produce unacceptable stockout rates.

This project bridges the gap by tracking **cumulative cash balance** and **cash volatility** as first-class evaluation metrics alongside operational ones.

---

## 3. Dataset Used

### 3.1 M5 (Walmart) Dataset

The project uses the **M5 Forecasting Competition dataset** published by Walmart. This is a large-scale, hierarchical, real-world retail sales dataset that provides daily unit sales at the store–item level for Walmart stores in the United States.

### 3.2 Files Used

| File                         | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `sales_train_evaluation.csv` | Daily unit sales for each item–store combination across ~1,940 days (`d_1` to `d_1941`), stored in wide format. |
| `calendar.csv`               | Maps each day identifier (`d_1`, `d_2`, …) to a calendar date, along with day-of-week, month, year, and event/holiday information. |
| `sell_prices.csv`            | Weekly selling prices for each item–store–week combination (`wm_yr_wk`), enabling revenue computation. |

### 3.3 Key Attributes

| Attribute         | Source / Derivation                               | Role in the Model                                   |
|-------------------|---------------------------------------------------|------------------------------------------------------|
| `sales`           | `sales_train_evaluation.csv` (melted to long)     | **Target variable** — daily unit demand              |
| `sell_price`      | `sell_prices.csv` (merged on store, item, week)   | **Feature** — price sensitivity; also used for revenue/cost computation |
| `dayofweek`       | Derived from `calendar.csv` date                  | **Feature** — captures weekday vs. weekend patterns  |
| `month`           | Derived from `calendar.csv` date                  | **Feature** — captures monthly/seasonal patterns     |
| `lag_7`           | 7-day lagged `sales`, grouped by `item_id`        | **Feature** — autoregressive signal from one week ago|
| `rolling_7`       | 7-day rolling mean of `sales`, grouped by `item_id`| **Feature** — smoothed short-term demand trend       |

### 3.4 Preprocessing Pipeline

1. **Store filtering:** Only store `CA_1` is retained to reduce computational load while maintaining a realistic single-store scenario.
2. **Wide-to-long transformation:** The wide-format sales data (columns `d_1` through `d_1941`) is melted into a long-format DataFrame with one row per item–day.
3. **Calendar merge:** The melted data is joined with `calendar.csv` on the day identifier (`d`) to attach calendar dates and temporal attributes.
4. **Price merge:** Selling prices from `sell_prices.csv` are merged on `store_id`, `item_id`, and `wm_yr_wk` (Walmart year–week identifier).
5. **Item filtering:** Only the top 3 most frequently occurring items are retained, focusing the analysis on high-volume SKUs.
6. **Type conversion:** `sales` is cast to numeric (with error coercion) and `date` is converted to datetime.
7. **Sorting and imputation:** Data is sorted by `[item_id, date]` and missing values are forward-filled.
8. **Feature engineering:** `dayofweek`, `month`, `lag_7`, and `rolling_7` features are computed as described above.
9. **Missing value removal:** Rows with NaN values (introduced by lag computation) are dropped.
10. **Time-based split:** Data before `2016-01-01` serves as the training set; data from `2016-01-01` onward serves as the test set.

---

## 4. What Is Being Predicted?

### 4.1 Demand Forecasting

The core prediction task is **daily unit demand forecasting** — predicting how many units of each item will be sold on a given day. Accurate demand forecasts are the foundation upon which the simulation engine makes inventory replenishment decisions.

### 4.2 Inputs to the ML Model

The feature vector for each data point consists of five engineered features:

```
features = ['sell_price', 'dayofweek', 'month', 'lag_7', 'rolling_7']
```

These capture price effects, calendar-driven demand patterns, and temporal autocorrelation in sales.

### 4.3 Target Variable

The target variable is `sales` — the actual daily unit sales for a given item on a given day.

### 4.4 Models Used

#### Primary Model: Linear Regression

The main pipeline (`digital_twin_main.py`) uses **sklearn's Linear Regression** as the demand forecasting model. Despite its simplicity, Linear Regression provides:

- Interpretable coefficients for each feature, aiding in understanding demand drivers.
- Fast training and inference, enabling rapid simulation iteration.
- A solid baseline for comparison with more complex models.

The model is trained on features from the training set (`date < 2016-01-01`) and evaluated on the test set (`date >= 2016-01-01`) using **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Squared Error).

#### Experimental Model: LSTM (Long Short-Term Memory)

A separate module (`lstm_forecasting.py`) implements an **LSTM neural network** for demand forecasting as an advanced alternative:

- **Data preparation:** Sales are normalized to [0, 1] using `MinMaxScaler`. Sliding windows of length 14 (`SEQ_LEN = 14`) are created, where each window of 14 consecutive days predicts the next day's sales.
- **Architecture:** A single LSTM layer with 32 units followed by a Dense output layer.
- **Training:** Adam optimizer, MSE loss, 20 epochs, batch size 16, with 10% validation split.
- **Evaluation:** Predictions are inverse-transformed back to original scale and evaluated with MAE and RMSE.

The LSTM model was developed as a reference implementation to demonstrate the extensibility of the framework to deep learning–based forecasting.

---

## 5. Digital Twin / Simulation Component

### 5.1 What Is a Digital Twin in This Context?

In this project, the **digital twin** is a computational simulation that mirrors the behavior of a real-world retail supply chain on a day-by-day basis. It takes ML-generated demand forecasts as input and simulates:

- Inventory depletion due to daily sales.
- Replenishment ordering based on a configurable policy.
- Order arrivals after a lead time delay.
- Financial transactions (revenue, supplier payments, holding costs).

The digital twin allows decision-makers to **test different inventory policies (adaptive vs. fixed), safety stock levels, and stress scenarios** without disrupting the real supply chain.

### 5.2 Day-by-Day Inventory Simulation

The core simulation function `run_inventory_simulation()` iterates through each day in the test dataset and executes the following logic:

```
For each day i:
    1. Reduce inventory by predicted_sales[i]
    2. Compute reorder point (adaptive or fixed)
    3. If inventory < reorder_point AND no pending orders:
         Place an order of order_quantity, arriving at day (i + lead_time)
    4. Process arriving orders: add quantity to inventory, deduct supplier payment from cash
    5. Deduct holding cost from cash
    6. Add net_cash (revenue − supplier cost − holding cost) to cash balance
```

### 5.3 Modeling of Key Parameters

| Parameter           | Default Value | Description                                                    |
|---------------------|---------------|----------------------------------------------------------------|
| `initial_inventory` | 500 units     | Starting inventory at day 0                                    |
| `lead_time`         | 5 days        | Number of days between placing and receiving an order          |
| `order_quantity`    | 500 units     | Fixed quantity ordered each time a reorder is triggered        |
| `supplier_cost_ratio` | 0.60        | Supplier cost as a fraction of revenue                        |
| `holding_cost_ratio`  | 0.02        | Daily holding cost as a fraction of inventory value           |
| `initial_cash`      | ₹100,000      | Starting cash balance                                         |

### 5.4 Reorder Point Logic

- **Adaptive policy:** `reorder_point = predicted_demand × lead_time + safety_stock`
  - Adjusts dynamically every day based on the ML forecast.
- **Fixed (baseline) policy:** `reorder_point = average_train_demand × lead_time`
  - Remains constant throughout the simulation, computed from historical average demand.

### 5.5 Safety Stock Modeling

Safety stock is computed statistically from forecast residuals:

$$\text{Safety Stock} = Z \times \sigma_e \times \sqrt{L}$$

where:
- $Z = 1.65$ (z-score for 95% service level)
- $\sigma_e$ = standard deviation of forecast errors (`y_test − predictions`)
- $L$ = lead time (5 days)

This provides a statistically grounded buffer against forecast inaccuracy, rather than an arbitrary fixed buffer.

### 5.6 Stress Scenario Simulation

The simulation supports **scenario-based stress testing** through the `prepare_scenario_data()` and `run_stress_scenario()` functions. Each scenario modifies the base test data and/or simulation parameters to evaluate policy resilience under adverse conditions:

| Scenario                         | Modification                           |
|----------------------------------|----------------------------------------|
| Base Case                        | No modifications (control scenario)    |
| Demand Spike (+30%)              | `demand_multiplier = 1.30`             |
| Lead Time Shock (5 → 8 days)     | `sim_lead_time = 8`                    |
| Supplier Cost Increase (0.6 → 0.75) | `sc_supplier_cost_ratio = 0.75`    |
| Revenue Drop (−20%)              | `price_multiplier = 0.80`              |

For each scenario, both adaptive and baseline policies are executed and compared.

### 5.7 Dynamic Reactivity

The digital twin reacts **dynamically** to parameter changes:

- If lead time increases, the simulation automatically adjusts order placement timing and delivery delays.
- If demand is amplified, the reorder point (for adaptive policy) adjusts via the forecast, while the baseline policy cannot adapt.
- If supplier costs increase, supplier payments are recomputed, directly affecting cash balance trajectories.
- Revenue and cost recomputations are performed within `prepare_scenario_data()` before simulation.

---

## 6. Cash Flow Modeling

### 6.1 Revenue Calculation

Revenue for each day is computed as:

$$\text{Revenue}_i = \text{predicted\_sales}_i \times \text{sell\_price}_i$$

This uses the ML-predicted demand (not actual sales), representing the revenue expectation based on the forecast.

### 6.2 Supplier Cost

Supplier cost represents the cost of goods procured:

$$\text{Supplier Cost}_i = \text{Revenue}_i \times \text{supplier\_cost\_ratio}$$

Default `supplier_cost_ratio = 0.60`, meaning 60% of revenue goes to procurement.

Additionally, when orders arrive, a separate supplier payment is deducted:

$$\text{Supplier Payment} = \text{order\_quantity} \times \text{sell\_price}_i \times \text{supplier\_cost\_ratio}$$

### 6.3 Holding Cost

Holding cost is incurred daily based on current inventory level:

$$\text{Holding Cost}_i = \text{inventory}_i \times \text{holding\_cost\_ratio}$$

Default `holding_cost_ratio = 0.02` per unit per day.

### 6.4 Net Cash and Cumulative Cash Tracking

Net cash per day is:

$$\text{Net Cash}_i = \text{Revenue}_i - \text{Supplier Cost}_i - \text{Holding Cost}_i$$

The cumulative cash balance is tracked throughout the simulation:

$$\text{Cash Balance}_i = \text{Cash Balance}_{i-1} + \text{Net Cash}_i - \text{Holding Cost (inventory-based)}_i - \text{Supplier Payments (arrivals)}_i$$

Starting from `initial_cash = 100,000`.

### 6.5 Cash Volatility Measurement

Cash volatility is measured as the **standard deviation** of the daily cash balance time series:

$$\text{Cash Volatility} = \text{std}(\text{Cash Balance}_1, \text{Cash Balance}_2, \ldots, \text{Cash Balance}_n)$$

Lower volatility indicates more predictable and stable cash flow, which is desirable for operational planning.

---

## 7. Policies Compared

The project evaluates **three distinct inventory replenishment policies** under identical demand conditions:

### 7.1 Baseline Fixed Reorder Policy

- **Reorder point:** Constant, set to `average_train_demand × lead_time`.
- **Behavior:** Does not react to changing demand patterns. Represents traditional inventory management.
- **Purpose:** Serves as the control/benchmark for evaluating ML-driven policies.

### 7.2 Adaptive ML-Based Policy (Without Safety Stock)

- **Reorder point:** `predicted_demand × lead_time` (recomputed daily).
- **Behavior:** Dynamically adjusts to forecasted demand. Orders more aggressively during predicted high-demand periods and conservatively during predicted low-demand periods.
- **Purpose:** Demonstrates the value of ML-driven adaptive decision-making versus static rules.

### 7.3 Adaptive ML-Based Policy with Safety Stock

- **Reorder point:** `predicted_demand × lead_time + safety_stock`.
- **Safety stock** is statistically computed from forecast error distributions (see Section 5.5).
- **Behavior:** Combines ML adaptivity with a statistically grounded buffer against forecast inaccuracy.
- **Purpose:** Represents the most robust policy, balancing responsiveness with risk mitigation.

### 7.4 Stress Test Scenarios

All three policies (adaptive with safety stock and baseline fixed) are further evaluated under five stress scenarios (see Section 5.6) to assess resilience and financial robustness under adverse conditions.

---

## 8. Evaluation Metrics

### 8.1 Forecasting Metrics

| Metric | Formula                                                  | Purpose                                           |
|--------|----------------------------------------------------------|---------------------------------------------------|
| **MAE**  | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$                  | Average absolute prediction error (scale of data) |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$           | Penalizes large errors more heavily than MAE      |

### 8.2 Operational Metrics

| Metric               | Computation                                                    | Interpretation                                    |
|----------------------|----------------------------------------------------------------|---------------------------------------------------|
| **Stockout Rate (%)**  | Percentage of simulation days where `inventory < 0`          | Lower is better; indicates supply reliability     |
| **Service Level (%)** | `100 − Stockout Rate`                                        | Higher is better; target is typically ≥ 95%       |
| **Average Inventory** | Mean inventory level across all simulation days               | Balance between too high (costly) and too low (risky) |
| **Total Reorders**    | Count of days where a reorder was triggered                   | Reflects ordering frequency and logistics burden  |

### 8.3 Financial Metrics

| Metric              | Computation                                            | Interpretation                                     |
|---------------------|--------------------------------------------------------|----------------------------------------------------|
| **Final Cash**       | Cash balance at the end of the simulation horizon      | Overall financial outcome of the policy           |
| **Cash Volatility**  | Standard deviation of the daily cash balance series    | Lower indicates more predictable, stable finances |

All six operational and financial metrics are computed by the `calculate_metrics()` function and presented in a comparison table across all policies and stress scenarios.

---

## 9. What the System Ultimately Does

### 9.1 Forecasting Feeds Simulation

The ML model (Linear Regression) generates daily demand predictions for the test period. These predictions are injected into each row of the simulation DataFrame as `predicted_sales`. The simulation engine **consumes** these forecasts to drive inventory depletion and reorder decisions. This creates a seamless pipeline:

```
Raw Data → Feature Engineering → ML Training → Predictions → Simulation Input
```

### 9.2 Simulation Evaluates Financial Impact

The digital twin simulation takes the demand forecasts and, for each policy configuration, produces a complete time series of:

- Daily inventory levels
- Daily cash balances
- Reorder events

These are aggregated into the six evaluation metrics (Section 8), enabling **quantitative comparison** of policies. Decision-makers can identify which policy delivers the best trade-off between service level, inventory efficiency, and financial performance.

### 9.3 Stress Testing Evaluates Resilience

The stress testing module subjects each policy to hypothetical adverse scenarios (demand spikes, lead time shocks, cost increases, revenue drops). By comparing policy performance **across scenarios**, the system reveals:

- Which policy degrades most gracefully under disruption.
- How sensitive financial outcomes are to specific parameter changes.
- Whether the adaptive policy's advantages over the baseline persist under stress.

This provides a **risk-aware evaluation framework** that goes beyond point-estimate optimization.

### 9.4 End-to-End Flow

```
┌──────────────┐    ┌────────────────┐    ┌──────────────────────┐    ┌──────────────┐
│  M5 Dataset  │───▶│  Preprocessing │───▶│  Feature Engineering │───▶│   ML Model   │
│  (Walmart)   │    │  & Merging     │    │  (lag, rolling, etc) │    │ (LinReg/LSTM)│
└──────────────┘    └────────────────┘    └──────────────────────┘    └──────┬───────┘
                                                                             │
                                                                     Predicted Demand
                                                                             │
                                                                             ▼
┌──────────────┐    ┌────────────────┐    ┌──────────────────────┐    ┌──────────────┐
│  Evaluation  │◀───│  Metrics &     │◀───│  Digital Twin        │◀───│  Safety Stock │
│  & Reporting │    │  Comparison    │    │  Simulation Engine   │    │  Computation  │
└──────────────┘    └────────────────┘    └──────────────────────┘    └──────────────┘
                                                   │
                                           ┌───────┴───────┐
                                           │               │
                                    Policy Comparison   Stress Testing
                                    (Adaptive vs Fixed) (5 Scenarios)
```

---

## 10. Final Contribution

### 10.1 Summary of Contributions

This project makes the following contributions as a minor project:

1. **Integrated ML-Simulation Framework:** Unlike conventional approaches that treat forecasting and inventory management as disjoint activities, this project builds a unified pipeline where ML predictions directly drive simulation-based inventory decisions and financial tracking.

2. **Adaptive Reorder Policy:** The project demonstrates that an ML-driven adaptive reorder point — which adjusts daily based on forecasted demand — outperforms static, average-based reorder policies across operational and financial metrics.

3. **Statistically Grounded Safety Stock:** Rather than using arbitrary safety buffers, the project computes safety stock from the standard deviation of forecast residuals at a specified service level (95%), providing a principled risk–buffer trade-off.

4. **Cash Flow as a First-Class Metric:** The project elevates cash flow from an afterthought to a primary evaluation dimension. Revenue, supplier costs, holding costs, net cash, cumulative cash balance, and cash volatility are all tracked and reported alongside traditional inventory metrics.

5. **Scenario-Based Stress Testing:** The project includes a structured stress testing module that evaluates policy robustness under five distinct adverse scenarios — demand surge, lead time shock, supplier cost escalation, and revenue decline — providing actionable insights into supply chain resilience.

6. **LSTM Reference Implementation:** An LSTM-based forecasting module is provided as a reference, demonstrating the framework's extensibility to deep learning–based demand prediction.

### 10.2 Novelty for a Minor Project

While individual components (demand forecasting, inventory simulation, cash flow analysis) exist in isolation in the literature, this project's contribution lies in their **integration into a cohesive digital twin framework** that enables:

- End-to-end evaluation from raw retail data to financial impact assessment.
- Policy comparison under both normal and stressed conditions.
- A modular architecture where the forecasting model, simulation parameters, and stress scenarios can be independently modified and extended.

This makes the project a meaningful demonstration of how **data-driven decision support systems** can be constructed for real-world supply chain operations, going beyond simple forecasting to encompass simulation, financial modeling, and risk analysis.

---

*Document prepared for academic project report, viva presentation, and technical documentation.*
