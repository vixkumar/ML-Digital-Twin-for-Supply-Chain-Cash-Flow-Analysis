# ML-Driven Digital Twin for Supply Chain and Cash Flow Optimization

### Under Demand Uncertainty

An ML-driven Digital Twin framework integrating demand forecasting, stochastic demand modeling, inventory simulation, cash-flow modeling, Monte Carlo risk analysis, and policy evaluation for risk-aware supply chain decision making.

**[🚀 Live Demo](https://supply-chain-digital-twin.streamlit.app/)** | **[📄 Research Paper](docs/research-paper.pdf)** | **[💻 GitHub Repository](https://github.com/vixkumar/supply-chain-digital-twin-)**

---

## 📸 Project Preview

*(Place your Streamlit application screenshots in the `docs/screenshots/` folder)*

| Dashboard | Demand Forecast |
|:---:|:---:|
| ![Dashboard](docs/screenshots/dashboard.png) | ![Demand Forecast](docs/screenshots/demand-forecast.png) |

| Inventory Simulation | Simulation Playback |
|:---:|:---:|
| ![Inventory Simulation](docs/screenshots/inventory-simulation.png) | ![Simulation Playback](docs/screenshots/simulation-playback.png) |

| Cash Flow Simulation | Monte Carlo Risk Analysis |
|:---:|:---:|
| ![Cash Flow](docs/screenshots/cash-flow.png) | ![Monte Carlo](docs/screenshots/monte-carlo.png) |

| Scenario Stress Testing | Policy Comparison |
|:---:|:---:|
| ![Scenario Analysis](docs/screenshots/scenario-analysis.png) | ![Policy Comparison](docs/screenshots/policy-comparison.png) |

---

## 🔍 Overview

Modern retail supply chains operate under constant demand uncertainty. While traditional forecasting models predict future sales, prediction alone is not enough. Knowing that demand will be exactly 100 units is helpful, but supply chain managers need to know *what happens* when demand is actually 130 or 70 units. They need to understand how inventory evolves, when replenishment should occur, the financial impact on cash flow, and how resilient their policies are to unexpected disruptions.

This project addresses this by integrating Machine Learning forecasting directly into a discrete-event Digital Twin simulation. The system uses the ML forecast as an input to a simulated operational environment, allowing decision-makers to evaluate different inventory policies, model stochastic demand, run Monte Carlo risk analysis, and perform scenario-based stress testing — all before making real-world financial commitments.

---

## 🎯 Problem Statement

Conventional forecasting systems primarily predict future demand but do not necessarily show:
- How inventory levels evolve day-by-day
- When physical replenishment should occur given specific lead times
- Whether stockouts will occur and at what rate
- How safety stock affects operations
- How inventory decisions directly affect running cash balances
- How demand uncertainty changes the distribution of financial outcomes
- How different replenishment policies perform under supply chain stress

This project addresses this gap using a unified ML + Digital Twin + Financial/Risk Simulation framework.

---

## 🔬 Research Gap

Existing approaches commonly focus on isolated components: demand forecasting, inventory optimization, Digital Twins, supply-chain resilience, or financial modeling. However, the translation of a statistical forecast error into a financial risk metric is often lost across different software silos. 

This project bridges these domains by integrating time-series ML, operational inventory dynamics, and financial cash-flow tracking into a single, unified decision-support workflow.

---

## 💡 Proposed Solution

A complete end-to-end pipeline from raw data to actionable decision support. The system trains ML models on historical retail data, embeds predictions into a discrete-event simulation engine (Digital Twin), models financial transactions day-by-day, and evaluates multiple inventory policies under both deterministic and stochastic conditions — including stress scenarios and Monte Carlo risk analysis.

---

## ⚙️ System Pipeline

```text
M5 Dataset
↓
Preprocessing
↓
Feature Engineering
↓
ML Demand Forecasting
↓
Model Evaluation
↓
Residual Analysis
↓
Stochastic Demand
↓
Digital Twin Simulation
↓
Inventory Simulation
↓
Cash Flow Modeling
↓
Monte Carlo Risk Analysis
↓
Scenario & Sensitivity Analysis
↓
Policy Evaluation
↓
Streamlit Dashboard
```

---

## ✨ Key Features

- M5 Walmart sales dataset processing
- Time-series preprocessing
- Feature engineering
- Demand forecasting
- Linear Regression
- Random Forest
- Ridge Regression
- LSTM experimental model
- MAE/RMSE model evaluation
- Residual analysis
- Stochastic demand modeling
- Digital Twin inventory simulation
- Reorder-point logic
- Safety stock modeling
- Adaptive inventory policy
- Cash-flow simulation
- Monte Carlo risk analysis
- Scenario/stress testing
- Sensitivity analysis
- Policy comparison
- Simulation playback
- Interactive dashboard

---

## 🤖 Machine Learning

### Models Evaluated
- **Linear Regression**
- **Random Forest** (100 estimators)
- **Ridge Regression** ($\alpha = 1.0$)
- **LSTM** (standalone experimental module)

### Methodology & Selection
The models were trained on historical data (prior to `2016-01-01`) and evaluated on a holdout test set using **MAE** and **RMSE**. 

Linear Regression was selected as the primary forecasting engine because it achieved the best performance **for the experimental dataset and evaluation setup used in this project**. 

**Input Features:** `sell_price`, `dayofweek`, `month`, `lag_7`, `rolling_7`  
**Target Variable:** `sales` (daily unit demand)

---

## 🏭 Digital Twin Simulation

The Digital Twin represents the evolution of the supply-chain inventory and financial state over simulated time.

```text
ML Forecast
     ↓
Uncertainty Modeling
     ↓
Simulated Demand
     ↓
Inventory State
     ↓
Reorder Decision
     ↓
Replenishment
     ↓
Financial Impact
```

---

## 📦 Inventory Simulation

**Inventory Evolution Formula:**  
$I_t = I_{t-1} - D_t + O_t$

Where:
- $I_t$ = Inventory at end of day $t$
- $D_t$ = Demand (forecasted or stochastic) on day $t$
- $O_t$ = Quantity of replenishment orders arriving on day $t$ (placed at $t - L$)

When inventory falls below the reorder point, a new order is triggered. The reorder point logic uses the simulated lead time. The system also calculates safety stock dynamically using the formula: $Z \times \sigma_e \times \sqrt{L}$.

---

## 💰 Cash Flow Simulation

Financial transactions are tracked daily in the simulation:
- **Revenue:** Computed from predicted demand and current selling price.
- **Supplier Cost:** Sourced as a ratio of revenue.
- **Holding Cost:** Charged against current physical inventory levels.
- **Order Payment:** Deducted from cash when a replenishment order physically arrives.

The running cash balance is updated continuously to track the holistic financial health of the supply chain.

---

## 🎲 Stochastic Demand Modeling

In a deterministic simulation, demand is exactly the forecast:  
$D_t = \hat{y}_t$

The simulation introduces realistic uncertainty using the forecast and residual variability:  
$D_t \sim \mathcal{N}(\hat{y}_t, \sigma_e)$

Where:
- $D_t$ = Simulated demand
- $\hat{y}_t$ = ML-predicted demand
- $\sigma_e$ = Residual standard deviation (derived from test set error)

---

## 🎯 Monte Carlo Risk Analysis

To quantify risk, the Monte Carlo module runs the inventory simulation **100 times**, with each run experiencing a different stochastic demand realization. 

Because demand fluctuates differently in each run, inventory and cash flow are recalculated 100 times. By analyzing the resulting distributions, the system produces actionable risk metrics:
- Expected final cash
- Best/worst case cash scenarios
- Stockout probability
- Service level distributions
- Inventory statistics

---

## ⚡ Scenario & Stress Testing

The Digital Twin evaluates policy robustness under adverse conditions through predefined stress scenarios:
- **Demand Spike** (+30% demand multiplier)
- **Lead Time Shock** (Delay increases from 5 to 8 days)
- **Supplier Cost Increase** (Cost ratio rises from 0.6 to 0.75)
- **Revenue Drop** (Selling price drops by 20%)

---

## 📊 Sensitivity Analysis

The system performs parameter sweeps to identify how changes in important parameters influence outcomes such as inventory, stockout risk, service level, or cash flow. The parameters varied in this analysis are:
- **Lead time** (3, 5, and 8 days)
- **Order quantity** (200, 400, and 600 units)

---

## 🔄 Policy Evaluation

Three distinct inventory replenishment policies are evaluated:
1. **Baseline** (Fixed static reorder point)
2. **Adaptive** (ML-driven dynamic reorder point)
3. **Adaptive + Safety Stock** (ML-driven + statistical safety buffer)

Policies are compared using actual KPIs such as stockout rate, service level, average inventory, final cash, and cash volatility.

---

## 📈 Results

*Note: The deterministic results depend on initialization parameters. The following represent standard baseline observations from the project.*

- **Model Accuracy:** The Linear Regression model minimized both MAE and RMSE on the holdout test set for this configuration.
- **Service Level:** The Adaptive + Safety Stock policy effectively maintains service levels $>95\%$ across standard operational conditions.
- **Financial Stability:** The Adaptive policy significantly reduces cash volatility compared to the fixed Baseline policy.
- **Monte Carlo Outcomes:** Stochastic runs reveal the distribution of expected final cash and quantify the specific probability of stockout events, demonstrating the necessity of the safety stock buffer.

---

## 🛠️ Technology Stack

**Language**
- Python 3.11+

**Data & ML**
- Pandas
- NumPy
- Scikit-learn
- TensorFlow/Keras (experimental LSTM)

**Visualization/UI**
- Streamlit
- Plotly
- Matplotlib

**Methods**
- Time-series forecasting
- Digital Twin simulation
- Stochastic modeling
- Monte Carlo simulation
- Sensitivity analysis

---

## 🚀 Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vixkumar/supply-chain-digital-twin-.git
   cd supply-chain-digital-twin-
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download missing dataset files:**
   Download `sales_train_evaluation.csv` and `sell_prices.csv` from the Kaggle M5 Competition and place them in the root directory.

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Live Demo

**[🚀 Launch the Digital Twin](https://supply-chain-digital-twin.streamlit.app/)**

*(This is the deployed Streamlit application hosted on Streamlit Community Cloud)*

---

## 📄 Research Publication

**ML-Driven Digital Twin for Supply Chain and Cash Flow Optimization Under Demand Uncertainty**

Accepted for oral presentation at **ICSSIT 2026**.

**Paper ID:** ICSSIT-477

**[📄 Read the Research Paper](docs/research-paper.pdf)**

---

## 📁 Project Structure

```text
supply-chain-digital-twin-/
├── app.py                      # Main Streamlit dashboard application
├── digital_twin_main.py        # Core ML and simulation engine
├── lstm_forecasting.py         # Experimental LSTM module
├── calendar.csv                # M5 calendar dataset (included)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusion rules
├── README.md                   # Project documentation
└── docs/
    ├── research-paper.pdf      # Research publication
    ├── screenshots/            # Application screenshots
    ├── TECHNICAL_CONTEXT.md    # Detailed technical context
    ├── PROJECT_EXPLANATION.md  # Detailed project explanation
    ├── TECHNICAL_DOCUMENTATION.md
    └── ER_DIAGRAM.md           # Entity relationship diagram
```

---

## 🔮 Future Improvements

- Integration with real-time POS/ERP data feeds
- Cloud-distributed Monte Carlo simulation for higher iteration counts
- Incorporation of probabilistic deep learning forecasting models
- Multi-store and multi-SKU scaling capabilities
