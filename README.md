# ⚡ Energy Price Forecaster

**Empowering smart energy decisions with advanced machine learning. Forecast European Day-Ahead electricity prices, mitigate market risks, and optimize operational schedules with precise pricing models tailored for both industrial facilities and residential consumers.**

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Regression-orange)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue)
![GCP](https://img.shields.io/badge/GCP-Cloud%20Run%20%7C%20GCS-4285F4)
![SHAP](https://img.shields.io/badge/XAI-SHAP-red)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green)

---

## 📖 Overview

European energy markets are highly volatile. To maintain profitability, stakeholders need to know exactly when electricity prices will peak and when they will drop. 

This project is a fully automated, end-to-end Machine Learning Operations (MLOps) pipeline. It provides highly accurate **Day-Ahead electricity price forecasts** combined with an advanced scheduling algorithm to mathematically identify the cheapest continuous operational windows for heavy machinery.

### 🌟 Key Features

1. **Dual-Sector Regional Pricing:** Calculates exact final prices by layering wholesale predictions with German statutory taxes (Stromsteuer, KWKG-Umlage, Offshore-Umlage, StromNEV-Umlage) and regional grid fees for Nuremberg, Munich, Berlin, and Hamburg.
2. **Advanced Machine Learning:** Utilizes LightGBM regression to track hourly wholesale trends and predict upcoming energy costs with high accuracy.
3. **Explainable AI (XAI):** Integrates SHAP (SHapley Additive exPlanations) Waterfall and Summary plots, uploading them directly to Google Cloud Storage to explain the macro market drivers behind the AI's latest forecast.
4. **Shift-Constrained Optimizer:** A highly efficient scheduling algorithm that instantly finds the cheapest continuous or interruptible block of hours for any given machine (kW) while adhering to strict operational shift times.
5. **Market Risk Backtesting:** A rigorous 90-day walk-forward backtest system that evaluates model error (MAE) and visualizes the financial variance between predictions and actuals over a rolling time window.
6. **Serverless Cloud Dashboard:** A high-performance Streamlit interface deployed via Docker on Google Cloud Run, allowing public, unauthenticated access to live market intelligence.

---

## 🏗️ Architecture

```text
energy-price-forecaster/
├── .github/workflows/
│   └── pipeline.yml          # Automated daily cron job for ETL & ML Ops
├── src/
│   ├── etl.py                # Live market data ingestion & database upserts
│   ├── db_utils.py           # Database connection & schema management
│   ├── features.py           # Feature engineering (Lags, weather, cyclical time)
│   ├── train.py              # LightGBM model training & SHAP generation
│   ├── backtest.py           # Walk-forward backtesting & Cloud Storage uploads
│   ├── inference.py          # Generates tomorrow's price forecast
│   ├── optimizer.py          # Continuous & interruptible scheduling math
│   ├── roi_calculator.py     # Calculates Return on Investment metrics
│   ├── cost_calculator.py    # Computes operational costs
│   └── compare_models.py     # Script to evaluate and compare multiple models
├── visualizations/           # Local temporary directory for generated SHAP/Backtest plots
├── Dockerfile                # Container blueprint for Google Cloud Run deployment
├── requirements.txt          # Python dependencies
└── streamlit_app.py          # Enterprise UI dashboard (Cloud Run entrypoint)
```

---

## 🧠 The Math & Optimization

### The Heavy Machinery Optimizer
Given an array of 24 predicted hourly prices and a required machine run-time duration, the custom optimizer calculates the cheapest route for operations. It uses a rolling-window summation algorithm to find the optimal contiguous block of time within a strictly constrained legal shift mask, dynamically calculating the final financial estimates against exact industrial grid rates.

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/miteshkumar07/The-Heavy-Machinery-Load-Shifter.git
cd The-Heavy-Machinery-Load-Shifter
conda create -n pro python=3.12
conda activate pro
pip install -r requirements.txt
```

### 2. Cloud & Database Setup
This project requires a PostgreSQL database and a Google Cloud Storage bucket. Set your connection variables in your terminal or a `.env` file before running the pipeline:
```bash
export DB_HOST="your_database_ip"
export DB_PASSWORD="your_database_password"
export GCP_CREDENTIALS='{"type": "service_account", ...}'
```

### 3. Execution (The Pipeline)
**Run the Data Pipeline & Update DB:**
```bash
python src/etl.py
```
**Train Models & Generate SHAP (Uploads to GCS):**
```bash
python src/train.py
```
**Run Walk-Forward Backtest (Uploads to GCS):**
```bash
python src/backtest.py
```
**Generate Tomorrow's Forecast:**
```bash
python src/inference.py
```
**Launch the Dashboard Locally:**
```bash
streamlit run streamlit_app.py
```

---

## ☁️ Cloud Deployment

This application is engineered for automated cloud deployment:
* **Backend:** GitHub Actions triggers `.github/workflows/pipeline.yml` daily to execute the `src/` scripts, pushing new predictions to PostgreSQL and fresh visualization binaries to Google Cloud Storage.
* **Frontend:** Deployed via Google Cloud Run. The service automatically scales, pulling live numerical data directly from the external database and fetching visual assets via the `google-cloud-storage` client for a seamless user experience.




## 📜 Data Attribution

This project relies on high-quality open-source and publicly available datasets to power its machine learning forecasts. We gratefully acknowledge the following data providers:

* **Meteorological Data:** Weather features (including wind speed, solar radiation, temperature, and HDD/CDD) are provided by the [Open-Meteo API](https://open-meteo.com/), which aggregates open-data weather forecasts from national weather services. Sourced under the **CC-BY 4.0** license.
* **Energy Market Data:** European Day-Ahead wholesale electricity prices and regional market data are sourced from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/). Sourced under the **CC-BY 4.0** license.
* **Macroeconomic Proxies:** Commodity proxy prices (Natural Gas, Carbon, Oil) are fetched via [Yahoo Finance](https://finance.yahoo.com/) for non-commercial research, providing the AI with critical global economic context.