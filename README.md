# Project Global-Grid - Predictive Maintenance of a Multi-National Wind Farm Fleet
https://multinational-wind-farm-fleet-predictive-maintenance-cey7btpcm.streamlit.app/

Why This Project Exists
Wind energy is the fastest-growing renewable energy source in the world. According to the Global Wind Energy Council, installed wind capacity has grown over 900% in the last two decades — and that growth is accelerating. By 2030, wind is projected to supply over 20% of global electricity demand.
But turbines fail. Generator overheating, rotor anomalies, and mechanical degradation cost the industry billions annually in unplanned downtime and emergency repairs. A single unscheduled maintenance event on an offshore turbine can cost upward of $200,000. Multiply that across a fleet of hundreds of turbines operating in remote or offshore environments, and the business case for predictive maintenance becomes undeniable.
Project Global-Grid was built to answer a practical question: can SCADA telemetry alone — without additional sensors or manual inspection — be used to predict turbine health risk in real time?
The answer is yes.

What Was Built
A full end-to-end predictive maintenance analytics pipeline, from raw SCADA ingestion to a live interactive monitoring dashboard — covering every stage a real data team would execute:
PhaseWhat Was DoneData AcquisitionAventa AV-7 SCADA dataset — 39M records at 1-second resolutionData CleaningDatetime parsing, deduplication, sensor error filtering, column standardisationSQL AnalyticsSQLite database with 6 operational queries (power trends, temperature spikes, idle/active ratio, efficiency)EDAPower curve analysis, wind speed distribution, temperature timelines, correlation heatmapFeature Engineering13 features including rolling statistics, lag features, rate-of-change, and cross-sensor interactionsPredictive Modelling4-class XGBoost risk classifier + LSTM 30-minute temperature forecasterDashboardPower BI operational dashboard (in progress)Interactive Demo3-tab Streamlit application — live risk monitor, LSTM forecast, degradation trends

The Dataset
Source: Aventa AV-7 IET OST SCADA Dataset
Turbine: Aventa AV-7 research-grade wind turbine
Resolution: 1 reading per second
Raw size: ~39 million rows, ~3GB
Operational period: December 2021 — July 2023 (19 months)
After resampling to 1-minute averages: ~712,000 operational records
Key telemetry channels:
SignalDescriptionWindSpeedFreestream wind velocity (m/s)RotorSpeedBlade rotation rate (rpm)GeneratorSpeedGenerator shaft speed (rpm)GeneratorTemperatureGenerator winding temperature (°C)PowerOutputActive power generation (kW)PitchDegBlade pitch angle (degrees)TurbineStatusOperational state code

Technical Architecture
Raw SCADA (39M rows, 1-sec)
        │
        ▼
  Data Cleaning & Resampling
  (1-min averages → 712K rows)
        │
        ▼
  SQL Analytics Layer (SQLite)
  Operational queries, KPI extraction
        │
        ▼
  Exploratory Data Analysis
  Power curves, distributions, correlation
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  Feature Engineering              LSTM Forecaster
  13 temporal features             60-min history → 30-min forecast
  rolling stats, lag, RoC          Temperature trajectory prediction
        │
        ▼
  Multi-Sensor Risk Labeling
  Anomaly scoring across 4 independent signals
  (Temperature + Power + Efficiency + Rate-of-Change)
        │
        ▼
  XGBoost Classifier
  4-class: Normal / Low / Medium / High Risk
  SMOTE oversampling on minority classes
        │
        ▼
  Streamlit Dashboard
  Tab 1: Real-time risk monitor
  Tab 2: LSTM temperature forecast
  Tab 3: Long-term degradation trends

The Modelling Approach
Why Multi-Sensor Scoring?
Early iterations of this project used temperature thresholds alone to generate risk labels. While this produced high weighted accuracy, it meant the model only learned to watch temperature — the other sensors contributed nothing. The labeling strategy was redesigned around independent anomaly scoring:
SignalRisk ContributionGenerator temperature level0–2 pointsTemperature rate-of-change0–2 pointsPower underperformance under high wind0–2 pointsEfficiency collapse (power/wind ratio)0–2 points
A turbine accumulates a score across all four signals. The total score maps to a risk class:
ScoreLabel0Normal1–2Low Risk3–4Medium Risk5+High Risk
This means any sensor can raise risk independently — and the model genuinely learned all channels.
Results
Macro F1 Score: 0.97

              precision    recall  f1-score   support
      Normal       1.00      1.00      1.00     16000
    Low Risk       0.99      0.99      0.99     16000
 Medium Risk       0.98      0.98      0.98     11455
   High Risk       0.83      0.98      0.90       607

    accuracy                           0.99     44062
   weighted avg    0.99      0.99      0.99     44062
High Risk F1 of 0.90 on 607 real test samples — not synthetic. The model correctly identifies critical conditions with high recall, minimising the missed detections that matter most in a maintenance context.
Why High Risk Is Rare (And That's Correct)
High Risk events constitute ~1% of all operational minutes. This is intentional and realistic. A model that flags High Risk constantly would be ignored by operators within days. The system is calibrated to alert on genuine anomalies — the 1% of minutes where multiple signals simultaneously indicate a developing fault.
LSTM Temperature Forecaster
A lightweight LSTM (64 → 32 units) was trained on the full 712K minute sequence to predict generator temperature 30 minutes ahead from the preceding 60-minute history. The model was trained on Kaggle with early stopping at 30 epochs. Forecasts for four representative operational scenarios (normal, warming, high-temperature sustained, cooling) are pre-computed and embedded in the deployment artifacts — removing the TensorFlow dependency from the Streamlit app entirely.

SQL Analytics Layer
Before any machine learning, the cleaned dataset was loaded into a SQLite database and interrogated with 6 operational queries that a turbine fleet analyst would run daily:

Daily power generation — total kWh output ranked by date
Average operating conditions — fleet-wide wind, temperature, power baselines
Temperature spike detection — all minutes above 70°C with timestamps
Idle vs active time — operational availability ratio
Power-to-wind efficiency — top 1,000 most efficient operating minutes
Hourly power trends — diurnal generation patterns by hour of day

This SQL layer simulates the analytics pipeline that would feed a real SCADA monitoring platform or BI tool.

Exploratory Data Analysis
Key findings from the EDA phase:

Power curve confirms correct turbine operation — power output follows the characteristic S-curve against wind speed, saturating above ~12 m/s
36.8% of minutes have zero rotor speed — the turbine is frequently idle or in standby, which heavily influences threshold calibration
Generator temperature ranges from -10°C to 83°C — with the 99th percentile at 68.3°C, used as the High Risk threshold
Efficiency (power/wind) shows high variance — some minutes produce over 40 kW per m/s, others near zero under identical wind conditions, indicating mechanical variability
Strong correlation between wind speed and power output (r = 0.72) and between rotor speed and generator speed (r = 0.94), as expected from turbine physics


Repository Structure
project-global-grid/
│
├── app.py                          # Streamlit interactive demo (Phase 10)
├── requirements.txt                # Python dependencies
│
├── models/
│   ├── polaris_risk_model.pkl      # Trained XGBoost classifier
│   ├── polaris_label_encoder.pkl   # Class label encoder
│   ├── polaris_features.json       # Feature list (inference order)
│   └── degradation_stats.json     # Weekly trends + LSTM forecasts
│
├── notebooks/
│   ├── project_polaris_data_analysis.ipynb   # EDA + SQL analytics (Phase 3–7)
│   └── polaris_phase8_predictive_maintenance.ipynb  # ML pipeline (Phase 8)
│
└── .streamlit/
    └── config.toml                 # Forces light theme

Live Demo
Open the Streamlit App
The app has three tabs:
Tab 1 — Risk Monitor
Adjust live sensor sliders for wind speed, rotor speed, generator speed, pitch angle, power output, and generator temperature. The XGBoost classifier predicts the current risk class in real time with class probabilities and a maintenance recommendation.
Tab 2 — Temperature Forecast
Select one of four operational scenarios (normal, warming, sustained high, cooling). The LSTM forecast is displayed as an annotated Plotly chart showing the last 60 minutes of history and the predicted temperature trajectory over the next 30 minutes, with threshold lines at 58°C (Medium Risk) and 68°C (High Risk).
Tab 3 — Degradation Trends
Long-term weekly trends across 525 weeks of operational data. Three views: generator temperature drift, efficiency degradation, and risk event frequency. Each view auto-generates a trend analysis statement.

Tech Stack
LayerTechnologyData processingPython, Pandas, NumPySQL analyticsSQLite, SQLAlchemyVisualisation (EDA)Matplotlib, SeabornML — ClassifierXGBoost, scikit-learn, imbalanced-learn (SMOTE)ML — ForecasterTensorFlow / Keras LSTMDashboardPower BI (in progress)Interactive demoStreamlit, PlotlyDeploymentStreamlit Community Cloud

Scalability to a Real Fleet
This pipeline was demonstrated on the Aventa AV-7, a single research turbine. The architecture is designed to scale:

Feature engineering is physics-based — rolling temperature statistics, efficiency ratios, and rate-of-change are meaningful for any wind turbine, not just the AV-7
Per-turbine calibration — each turbine in a fleet generates its own degradation_stats.json from its historical SCADA data, establishing a turbine-specific baseline
Classifier generalisation — the multi-sensor anomaly scoring approach applies across turbine classes, with threshold percentiles recalculated from each turbine's operational history
Production extension — a REST API wrapping the inference pipeline would allow each turbine to stream its latest 60 readings and receive a risk score and 30-minute temperature forecast in response


Project Phases
This project followed the Project Polaris workflow — a structured 12-phase analytics pipeline:
PhaseDescriptionStatus1Problem definition & business caseComplete2Data acquisitionComplete3Data cleaning & preparationComplete4Data reduction (resampling)Complete5SQL database layerComplete6SQL analytics queriesComplete7Exploratory data analysisComplete8Predictive maintenance modellingComplete9Power BI dashboardIn progress10Interactive Streamlit demoComplete11DocumentationComplete12Portfolio integrationComplete

Author
Built as a portfolio project demonstrating end-to-end data analytics and machine learning engineering skills — from raw industrial sensor data to a deployed predictive maintenance system.


