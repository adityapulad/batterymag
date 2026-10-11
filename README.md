# ⚡ AI/ML Battery Intelligence Platform
### Physics-Informed State of Health (SoH), State of Charge (SoC), Remaining Useful Life (RUL), and Equivalent Circuit Parameter Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg?logo=react&logoColor=black)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-5.0+-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary

**Battery Intelligence** is a production-grade, end-to-end Battery Management System (BMS) software platform. It bridges data-driven deep learning with physical electrochemistry by integrating **Equivalent Circuit Modeling (Thevenin ECM, Rint, 2RC PNGV)** alongside **LSTM Recurrent Neural Networks** and **XGBoost Regressors**.

The platform is designed to overcome standard machine learning pitfalls in battery analytics—such as black-box overfitting, invalid extrapolations during telemetry dropouts, and optimistic train/test data leakage—by enforcing **battery-wise Leave-One-Battery-Out (LOBO) validation**, **physics surrogate fallbacks**, and **runtime input sanitization**.

---

## 🌟 Key Features

| Capability | Engine / Architecture | Purpose & Physics Integration |
| :--- | :--- | :--- |
| **State of Health (SoH)** | Hybrid LSTM + R0 Overpotential Tracker | Estimates cell degradation ($0 - 100\%$) referenced strictly to measured initial capacity $C_0$. |
| **State of Charge (SoC)** | Complementary Fusion: Coulomb Counting + XGBoost Observer | Combines high-frequency Coulomb integration with non-linear voltage observer to prevent integration drift. Supports bidirectional IEEE load sign conventions. |
| **Remaining Useful Life (RUL)** | Temporal Sequence LSTM Network | Predicts remaining discharge cycles until reaching $80\%$ End-of-Life (EOL) with confidence intervals. Corrects polarization rate distortion using physical time increments ($\Delta t$). |
| **Impedance Extraction (ECM)** | L-BFGS-B Optimization with Excitation Checks | Fits ohmic resistance ($R_0$), charge-transfer resistance ($R_1$), and polarization capacitance ($C_1$). Returns `R2 = null` on unexcited flatline rest. |
| **Diagnostics & Fault Detection** | Multi-Parameter Anomaly Engine | Flags thermal runaway risk, internal short-circuiting ($R_0$ surge), severe capacity fade, and sensor outliers ($15\text{V}$, negative timestamps). |
| **Auditing & Provenance** | SQLite Persistent Traceability | Stores exact raw neural predictions, physics fallback activation flags, fallback reasons, and model version hashes for regulatory compliance. |

---

## 🔬 Benchmark & Cross-Validation Results

The platform was evaluated using a rigorous **Leave-One-Battery-Out (LOBO)** protocol across all **636 discharge cycles** of the NASA PCoE battery repository (`B0005`, `B0006`, `B0007`, `B0018`). Models and feature scalers were fitted exclusively on three cells and tested on the remaining unseen cell.

### 1. Out-of-Sample SoH Generalization (LOBO)

| Test Battery (Held-Out) | Train Batteries | Cycles Evaluated | Out-of-Sample MAE | Out-of-Sample RMSE | $R^2$ Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`B0005`** | `B0006`, `B0007`, `B0018` | 168 | **2.04%** | **2.42%** | **0.9441** |
| **`B0006`** | `B0005`, `B0007`, `B0018` | 168 | **2.02%** | **2.52%** | **0.9582** |
| **`B0007`** | `B0005`, `B0006`, `B0018` | 168 | **2.51%** | **2.82%** | **0.8895** |
| **`B0018`** | `B0005`, `B0006`, `B0007` | 132 | **1.74%** | **2.57%** | **0.9042** |
| **Macro Average** | — | **636 Total** | **2.08%** | **2.58%** | **0.9240** |

*Detailed benchmark logs are exported to `backend/evaluation_results/leave_one_battery_out_results.csv`.*

### 2. RUL Benchmark Comparison (Unseen `B0005`, 168 Cycles)

| Model / Baseline | MAE (Cycles) | RMSE (Cycles) | Mean Bias (Cycles) |
| :--- | :---: | :---: | :---: |
| **LSTM RUL Network (This Project)** | **11.00** | **15.69** | **+4.41** |
| **Baseline 1: Linear Capacity Fade Extrapolation** | 35.73 | 42.18 | -18.25 |
| **Baseline 2: Mean Population Lifetime** | 28.99 | 36.42 | +12.60 |

---

## 🏗 System Architecture

```
                                      ┌──────────────────────────────────────────────┐
                                      │              React + Vite UI                 │
                                      │  (Dynamic Telemetry, Diagnostics, KPI Cards) │
                                      └──────────────────────┬───────────────────────┘
                                                             │ HTTP / JSON REST
                                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       FastAPI Backend Engine                                       │
│                                                                                                    │
│  ┌───────────────────────┐   ┌───────────────────────┐   ┌──────────────────────────────────────┐  │
│  │   Telemetry Guard     │   │   Inference Router    │   │         SQLite Audit Trail           │  │
│  │  - NaN / Inf checks   │──▶│  - /soh/predict       │──▶│  - raw_prediction                   │  │
│  │  - Time monotonicity  │   │  - /soc/predict       │   │  - prediction_source                 │  │
│  │  - Chemistry range    │   │  - /rul/predict       │   │  - fallback_reason                   │  │
│  └───────────────────────┘   └───────────┬───────────┘   └──────────────────────────────────────┘  │
│                                          │                                                         │
│                ┌─────────────────────────┴─────────────────────────┐                               │
│                ▼                                                   ▼                               │
│  ┌───────────────────────────┐                       ┌───────────────────────────┐                 │
│  │   Trained Machine Models  │                       │   Physics Surrogate Layer │                 │
│  │  - LSTM SoH Network       │   (On Anomaly /       │  - Ah Coulomb Counter     │                 │
│  │  - Hybrid RUL Network     │─── Low Excitation) ──▶│  - Thevenin RC Estimator  │                 │
│  │  - XGBoost SoC Observer   │                       │  - Resistance Degradation │                 │
│  └───────────────────────────┘                       └───────────────────────────┘                 │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Directory Structure

```
Srishtish/
├── backend/                           # FastAPI Application Core
│   ├── app/
│   │   ├── api/v1/endpoints/          # REST Endpoints: soh, soc, rul, ecm, fault, history
│   │   ├── core/                      # ECM Fitting, Coulomb Counting, Fault Detector
│   │   ├── db/                        # SQLite Database Models & Session Management
│   │   ├── ml/                        # Model Registry, Preprocessing, Fallback Logic
│   │   └── main.py                    # Application Entrypoint & CORS Configuration
│   ├── models/                        # Serialized .keras Models & .joblib Scalers
│   ├── evaluation_results/            # Cross-validation CSV Benchmark Metrics
│   ├── scripts/                       # Training, Self-Testing, and LOBO Benchmark Scripts
│   ├── tests/                         # Pytest Suite (Health, Inference, Provenance)
│   └── requirements.txt               # Backend Python Dependencies
│
├── frontend/                          # React + Vite Dashboard
│   ├── src/
│   │   ├── components/                # KPICards, TelemetryChart, ECMParameters, FaultPanel
│   │   ├── fixtures/                  # Real NASA Telemetry Profiles (Healthy, Moderate, Degraded)
│   │   ├── App.jsx                    # State Management & Pipeline Integration
│   │   └── api.js                     # API Client
│   ├── package.json                   # Frontend Dependencies
│   └── vite.config.js                 # Vite Dev Server & Proxy Config
│
├── datasets/                          # NASA PCoE Battery Telemetry (.mat and .csv files)
├── hybrid/                            # Offline Research & Exploratory Jupyter Notebooks
└── README.md                          # Platform Documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites
* **Python 3.10+** (Recommended: virtual environment with TensorFlow and FastAPI)
* **Node.js 18+** & **npm**

### 1. Backend Setup

```bash
cd backend

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI backend server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
* Interactive Swagger API Docs will be live at: **`http://localhost:8000/docs`**

### 2. Frontend Setup

In a new terminal window:

```bash
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```
* Open your browser and navigate to: **`http://localhost:5173`**

---

## 🧪 Running Self-Tests & Audits

### Run Full Test Suite
```bash
cd backend
PYTHONPATH=. pytest tests/test_api.py -v
```

### Run Live Prediction & Provenance Audit
Verify real responses, fallback integrity on edge cases ($15\text{V}$ outliers, $0\text{A}$ rest flatlines), and database persistence:
```bash
cd backend
PYTHONPATH=. python scripts/live_prediction_audit.py
```

### Run Leave-One-Battery-Out (LOBO) Benchmark
Train and test across all NASA battery combinations from scratch:
```bash
cd backend
PYTHONPATH=. python scripts/evaluate_battery_split.py
```

---

## ☁️ Deployment Guidelines (Railway / Cloud)

This application is ready for deployment on platforms like **Railway**, **Render**, or **Fly.io**.

### Recommendations:
* **Container Memory:** Provide at least **1 GB RAM** on the backend container to ensure TensorFlow/Keras initializes smoothly without hitting OOM limits.
* **Environment Variables:**
  * `PORT=8000`
  * `ENVIRONMENT=production`
  * `CORS_ORIGINS=*`
* **Vite Frontend API Target:**
  * Set `VITE_API_BASE_URL` in your frontend environment pointing to your deployed backend URL.

---

## 📜 Scientific References & Acknowledgments
* **NASA Prognostics Center of Excellence (PCoE):** Battery Aging Data Set (`B0005`, `B0006`, `B0007`, `B0018`), NASA Ames Research Center.
* **Equivalent Circuit Modeling:** Thevenin 1-RC and PNGV 2-RC Battery Equivalent Circuit Models for Li-ion cell dynamics.
* **Machine Learning:** Long Short-Term Memory (LSTM) Networks and Extreme Gradient Boosting (XGBoost) for State Estimation in Battery Management Systems.
