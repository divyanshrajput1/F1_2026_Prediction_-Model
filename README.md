![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange)
![Simulation](https://img.shields.io/badge/Simulation-Monte%20Carlo-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

# 🏎️ F1 2026 Race & Season Prediction Engine

A full end-to-end **machine learning + Monte-Carlo simulation system** for predicting Formula 1 race outcomes and season championships for the **2026 season**.

This project goes far beyond simple ML models — it integrates:
- historical race data
- probabilistic ML models
- calibration & explainability
- config-driven race logic
- Monte-Carlo simulations
- scenario analysis

Built entirely in **Python**, following **real ML engineering practices**.

---

## 🚀 What This Project Does

### 🔮 Per-Driver Predictions
For each race:
- DNF probability
- Top-10 probability
- Podium probability
- Expected championship points

### 🎲 Monte-Carlo Race Simulation
- 100,000+ simulations per race
- Grid position scenarios (good / baseline / bad)
- Chaos modeling (DNFs, randomness)
- Points distribution

### 🏆 Full Season Simulation
- Race-by-race simulation
- Driver championship standings
- Constructor championship standings
- Consistency & volatility tracking

### 📊 Model Reliability
- ROC-AUC evaluation
- Brier score
- Probability calibration curves
- SHAP explainability

## 🧪 Models Used

| Task | Model |
|----|------|
| DNF Prediction | XGBoost Classifier |
| Top-10 Finish | XGBoost Classifier |
| Podium Finish | XGBoost Classifier |

### 📈 Model Performance (Typical)
| Model | ROC-AUC | Brier Score |
|----|--------|-------------|
| DNF | ~0.56 | ~0.20 |
| Top-10 | ~0.85 | ~0.13 |
| Podium | ~0.87 | ~0.11 |

## 🧠 System Architecture

Raw Data (FastF1)
↓
Feature Engineering
↓
ML Models
(DNF / Top-10 / Podium)
↓
Probability Calibration
↓
Race Prediction
↓
Monte-Carlo Simulation
↓
Season Championship

## ⚙️ Installation

1️⃣ Clone repository
```bash
git clone https://github.com/divyanshrajput1/F1_2026_Prediction_ Model.git
cd F1_2026_Prediction_ Model

2️⃣ Create environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ How to Run

🔹 Build features
python -m features.build_features

🔹 Train models
python models/dnf_model.py
python models/top10_model.py
python models/podium_model.py

🔹 Apply models to a race
python simulation/apply_models_2026.py

🔹 Run Monte-Carlo race simulation
python -m simulation.monte_carlo_2026

🧩 Configuration-Driven Simulation

All race behavior is controlled via YAML:
simulation:
  n_simulations: 100,000

qualifying:
  grid_source: baseline
  grid_effect_weight: -0.03

chaos:
  dnf_multiplier: 1.2
  noise_std: 0.05

Change config → rerun → new race scenario.

📊 Outputs

Race results CSV
Season standings CSV
Probability calibration plots
SHAP explainability plots


👤 Author :-
Divyansh Rajput
Data Science / Machine Learning
Project built for advanced ML & simulation practice.

⚠️ Disclaimer
This project is for educational and analytical purposes only.
Not affiliated with Formula 1 or FIA.
