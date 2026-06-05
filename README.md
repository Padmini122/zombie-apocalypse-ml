# 🧟 Zombie Apocalypse Supply Chain Optimizer

🚀 Live Demo: [Launch Interactive Dashboard](https://zombie-apocalypse-ml-mpge6vurp5yzlhh6kwctpx.streamlit.app/)
Built as an End-to-End Machine Learning & Simulation Project 

## 🎯 The Problem

In a large-scale zombie outbreak, evacuation routes can become inaccessible within hours due to infection spread, infrastructure failure, and resource shortages. Traditional evacuation planning assumes static conditions and cannot adapt to rapidly changing outbreak dynamics.

This system simulates a zombie apocalypse across interconnected cities and uses machine learning to predict route failures, identify high-risk zones, and support data-driven evacuation decisions before collapse occurs.

---

## 🌍 Simulation Environment

| Property          | Details                                                      |
| ----------------- | ------------------------------------------------------------ |
| Cities Simulated  | Multiple interconnected urban zones                          |
| Network Model     | Graph-based transportation network                           |
| Outbreak Duration | Multi-day progression simulation                             |
| Risk Factors      | Infection rate, population density, elevation, defense level |
| Objective         | Predict evacuation route survivability                       |

---

## 🏗️ What I Built

### Phase 1 — Zombie Outbreak Simulation

* Generated a graph-based city network using NetworkX.
* Simulated infection spread across connected regions.
* Modeled daily outbreak progression.
* Tracked city-level infection growth and survivability.

### Phase 2 — Feature Engineering

Created predictive features including:

* Infection probability
* Neighbor infection pressure
* Population density
* Defensive infrastructure score
* Elevation factor
* Distance from outbreak source

Generated machine-learning-ready datasets from simulation outputs.

### Phase 3 — Machine Learning Models

| Model                   | Algorithm         | Purpose                         |
| ----------------------- | ----------------- | ------------------------------- |
| Route Failure Predictor | Linear Regression | Baseline prediction             |
| Risk Classification     | Decision Tree     | Explainable predictions         |
| Survival Forecasting    | Random Forest     | High-performance ensemble model |

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Phase 4 — Risk Intelligence Engine

* Predicts evacuation route collapse probability.
* Detects critical outbreak hotspots.
* Ranks cities by survival likelihood.
* Identifies safest evacuation paths.

### Phase 5 — Interactive Streamlit Dashboard

#### 📈 Overview

Simulation statistics, outbreak progression, and city risk summaries.

#### 🧠 Model Performance

Model comparison, ROC curves, confusion matrices, and prediction metrics.

#### 🌍 Outbreak Visualizations

Network spread maps and infection progression analytics.

#### ⚠️ Risk Monitoring

Early-warning indicators for route failures and high-risk regions.

---

## 📊 Visual Analytics

The project automatically generates:

* Outbreak Progression Maps
* Confusion Matrices
* ROC Curves
* Feature Importance Rankings

---

## 🔍 Key Findings

* Infection probability is the strongest predictor of evacuation failure.
* Population density significantly accelerates outbreak spread.
* Elevated regions exhibit longer survival durations.
* Defensive infrastructure substantially reduces collapse risk.
* Random Forest consistently achieves the strongest predictive performance.

---

## 🛠️ Tech Stack

| Category         | Tools               |
| ---------------- | ------------------- |
| Programming      | Python              |
| Data Processing  | Pandas, NumPy       |
| Machine Learning | Scikit-learn        |
| Simulation       | NetworkX            |
| Visualization    | Matplotlib, Seaborn |
| Dashboard        | Streamlit           |
| Version Control  | Git, GitHub         |

---

## 🚀 Run Locally

```bash
# Clone repository
git clone https://github.com/Padmini122/zombie-apocalypse-ml.git

# Enter project folder
cd zombie-apocalypse-ml

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py
```

---

## 📜 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

Padmini Kovvuri

GitHub: https://github.com/Padmini122

LinkedIn: https://www.linkedin.com/in/padmini-kovvuri-8063a72b5

---

> "In a zombie apocalypse, data-driven decisions can be the difference between survival and extinction."
