
# 🧟 Zombie Apocalypse Supply Chain Optimizer

## 📌 Project Overview

The Zombie Apocalypse Supply Chain Optimizer is an end-to-end Machine Learning project that simulates a zombie outbreak across interconnected cities and predicts evacuation route failures using predictive analytics.

The system combines simulation modeling, feature engineering, machine learning, and interactive visualization to help identify the safest evacuation strategies during a hypothetical apocalypse.

---

## 🎯 Objectives

- Simulate zombie outbreak progression across multiple cities.
- Generate realistic outbreak datasets.
- Engineer meaningful predictive features.
- Train and evaluate multiple machine learning models.
- Visualize outbreak spread and model performance.
- Provide an interactive dashboard using Streamlit.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- NetworkX
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

## 📂 Project Structure

```text
zombie-apocalypse-ml/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── simulator.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── visualization.py
│
├── data/
│
├── models/
│   ├── linear_regression.pkl
│   ├── decision_tree.pkl
│   └── random_forest.pkl
│
├── visualizations/
│   ├── outbreak_progression.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── feature_importance.png
│
└── docs/
    ├── methodology.md
    └── results_summary.md
```

---

## 🤖 Machine Learning Models

The project trains and evaluates:

| Model | Purpose |
|---------|---------|
| Linear Regression | Baseline predictor |
| Decision Tree | Interpretable classification |
| Random Forest | High-performance ensemble model |

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## 📊 Visualizations

The project automatically generates:

### Outbreak Progression
Tracks zombie infection spread across cities over time.

### Confusion Matrices
Compares prediction performance of all models.

### ROC Curves
Displays classifier effectiveness using AUC scores.

### Feature Importance
Highlights the most influential factors affecting evacuation route failures.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Padmini122/zombie-apocalypse-ml.git
cd zombie-apocalypse-ml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Command Line Version

```bash
python main.py
```

### Streamlit Dashboard

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 📈 Key Features

- Zombie outbreak simulation
- Supply chain disruption analysis
- Feature engineering pipeline
- Multiple ML model comparison
- Interactive dashboard
- Automated visualization generation
- Model persistence using Joblib

---

## 🔍 Key Insights

- Infection probability is the strongest predictor of route failure.
- Population density accelerates outbreak spread.
- Elevated regions survive longer.
- Defensive infrastructure significantly improves survival rates.
- Random Forest consistently outperforms other models.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Padmini Kovvuri**

GitHub: https://github.com/Padmini122

LinkedIn:https://www.linkedin.com/in/padmini-kovvuri-8063a72b5

---

> "In a zombie apocalypse, data-driven decisions can be the difference between survival and extinction."
=======
# zombie-apocalypse-ml
ML-powered zombie outbreak survival route predictor  Linear Regression, Decision Tree, Random Forest
<<<<<<< HEAD
=======

>>>>>>> dd2b138 (Fix Streamlit Cloud dependencies)
