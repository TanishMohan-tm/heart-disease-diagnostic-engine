# 🩺 Heart Disease Diagnostic Engine

*Machine Learning · Explainable AI · Healthcare Analytics*

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-Open-2ea44f)](https://tanishmohan-tm.github.io/heart-disease-diagnostic-engine/dashboard/)

An explainable cardiovascular risk classification system trained on the UCI Cleveland Heart Disease dataset.  
The project combines interpretable machine learning with a fully client-side diagnostic dashboard for real-time risk exploration.

> ⚠️ **Disclaimer**  
> This project is intended strictly for educational and research purposes.  
> It is trained on the 1988 Cleveland dataset (`n=304`) and is **not clinically validated**, FDA-approved, or suitable for medical diagnosis.

---

# 📌 Overview

This project investigates the trade-off between predictive performance and clinical interpretability in healthcare ML systems.

Three classification models were evaluated:

- Logistic Regression
- Random Forest
- Decision Tree

Although Random Forest achieved the highest raw accuracy, Logistic Regression was ultimately selected due to:

- coefficient-level explainability
- calibrated probability outputs
- transparent feature attribution
- suitability for clinical reasoning

The final system includes:

- end-to-end preprocessing pipeline
- comparative model evaluation
- explainability-focused inference
- interactive browser dashboard
- zero-backend deployment architecture

---

# 🧠 System Architecture

## High-Level Architecture

```text
                           ┌────────────────────┐
                           │  UCI Dataset       │
                           │  Cleveland (CSV)   │
                           └─────────┬──────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │ Data Processing Pipeline       │
                    │ • Cleaning                     │
                    │ • Missing-value handling       │
                    │ • Encoding                     │
                    │ • Feature scaling              │
                    └──────────────┬─────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────┐
                    │ Feature Matrix Construction    │
                    │ Stratified Train/Test Split    │
                    └──────────────┬─────────────────┘
                                   │
                 ┌─────────────────┼──────────────────┐
                 ▼                 ▼                  ▼
        ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
        │ Logistic       │ │ Random Forest  │ │ Decision Tree  │
        │ Regression     │ │ Classifier     │ │ Classifier     │
        └────────┬───────┘ └────────┬───────┘ └────────┬───────┘
                 │                  │                  │
                 └──────────────────┼──────────────────┘
                                    ▼
                    ┌────────────────────────────────┐
                    │ Evaluation Layer               │
                    │ • Accuracy                     │
                    │ • F1 Score                     │
                    │ • ROC-AUC                      │
                    │ • Confusion Matrix             │
                    └──────────────┬─────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────┐
                    │ Explainability Layer           │
                    │ • Logistic coefficients        │
                    │ • Feature attribution          │
                    │ • Risk interpretation          │
                    └──────────────┬─────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────┐
                    │ Frontend Dashboard             │
                    │ Static HTML + Vanilla JS       │
                    │ Browser-side inference         │
                    └────────────────────────────────┘
```

---

# ⚙️ ML Pipeline

## 1. Data Preprocessing

### Cleaning & Feature Engineering

| Step | Description |
|---|---|
| Missing-value handling | `slope` imputed using mode |
| Feature removal | `ca` and `thal` dropped due to high missingness |
| Encoding | Ordinal + binary encoding |
| Scaling | `StandardScaler` z-score normalization |
| Splitting | Stratified 80/20 train-test split |

---

## 2. Model Training

Each model was trained and evaluated using:

- 5-fold cross-validation
- ROC-AUC analysis
- F1-score comparison
- confusion matrix evaluation
- hyperparameter tuning via `GridSearchCV`

---

## 3. Inference Design

The deployed dashboard performs inference entirely in-browser.

### Deployment Strategy

Instead of serving predictions through a backend API:

- trained coefficients
- normalization statistics
- thresholds
- feature metadata

are embedded directly into JavaScript.

This creates a:

- zero-infrastructure deployment
- privacy-preserving inference system
- instant prediction pipeline
- GitHub Pages compatible architecture

---

# 📊 Model Performance

| Model | Accuracy | ROC-AUC | F1 Score | Notes |
|---|---|---|---|---|
| Random Forest | 90.16% | 0.9535 | 0.8966 | Highest raw accuracy |
| Logistic Regression | 86.89% | 0.9502 | 0.8621 | ✅ Selected |
| Decision Tree | 80.33% | 0.8317 | 0.7931 | Severe overfitting |

---

# 🧬 Why Logistic Regression Was Selected

Despite slightly lower accuracy than Random Forest, Logistic Regression was chosen because healthcare systems require:

- interpretability
- transparent reasoning
- calibrated probabilities
- feature-level attribution
- reproducibility

A clinically deployable system must allow practitioners to understand *why* a prediction was generated.

---

# 🔍 Clinical Features Used

| Feature | Role |
|---|---|
| Age | Risk-associated |
| Sex | Risk-associated |
| Chest Pain Type | Risk-associated |
| Resting Blood Pressure | Risk-associated |
| Cholesterol | Risk-associated |
| Fasting Blood Sugar | Weak association |
| Resting ECG | Risk-associated |
| Max Heart Rate | Protective |
| Exercise-induced Angina | Risk-associated |
| ST Depression | Risk-associated |
| ST Slope | Risk-associated |

---

# 🖥️ Dashboard Features

The interactive dashboard includes:

- real-time probability inference
- calibrated risk gauge
- top contributing feature analysis
- clinician-oriented explanations
- patient-friendly explanations
- browser-only execution
- responsive UI

### Dashboard Architecture

```text
User Input
    │
    ▼
Normalization Layer
    │
    ▼
Logistic Regression Inference
    │
    ▼
Sigmoid Probability Output
    │
    ├── Risk Classification
    ├── Feature Attribution
    └── Explanation Engine
```

---

# 📁 Repository Structure

```text
heart-disease-diagnostic-engine/
│
├── dashboard/
│   └── index.html
│
├── data/
│   └── heart_disease_uci.csv
│
├── notebooks/
│   └── 01.ipynb
│
├── assets/
│   ├── eda_plots.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── feature_importance.png
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 📈 Generated Outputs

| File | Purpose |
|---|---|
| `eda_plots.png` | Exploratory data analysis |
| `confusion_matrices.png` | Classification performance |
| `roc_curves.png` | ROC-AUC comparison |
| `feature_importance.png` | Coefficient analysis |

---

# 🛠️ Tech Stack

## Machine Learning

- Python
- scikit-learn
- pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn

## Frontend

- HTML5
- Vanilla JavaScript
- SVG

## Deployment

- GitHub Pages
- Static hosting

---

# 🚀 Quick Start

## Clone Repository

```bash
git clone https://github.com/TanishMohan-tm/heart-disease-diagnostic-engine.git
cd heart-disease-diagnostic-engine
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Notebook

```bash
jupyter notebook notebooks/01.ipynb
```

---

## Launch Dashboard

### macOS

```bash
open dashboard/index.html
```

### Linux

```bash
xdg-open dashboard/index.html
```

### Windows

Open `dashboard/index.html` in your browser.

---

# 💡 Key Engineering Learnings

### Interpretability is a system requirement

Healthcare models cannot rely solely on predictive accuracy.

---

### Simpler models scale better in low-data environments

With only 304 samples, regularization and preprocessing mattered more than model complexity.

---

### Frontend-only ML deployment is viable

Lightweight inference systems can run entirely inside the browser without cloud infrastructure.

---

### Explainability improves trust

Feature attribution significantly improves usability for both clinicians and non-technical users.

---

# 📚 References

### Dataset

UCI Machine Learning Repository — Heart Disease Dataset  
https://archive.ics.uci.edu/ml/datasets/heart+disease

---

# 📄 License

MIT License — see `LICENSE` for details.