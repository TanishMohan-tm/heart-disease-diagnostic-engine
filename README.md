# 🩺 Heart Disease Diagnostic Engine
**Figure 01 · Machine Learning · Healthcare**

A comparative study and client-side deployment of a cardiovascular risk stratification system trained on the UCI Cleveland Heart Disease Dataset.

> ⚠️ **Disclaimer:** For educational and research use only. Trained on 1988 data. Not FDA-cleared or clinically validated. Must not replace evaluation by a qualified medical professional.

---

## 📌 Overview

This project benchmarks three classification algorithms on a clinical dataset, selects the optimal model based on interpretability and calibration requirements in healthcare, then deploys it as a **serverless, client-side inference engine** with zero infrastructure cost.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | UCI Cleveland Heart Disease Dataset |
| Samples | 304 |
| Features | 11 clinical parameters |
| Target | Binary — presence vs. absence of heart disease |
| Missing Values | 6 (imputed via median) |

**Preprocessing pipeline:**
- Median imputation for missing values
- Z-score normalization (standardization) — improved LR accuracy by ~6pp
- One-hot encoding for categorical variables
- Stratified 80/20 train-test split + 5-fold cross-validation

---

## 🧠 Model Comparison

| Algorithm | Test Accuracy | ROC-AUC | F1-Score | Notes |
|---|---|---|---|---|
| **Random Forest** | 90% | 0.96 | 0.90 | Highest raw accuracy; less interpretable |
| **Logistic Regression** | 87% | 0.95 | 0.86 | ✅ **Deployed** |
| Decision Tree | 80% | 0.83 | 0.79 | Overfits — 100% train vs 71% test (unpruned) |

**Deployment rationale:** Logistic Regression was selected over Random Forest despite a 3% accuracy gap. In a clinical context, coefficient-level explainability and well-calibrated probability outputs are non-negotiable. Random Forest operates as a black box — unacceptable when decisions affect patient outcomes.

---

## ⚙️ Clinical Features

| # | Feature | Role | Coefficient (β) |
|---|---|---|---|
| 1 | Age | Input | — |
| 2 | Biological Sex | Risk factor | +0.626 |
| 3 | Chest Pain Type | Risk factor | +0.586 |
| 4 | Resting BP (mmHg) | Input | — |
| 5 | Cholesterol (mg/dL) | Input | — |
| 6 | Fasting Blood Sugar >120 mg/dL | Input | — |
| 7 | Resting ECG | Input | — |
| 8 | Max Heart Rate Achieved | Protective factor | −0.336 |
| 9 | Exercise-Induced Angina | Input | — |
| 10 | ST Depression (mm) | Input | — |
| 11 | Peak Exercise ST Slope | Input | — |

---

## 🚀 Deployment Architecture

```
Training (Python / Scikit-learn)
        │
        ▼
  Model Weights Serialized
  to JavaScript Object
        │
        ▼
  Client-Side Inference (JS)
  ┌─────────────────────────┐
  │  Inference time: <1ms   │
  │  Server cost:    $0     │
  │  Offline-capable: ✅    │
  └─────────────────────────┘
```

Trained weights and intercept are exported from Python and embedded directly in the browser — no backend, no API, no latency.

---

## 🗂️ Repository Structure

```
heart-disease-diagnostic-engine/
│
├── data/
│   └── heart_disease_uci.csv
│
├── notebooks/
│   └── 01.ipynb
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🛠️ Stack

`Python` · `Scikit-learn` · `pandas` · `NumPy` · `JavaScript`

---

## 💡 Key Learnings

1. **Feature quality beats model complexity** — on small datasets (~300 samples), normalization and encoding matter more than algorithm depth.
2. **Interpretability is a hard constraint in healthcare** — a 3% accuracy gain is not worth losing coefficient-level attribution.
3. **Bias-variance trade-off is textbook here** — unconstrained Decision Trees scored 100% training accuracy and 71% test; regularization is mandatory.

---

## 🏃 Quickstart

```bash
git clone https://github.com/<your-username>/heart-disease-diagnostic-engine.git
cd heart-disease-diagnostic-engine
pip install -r requirements.txt
jupyter notebook notebooks/01_heart_disease.ipynb
```

---

## 📄 License

MIT License — see [`LICENSE`](LICENSE) for details.
