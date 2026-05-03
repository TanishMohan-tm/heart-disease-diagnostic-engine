# 🩺 Heart Disease Diagnostic Engine
**Figure 01 · Machine Learning · Healthcare**

A comparative study of cardiovascular risk classification trained on the UCI Cleveland Heart Disease Dataset.

> ⚠️ **Disclaimer:** For educational and research use only. Trained on 1988 data. Not FDA-cleared or clinically validated. Must not replace evaluation by a qualified medical professional.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | UCI Cleveland Heart Disease Dataset |
| Samples | 304 |
| Features | 11 clinical parameters |
| Target | Binary — presence vs. absence of heart disease |
| Missing Values | 1 (slope — imputed via mode) |

**Preprocessing pipeline:**
- `ca` and `thal` dropped due to high missingness
- Ordinal encoding for categorical variables (cp, restecg, slope)
- Binary encoding for sex, fbs, exang
- Z-score normalization (StandardScaler) — improved Logistic Regression accuracy by ~6pp
- Stratified 80/20 train-test split + 5-fold cross-validation

---

## 🧠 Model Comparison

| Algorithm | Test Accuracy | ROC-AUC | F1-Score | Notes |
|---|---|---|---|---|
| **Random Forest** | 90.16% | 0.9535 | 0.8966 | Highest raw accuracy; less interpretable |
| **Logistic Regression** | 86.89% | 0.9502 | 0.8621 | ✅ **Selected** |
| Decision Tree | 80.33% | 0.8317 | 0.7931 | Overfits — 100% train vs 71% test (unpruned) |

**Why Logistic Regression over Random Forest:** In a clinical context, coefficient-level explainability and well-calibrated probability outputs are non-negotiable. A 3% accuracy gain does not justify a black box when decisions affect patient outcomes. Hyperparameter tuning via GridSearchCV selected C=0.01 as the optimal regularization strength.

---

## ⚙️ Clinical Features

| # | Feature | Role | Coefficient (β) |
|---|---|---|---|
| 1 | Age | Input | +0.1770 |
| 2 | Biological Sex | Risk factor | +0.6257 |
| 3 | Chest Pain Type | Risk factor | +0.5860 |
| 4 | Resting BP (mmHg) | Input | +0.1871 |
| 5 | Cholesterol (mg/dL) | Input | +0.1418 |
| 6 | Fasting Blood Sugar >120 mg/dL | Input | −0.0429 |
| 7 | Resting ECG | Input | +0.1921 |
| 8 | Max Heart Rate Achieved | Protective factor | −0.3364 |
| 9 | Exercise-Induced Angina | Input | +0.3381 |
| 10 | ST Depression (mm) | Input | +0.3934 |
| 11 | Peak Exercise ST Slope | Input | +0.1894 |

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
├── assets/
│   ├── eda_plots.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── feature_importance.png
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🛠️ Stack

`Python` · `Scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

---

## 🏃 Quickstart

```bash
git clone https://github.com/TanishMohan-tm/heart-disease-diagnostic-engine.git
cd heart-disease-diagnostic-engine
pip install -r requirements.txt
jupyter notebook notebooks/01.ipynb
```

Running the notebook end-to-end will generate the 4 evaluation plots saved to the project root.

---

## 💡 Key Learnings

1. **Feature quality beats model complexity** — on 304 samples, normalization and encoding matter more than algorithm depth.
2. **Interpretability is a hard constraint in healthcare** — a 3% accuracy gain is not worth losing coefficient-level attribution.
3. **Bias-variance trade-off is textbook here** — unconstrained Decision Trees scored 100% training accuracy and 71% test; regularization is mandatory.

---

## 📄 License

MIT License — see [`LICENSE`](LICENSE) for details.
