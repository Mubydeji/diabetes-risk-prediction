# Diabetes Risk Prediction: End-to-End ML Pipeline
### From Raw Clinical Data to a Deployed Prediction Model

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Overview

This project builds a complete machine learning pipeline to predict diabetes risk in patients based on clinical measurements. It simulates a real-world clinical decision support tool — flagging high-risk patients for priority follow-up before diabetes develops.

The pipeline covers every stage: data cleaning, feature engineering, model training and comparison, evaluation, threshold optimisation, and a structured clinical recommendation.

---

## Clinical Question

> *Given a patient's clinical measurements, can we accurately predict whether they are at risk of diabetes — and which factors drive that risk?*

---

## Dataset

**Pima Indians Diabetes Dataset**
- Source: UCI Machine Learning Repository via Kaggle
- 768 female patients of Pima Indian heritage, aged 21+
- 9 features including Glucose, BMI, Insulin, Blood Pressure, and Diabetes Outcome
- Target variable: Outcome (1 = diabetic, 0 = non-diabetic)

---

## Project Structure
diabetes-risk-prediction/
│
├── notebooks/
│   └── diabetes_risk_prediction.ipynb
├── README.md
└── requirements.txt

---

## Methodology

### 1. Data Cleaning
- Identified and replaced biologically impossible zero values in Glucose, BMI, Blood Pressure, Skin Thickness, and Insulin with column medians

### 2. Exploratory Data Analysis
- Correlation analysis identified Glucose (0.49), BMI (0.31), and Age (0.24) as the strongest predictors
- Boxplots and distribution plots confirmed clear separation between diabetic and non-diabetic groups across key features

### 3. Feature Engineering
Four new features created from domain knowledge:
- **Glucose_BMI**: Interaction between the two strongest predictors
- **Age_Pregnancies**: Combined age and pregnancy risk factor
- **Insulin_Glucose_Ratio**: Proxy for insulin resistance
- **Risk_Score**: Weighted composite of Glucose, BMI, and Age

### 4. Model Training
Four models trained and compared using 5-fold stratified cross-validation:

| Model | ROC-AUC | Accuracy | F1 Score |
|---|---|---|---|
| Logistic Regression | **0.8463** | **0.7882** | **0.6588** |
| Support Vector Machine | 0.8275 | 0.7638 | 0.6131 |
| Gradient Boosting | 0.8167 | 0.7459 | 0.6173 |
| Random Forest | 0.8146 | 0.7540 | 0.6196 |

### 5. Model Evaluation
Best model (Logistic Regression) evaluated on held-out test set:
- ROC-AUC: 0.8213
- Default threshold recall: 0.50 (27 of 54 diabetic patients missed)

### 6. Threshold Optimisation
Classification threshold adjusted from 0.50 to 0.26 to prioritise recall in a clinical context:

| Metric | Default (0.50) | Optimised (0.26) |
|---|---|---|
| Recall | 0.5000 | **0.8889** |
| Precision | 0.5745 | 0.6076 |
| F1 Score | 0.5347 | **0.7218** |
| Missed diabetics | 27/54 (50%) | **6/54 (11.1%)** |

---

## Key Findings

- Logistic Regression outperformed all ensemble models on ROC-AUC, and its interpretability makes it suitable for clinical deployment
- Top risk factors: number of pregnancies (+0.845), glucose level (+0.796), and composite risk score (+0.761)
- Engineered features ranked among the top 6 predictors, confirming the value of domain-informed feature creation
- Threshold optimisation reduced missed diabetic patients by 38.9 percentage points with only a modest reduction in precision

---

## Clinical Recommendation

Deploy with the optimised threshold of 0.26 as a first-pass screening tool. Flag any patient with predicted probability ≥ 0.26 for priority clinical follow-up.

Focus clinical attention on the top three risk factors: number of pregnancies, fasting glucose, and BMI. Retrain every 6 months as new data accumulates.

---

## Limitations

1. Dataset limited to female patients of Pima Indian heritage aged 21+ — generalisation requires validation on broader populations
2. Cross-sectional data predicts current diabetes status, not future onset
3. Class imbalance handled via threshold optimisation — future work should explore SMOTE and cost-sensitive learning

---

## Tools and Libraries

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Scikit-learn | Modelling and evaluation |
| Matplotlib | Visualisation |
| Seaborn | Statistical plots |

---

## Author

**Mubarak Adesola Adedeji**
Data Analyst | Python · SQL · R · Power BI
[LinkedIn](https://linkedin.com/in/mubarak-adedeji-776804273) · [GitHub](https://github.com/Mubydeji)

---

## License

MIT License — free to use, adapt, and build on with attribution.
