# Diabetes Readmission Prediction & Analytics

## 📌 Project Overview

This project presents an end-to-end healthcare data science workflow using the
Diabetes 130-US Hospitals dataset. The project combines data preprocessing,
SQL, exploratory data analysis, machine learning, explainable AI, and
interactive visualization to investigate hospital readmission patterns among
patients with diabetes.

The project focuses on three readmission outcomes:

- Within 30 Days
- After 30 Days
- Not Readmitted

The final machine-learning model is deployed through a Streamlit application,
while Power BI is used to present clinical and model-performance analytics.

---

## 🎯 Project Objectives

- Clean and preprocess a large healthcare dataset.
- Explore demographic, clinical, diagnostic, and utilization patterns.
- Investigate factors associated with hospital readmission.
- Develop a multiclass machine-learning model for readmission prediction.
- Evaluate model performance using appropriate multiclass metrics.
- Use SHAP to improve model interpretability.
- Build an interactive Power BI dashboard.
- Develop an interactive Streamlit prediction application.

---

## 📊 Dataset

The project uses the **Diabetes 130-US Hospitals for Years 1999-2008**
dataset.

The dataset contains approximately 101,766 hospital encounters and includes
information related to:

- Patient demographics
- Admission information
- Hospital utilization
- Diagnoses
- Laboratory procedures
- Medications
- Length of stay
- Diabetes management
- Readmission outcomes

The target variable contains three classes:

| Original Outcome | Project Outcome |
|---|---|
| `<30` | Within 30 Days |
| `>30` | After 30 Days |
| `NO` | Not Readmitted |

---

## 🔧 Data Preparation

The preprocessing workflow included:

- Handling missing-value placeholders
- Removing the `weight` variable
- Mapping ICD-9 diagnosis codes into broader diagnostic categories
- Creating 3-digit diagnosis features
- Creating medication-related features
- Creating prior healthcare utilization features
- Checking duplicate records
- Checking remaining missing values
- Patient-level train/test splitting to prevent patient overlap

The final cleaned dataset contains **101,766 encounters and 60 variables**.

---

## 🤖 Machine Learning

Several modelling approaches were investigated during development,
including:

- Logistic Regression
- XGBoost
- LightGBM
- CatBoost
- Two-stage classification approaches

The final model selected for the portfolio application was an
**Enhanced CatBoost multiclass classifier**.

The model uses 49 features, including demographic, admission, utilization,
clinical, medication, and diagnosis-related variables.

### Model Performance

Evaluation was performed on a held-out test set of 20,679 encounters.

| Metric | Result |
|---|---:|
| Accuracy | 50.82% |
| Balanced Accuracy | 45.85% |
| Macro F1 | 44.09% |
| Weighted F1 | 52.21% |

Because the target classes are imbalanced, balanced accuracy and macro F1
are reported alongside conventional accuracy.

---

## 🔍 Explainable AI

SHAP (SHapley Additive exPlanations) was used to investigate how model
features contributed to predictions.

Important features included:

- Previous inpatient visits
- 3-digit primary diagnosis
- Prior patient encounters
- Length of hospital stay
- Number of diagnoses
- Secondary diagnosis
- Previous outpatient visits
- Payer information
- Admission source
- Medical specialty

Feature importance represents model associations and should not be
interpreted as causal relationships.

---

## 📈 Power BI Dashboard

The Power BI dashboard contains multiple analytical components covering:

### Clinical & Utilization Insights

- Overall encounter volume
- Readmission outcome distribution
- Readmission rates
- Average length of stay
- Average number of diagnoses
- Readmission patterns by age
- Length of stay
- Previous inpatient visits
- Emergency visits
- Outpatient visits
- Diagnosis categories

### Machine Learning Performance

- Accuracy
- Balanced accuracy
- Macro F1
- Confusion matrix
- Class-level precision
- Class-level recall
- Class-level F1 score
- Actual vs predicted outcomes

---

## 🌐 Streamlit Prediction App

The Streamlit application provides an interactive demonstration of the
trained model.

Users can enter hypothetical encounter information including:

- Demographics
- Admission information
- Healthcare utilization
- Clinical measurements
- Diagnosis codes
- Medication information

The application returns:

- Predicted readmission outcome
- Probability of each outcome
- Model input information

The application is intended as a **portfolio demonstration of machine
learning deployment**, rather than a clinical decision-support system.

---

## 🗂️ Project Structure

```text
diabetes-readmission-prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── final_enhanced_catboost.cbm

```
## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- CatBoost
- SHAP
- Scikit-learn
- PostgreSQL / Supabase
- Power BI
- Streamlit
- GitHub


## ⚠️ Disclaimer

This project is an educational and portfolio demonstration of healthcare
data science and machine learning.

The model has not been clinically validated and should not be used to make
clinical decisions, determine patient treatment, or replace professional
medical judgement.

The Streamlit application should only be demonstrated using appropriate
non-identifiable or hypothetical information.


## 👩‍💻 Author

**HOVINYAA SELVARAJU**

Master's in Data Science | Bioinformatics & Computational Biology

**Areas demonstrated in this project:**

Healthcare Analytics • Machine Learning • SQL • Python • Power BI •  
Explainable AI • Data Visualization • Model Deployment
