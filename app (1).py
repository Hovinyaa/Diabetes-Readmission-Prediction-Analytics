import json
from pathlib import Path
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

st.set_page_config(page_title="Diabetes Readmission Predictor", page_icon="🏥", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_enhanced_catboost.cbm"
METADATA_PATH = BASE_DIR / "feature_metadata.json"
MAPPING_PATH = BASE_DIR / "diagnosis_mapping.csv"

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    return model

@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_shap_explainer():
    return shap.TreeExplainer(model)

@st.cache_data
def load_diagnosis_mapping():
    df = pd.read_csv(MAPPING_PATH, dtype=str).fillna("")
    return df.drop_duplicates("diag_3digit")

model = load_model()
shap_explainer = load_shap_explainer()
metadata = load_metadata()
diagnosis_mapping = load_diagnosis_mapping()
CLASSES = metadata["classes"]
MODEL_FEATURES = metadata["features"]
DIAG_MAP = dict(zip(diagnosis_mapping["diag_3digit"], diagnosis_mapping["category"]))

def diagnosis_category(code):
    return DIAG_MAP.get(str(code).strip(), "Unknown")

def build_input_row(race, gender, age, admission_type_id, admission_source_id,
                    time_in_hospital, payer_code, medical_specialty,
                    num_lab_procedures, num_procedures, num_medications,
                    number_outpatient, number_emergency, number_inpatient,
                    number_diagnoses, max_glu_serum, a1cresult, insulin, change,
                    diabetes_med, diag_1, diag_2, diag_3, prior_patient_encounters):
    row = {feature: "No" for feature in MODEL_FEATURES}
    row.update({
        "race": race, "gender": gender, "age": age,
        "admission_type_id": str(admission_type_id),
        "admission_source_id": str(admission_source_id),
        "payer_code": payer_code, "medical_specialty": medical_specialty,
        "max_glu_serum": max_glu_serum, "A1Cresult": a1cresult,
        "insulin": insulin, "change": change, "diabetesMed": diabetes_med,
        "diag_1_category": diagnosis_category(diag_1),
        "diag_2_category": diagnosis_category(diag_2),
        "diag_3_category": diagnosis_category(diag_3),
        "diag_1_3digit": str(diag_1), "diag_2_3digit": str(diag_2),
        "diag_3_3digit": str(diag_3),
        "time_in_hospital": int(time_in_hospital),
        "num_lab_procedures": int(num_lab_procedures),
        "num_procedures": int(num_procedures),
        "num_medications": int(num_medications),
        "number_outpatient": int(number_outpatient),
        "number_emergency": int(number_emergency),
        "number_inpatient": int(number_inpatient),
        "number_diagnoses": int(number_diagnoses),
        "prior_patient_encounters": int(prior_patient_encounters),
        "active_diabetes_med_count": 1 if diabetes_med == "Yes" else 0,
        "medication_change_count": 1 if change == "Ch" else 0,
    })
    return pd.DataFrame([row], columns=MODEL_FEATURES)

def predict(row):
    probabilities = model.predict_proba(row)[0]
    prediction = model.predict(row).flatten()[0]
    return prediction, dict(zip(CLASSES, probabilities))

def explain_prediction(row, prediction):
    # SHAP returns (rows, features, classes) for this multiclass CatBoost model.
    shap_values = shap_explainer.shap_values(row)
    class_index = CLASSES.index(prediction)
    values = shap_values[0, :, class_index]

    explanation = pd.DataFrame({
        "Feature": MODEL_FEATURES,
        "SHAP Value": values,
        "Input Value": row.iloc[0].astype(str).values,
    })
    explanation["Absolute SHAP"] = explanation["SHAP Value"].abs()
    explanation["Direction"] = explanation["SHAP Value"].apply(
        lambda x: "Toward prediction" if x > 0 else "Away from prediction"
    )
    return explanation.sort_values("Absolute SHAP", ascending=False)

st.title("🏥 Diabetes Readmission Predictor")
st.markdown("**Machine-learning portfolio project** — enter a hypothetical diabetes encounter to estimate the model's predicted readmission outcome.")

st.info("Portfolio demonstration only — this model is not clinically validated and should not be used for patient care or clinical decision-making.")

st.header("1. Patient Information")
c1, c2, c3 = st.columns(3)
with c1:
    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Unknown"])
with c2:
    gender = st.selectbox("Gender", ["Female", "Male", "Unknown/Invalid"])
with c3:
    age = st.selectbox("Age Group", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=6)

st.header("2. Admission Information")
c1, c2, c3, c4 = st.columns(4)
with c1:
    admission_type_id = st.selectbox("Admission Type", ["1","2","3","4","5","6","7","8"])
with c2:
    admission_source_id = st.selectbox("Admission Source", ["1","14","17","2","20","22","3","4","5","6","7","8","9"])
with c3:
    payer_code = st.selectbox("Payer Code", ["BC","CH","CM","CP","DM","HM","MC","MD","MP","OG","OT","PO","SI","SP","UN","Unknown","WC"], index=6)
with c4:
    medical_specialty = st.selectbox("Medical Specialty", ["InternalMedicine","Family/GeneralPractice","Cardiology","Emergency/Trauma","Endocrinology","Nephrology","Hospitalist","InfectiousDiseases","Gastroenterology","Other"])

st.header("3. Current Encounter & Utilization")
c1, c2, c3, c4, c5 = st.columns(5)
with c1: time_in_hospital = st.number_input("Length of Stay (days)", 1, 14, 5)
with c2: num_lab_procedures = st.number_input("Lab Procedures", 0, 150, 40)
with c3: num_procedures = st.number_input("Procedures", 0, 10, 1)
with c4: num_medications = st.number_input("Number of Medications", 0, 80, 15)
with c5: number_diagnoses = st.number_input("Number of Diagnoses", 1, 20, 8)

c1, c2, c3, c4 = st.columns(4)
with c1: number_outpatient = st.number_input("Previous Outpatient Visits", 0, 50, 1)
with c2: number_emergency = st.number_input("Previous Emergency Visits", 0, 50, 0)
with c3: number_inpatient = st.number_input("Previous Inpatient Visits", 0, 20, 2)
with c4: prior_patient_encounters = st.number_input("Prior Patient Encounters", 0, 50, 2)

st.header("4. Clinical Information")
c1, c2, c3 = st.columns(3)
with c1: max_glu_serum = st.selectbox("Maximum Glucose Serum", [">200", ">300", "Norm", "Not Tested"], index=2)
with c2: a1cresult = st.selectbox("HbA1c Result", [">7", ">8", "Norm", "Not Tested"], index=0)
with c3: diabetes_med = st.selectbox("Diabetes Medication", ["Yes", "No"])

st.header("5. Diagnosis")
diag_codes = diagnosis_mapping["diag_3digit"].tolist() or ["Unknown"]
c1, c2, c3 = st.columns(3)
with c1:
    diag_1 = st.selectbox("Primary Diagnosis (3-digit ICD-9)", diag_codes, index=diag_codes.index("428") if "428" in diag_codes else 0)
with c2:
    diag_2 = st.selectbox("Secondary Diagnosis (3-digit ICD-9)", diag_codes, index=diag_codes.index("250") if "250" in diag_codes else 0)
with c3:
    diag_3 = st.selectbox("Tertiary Diagnosis (3-digit ICD-9)", diag_codes, index=diag_codes.index("401") if "401" in diag_codes else 0)
st.caption(f"Categories — Primary: **{diagnosis_category(diag_1)}** | Secondary: **{diagnosis_category(diag_2)}** | Tertiary: **{diagnosis_category(diag_3)}**")

st.header("6. Medication Changes")
c1, c2 = st.columns(2)
with c1: insulin = st.selectbox("Insulin Change", ["No", "Steady", "Up", "Down"], index=1)
with c2: change = st.selectbox("Medication Change", ["No", "Ch"], index=1)

st.divider()
if st.button("🔮 Predict Readmission Outcome", type="primary", use_container_width=True):
    input_row = build_input_row(
        race, gender, age, admission_type_id, admission_source_id,
        time_in_hospital, payer_code, medical_specialty, num_lab_procedures,
        num_procedures, num_medications, number_outpatient, number_emergency,
        number_inpatient, number_diagnoses, max_glu_serum, a1cresult, insulin,
        change, diabetes_med, diag_1, diag_2, diag_3, prior_patient_encounters
    )
    prediction, probabilities = predict(input_row)
    st.subheader("Prediction")
    st.success(f"Predicted outcome: **{prediction}**")
    st.subheader("Predicted Probability")
    c1, c2, c3 = st.columns(3)
    for col, outcome in zip((c1, c2, c3), CLASSES):
        with col:
            st.metric(outcome, f"{probabilities[outcome] * 100:.2f}%")
    st.bar_chart(pd.DataFrame({"Probability": probabilities}))

    st.subheader("🔍 Why did the model make this prediction?")
    st.markdown(
        f"SHAP explains how each feature contributed to the model's **{prediction}** prediction. "
        "Positive values push the model toward the predicted class, while negative values push it away. "
        "These are model contributions, not causal effects."
    )

    explanation = explain_prediction(input_row, prediction)
    top_explanation = explanation.head(10).copy()
    top_explanation = top_explanation.sort_values("SHAP Value")

    # Horizontal SHAP chart keeps long feature names readable.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(top_explanation["Feature"], top_explanation["SHAP Value"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("SHAP Value")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top Feature Contributions to {prediction} Prediction")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    display_explanation = explanation.head(10)[
        ["Feature", "SHAP Value", "Input Value", "Direction"]
    ].copy()
    display_explanation["SHAP Value"] = display_explanation["SHAP Value"].round(4)
    st.dataframe(display_explanation, use_container_width=True, hide_index=True)

    with st.expander("View model input"):
        st.dataframe(input_row.T.rename(columns={0: "Value"}))

st.divider()
st.caption(f"Model: {metadata['model_name']} | 49 features | Test accuracy: {metadata['accuracy']*100:.1f}% | Macro F1: {metadata['macro_f1']*100:.1f}%")
st.caption("Educational portfolio demonstration using the Diabetes 130-US Hospitals dataset. Not intended for clinical use.")
