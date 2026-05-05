import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def train_model():
    from sklearn.datasets import make_classification
    import pandas as pd
    import numpy as np

    np.random.seed(42)

    data = {
        "Pregnancies"             : [6,1,8,1,0,5,3,10,2,8,4,10,10,1,5,7,0,7,1,1,3,8,7,9,11,10,7,1,13,5,4,1,3,7,5,0,2,7,0,5,5,3,1,4,5,5,6,8,3,5],
        "Glucose"                 : [148,85,183,89,137,116,78,115,197,125,110,168,139,189,166,100,118,107,103,115,126,99,196,119,143,125,147,97,145,117,111,102,128,158,105,103,130,105,115,117,121,91,129,130,125,130,125,120,102,129],
        "BloodPressure"           : [72,66,64,66,40,74,50,70,70,96,74,74,80,60,72,78,84,74,30,70,88,64,90,80,94,72,70,66,82,66,72,54,68,76,72,72,68,76,60,92,72,54,72,70,70,80,72,74,54,66],
        "SkinThickness"           : [35,29,0,23,35,0,32,0,45,0,0,0,0,23,19,31,47,0,38,30,41,0,0,47,33,0,21,23,0,0,37,27,30,43,14,25,28,23,0,30,23,17,0,30,26,56,43,57,27,30],
        "Insulin"                 : [0,0,0,94,168,0,88,0,543,0,0,0,0,846,175,162,230,0,83,96,235,0,0,203,146,0,0,83,0,166,135,0,55,0,0,0,0,0,0,0,112,66,176,320,225,92,0,0,0,0],
        "BMI"                     : [33.6,26.6,23.3,28.1,43.1,25.6,31.0,35.3,30.5,0.0,26.2,38.0,27.1,30.1,25.8,36.2,45.8,29.6,43.3,34.6,39.3,29.0,39.8,32.4,36.6,26.0,37.4,28.2,32.9,25.6,43.1,52.6,40.1,29.0,32.0,37.0,39.5,34.6,29.2,26.9,26.2,26.5,38.5,34.2,30.5,40.6,37.6,44.0,25.5,39.5],
        "DiabetesPedigreeFunction": [0.627,0.351,0.672,0.167,2.288,0.201,0.248,0.134,0.158,0.232,0.111,0.537,1.441,0.398,0.587,0.422,0.551,0.254,0.183,0.529,0.704,0.388,0.451,0.299,0.254,0.587,0.426,0.498,0.352,0.201,0.183,0.186,0.295,0.302,0.167,0.251,0.389,0.237,0.162,0.217,0.245,0.274,0.245,0.315,0.263,0.434,0.537,0.387,0.455,0.637],
        "Age"                     : [50,31,32,21,33,30,26,29,53,54,30,34,57,59,26,44,49,39,20,32,27,23,41,31,51,33,29,21,57,22,24,27,23,32,23,25,29,24,29,45,40,35,26,27,30,28,40,39,26,29],
        "Outcome"                 : [1,0,1,0,1,0,1,0,1,1,0,1,0,1,1,1,1,0,0,0,0,0,1,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0]
    }

    df = pd.DataFrame(data)

    zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    df["Glucose_BMI"]           = df["Glucose"] * df["BMI"]
    df["Age_Pregnancies"]       = df["Age"] * df["Pregnancies"]
    df["Insulin_Glucose_Ratio"] = df["Insulin"] / (df["Glucose"] + 1)
    df["Risk_Score"]            = (
        0.5 * (df["Glucose"] / df["Glucose"].max()) +
        0.3 * (df["BMI"]     / df["BMI"].max())     +
        0.2 * (df["Age"]     / df["Age"].max())
    )

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline


def prepare_input(pregnancies, glucose, blood_pressure, skin_thickness,
                  insulin, bmi, dpf, age):
    glucose_bmi           = glucose * bmi
    age_pregnancies       = age * pregnancies
    insulin_glucose_ratio = insulin / (glucose + 1)
    risk_score            = (
        0.5 * (glucose / 200) +
        0.3 * (bmi     / 70)  +
        0.2 * (age     / 100)
    )
    return np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age,
        glucose_bmi, age_pregnancies, insulin_glucose_ratio, risk_score
    ]])


model = train_model()

st.title("🩺 Diabetes Risk Predictor")
st.markdown(
    "Enter patient clinical measurements below to assess diabetes risk. "
    "This tool uses a Logistic Regression model trained on the Pima Indians Diabetes Dataset."
)
st.divider()

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies    = st.number_input("Number of Pregnancies",    min_value=0,   max_value=20,  value=1,    step=1)
    glucose        = st.number_input("Glucose Level (mg/dL)",    min_value=50,  max_value=300, value=120,  step=1)
    blood_pressure = st.number_input("Blood Pressure (mmHg)",    min_value=40,  max_value=200, value=80,   step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)",      min_value=0,   max_value=100, value=20,   step=1)

with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)",         min_value=0,   max_value=900, value=80,   step=1)
    bmi     = st.number_input("BMI (kg/m²)",                     min_value=10.0,max_value=70.0,value=25.0, step=0.1)
    dpf     = st.number_input("Diabetes Pedigree Function",       min_value=0.0, max_value=3.0, value=0.5,  step=0.01)
    age     = st.number_input("Age (years)",                      min_value=18,  max_value=100, value=30,   step=1)

st.divider()

THRESHOLD = 0.26

if st.button("Assess Diabetes Risk", type="primary", use_container_width=True):

    input_data   = prepare_input(pregnancies, glucose, blood_pressure,
                                 skin_thickness, insulin, bmi, dpf, age)
    probability  = model.predict_proba(input_data)[0][1]
    prediction   = int(probability >= THRESHOLD)

    st.subheader("Risk Assessment Result")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Risk Probability", f"{probability * 100:.1f}%")

    with col_b:
        risk_level = "High Risk" if probability >= 0.5 else "Moderate Risk" if probability >= THRESHOLD else "Low Risk"
        st.metric("Risk Level", risk_level)

    with col_c:
        st.metric("Classification Threshold", f"{THRESHOLD}")

    if prediction == 1:
        st.error(
            f" **This patient is flagged as HIGH RISK for diabetes** "
            f"(probability: {probability*100:.1f}%). "
            f"Priority clinical follow-up is recommended."
        )
    else:
        st.success(
            f" **This patient is currently LOW RISK for diabetes** "
            f"(probability: {probability*100:.1f}%). "
            f"Routine monitoring is advised."
        )

    st.divider()
    st.subheader("Key Risk Factors")

    factors = {
        "Glucose Level"   : (glucose,  70,  200, "Higher glucose significantly increases risk"),
        "BMI"             : (bmi,      18,  45,  "BMI above 30 is associated with increased risk"),
        "Age"             : (age,      20,  80,  "Risk increases with age"),
        "Pregnancies"     : (pregnancies, 0, 10, "Higher number of pregnancies increases risk"),
    }

    for factor, (value, low, high, note) in factors.items():
        normalized = min((value - low) / (high - low), 1.0)
        st.write(f"**{factor}**: {value}")
        st.progress(normalized)
        st.caption(note)

st.divider()
st.caption(
    "⚕️ This tool is intended for research and educational purposes only. "
    "It is not a substitute for clinical diagnosis. "
    "All predictions should be reviewed by a qualified healthcare professional."
)
st.caption("Built by Mubarak Adesola Adedeji · [LinkedIn](https://linkedin.com/in/mubarak-adedeji-776804273) · [GitHub](https://github.com/Mubydeji)")
