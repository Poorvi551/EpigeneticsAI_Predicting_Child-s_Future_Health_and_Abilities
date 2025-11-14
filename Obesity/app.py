import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("obesity_predictor.pkl")
encoders = joblib.load("encoders.pkl")

# Define expected feature order
model_features = [
    "INDICATOR", "PANEL", "PANEL_NUM", "UNIT", "UNIT_NUM",
    "STUB_NAME", "STUB_NAME_NUM", "STUB_LABEL_NUM", "STUB_LABEL",
    "YEAR", "YEAR_NUM", "AGE", "AGE_NUM"
]

# Page setup
st.set_page_config(page_title="Obesity Predictor", layout="centered")
st.title("🧠 Obesity Prediction for Children & Adolescents")
st.markdown("Enter demographic and survey details to predict obesity percentage.")

# Load full dataset for dropdowns
raw_df = pd.read_csv("obesity data.csv")

# Helper to filter dropdowns from dataset column
def safe_select(label, column_name):
    options = raw_df[column_name].dropna().unique()
    clean = [opt for opt in options if str(opt).strip() != ""]
    return st.selectbox(label, sorted(set(clean)))

# Input form
with st.form("prediction_form"):
    indicator = safe_select("Indicator", "INDICATOR")
    panel = safe_select("Panel", "PANEL")
    panel_num = st.number_input("Panel Num", min_value=0, max_value=10, value=1)
    unit = safe_select("Unit", "UNIT")
    unit_num = st.number_input("Unit Num", min_value=0, max_value=10, value=1)
    stub_name = safe_select("Stub Name", "STUB_NAME")
    stub_name_num = st.number_input("Stub Name Num", min_value=0, max_value=10, value=0)
    stub_label_num = st.number_input("Stub Label Num", min_value=0, max_value=10, value=0)
    stub_label = safe_select("Stub Label", "STUB_LABEL")
    year = safe_select("Year", "YEAR")
    year_num = st.number_input("Year Num", min_value=1, max_value=20, value=1)
    age = safe_select("Age", "AGE")
    age_num = st.number_input("Age Num", min_value=0, max_value=10, value=0)

    submitted = st.form_submit_button("🔍 Predict Obesity")

def preprocess_input(user_data):
    processed = {}
    for col in model_features:
        val = user_data.get(col)
        if col in encoders:
            try:
                processed[col] = encoders[col].transform([val])[0]
            except ValueError:
                st.error(f"⚠️ Value '{val}' for '{col}' is not recognized by the model encoder.")
                return None
        else:
            processed[col] = val
    return pd.DataFrame([processed])

# Prediction logic
if submitted:
    # 🔹 Add this slider first
    threshold = st.slider(
        "Set Obesity Threshold (%)",
        min_value=5.0,
        max_value=30.0,
        value=15.0,
        help="If predicted obesity is above this threshold, the child is considered obese."
    )

    # 🔹 Collect user inputs
    user_data = {
        "INDICATOR": indicator,
        "PANEL": panel,
        "PANEL_NUM": panel_num,
        "UNIT": unit,
        "UNIT_NUM": unit_num,
        "STUB_NAME": stub_name,
        "STUB_NAME_NUM": stub_name_num,
        "STUB_LABEL_NUM": stub_label_num,
        "STUB_LABEL": stub_label,
        "YEAR": year,
        "YEAR_NUM": year_num,
        "AGE": age,
        "AGE_NUM": age_num,
    }

    # 🔹 Preprocess and predict
    input_df = preprocess_input(user_data)
    if input_df is not None:
        prediction = model.predict(input_df)[0]
        st.markdown(f"📊 **Predicted Obesity Estimate:** `{round(prediction, 2)}%`")

        # 🔹 Diagnosis logic using slider value
        if prediction >= threshold:
            st.error("🧒 **Diagnosis:** The child is likely facing obesity.")
        else:
            st.success("✅ **Diagnosis:** The child is not facing obesity.")