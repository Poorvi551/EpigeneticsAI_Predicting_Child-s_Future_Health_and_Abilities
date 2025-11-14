import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('autism_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

st.set_page_config(page_title="Autism Detection", layout="centered")
st.title("🧠 Autism Detection in Children")
st.markdown("Enter the child's responses and background info to assess autism risk.")

# Input form
user_input = {
    'A1': st.selectbox("Responds to name?", [0, 1]),
    'A2': st.selectbox("Makes eye contact?", [0, 1]),
    'A3': st.selectbox("Shows interest in others?", [0, 1]),
    'A4': st.selectbox("Engages in pretend play?", [0, 1]),
    'A5': st.selectbox("Points to show interest?", [0, 1]),
    'A6': st.selectbox("Follows instructions?", [0, 1]),
    'A7': st.selectbox("Shows repetitive behaviors?", [0, 1]),
    'A8': st.selectbox("Reacts to loud sounds?", [0, 1]),
    'A9': st.selectbox("Prefers routines?", [0, 1]),
    'A10': st.selectbox("Avoids physical contact?", [0, 1]),
    'Age': st.slider("Age", 1, 18, 5),
    'Sex': st.selectbox("Sex", ['m', 'f']),
    'Jauundice': st.selectbox("History of jaundice?", ['yes', 'no']),
    'Family_ASD': st.selectbox("Family history of ASD?", ['yes', 'no'])
}

# Prediction
if st.button("🔍 Predict Autism Risk"):
    input_df = pd.DataFrame([user_input])
    for col in ['Sex', 'Jauundice', 'Family_ASD']:
        input_df[col] = label_encoders[col].transform(input_df[col])
    prediction = model.predict(input_df)[0]
    result = label_encoders['Class'].inverse_transform([prediction])[0]

    if result == 'YES':
        st.error("⚠️ The child may be at risk for autism. Please consult a specialist.")
    else:
        st.success("✅ No signs of autism detected based on this screening.")