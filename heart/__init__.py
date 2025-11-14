import pandas as pd
import joblib
def predict_child_heart_disease(user_input_dict):
    model = joblib.load('child_heart_model.pkl')
    scaler = joblib.load('child_heart_scaler.pkl')
    input_df = pd.DataFrame([user_input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    return "💓 High risk of heart disease, please consult a DOCTOR!!." if prediction == 1 else "✅ Low risk of heart disease, You are HEALTHY!"