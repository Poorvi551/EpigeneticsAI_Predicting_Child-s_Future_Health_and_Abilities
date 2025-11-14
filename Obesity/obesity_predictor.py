import joblib
import pandas as pd
def preprocess_user_input(data):
    encoders = joblib.load("encoders.pkl")
    processed = {}

    for col, val in data.items():
        if col in encoders:
            processed[col] = encoders[col].transform([val])[0]
        else:
            processed[col] = float(val)  # Assume numeric if not categorical

    return pd.DataFrame([processed])

def predict_obesity(data):
    model = joblib.load("obesity_predictor.pkl")
    input_df = preprocess_user_input(data)
    prediction = model.predict(input_df)[0]
    return round(prediction, 2)