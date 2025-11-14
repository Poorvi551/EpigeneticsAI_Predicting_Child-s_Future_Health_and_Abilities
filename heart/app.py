import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('child_heart_model.pkl')
        scaler = joblib.load('child_heart_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found! Please ensure 'child_heart_model.pkl' and 'child_heart_scaler.pkl' are in the same directory.")
        return None, None

def predict_child_heart_disease(user_input_dict, model, scaler):
    """
    Predict heart disease risk based on user input.
    
    Parameters:
    -----------
    user_input_dict : dict
        Dictionary containing all required features
    model : sklearn model
        Trained stacking classifier
    scaler : sklearn scaler
        Fitted StandardScaler
        
    Returns:
    --------
    tuple : (prediction, probability)
    """
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([user_input_dict])
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Get prediction and probability
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        return prediction, probability
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# Title and description
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("Enter the patient's health metrics to assess potential heart disease risk.")
st.markdown("---")

# Load model
model, scaler = load_model()

if model is not None and scaler is not None:
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Patient Information")
        age = st.slider("Age (years)", 1, 100, 50, help="Patient's age in years")
        sex = st.selectbox(
            "Sex", 
            [0, 1], 
            format_func=lambda x: "Female" if x == 0 else "Male",
            help="Biological sex"
        )
        cp = st.selectbox(
            "Chest Pain Type", 
            [0, 1, 2, 3],
            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x],
            help="Type of chest pain experienced"
        )
        trestbps = st.slider(
            "Resting Blood Pressure (mm Hg)", 
            80, 200, 120,
            help="Resting blood pressure measurement"
        )
        chol = st.slider(
            "Serum Cholesterol (mg/dl)", 
            100, 600, 200,
            help="Serum cholesterol level"
        )
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl", 
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Whether fasting blood sugar is greater than 120 mg/dl"
        )
        restecg = st.selectbox(
            "Resting ECG Results", 
            [0, 1, 2],
            format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x],
            help="Resting electrocardiographic results"
        )
    
    with col2:
        st.subheader("🏥 Clinical Measurements")
        thalach = st.slider(
            "Max Heart Rate Achieved", 
            60, 220, 150,
            help="Maximum heart rate achieved during exercise"
        )
        exang = st.selectbox(
            "Exercise Induced Angina", 
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Whether exercise induces angina"
        )
        oldpeak = st.slider(
            "ST Depression", 
            0.0, 6.0, 1.0, 0.1,
            help="ST depression induced by exercise relative to rest"
        )
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment", 
            [0, 1, 2],
            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
            help="Slope of the peak exercise ST segment"
        )
        ca = st.selectbox(
            "Major Vessels Colored by Fluoroscopy", 
            [0, 1, 2, 3],
            help="Number of major vessels (0-3) colored by fluoroscopy"
        )
        thal = st.selectbox(
            "Thalassemia", 
            [0, 1, 2, 3],
            format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x],
            help="Thalassemia test result"
        )
    
    st.markdown("---")
    
    # Create user input dictionary
    user_input = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    # Center the predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔍 Predict Heart Disease Risk", use_container_width=True)
    
    # Predict
    if predict_button:
        with st.spinner("Analyzing..."):
            prediction, probability = predict_child_heart_disease(user_input, model, scaler)
            
            if prediction is not None and probability is not None:
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Display results based on prediction
                if prediction == 1:
                    st.error("⚠️ **High Risk of Heart Disease Detected**")
                    st.metric(
                        label="Risk Probability", 
                        value=f"{probability[1]:.1%}",
                        delta=None
                    )
                    st.warning("**Recommendation:** Please consult a cardiologist for further evaluation and medical advice.")
                else:
                    st.success("✅ **Low Risk of Heart Disease**")
                    st.metric(
                        label="Health Probability", 
                        value=f"{probability[0]:.1%}",
                        delta=None
                    )
                    st.info("**Recommendation:** Continue maintaining a healthy lifestyle. Regular check-ups are advised.")
                
                # Show probability breakdown
                st.markdown("#### Probability Breakdown")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("No Disease", f"{probability[0]:.2%}")
                with col2:
                    st.metric("Disease Present", f"{probability[1]:.2%}")
                
                # Display input summary in expander
                with st.expander("📝 View Input Summary"):
                    input_df = pd.DataFrame([user_input])
                    st.dataframe(input_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    ⚠️ <b>Disclaimer:</b> This tool is for educational purposes only and should not replace professional medical advice.
    Always consult with a qualified healthcare provider for medical decisions.
    </div>
    """,
    unsafe_allow_html=True
)