import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Child Stress Prediction Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stressed-result {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .not-stressed-result {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .recommendation-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the trained model and scaler"""
    try:
        # Try to load binary model first (preferred)
        if os.path.exists('best_binary_stress_model.pkl'):
            model = joblib.load('best_binary_stress_model.pkl')
            scaler = joblib.load('binary_scaler.pkl')
            model_type = 'binary'
        else:
            # Fallback to multi-class model
            model = joblib.load('best_stress_model.pkl')
            scaler = joblib.load('scaler.pkl')
            model_type = 'multi_class'
        
        return model, scaler, model_type
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure you have trained the model first by running the Jupyter notebook.")
        return None, None, None

def predict_stress(features_dict, model, scaler, model_type):
    """Make stress prediction"""
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([features_dict])
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled).max()
        
        if model_type == 'binary':
            # Binary classification: 0 = Not Stressed, 1 = Stressed
            stress_level = "Stressed" if prediction == 1 else "Not Stressed"
        else:
            # Multi-class classification
            stress_levels = {0: "Low Stress", 1: "Medium Stress", 2: "High Stress"}
            stress_level = stress_levels.get(prediction, "Unknown")
        
        return prediction, confidence, stress_level
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def get_recommendations(prediction, confidence, model_type):
    """Get recommendations based on prediction"""
    if model_type == 'binary':
        if prediction == 1:  # Stressed
            return [
                "Consider consulting with a child psychologist or counselor",
                "Monitor the child's behavior and emotional state closely",
                "Provide emotional support and create a safe environment",
                "Encourage open communication about their feelings",
                "Consider reducing academic or social pressures if possible",
                "Ensure the child gets adequate sleep and nutrition"
            ]
        else:  # Not Stressed
            return [
                "Continue monitoring the child's well-being",
                "Maintain current supportive environment",
                "Encourage healthy coping mechanisms",
                "Regular check-ins to ensure continued well-being",
                "Promote positive social interactions",
                "Support their interests and hobbies"
            ]
    else:
        # Multi-class recommendations
        if prediction == 2:  # High Stress
            return [
                "Immediate professional intervention recommended",
                "Create a calm and supportive environment",
                "Reduce all sources of pressure and stress",
                "Consider temporary academic accommodations",
                "Seek family counseling if needed"
            ]
        elif prediction == 1:  # Medium Stress
            return [
                "Monitor closely and provide extra support",
                "Identify and address specific stress sources",
                "Teach stress management techniques",
                "Ensure adequate rest and relaxation time"
            ]
        else:  # Low Stress
            return [
                "Continue current supportive practices",
                "Maintain healthy routines",
                "Encourage positive activities",
                "Regular emotional check-ins"
            ]

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Child Stress Prediction Tool</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Assess whether a child is experiencing stress based on psychological, physical, environmental, and social factors.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler, model_type = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar for model info
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        st.info(f"**Model Type:** {'Binary Classification' if model_type == 'binary' else 'Multi-class Classification'}")
        st.info("**Features:** 20 psychological, physical, environmental, and social factors")
        st.info("**Accuracy:** 85-95% (varies by model)")
        
        st.markdown("## 📝 Instructions")
        st.markdown("""
        1. Fill in all the assessment fields
        2. Use the sliders to indicate levels (0-5 scale)
        3. Click 'Predict Stress Level' to get results
        4. Review recommendations based on the prediction
        """)
        
        st.markdown("## ⚠️ Disclaimer")
        st.warning("""
        This tool is for educational and preliminary assessment purposes only. 
        It should not replace professional medical or psychological evaluation.
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">📋 Child Assessment Form</h2>', unsafe_allow_html=True)
        
        # Create form
        with st.form("stress_assessment_form"):
            # Psychological Factors
            st.markdown("### 🧠 Psychological Factors")
            col1_1, col1_2 = st.columns(2)
            
            with col1_1:
                anxiety_level = st.slider("Anxiety Level (0-30)", 0, 30, 15, 
                                        help="Higher values indicate more anxiety")
                self_esteem = st.slider("Self Esteem (0-30)", 0, 30, 20, 
                                      help="Higher values indicate better self-esteem")
                depression = st.slider("Depression Level (0-30)", 0, 30, 10, 
                                     help="Higher values indicate more depression")
            
            with col1_2:
                mental_health_history = st.selectbox("Mental Health History", 
                                                   ["No", "Yes"], 
                                                   help="History of mental health issues")
                mental_health_history = 1 if mental_health_history == "Yes" else 0
            
            # Physical Factors
            st.markdown("### 🏥 Physical Factors")
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                headache = st.slider("Headache Frequency (0-5)", 0, 5, 2, 
                                   help="How often the child experiences headaches")
                blood_pressure = st.slider("Blood Pressure Issues (0-5)", 0, 5, 1, 
                                         help="Blood pressure related problems")
            
            with col2_2:
                sleep_quality = st.slider("Sleep Quality (0-5)", 0, 5, 3, 
                                        help="5 = excellent sleep, 0 = very poor sleep")
                breathing_problem = st.slider("Breathing Problems (0-5)", 0, 5, 2, 
                                            help="Breathing difficulties or issues")
            
            # Environmental Factors
            st.markdown("### 🏠 Environmental Factors")
            col3_1, col3_2 = st.columns(2)
            
            with col3_1:
                noise_level = st.slider("Noise Level at Home (0-5)", 0, 5, 2, 
                                      help="Level of noise in the home environment")
                living_conditions = st.slider("Living Conditions Quality (0-5)", 0, 5, 3, 
                                            help="Quality of living conditions")
            
            with col3_2:
                safety = st.slider("Safety Level (0-5)", 0, 5, 3, 
                                 help="How safe the child feels in their environment")
                basic_needs = st.slider("Basic Needs Met (0-5)", 0, 5, 3, 
                                      help="Whether basic needs are adequately met")
            
            # Academic Factors
            st.markdown("### 📚 Academic Factors")
            col4_1, col4_2 = st.columns(2)
            
            with col4_1:
                academic_performance = st.slider("Academic Performance (0-5)", 0, 5, 3, 
                                               help="Current academic performance level")
                study_load = st.slider("Study Load (0-5)", 0, 5, 3, 
                                     help="Amount of academic workload")
            
            with col4_2:
                teacher_student_relationship = st.slider("Teacher-Student Relationship (0-5)", 0, 5, 3, 
                                                       help="Quality of relationship with teachers")
                future_career_concerns = st.slider("Future Career Concerns (0-5)", 0, 5, 3, 
                                                 help="Level of concern about future career")
            
            # Social Factors
            st.markdown("### 👥 Social Factors")
            col5_1, col5_2 = st.columns(2)
            
            with col5_1:
                social_support = st.slider("Social Support (0-5)", 0, 5, 3, 
                                         help="Level of social support available")
                peer_pressure = st.slider("Peer Pressure (0-5)", 0, 5, 2, 
                                        help="Level of peer pressure experienced")
            
            with col5_2:
                extracurricular_activities = st.slider("Extracurricular Activities (0-5)", 0, 5, 3, 
                                                     help="Participation in activities outside school")
                bullying = st.slider("Bullying Experience (0-5)", 0, 5, 1, 
                                   help="Experience with bullying (0 = none, 5 = severe)")
            
            # Submit button
            submitted = st.form_submit_button("🔍 Predict Stress Level", use_container_width=True)
            
            if submitted:
                # Prepare features dictionary
                features = {
                    'anxiety_level': anxiety_level,
                    'self_esteem': self_esteem,
                    'mental_health_history': mental_health_history,
                    'depression': depression,
                    'headache': headache,
                    'blood_pressure': blood_pressure,
                    'sleep_quality': sleep_quality,
                    'breathing_problem': breathing_problem,
                    'noise_level': noise_level,
                    'living_conditions': living_conditions,
                    'safety': safety,
                    'basic_needs': basic_needs,
                    'academic_performance': academic_performance,
                    'study_load': study_load,
                    'teacher_student_relationship': teacher_student_relationship,
                    'future_career_concerns': future_career_concerns,
                    'social_support': social_support,
                    'peer_pressure': peer_pressure,
                    'extracurricular_activities': extracurricular_activities,
                    'bullying': bullying
                }
                
                # Make prediction
                prediction, confidence, stress_level = predict_stress(features, model, scaler, model_type)
                
                if prediction is not None:
                    # Display results
                    st.markdown('<h2 class="section-header">📊 Assessment Results</h2>', unsafe_allow_html=True)
                    
                    # Result display
                    if model_type == 'binary':
                        if prediction == 1:  # Stressed
                            st.markdown('<div class="stressed-result">', unsafe_allow_html=True)
                            st.markdown("### 🚨 **RESULT: CHILD IS EXPERIENCING STRESS**")
                            st.markdown("⚠️ This child shows signs of stress and may need support.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:  # Not Stressed
                            st.markdown('<div class="not-stressed-result">', unsafe_allow_html=True)
                            st.markdown("### ✅ **RESULT: CHILD IS NOT EXPERIENCING STRESS**")
                            st.markdown("😊 This child appears to be coping well.")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        # Multi-class result
                        if prediction == 2:  # High Stress
                            st.markdown('<div class="stressed-result">', unsafe_allow_html=True)
                            st.markdown("### 🚨 **HIGH STRESS LEVEL DETECTED**")
                            st.markdown("⚠️ This child is experiencing high levels of stress.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        elif prediction == 1:  # Medium Stress
                            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                            st.markdown("### 🟡 **MEDIUM STRESS LEVEL DETECTED**")
                            st.markdown("⚠️ This child is experiencing moderate stress.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:  # Low Stress
                            st.markdown('<div class="not-stressed-result">', unsafe_allow_html=True)
                            st.markdown("### ✅ **LOW STRESS LEVEL DETECTED**")
                            st.markdown("😊 This child appears to be coping well.")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Metrics
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    
                    with col_metric1:
                        st.metric("Predicted Level", stress_level)
                    
                    with col_metric2:
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    with col_metric3:
                        risk_level = "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.6 else "LOW"
                        st.metric("Risk Level", risk_level)
                    
                    # Recommendations
                    st.markdown('<h2 class="section-header">💡 Recommendations</h2>', unsafe_allow_html=True)
                    recommendations = get_recommendations(prediction, confidence, model_type)
                    
                    st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"**{i}.** {rec}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Key factors analysis
                    st.markdown('<h2 class="section-header">🔍 Key Factors Analysis</h2>', unsafe_allow_html=True)
                    
                    # Identify concerning factors
                    concerning_factors = []
                    for factor, value in features.items():
                        if factor in ['anxiety_level', 'depression', 'headache', 'bullying'] and value > 3:
                            concerning_factors.append(factor.replace('_', ' ').title())
                        elif factor in ['self_esteem', 'sleep_quality', 'social_support'] and value < 2:
                            concerning_factors.append(factor.replace('_', ' ').title())
                    
                    if concerning_factors:
                        st.warning("⚠️ **Areas of concern identified:**")
                        for factor in concerning_factors:
                            st.markdown(f"• {factor}")
                    else:
                        st.success("✅ **No major risk factors identified**")
    
    with col2:
        st.markdown('<h2 class="section-header">📈 Model Performance</h2>', unsafe_allow_html=True)
        
        # Performance metrics
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Model Accuracy", "85-95%")
        st.metric("Features Analyzed", "20")
        st.metric("Training Samples", "1,100")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.markdown("### 🎯 Top Important Features")
            feature_names = [
                'Anxiety Level', 'Self Esteem', 'Depression', 'Headache',
                'Blood Pressure', 'Sleep Quality', 'Breathing Problem',
                'Noise Level', 'Living Conditions', 'Safety', 'Basic Needs',
                'Academic Performance', 'Study Load', 'Teacher Relationship',
                'Career Concerns', 'Social Support', 'Peer Pressure',
                'Extracurricular Activities', 'Bullying'
            ]
            
            importances = model.feature_importances_
            top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
            
            for feature, importance in top_features:
                st.progress(importance, text=f"{feature}: {importance:.3f}")
        
        # Quick tips
        st.markdown("### 💡 Quick Tips")
        st.info("""
        **For Parents/Guardians:**
        - Monitor changes in behavior
        - Maintain open communication
        - Create a supportive environment
        - Seek professional help when needed
        """)
        
        st.info("""
        **For Educators:**
        - Watch for academic performance changes
        - Provide emotional support
        - Communicate with parents
        - Create inclusive classroom environment
        """)

if __name__ == "__main__":
    main()
