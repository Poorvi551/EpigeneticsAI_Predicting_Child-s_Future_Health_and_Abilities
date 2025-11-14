import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# === Safe Model Loading ===
try:
    model = pickle.load(open('model.pkl', 'rb'))
    encoders = pickle.load(open('encoders.pkl', 'rb'))
    X_columns = pickle.load(open('X_columns.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"❌ Missing file: {e.filename}. Please make sure 'model.pkl', 'encoders.pkl', and 'X_columns.pkl' are in the same folder.")
    st.stop()

# === Prediction Function ===
def predict_anxiety(user_input):
    try:
        input_df = pd.DataFrame([user_input])
        
        # Add missing columns with default values first
        for col in X_columns:
            if col not in input_df.columns:
                if col in encoders:
                    input_df[col] = encoders[col].classes_[0]
                else:
                    input_df[col] = 0

        # Reorder columns to match training data
        input_df = input_df.reindex(columns=X_columns, fill_value=0)

        # Process each column
        for col in input_df.columns:
            if col in encoders:
                try:
                    input_df[col] = input_df[col].astype(str)
                    known_classes = set(encoders[col].classes_)
                    input_classes = set(input_df[col])
                    unseen = input_classes - known_classes

                    if unseen:
                        encoders[col].classes_ = list(encoders[col].classes_) + list(unseen)

                    input_df[col] = encoders[col].transform(input_df[col])
                except Exception:
                    input_df[col] = 0
            else:
                try:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                    input_df[col] = input_df[col].fillna(0)
                except Exception:
                    input_df[col] = 0

        # Ensure all columns are numeric
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        # Make prediction
        prediction = model.predict(input_df)[0]
        return prediction
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# === Streamlit UI ===
st.set_page_config(page_title="Student Anxiety Predictor", layout="wide")
st.title("🎓 Student Anxiety Prediction App")

st.info("📋 **Instructions:** Please answer all questions honestly based on your experiences over the last 2 weeks. This assessment takes approximately 5-7 minutes to complete.")

with st.form("anxiety_form"):
    # === GAD-7 Assessment ===
    st.subheader("🧠 Part 1: GAD-7 - Generalized Anxiety Disorder Scale")
    st.markdown("**Over the last 2 weeks, how often have you been bothered by the following problems?**")
    st.caption("Scale: 0 = Not at all | 1 = Several days | 2 = More than half the days | 3 = Nearly every day")
    
    gad_questions = [
        "Feeling nervous, anxious, or on edge",
        "Not being able to stop or control worrying",
        "Worrying too much about different things",
        "Trouble relaxing",
        "Being so restless that it's hard to sit still",
        "Becoming easily annoyed or irritable",
        "Feeling afraid as if something awful might happen"
    ]
    
    gad_scores = []
    for i, question in enumerate(gad_questions, 1):
        score = st.slider(
            f"**GAD{i}:** {question}", 
            min_value=0, max_value=3, value=0, 
            format="%d",
            help="0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day"
        )
        gad_scores.append(score)
    
    st.divider()
    
    # === SWL Assessment ===
    st.subheader("😊 Part 2: SWL - Satisfaction With Life Scale")
    st.markdown("**Please indicate your agreement with each statement:**")
    st.caption("Scale: 1 = Strongly Disagree | 2 = Disagree | 3 = Slightly Disagree | 4 = Neutral | 5 = Slightly Agree | 6 = Agree | 7 = Strongly Agree")
    
    swl_questions = [
        "In most ways my life is close to my ideal",
        "The conditions of my life are excellent",
        "I am satisfied with my life",
        "So far I have gotten the important things I want in life",
        "If I could live my life over, I would change almost nothing"
    ]
    
    swl_scores = []
    for i, question in enumerate(swl_questions, 1):
        score = st.slider(
            f"**SWL{i}:** {question}", 
            min_value=1, max_value=7, value=4, 
            format="%d",
            help="1=Strongly Disagree to 7=Strongly Agree"
        )
        swl_scores.append(score)
    
    st.divider()
    
    # === SPIN Assessment ===
    st.subheader("👥 Part 3: SPIN - Social Phobia Inventory")
    st.markdown("**How much do the following problems bother you?**")
    st.caption("Scale: 0 = Not at all | 1 = A little bit | 2 = Somewhat | 3 = Very much | 4 = Extremely")
    
    spin_questions = [
        "I am afraid of people in authority",
        "I am bothered by blushing in front of people",
        "Parties and social events scare me",
        "I avoid talking to people I don't know",
        "Being criticized scares me a lot",
        "I avoid doing things or speaking to people for fear of embarrassment",
        "Sweating in front of people causes me distress",
        "I avoid going to parties",
        "I avoid activities in which I am the center of attention",
        "Talking to strangers scares me",
        "I avoid having to give speeches",
        "I would do anything to avoid being criticized",
        "Heart palpitations bother me when I am around people",
        "I am afraid of doing things when people might be watching",
        "Being embarrassed or looking stupid are among my worst fears",
        "I avoid speaking to anyone in authority",
        "Trembling or shaking in front of others is distressing to me"
    ]
    
    with st.expander("📋 Click to expand and complete SPIN questions (17 items)", expanded=False):
        spin_scores = []
        for i, question in enumerate(spin_questions, 1):
            score = st.slider(
                f"**SPIN{i}:** {question}", 
                min_value=0, max_value=4, value=0, 
                format="%d",
                help="0=Not at all to 4=Extremely"
            )
            spin_scores.append(score)
    
    st.divider()
    
    # === Narcissism ===
    st.subheader("💫 Part 4: Personality Assessment")
    narcissism = st.slider(
        "**I see myself as someone who is assertive and confident**", 
        min_value=0, max_value=5, value=2, 
        format="%d",
        help="0=Strongly Disagree to 5=Strongly Agree"
    )
    
    st.divider()
    
    # === Personal Information ===
    st.subheader("👤 Part 5: Personal Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "Age", 
            min_value=10, max_value=100, value=21, step=1,
            help="Enter your current age in years"
        )
        
        gender = st.selectbox(
            "Gender", 
            ["Male", "Female", "Other"],
            help="Select your gender identity"
        )
        
        work = st.selectbox(
            "Current Work/Study Status", 
            ["Student at college / university", "Student at school", "Employed", 
             "Unemployed / between jobs", "Retired"],
            help="What is your current primary occupation?"
        )
        
        degree = st.selectbox(
            "Highest Education Level Completed", 
            ["High school diploma (or equivalent)", "Bachelor (or equivalent)", 
             "Master (or equivalent)", "PhD (or equivalent)", "Other"],
            help="What is your highest completed education level?"
        )
    
    with col2:
        platform = st.selectbox(
            "Primary Gaming Platform", 
            ["PC", "Console", "Mobile", "Other"],
            help="Which platform do you mainly use for gaming?"
        )
        
        playstyle = st.selectbox(
            "Preferred Playstyle", 
            ["Singleplayer", "Multiplayer - online - with real life friends",
             "Multiplayer - online - with online acquaintances",
             "Multiplayer - online - with strangers", "Multiplayer - local"],
            help="How do you prefer to play games?"
        )
        
        accept = st.selectbox(
            "Terms and Consent", 
            ["Accept", "Decline"],
            help="Do you consent to participate in this assessment?"
        )
    
    st.divider()
    
    # === Gaming Information ===
    st.subheader("🎮 Part 6: Gaming Habits and Lifestyle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        game = st.selectbox(
            "Favorite Game Genre", 
            ["Other", "RPG", "FPS", "Strategy", "Puzzle", "Sports", 
             "Racing", "Fighting", "Adventure"],
            help="What type of games do you enjoy most?"
        )
        
        hours = st.number_input(
            "Hours per week spent gaming", 
            min_value=0.0, max_value=168.0, value=10.0, step=0.5,
            help="How many hours do you spend playing games per week?"
        )
        
        streams = st.number_input(
            "Hours per week watching gaming streams/content", 
            min_value=0.0, max_value=168.0, value=0.0, step=0.5,
            help="How many hours do you spend watching gaming-related content?"
        )
    
    with col2:
        earnings = st.selectbox(
            "Do you earn money from gaming?", 
            ["No", "Sometimes", "Yes"],
            help="Do you monetize your gaming through streaming, tournaments, etc.?"
        )
        
        whyplay = st.selectbox(
            "Primary Reason for Gaming", 
            ["Fun", "Stress relief", "Social", "Competition", "Achievement", "Other"],
            help="What is your main motivation for playing games?"
        )
        
        league = st.selectbox(
            "Competitive Gaming Rank/League", 
            ["Unranked", "Bronze", "Silver", "Gold", "Platinum", 
             "Diamond", "Master", "Grandmaster"],
            help="What is your rank in competitive gaming? (Select Unranked if not applicable)"
        )
    
    st.divider()
    
    submitted = st.form_submit_button("🔍 Predict Anxiety Level", use_container_width=True)

# === Results Display ===
if submitted:
    # Validate consent
    if accept == "Decline":
        st.error("❌ You must accept the terms to proceed with the assessment.")
        st.stop()
    
    user_input = {
        **{f"GAD{i+1}": gad_scores[i] for i in range(7)},
        **{f"SWL{i+1}": swl_scores[i] for i in range(5)},
        **{f"SPIN{i+1}": spin_scores[i] for i in range(17)},
        "Narcissism": narcissism,
        "Age": age,
        "Gender": gender,
        "Work": work,
        "Degree": degree,
        "Platform": platform,
        "Playstyle": playstyle,
        "accept": accept,
        "Game": game,
        "Hours": hours,
        "earnings": earnings,
        "whyplay": whyplay,
        "League": league,
        "streams": streams
    }

    with st.spinner("🔄 Analyzing your responses..."):
        prediction = predict_anxiety(user_input)

    st.divider()
    st.subheader("📊 Assessment Results")
    
    if prediction == 1:
        st.error("### 😟 Anxiety Indicators Detected")
        st.markdown("""
        Based on your responses, the model has identified patterns consistent with anxiety symptoms.
        
        **💡 Recommended Next Steps:**
        - **Talk to a professional:** Consider scheduling an appointment with a mental health counselor or therapist
        - **Reach out for support:** Connect with trusted friends, family members, or campus counseling services
        - **Practice self-care:** Engage in stress-reduction activities like exercise, meditation, or hobbies
        - **Emergency help:** If you're in crisis, contact a crisis helpline immediately
        
        **📞 Helpful Resources:**
        - **National Crisis Helpline:** Available 24/7 for immediate support
        - **Campus Counseling Services:** Free or low-cost services for students
        - **Online Therapy Platforms:** Accessible professional mental health support
        - **Support Groups:** Connect with others experiencing similar challenges
        
        ---
        
        ⚠️ **Important:** This is a screening tool, not a clinical diagnosis. Please consult with a qualified mental health professional for proper evaluation and treatment.
        """)
        
    elif prediction == 0:
        st.success("### 😊 No Significant Anxiety Indicators Detected")
        st.markdown("""
        Based on your responses, the model did not identify significant patterns associated with anxiety.
        
        **💡 Continue Supporting Your Mental Health:**
        - **Maintain healthy habits:** Keep up with exercise, sleep, and balanced nutrition
        - **Stay connected:** Nurture relationships with friends and family
        - **Engage in activities:** Continue doing things you enjoy and find meaningful
        - **Monitor yourself:** Be aware of changes in your mood or stress levels
        - **Seek help when needed:** Don't hesitate to reach out if things change
        
        **Remember:** Mental health exists on a spectrum and can fluctuate. Regular self-check-ins are valuable.
        """)
        
    else:
        st.warning("### ⚠️ Unable to Make Prediction")
        st.markdown("""
        The system was unable to generate a prediction. This may be due to:
        - Incomplete or inconsistent response patterns
        - Technical issues with the prediction model
        - Data processing errors
        
        **What to do:**
        - Review your responses and try submitting again
        - If the problem persists, contact support
        - Consider speaking with a mental health professional for a proper assessment
        """)
    
    st.divider()
    
    # Calculate and display basic scores
    with st.expander("📈 View Your Assessment Scores"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gad_total = sum(gad_scores)
            st.metric("GAD-7 Total Score", f"{gad_total}/21")
            st.caption("0-4: Minimal | 5-9: Mild | 10-14: Moderate | 15-21: Severe")
        
        with col2:
            swl_total = sum(swl_scores)
            st.metric("SWL Total Score", f"{swl_total}/35")
            st.caption("5-9: Extremely dissatisfied | 20: Neutral | 31-35: Extremely satisfied")
        
        with col3:
            spin_total = sum(spin_scores)
            st.metric("SPIN Total Score", f"{spin_total}/68")
            st.caption("0-20: Minimal | 21-30: Mild | 31-40: Moderate | 41+: Severe")
    
    st.divider()
    st.caption("""
    **⚠️ Disclaimer:** This tool is designed for screening and educational purposes only. It does not provide medical advice, 
    diagnosis, or treatment. The predictions are based on statistical patterns and should not replace professional medical or 
    psychological evaluation. If you have concerns about your mental health, please consult with a qualified healthcare provider.
    """)