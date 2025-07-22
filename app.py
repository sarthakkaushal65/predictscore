import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("student_score_model-3.pkl")

# Streamlit page config
st.set_page_config(page_title="Student Exam Score Predictor", page_icon="🎓", layout="centered")

# Title
st.title("🎯 Student Exam Score Predictor")
st.markdown("Enter student details below to predict their final exam score.")

# Input form
with st.form("prediction_form"):
    st.subheader("📋 Input Student Factors")

    Hours_Studied = st.slider("Hours Studied per Week", 1, 50, 20)
    Attendance = st.slider("Attendance (%)", 50, 100, 80)
    Parental_Involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
    Access_to_Resources = st.selectbox("Access to Resources", ["Poor", "Average", "Good"])
    Extracurricular_Activities = st.selectbox("Extracurricular Activities", ["None", "Some", "Active"])
    Sleep_Hours = st.slider("Sleep Hours per Day", 4, 10, 7)
    Previous_Scores = st.slider("Previous Score (%)", 50, 100, 75)
    Motivation_Level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
    Internet_Access = st.selectbox("Internet Access", ["No", "Yes"])
    Tutoring_Sessions = st.slider("Tutoring Sessions per Week", 0, 10, 2)
    Family_Income = st.selectbox("Family Income", ["Low", "Medium", "High"])
    Teacher_Quality = st.selectbox("Teacher Quality", ["Poor", "Average", "Excellent"])
    School_Type = st.selectbox("School Type", ["Public", "Private"])
    Peer_Influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
    Physical_Activity = st.slider("Physical Activity (hrs/week)", 0, 10, 3)
    Learning_Disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])
    Parental_Education_Level = st.selectbox("Parental Education", ["None", "High School", "Graduate", "Postgraduate"])
    Distance_from_Home = st.selectbox("Distance from Home", ["<1 km", "1-5 km", "5-10 km", ">10 km"])
    Gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    submit = st.form_submit_button("🎓 Predict Score")

# Label encoding map (based on previous training)
def encode_inputs():
    return {
        "Parental_Involvement": {"Low": 0, "Medium": 1, "High": 2},
        "Access_to_Resources": {"Poor": 0, "Average": 1, "Good": 2},
        "Extracurricular_Activities": {"None": 0, "Some": 1, "Active": 2},
        "Motivation_Level": {"Low": 0, "Medium": 1, "High": 2},
        "Internet_Access": {"No": 0, "Yes": 1},
        "Family_Income": {"Low": 0, "Medium": 1, "High": 2},
        "Teacher_Quality": {"Poor": 0, "Average": 1, "Excellent": 2},
        "School_Type": {"Public": 0, "Private": 1},
        "Peer_Influence": {"Negative": 0, "Neutral": 1, "Positive": 2},
        "Learning_Disabilities": {"No": 0, "Yes": 1},
        "Parental_Education_Level": {"None": 0, "High School": 1, "Graduate": 2, "Postgraduate": 3},
        "Distance_from_Home": {"<1 km": 0, "1-5 km": 1, "5-10 km": 2, ">10 km": 3},
        "Gender": {"Male": 0, "Female": 1, "Other": 2}
    }

# Prediction
if submit:
    encoders = encode_inputs()

    input_data = pd.DataFrame({
        "Hours_Studied": [Hours_Studied],
        "Attendance": [Attendance],
        "Parental_Involvement": [encoders["Parental_Involvement"][Parental_Involvement]],
        "Access_to_Resources": [encoders["Access_to_Resources"][Access_to_Resources]],
        "Extracurricular_Activities": [encoders["Extracurricular_Activities"][Extracurricular_Activities]],
        "Sleep_Hours": [Sleep_Hours],
        "Previous_Scores": [Previous_Scores],
        "Motivation_Level": [encoders["Motivation_Level"][Motivation_Level]],
        "Internet_Access": [encoders["Internet_Access"][Internet_Access]],
        "Tutoring_Sessions": [Tutoring_Sessions],
        "Family_Income": [encoders["Family_Income"][Family_Income]],
        "Teacher_Quality": [encoders["Teacher_Quality"][Teacher_Quality]],
        "School_Type": [encoders["School_Type"][School_Type]],
        "Peer_Influence": [encoders["Peer_Influence"][Peer_Influence]],
        "Physical_Activity": [Physical_Activity],
        "Learning_Disabilities": [encoders["Learning_Disabilities"][Learning_Disabilities]],
        "Parental_Education_Level": [encoders["Parental_Education_Level"][Parental_Education_Level]],
        "Distance_from_Home": [encoders["Distance_from_Home"][Distance_from_Home]],
        "Gender": [encoders["Gender"][Gender]]
    })

    predicted_score = model.predict(input_data)[0]
    st.success(f"✅ Predicted Exam Score: **{predicted_score:.2f}** out of 100")
