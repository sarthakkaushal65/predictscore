import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Student Exam Score Predictor", layout="centered")

# Load the model
model = joblib.load("student_score_model-3.pkl")

# Title
st.markdown("<h1 style='text-align: center; color: navy;'>🎓 Student Exam Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict a student's exam score based on key academic and behavioral factors.</p>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Input Student Details")

hours_studied = st.sidebar.slider("Hours Studied per Day", 0.0, 10.0, 2.0, 0.1)
previous_scores = st.sidebar.slider("Previous Exam Score (%)", 0, 100, 70)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 85)
participation = st.sidebar.slider("Class Participation (1-10)", 1, 10, 6)
sleep_hours = st.sidebar.slider("Sleep Hours per Night", 0.0, 10.0, 6.5, 0.5)
assignments_submitted = st.sidebar.slider("Assignments Submitted (%)", 0, 100, 80)
internet_access = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
study_group = st.sidebar.selectbox("Studies in Group", ["Yes", "No"])
parental_support = st.sidebar.selectbox("Parental Support", ["Strong", "Moderate", "None"])
extra_classes = st.sidebar.selectbox("Takes Extra Classes", ["Yes", "No"])

# Map categorical inputs
internet_access_bin = 1 if internet_access == "Yes" else 0
study_group_bin = 1 if study_group == "Yes" else 0
parental_support_map = {"Strong": 2, "Moderate": 1, "None": 0}
extra_classes_bin = 1 if extra_classes == "Yes" else 0

# Create input DataFrame
input_data = pd.DataFrame({
    "Hours_Studied": [hours_studied],
    "Previous_Score": [previous_scores],
    "Attendance": [attendance],
    "Class_Participation": [participation],
    "Sleep_Hours": [sleep_hours],
    "Assignments_Submitted": [assignments_submitted],
    "Internet_Access": [internet_access_bin],
    "Study_Group": [study_group_bin],
    "Parental_Support": [parental_support_map[parental_support]],
    "Extra_Classes": [extra_classes_bin]
})

# Predict
if st.button("Predict Score"):
    prediction = model.predict(input_data)[0]
    st.success(f"📊 Predicted Exam Score: **{round(prediction, 2)}%**")

    # Visualization
    st.subheader("📈 Visual Interpretation")
    fig, ax = plt.subplots()
    sns.barplot(x=input_data.columns, y=input_data.values[0], palette="coolwarm", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>SARTHAK KA AAVISHKAR 😎</h5>", unsafe_allow_html=True)
