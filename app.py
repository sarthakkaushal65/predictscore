import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore') # To suppress potential sklearn version warnings

# Set page config
st.set_page_config(page_title="Student Exam Score Predictor", layout="centered")

# Load the model
# Ensure 'student_score_model-3.pkl' is in the same directory as app.py on GitHub
try:
    model = joblib.load("student_score_model-3.pkl")
except FileNotFoundError:
    st.error("Error: Model file 'student_score_model-3.pkl' not found. Please ensure it's in your GitHub repository.")
    st.stop() # Stop the app if model isn't found

# Title
st.markdown("<h1 style='text-align: center; color: navy;'>🎓 Student Exam Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict a student's exam score based on various academic, behavioral, and demographic factors.</p>", unsafe_allow_html=True)

# Sidebar inputs for ALL 23 features in the correct order as per training data
st.sidebar.header("Input Student Details")

# Define mappings for categorical features (based on typical LabelEncoder behavior or your assumed ordinality)
gender_map = {"Male": 1, "Female": 0} # Assuming alphabetical or common mapping for LabelEncoder
yes_no_map = {"Yes": 1, "No": 0}
parental_support_map = {"None": 0, "Moderate": 1, "Strong": 2} # Adjusting order based on common LabelEncoder output (alphabetical)
family_income_map = {"Low": 0, "Medium": 1, "High": 2}


# --- Feature Inputs (arranged to match original training order as much as possible) ---

st.sidebar.subheader("Demographics & Home Environment")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 10, 20, 15)
distance_from_home = st.sidebar.slider("Distance from Home (km)", 0.0, 50.0, 5.0, 0.5)
family_income = st.sidebar.selectbox("Family Income", ["Low", "Medium", "High"])
internet_access = st.sidebar.selectbox("Internet Access at Home", ["Yes", "No"])
parental_support = st.sidebar.selectbox("Parental Support Level", ["None", "Moderate", "Strong"]) # Re-ordered for mapping consistency


st.sidebar.subheader("Academic Factors")
hours_studied = st.sidebar.slider("Hours Studied per Day", 0.0, 10.0, 2.0, 0.1)
previous_scores = st.sidebar.slider("Previous Exam Score (%)", 0, 100, 70)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 85)
class_participation = st.sidebar.slider("Class Participation (1-10)", 1, 10, 6)
assignments_submitted = st.sidebar.slider("Assignments Submitted (%)", 0, 100, 80)
sample_papers_solved = st.sidebar.slider("Sample Papers Solved", 0, 50, 5)
library_usage = st.sidebar.slider("Library Usage (Hours per Week)", 0.0, 20.0, 2.0, 0.5)


st.sidebar.subheader("Lifestyle & Activities")
sleep_hours = st.sidebar.slider("Sleep Hours per Night", 0.0, 10.0, 6.5, 0.5)
extracurricular_activities = st.sidebar.selectbox("Participates in Extracurricular Activities", ["Yes", "No"])
sports_participation = st.sidebar.selectbox("Participates in Sports", ["Yes", "No"])
health_issues = st.sidebar.selectbox("Has Health Issues", ["Yes", "No"])
mental_health_score = st.sidebar.slider("Mental Health Score (1-10)", 1, 10, 7) # Assuming higher is better
travel_time = st.sidebar.slider("Travel Time to School (minutes)", 0, 120, 15, 5)


st.sidebar.subheader("Additional Support")
tutoring = st.sidebar.selectbox("Receives Tutoring", ["Yes", "No"])
study_group = st.sidebar.selectbox("Studies in Group", ["Yes", "No"])
extra_classes = st.sidebar.selectbox("Takes Extra Classes", ["Yes", "No"])
access_to_resources = st.sidebar.selectbox("Access to Study Resources", ["Yes", "No"])


# --- Map categorical inputs to numerical values consistent with LabelEncoder ---
# IMPORTANT: These mappings assume LabelEncoder assigns 0/1 based on alphabetical order for Yes/No, and specific orders for others.
# If your LabelEncoder produced different mappings during training, adjust these.
gender_encoded = gender_map[gender]
extracurricular_activities_encoded = yes_no_map[extracurricular_activities]
tutoring_encoded = yes_no_map[tutoring]
sports_participation_encoded = yes_no_map[sports_participation]
health_issues_encoded = yes_no_map[health_issues]
internet_access_encoded = yes_no_map[internet_access]
study_group_encoded = yes_no_map[study_group]
extra_classes_encoded = yes_no_map[extra_classes]
access_to_resources_encoded = yes_no_map[access_to_resources]
family_income_encoded = family_income_map[family_income]
parental_support_encoded = parental_support_map[parental_support]


# Create input DataFrame with ALL features in the EXACT ORDER used during training
# This order is crucial for the model to interpret the inputs correctly.
input_data = pd.DataFrame({
    "Gender": [gender_encoded],
    "Age": [age],
    "Hours_Studied": [hours_studied],
    "Previous_Score": [previous_scores],
    "Extracurricular_Activities": [extracurricular_activities_encoded],
    "Sleep_Hours": [sleep_hours],
    "Sample_Papers_Solved": [sample_papers_solved],
    "Class_Participation": [class_participation],
    "Assignments_Submitted": [assignments_submitted],
    "Tutoring": [tutoring_encoded],
    "Parental_Support": [parental_support_encoded],
    "Sports_Participation": [sports_participation_encoded],
    "Health_Issues": [health_issues_encoded],
    "Internet_Access": [internet_access_encoded],
    "Study_Group": [study_group_encoded],
    "Travel_Time": [travel_time],
    "Library_Usage": [library_usage],
    "Family_Income": [family_income_encoded],
    "Distance_from_Home": [distance_from_home],
    "Extra_Classes": [extra_classes_encoded],
    "Attendance": [attendance],
    "Access_to_Resources": [access_to_resources_encoded],
    "Mental_Health_Score": [mental_health_score]
})

# Predict
if st.button("Predict Score"):
    prediction = model.predict(input_data)[0]
    st.success(f"📊 Predicted Exam Score: **{round(prediction, 2)}%**")

    # Visualization - Feature Impact (Using actual input data's values for relevant features)
    st.subheader("📈 Visual Interpretation of Key Inputs")

    # Select a few key features for visualization based on the inputs you have
    # If you had feature importances from the model, you'd use those.
    # For now, let's pick some directly from the user's input
    display_features = ['Hours_Studied', 'Previous_Score', 'Attendance', 'Class_Participation', 'Sleep_Hours']
    
    # Create a DataFrame for plotting, including the predicted score
    plot_df = input_data[display_features].iloc[0].to_frame().T
    plot_df['Predicted Score'] = prediction

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # You might want to plot these against something meaningful, or just show their values.
    # A bar plot of the input values for context is a simple approach.
    plot_data = plot_df.drop(columns=['Predicted Score']).iloc[0]
    sns.barplot(x=plot_data.index, y=plot_data.values, ax=ax, palette="viridis")
    ax.set_title("Input Values for Key Academic Factors")
    ax.set_ylabel("Value")
    ax.set_xlabel("Feature")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    st.markdown("---")
    st.info("💡 Note: The above visualization shows the input values for some key features that influence the prediction. A more advanced visualization could show how each input feature impacts the predicted score based on the model's logic (e.g., SHAP values), but that requires more complex setup.")
