import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="🎯",
    layout="centered"
)

# Load model & scaler
model = joblib.load("job_acceptance_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🎓 Job Acceptance Prediction System")
st.write("Predict whether a candidate will be accepted for a job using Machine Learning.")

st.divider()

# Inputs (EXACT training order)
ssc = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, step=0.1)
hsc = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, step=0.1)
degree = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, step=0.1)
emp_test = st.number_input("Employability Test Percentage", min_value=0.0, max_value=100.0, step=0.1)
mba = st.number_input("MBA Percentage", min_value=0.0, max_value=100.0, step=0.1)
experience = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, step=0.5)
skills = st.number_input("Skills Match Percentage", min_value=0.0, max_value=100.0, step=0.1)
certs = st.number_input("Number of Certifications", min_value=0, max_value=20, step=1)
internship = st.selectbox("Internship Completed", [0, 1])
interview = st.number_input("Interview Score", min_value=0.0, max_value=100.0, step=0.1)

# Predict
if st.button("🔍 Predict Job Acceptance"):
    X = np.array([
        ssc,
        hsc,
        degree,
        emp_test,
        mba,
        experience,
        skills,
        certs,
        internship,
        interview
    ]).reshape(1, -1)

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]

    st.divider()

    if prediction == 1:
        st.success("✅ Candidate is likely to be ACCEPTED")
    else:
        st.error("❌ Candidate is likely to be REJECTED")

    st.subheader("📊 Prediction Probability")
    st.bar_chart({
        "Rejected": probability[0],
        "Accepted": probability[1]
    })

st.caption("Final Machine Learning Project")
