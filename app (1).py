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
model = joblib.load("model/acceptance.pkl")
scaler = joblib.load("model/scaler.pkl")

# Title
st.title("🎓 Job Acceptance Prediction System")
st.write("Predict whether a candidate will be accepted for a job using Machine Learning.")

st.divider()

# Input fields (match training order!)
labels = [
    "SSC Percentage",
    "HSC Percentage",
    "Degree Percentage",
    "Employability Test Percentage",
    "MBA Percentage",
    "Years of Experience",
    "Skills Match Percentage",
    "Number of Certifications",
    "Internship Completed (0 = No, 1 = Yes)",
    "Interview Score"
]

inputs = []
for label in labels:
    inputs.append(st.number_input(label, step=0.1))

# Predict button
if st.button("🔍 Predict Job Acceptance"):
    X = np.array(inputs).reshape(1, -1)
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

st.caption("Final Year Machine Learning Project")
