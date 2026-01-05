# Job Acceptance Prediction System

## Project Description
A machine learning application that predicts whether a candidate will be accepted for a job based on academic and professional attributes.

## Project Objective
To build a predictive model that assists in job acceptance decision-making using machine learning techniques.

## Dataset
- Custom CSV dataset
- Target variable: Job Acceptance (Accepted / Rejected)

## Machine Learning Model
- Algorithm: Random Forest Classifier
- Feature Scaling: StandardScaler
- Model saved using Joblib

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib

## Project Structure
job-acceptance-prediction/
├── app.py
├── requirements.txt
├── README.md
├── job_acceptance_model.pkl
└── scaler.pkl
## How to Run
pip install -r requirements.txt
streamlit run app.py

## Deployment
Deployed using Streamlit Cloud.

## Author
Muhammad Faizan
