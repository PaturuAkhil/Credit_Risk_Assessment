# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load artifacts
pipe      = joblib.load('credit_risk_pipeline.pkl')
explainer = joblib.load('shap_explainer.pkl')

st.title("Credit Risk Prediction Demo")

st.markdown("Enter applicant details on the left, then click **Predict**.")

required_columns = ['Duration', 'Credit amount', 'Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

# 1. Build input form
with st.sidebar.form("app_form"):
    # numeric inputs
    duration = st.number_input("Duration (months)", min_value=1, max_value=1000, value=12)
    credit_amt = st.number_input("Credit Amount", min_value=100, max_value=100000, value=5000)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    # categorical examples (repeat for your cat_features)
    sex = st.selectbox("Sex", ['male','female'])
    job = st.selectbox("Job Type", ['unemployed','unskilled','skilled','management'])
    # … add all other required fields …

    submitted = st.form_submit_button("Predict")

if submitted:
    # 2. Collect into dataframe
    input_df = pd.DataFrame([{
        'Duration': duration,
        'Credit amount': credit_amt,
        'Age': age,
        'Sex': sex,
        'Job': job,
        # … include all features exactly as in training …
        'Housing': 'own',  # Example: default value for Housing
        'Saving accounts': 'little',  # Example: default value for Saving accounts
        'Checking account': 'little',  # Example: default value for Checking account
        'Purpose': 'radio/tv'  # Example: default value for Purpose
    }])

    input_df = input_df.reindex(columns=required_columns, fill_value='unknown')  # Use 'unknown' or another placeholder

    # 3. Predict
    pred_class = pipe.predict(input_df)[0]
    pred_proba  = pipe.predict_proba(input_df)[0,1]

    st.write("### Prediction")
    st.write("**Credit Risk:**", "🔴 Bad" if pred_class else "🟢 Good")
    st.write(f"**Probability of Bad Risk:** {pred_proba:.2%}")
