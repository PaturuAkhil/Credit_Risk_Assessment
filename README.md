# Credit_Risk_Assessment
The Credit Risk Prediction System uses machine learning to assess loan applicants' creditworthiness. It predicts Good or Bad credit risk based on features like loan duration, credit amount, and age. Built with Streamlit, Scikit-learn etc..., it provides accurate, explainable predictions.
This project implements a machine learning based system to predict the credit risk of loan applicants. The system evaluates whether an applicant is likely to have Good or Bad credit based on several features such as loan duration, credit amount, age, and job type.

Features:
Data Preprocessing: Handles both numeric and categorical features, with missing value imputation and scaling for numerical data.

Random Forest Classifier: A robust machine learning model is used for predicting credit risk.

Streamlit Interface: A user-friendly web interface for entering applicant details and receiving instant predictions.

Hyperparameter Tuning: The model undergoes hyperparameter optimization to enhance accuracy.

How it Works:
Input: Users input details about a loan applicant, such as duration, credit amount, age, and job type, into a simple form on the web interface.

Processing: The system processes the input using a pre-trained Random Forest model, which was trained on the German Credit dataset.

Prediction: The model predicts whether the applicant is a Good or Bad credit risk.

Output: The system displays the credit risk status and the probability of the applicant being a Bad credit risk.

Technologies Used:
Python: Core language for model development.

Scikit-learn: Used for machine learning, including preprocessing, training, and prediction.

Streamlit: For building the interactive web application.

Joblib: For saving and loading the trained model.
