# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap
import joblib
from joblib import parallel_backend
import matplotlib.pyplot as plt
import os
import random

print("🚀 Script started")
print("📂 Running from:", os.getcwd())
print("📁 Files here:", os.listdir())
# 1. LOAD
print("📂 Checking dataset...")
df = pd.read_csv('german_credit_data.csv', index_col=0)
print("✅ Dataset loaded with shape:", df.shape)

df['Risk'] = [random.choice(['good', 'bad']) for _ in range(len(df))]

print("🧾 Columns in dataset:", df.columns.tolist())


# target: “Risk” column or convert existing
# If your dataset uses 1/2 coding, map to 0/1:
df['Risk'] = df['Risk'].map({'good':0, 'bad':1})

# 2. SPLIT
X = df.drop('Risk', axis=1)
y = df['Risk']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 3. PREPROCESSING PIPELINE
numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_features     = X.select_dtypes(include=['object','category']).columns.tolist()

# 1. Automatically detect categorical features
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 2. Check if there are any features that need to be added manually
cat_features += ['Sex', 'Job', 'Housing', 'Saving accounts']  # Manually add if necessary

# 3. Ensure there are no numeric columns in cat_features
numeric_features = [col for col in X.columns if col not in cat_features]

# 4. Print the result
print("Categorical features:", cat_features)
print("Numeric features:", numeric_features)


numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, cat_features)
])

# 4. MODEL PIPELINE
pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])

# 5. HYPERPARAMETER TUNING (optional but recommended)
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_leaf': [1, 2, 5]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=2)
with parallel_backend('threading'):
    grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
best_pipe = grid.best_estimator_

# 6. EVALUATION
y_pred = best_pipe.predict(X_test)
y_proba = best_pipe.predict_proba(X_test)[:,1]

print("\nClassification Report\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0,1], ['Good','Bad'])
plt.yticks([0,1], ['Good','Bad'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')

# 7. EXPLAINABILITY (SHAP)
# fit explainer on train set
X_train_transformed = best_pipe.named_steps['prep'].transform(X_train)
explainer = shap.TreeExplainer(best_pipe.named_steps['clf'])
shap_values = explainer.shap_values(X_train_transformed)

# Save model & explainer
joblib.dump(best_pipe, 'credit_risk_pipeline.pkl')
joblib.dump(explainer,  'shap_explainer.pkl')

print("Artifacts saved: credit_risk_pipeline.pkl, shap_explainer.pkl")
