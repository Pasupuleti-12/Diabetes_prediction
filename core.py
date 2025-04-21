# core.py - ML Model Training and Saving

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv('diabetes.csv')

# Preprocessing function
def preprocess_data(df):
    # Replace zeros with NaN then fill with median values
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_features] = df[zero_features].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    
    # Feature engineering: Adding interaction terms if needed
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    return df

df = preprocess_data(df)

# Split data into features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
)
model.fit(X_train_scaled, y_train)

# Save model and scaler to disk
joblib.dump(model, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model trained and saved successfully.")
