import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('flask-backend/model/final_model_week5.pkl', 'rb'))
model_columns = pickle.load(open('flask-backend/model/model_columns.pkl', 'rb'))

print("="*60)
print("MODEL ANALYSIS")
print("="*60)
print(f"Model type: {type(model)}")
print(f"Expected features: {model.n_features_in_}")
print(f"\nModel columns ({len(model_columns)}):")
print(model_columns)

df = pd.read_csv('data/insurance_fraud_data.csv')
print(f"\n{'='*60}")
print(f"DATASET INFO")
print(f"{'='*60}")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

print(f"\n{'='*60}")
print(f"DATA TYPES IN TRAINING DATA")
print(f"{'='*60}")
print(df.dtypes)

categorical_cols = ['gender', 'marital_status', 'property_status', 'claim_day_of_week', 
                   'accident_site', 'channel', 'vehicle_category', 'vehicle_color']
print(f"\n{'='*60}")
print(f"CATEGORICAL COLUMN VALUES")
print(f"{'='*60}")
for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col}: {df[col].unique()}")

print(f"\n{'='*60}")
print(f"TESTING PREDICTION")
print(f"{'='*60}")
sample = df.iloc[0:1].copy()
sample = sample.drop(columns=['fraud reported'], errors='ignore')

print(f"\nSample data:")
print(sample.iloc[0].to_dict())

X_test = sample[model_columns]
print(f"\nInput shape: {X_test.shape}")
print(f"Input columns: {list(X_test.columns)}")

try:
    prediction = model.predict(X_test)
    proba = model.predict_proba(X_test)
    print(f"\nPrediction: {prediction[0]}")
    print(f"Probability: {proba[0]}")
    print(f"Confidence: {max(proba[0])}")
except Exception as e:
    print(f"\nERROR during prediction: {e}")
