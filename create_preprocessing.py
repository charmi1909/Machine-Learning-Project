
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

print("="*80)
print("RECREATING PREPROCESSING PIPELINE")
print("="*80)

df_original = pd.read_csv('data/insurance_fraud_data.csv')
print(f"\n1. Loaded original data: {df_original.shape}")

df_original.columns = df_original.columns.str.replace(' ', '_')
print(f"   Fixed column names: {list(df_original.columns)}")

df_preprocessed = pd.read_csv('data/pre_processed_data.csv')
print(f"\n2. Loaded preprocessed data: {df_preprocessed.shape}")

X_train_scaled = pd.read_csv('data/X_train_week4.csv')
X_test_scaled = pd.read_csv('data/X_test_week4.csv')
print(f"\n3. Loaded scaled training data: X_train={X_train_scaled.shape}, X_test={X_test_scaled.shape}")

categorical_cols = ['gender', 'marital_status', 'property_status', 'claim_day_of_week', 
                   'accident_site', 'channel', 'vehicle_category', 'vehicle_color']
print(f"\n4. Categorical columns: {categorical_cols}")

print(f"\n5. Analyzing categorical encodings...")
label_encoders = {}

for col in categorical_cols:
    if col in df_preprocessed.columns:
        unique_original = df_original[col].unique()
        unique_preprocessed = df_preprocessed[col].unique()
        
        print(f"\n   {col}:")
        print(f"      Original: {unique_original}")
        print(f"      Preprocessed: {unique_preprocessed}")
        
        le = LabelEncoder()
        df_preprocessed_clean = df_preprocessed[col].fillna('Missing').astype(str)
        le.fit(df_preprocessed_clean)
        label_encoders[col] = le
        print(f"      Encoded mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

feature_cols = [col for col in df_preprocessed.columns if col != 'fraud_reported']
print(f"\n6. Feature columns ({len(feature_cols)}): {feature_cols}")

X = df_preprocessed[feature_cols].copy()
y = df_preprocessed['fraud_reported'].copy()

print(f"\n7. Encoding categorical columns...")
X_encoded = X.copy()
for col in categorical_cols:
    if col in X_encoded.columns:
        X_encoded[col] = X_encoded[col].fillna('Missing').astype(str)
        X_encoded[col] = label_encoders[col].transform(X_encoded[col])

if 'claim_date' in X_encoded.columns:
    print(f"\n   Converting claim_date to numeric...")
    X_encoded['claim_date'] = pd.to_datetime(X_encoded['claim_date'], errors='coerce')
    X_encoded['claim_date'] = X_encoded['claim_date'].astype('int64') // 10**9  # Convert to Unix timestamp
    print(f"      claim_date converted to timestamp")

print(f"\n   Converting all columns to numeric...")
for col in X_encoded.columns:
    X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')

X_encoded = X_encoded.fillna(0)

print(f"   Encoded data shape: {X_encoded.shape}")
print(f"   Data types: {X_encoded.dtypes.value_counts()}")
print(f"   Sample encoded values:")
print(X_encoded.iloc[0])

print(f"\n8. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")

print(f"\n9. Fitting StandardScaler...")
scaler = StandardScaler()
X_train_scaled_new = scaler.fit_transform(X_train)
X_test_scaled_new = scaler.transform(X_test)

print(f"   Scaled data - mean: {X_train_scaled_new.mean():.6f}, std: {X_train_scaled_new.std():.6f}")
print(f"   Sample scaled values (first row):")
print(X_train_scaled_new[0])

print(f"\n10. Comparing with existing scaled data...")
print(f"    Existing scaled mean: {X_train_scaled.values.mean():.6f}, std: {X_train_scaled.values.std():.6f}")
print(f"    Shape match: X_train={X_train_scaled.shape == X_train_scaled_new.shape}, X_test={X_test_scaled.shape == X_test_scaled_new.shape}")

print(f"\n11. Saving preprocessing artifacts...")

with open('flask-backend/model/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"    ✓ Saved label_encoders.pkl")

with open('flask-backend/model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"    ✓ Saved scaler.pkl")

with open('flask-backend/model/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"    ✓ Saved feature_columns.pkl")

with open('flask-backend/model/categorical_columns.pkl', 'wb') as f:
    pickle.dump(categorical_cols, f)
print(f"    ✓ Saved categorical_columns.pkl")

print(f"\n{'='*80}")
print("PREPROCESSING ARTIFACTS CREATED SUCCESSFULLY!")
print("="*80)
print("\nFiles created:")
print("  - flask-backend/model/label_encoders.pkl")
print("  - flask-backend/model/scaler.pkl")
print("  - flask-backend/model/feature_columns.pkl")
print("  - flask-backend/model/categorical_columns.pkl") 