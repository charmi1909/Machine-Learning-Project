from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import logging
import pandas as pd
import numpy as np

# ----------------------------
# App initialization
# ----------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FraudDetectionAPI")

# ----------------------------
# Model and preprocessing artifacts
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model_path = os.path.join(MODEL_DIR, "final_model_week5.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)

with open(os.path.join(MODEL_DIR, "categorical_columns.pkl"), "rb") as f:
    categorical_columns = pickle.load(f)

logger.info(f"Fraud Detection API Initialized | Model features: {len(feature_columns)} | Categorical: {len(categorical_columns)}")

# ----------------------------
# Health endpoint
# ----------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "model": "RandomForestClassifier (Week 5)",
        "features": len(feature_columns),
        "message": "Fraud Detection API - Send POST to /predict"
    })

# ----------------------------
# Categorical normalization rules
# ----------------------------
CATEGORICAL_NORMALIZE = {
    'accident_site': {'highway': 'Highway', 'local': 'Local', 'parking lot': 'Parking Lot'},
    'property_status': {'own': 'Own', 'rent': 'Rent'},
    'channel': {'broker': 'Broker', 'phone': 'Phone', 'online': 'Online'},
    'vehicle_category': {'compact': 'Compact', 'medium': 'Medium', 'large': 'Large'},
    'vehicle_color': {'grey': 'gray'},
}

# ----------------------------
# Input preprocessing
# ----------------------------
def preprocess_input(data):
    df = pd.DataFrame([data])
    df.columns = df.columns.str.replace(' ', '_')
    
    # Normalize categorical values
    for col in categorical_columns:
        if col in df.columns and col in CATEGORICAL_NORMALIZE:
            val = str(df[col].iloc[0]).strip().lower()
            if col == 'vehicle_color':
                df.loc[df.index[0], col] = val.replace('grey', 'gray')
            else:
                for k, v in CATEGORICAL_NORMALIZE[col].items():
                    if val == k.lower():
                        df.loc[df.index[0], col] = v
                        break

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            if col == 'claim_date':
                df[col] = pd.Timestamp.now()  
            elif col in categorical_columns:
                df[col] = 'Missing'  
            else:
                df[col] = 0  

    df = df[feature_columns]

    # Convert claim_date to timestamp
    if 'claim_date' in df.columns:
        df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
        df['claim_date'] = df['claim_date'].astype('int64') // 10**9

    # Encode categorical columns safely
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Missing').astype(str)
            le = label_encoders[col]
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    # Convert everything to numeric and fill missing
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)

    # Scale
    df_scaled = scaler.transform(df)
    return df_scaled

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.json
        if not payload:
            return jsonify({"error": "No JSON data received"}), 400
        data = payload.get("data", payload)
        logger.info(f"New prediction request: {list(data.keys())}")

        X_processed = preprocess_input(data)
        prediction = model.predict(X_processed)[0]
        prediction_label = "Fraud Detected" if prediction == 1 else "No Fraud Detected"
        probabilities = model.predict_proba(X_processed)[0]
        prob_dict = {str(cls): float(prob) for cls, prob in zip(model.classes_, probabilities)}
        confidence = float(max(probabilities))

        # Convert numpy types to native Python types
        result = {
            "prediction": prediction_label,
            "prediction_code": int(prediction),
            "confidence": round(confidence, 2),
            "probabilities": {k: round(v, 2) for k, v in prob_dict.items()}
        }

        logger.info(f"Prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        import traceback
        logger.error(f"ERROR in /predict: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))