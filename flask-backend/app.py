from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

model_path = os.path.join("model", "final_model_week5.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(os.path.join("model", "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

with open(os.path.join("model", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join("model", "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)

with open(os.path.join("model", "categorical_columns.pkl"), "rb") as f:
    categorical_columns = pickle.load(f)

print("\n" + "="*60)
print("FRAUD DETECTION API INITIALIZED")
print("="*60)
print(f"Model: RandomForestClassifier")
print(f"Expected features: {len(feature_columns)}")
print(f"Categorical columns: {len(categorical_columns)}")
print("="*60 + "\n")


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "model": "RandomForestClassifier (Week 5)",
        "features": len(feature_columns),
        "message": "Fraud Detection API - Send POST to /predict"
    })


CATEGORICAL_NORMALIZE = {
    'accident_site': {'highway': 'Highway', 'local': 'Local', 'parking lot': 'Parking Lot'},
    'property_status': {'own': 'Own', 'rent': 'Rent'},
    'channel': {'broker': 'Broker', 'phone': 'Phone', 'online': 'Online'},
    'vehicle_category': {'compact': 'Compact', 'medium': 'Medium', 'large': 'Large'},
    'vehicle_color': {'grey': 'gray'},
}

def preprocess_input(data):
    df = pd.DataFrame([data])
    df.columns = df.columns.str.replace(' ', '_')
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
    for col in feature_columns:
        if col not in df.columns:
            if col == 'claim_date':
                df[col] = pd.Timestamp.now()  
            elif col in categorical_columns:
                df[col] = 'Missing'  
            else:
                df[col] = 0  
    
    df = df[feature_columns]
    
    if 'claim_date' in df.columns:
        df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
        df['claim_date'] = df['claim_date'].astype('int64') // 10**9
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Missing').astype(str)
            
            if df[col].iloc[0] in label_encoders[col].classes_:
                df[col] = label_encoders[col].transform(df[col])
            else:
                print(f"Warning: Unknown value '{df[col].iloc[0]}' for {col}, using default")
                df[col] = 0
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.fillna(0)
    
    df_scaled = scaler.transform(df)
    
    return df_scaled


@app.route("/predict", methods=["POST"])
def predict():
    
    try:
        payload = request.json
        if not payload:
            return jsonify({"error": "No JSON data received"}), 400
        data = payload.get("data", payload)
        print(f"\n{'='*60}")
        print("NEW PREDICTION REQUEST")
        print(f"{'='*60}")
        print(f"Input data keys: {list(data.keys())}")
        X_processed = preprocess_input(data)
        print(f"Preprocessed shape: {X_processed.shape}")
        print(f"Sample values: {X_processed[0][:5]}...")
        
        prediction = model.predict(X_processed)[0]
        prediction_label = "Fraud Detected" if prediction == 1 else "No Fraud Detected"
        probabilities = model.predict_proba(X_processed)[0]
        
        classes = model.classes_
        prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
        confidence = float(max(probabilities))
        
        result = {
    "prediction": prediction_label,
    "prediction_code": int(prediction), 
    "confidence": round(confidence, 2),
    "probabilities": {k: round(v, 2) for k, v in prob_dict.items()}
}


        
        print(f"\nResult: {result}")
        print(f"{'='*60}\n")
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"\nERROR in /predict:")
        print(error_trace)
        return jsonify({
            "error": error_msg,
            "traceback": error_trace
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))