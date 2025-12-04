# predict.py
import pandas as pd
import json
import os
import pickle
from sklearn.preprocessing import LabelEncoder

def load_model(path: str):
    """Load a trained model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model file not found at {path}")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"📦 Model loaded from {path}")
    return model

def encode_non_numeric(df):
    """Encode categorical columns."""
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    return df

def align_features(new_data: pd.DataFrame, feature_json_path: str):
    """Ensure new data has same columns and order as during training."""
    if not os.path.exists(feature_json_path):
        raise FileNotFoundError(f"❌ Feature file not found at {feature_json_path}")
    with open(feature_json_path, 'r') as f:
        train_features = json.load(f)

    # Add any missing columns with 0
    for col in train_features:
        if col not in new_data.columns:
            new_data[col] = 0

    # Keep same column order
    new_data = new_data[train_features]
    return new_data

def predict_new_data(input_data: pd.DataFrame,
                     model_path='model.pkl',
                     feature_path=None):
    """Predict readmission for new data."""
    model = load_model(model_path)
    if feature_path is None:
        feature_path = model_path.replace('.pkl', '_features.json')

    df = encode_non_numeric(input_data)
    df = align_features(df, feature_path)

    preds = model.predict(df)
    return preds

if __name__ == "__main__":
    # 🧾 Example input (replace with your own patient data)
    new_data = pd.DataFrame([{
        'race': 'Asian',
        'gender': 'Male',
        'age_group': '[50-60)',
        'time_in_hospital': 3,
        'num_lab_procedures': 40,
        'num_medications': 10,
        'number_diagnoses': 5
        # ⚠️ Include all other important columns you used in training
    }])

    preds = predict_new_data(new_data,
                             model_path='model.pkl',
                             feature_path='model_features.json')

    label_map = {
        0: "Not Readmitted",
        1: "Readmitted within 30 days",
        2: "Readmitted after 30 days"
    }

    decoded_preds = [label_map.get(p, "Unknown") for p in preds]
    print(f"\n🔮 Predicted Readmission: {decoded_preds[0]}")
