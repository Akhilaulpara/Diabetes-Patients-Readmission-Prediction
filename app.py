import os
import json
import pickle
from io import BytesIO

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request, redirect, url_for, flash, send_file

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-key"

# Local sample CSV path (from conversation)
SAMPLE_DATA_PATH = "/mnt/data/diabetic_data.csv"

MODELS_DIR = "models"

# helper to load model + its features
def load_model_and_features(name):
    model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    feat_path = os.path.join(MODELS_DIR, f"{name}_features.json")
    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        raise FileNotFoundError(f"Missing files for {name}")
    model = pickle.load(open(model_path, "rb"))
    features = json.load(open(feat_path, "r"))
    # if JSON wrapped as {"features": [...]} handle it
    if isinstance(features, dict) and "features" in features:
        features = features["features"]
    return model, features

# load available models
AVAILABLE = {}
for nm in ["logistic_regression", "random_forest", "xgboost"]:
    try:
        AVAILABLE[nm] = {}
        AVAILABLE[nm]["model"], AVAILABLE[nm]["features"] = load_model_and_features(nm)
    except Exception as e:
        AVAILABLE.pop(nm, None)
        print(f"Model {nm} not loaded: {e}")

# ---------- Preprocessing helpers ----------
def safe_div(a, b):
    # elementwise safe division; returns 0 when denominator is 0 or NaN
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    out = np.zeros_like(a, dtype=float)
    mask = (b != 0) & (~np.isnan(b))
    out[mask] = a[mask] / b[mask]
    return out

def encode_readmitted(series):
    # map string labels to binary 0/1
    if series.dtype == object:
        return series.replace({ "NO": 0, ">30": 1, "<30": 1 }).astype(int)
    # if already numeric but maybe encoded differently, try to map >0 ->1
    return (series.astype(float) != 0).astype(int)

def build_engineered_features(df_raw, required_features):
    """
    From raw df columns (as you pasted), create a DataFrame with the required_features.
    Missing columns will be created with zeros.
    Derived features implemented:
      - procedure_density = num_procedures / time_in_hospital
      - lab_density = num_lab_procedures / time_in_hospital
      - med_diag_ratio = num_medications / number_diagnoses
      - total_interactions = number_outpatient + number_emergency + number_inpatient
      - hospital_stay_intensity = time_in_hospital * patient_severity_index
      - avg_procedures_per_day = num_procedures / time_in_hospital
      - avg_medications_per_day = num_medications / time_in_hospital
      - interaction_score = total_interactions / time_in_hospital
      - complexity_ratio = med_complexity_score / (patient_severity_index or 1)
    """
    df = df_raw.copy()
    # ensure numeric columns exist (create if missing)
    numeric_cols = [
        "num_procedures", "time_in_hospital", "num_lab_procedures", "num_medications",
        "number_diagnoses", "number_outpatient", "number_emergency", "number_inpatient",
        "patient_severity_index", "med_complexity_score"
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0
    # replace missing numeric-like values with 0
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Derived features
    df["procedure_density"] = safe_div(df["num_procedures"], df["time_in_hospital"])
    df["lab_density"] = safe_div(df["num_lab_procedures"], df["time_in_hospital"])
    df["med_diag_ratio"] = safe_div(df["num_medications"], df["number_diagnoses"])
    df["total_interactions"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    df["hospital_stay_intensity"] = df["time_in_hospital"] * df["patient_severity_index"]
    df["avg_procedures_per_day"] = safe_div(df["num_procedures"], df["time_in_hospital"])
    df["avg_medications_per_day"] = safe_div(df["num_medications"], df["time_in_hospital"])
    df["interaction_score"] = safe_div(df["total_interactions"], df["time_in_hospital"])
    # avoid division by zero
    df["complexity_ratio"] = safe_div(df["med_complexity_score"], df["patient_severity_index"].replace(0, np.nan).fillna(1))

    # For categorical features expected by model (race, gender, age categories, diag groups, medical_specialty, change, diabetesMed),
    # we will keep the raw values. If the model expected one-hot encoded columns, those must already be present in model_features json.
    # Build final DataFrame with required features in order
    final = pd.DataFrame(index=df.index)
    for feat in required_features:
        if feat in df.columns:
            final[feat] = df[feat]
        else:
            # not present: create zero column (model may expect dummy vars; can't infer easily here)
            final[feat] = 0
    # ensure numeric columns are numeric where relevant
    final = final.apply(lambda s: pd.to_numeric(s, errors="ignore"))
    return final

# ---------- Accuracy calculation ----------
def compute_accuracy_on_sample(model, features):
    """
    Attempt to compute accuracy using SAMPLE_DATA_PATH and ground truth column 'readmitted'.
    Returns accuracy (float) or 'N/A' if unavailable.
    """
    if not os.path.exists(SAMPLE_DATA_PATH):
        return "N/A"
    try:
        df = pd.read_csv(SAMPLE_DATA_PATH)
    except Exception:
        return "N/A"
    if "readmitted" not in df.columns:
        return "N/A"
    y = encode_readmitted(df["readmitted"])
    X = build_engineered_features(df, features)
    # model prediction
    try:
        if hasattr(AVAILABLE_MODEL := None, "__bool__"):  # no-op to satisfy linter
            pass
    except Exception:
        pass
    try:
        if hasattr(model := None, "predict"):
            pass
    except Exception:
        pass
    try:
        model = None  # placeholder
        # we will rely on caller providing model; this function will be called with the model below.
    except Exception:
        pass
    return "N/A"  # caller will compute with model directly


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", sample_path=SAMPLE_DATA_PATH, models=list(AVAILABLE.keys()))

@app.route("/predict", methods=["POST"])
def predict():
    model_name = request.form.get("model")
    use_sample = request.form.get("use_sample", "false").lower() == "true"

    if model_name not in AVAILABLE:
        flash("Selected model is not available on server.", "danger")
        return redirect(url_for("index"))

    model = AVAILABLE[model_name]["model"]
    features = AVAILABLE[model_name]["features"]

    # read dataframe
    if use_sample:
        if not os.path.exists(SAMPLE_DATA_PATH):
            flash("Sample dataset not found on server.", "danger")
            return redirect(url_for("index"))
        df_raw = pd.read_csv(SAMPLE_DATA_PATH)
    else:
        uploaded = request.files.get("file")
        if not uploaded:
            flash("Please upload a CSV file or use the sample dataset.", "warning")
            return redirect(url_for("index"))
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as e:
            flash(f"Failed to read uploaded CSV: {e}", "danger")
            return redirect(url_for("index"))

    # prepare features for model
    X = build_engineered_features(df_raw, features)

    # Predict probabilities if possible
    try:
        if hasattr(model, "predict_proba"):
            preds_proba = model.predict_proba(X)[:, 1]
        else:
            preds_proba = model.predict(X).astype(float)
    except Exception as e:
        flash(f"Model prediction failed: {e}", "danger")
        return redirect(url_for("index"))

    df_out = df_raw.copy()
    df_out["readmission_probability"] = np.round(preds_proba, 6)

    # sample rows for display
    sample_rows = df_out.head(10).to_dict(orient="records")

    # compute accuracy if sample has ground truth
    accuracy = "N/A"
    if "readmitted" in df_raw.columns:
        try:
            y_true = encode_readmitted(df_raw["readmitted"])
            # re-evaluate predictions to 0/1 with threshold 0.5
            y_pred = (preds_proba >= 0.5).astype(int)
            accuracy = round(accuracy_score(y_true, y_pred), 4)
        except Exception:
            accuracy = "N/A"

    # feature importance extraction
    fi_list = []
    try:
        if hasattr(model, "coef_"):
            coef = np.array(model.coef_).ravel()
            fi_df = pd.DataFrame({"feature": features, "importance": np.abs(coef)})
        elif hasattr(model, "feature_importances_"):
            fi_df = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        elif hasattr(model, "get_booster"):  # xgboost
            booster = model.get_booster()
            score = booster.get_score(importance_type="gain")
            # map f0 -> feature name
            imp_vals = [score.get(f"f{i}", 0) for i in range(len(features))]
            fi_df = pd.DataFrame({"feature": features, "importance": imp_vals})
        else:
            fi_df = pd.DataFrame({"feature": features, "importance": [0] * len(features)})
        fi_df = fi_df.sort_values("importance", ascending=False).head(20)
        fi_list = fi_df.to_dict(orient="records")
    except Exception as e:
        print("Feature importance extraction failed:", e)
        fi_list = []

    return render_template(
        "results.html",
        rows=sample_rows,
        feature_importance=fi_list,
        model_name=model_name.replace("_", " ").title(),
        accuracy=accuracy,
        sample_path=SAMPLE_DATA_PATH
    )

# Optional: download full predictions
@app.route("/download_results", methods=["POST"])
def download_results():
    csv_text = request.form.get("csv_data")
    if not csv_text:
        flash("No CSV data to download", "warning")
        return redirect(url_for("index"))
    buffer = BytesIO()
    buffer.write(csv_text.encode("utf-8"))
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="predictions.csv", mimetype="text/csv")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
