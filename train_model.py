import pandas as pd
import numpy as np
import os
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# -------------------- Helper Functions --------------------

def encode_non_numeric(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    return df

def add_engineered_features(df):
    """Add strong numerical interactions and ratio-based features"""
    df['procedure_density'] = df['num_procedures'] / (df['time_in_hospital'] + 1)
    df['lab_density'] = df['num_lab_procedures'] / (df['time_in_hospital'] + 1)
    df['med_diag_ratio'] = df['num_medications'] / (df['number_diagnoses'] + 1)
    df['total_interactions'] = df['num_procedures'] + df['num_medications'] + df['num_lab_procedures']
    df['hospital_stay_intensity'] = df['total_interactions'] / (df['time_in_hospital'] + 1)
    df['avg_procedures_per_day'] = df['num_procedures'] / (df['time_in_hospital'] + 1)
    df['avg_medications_per_day'] = df['num_medications'] / (df['time_in_hospital'] + 1)
    df['interaction_score'] = df['num_lab_procedures'] * df['num_medications']
    df['complexity_ratio'] = (df['num_procedures'] + df['number_diagnoses']) / (df['time_in_hospital'] + 1)
    return df

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"💾 Model saved at: {path}")


def safe_f1(y_true, y_pred):
    # Automatically pick correct averaging
    unique_labels = len(np.unique(y_true))
    avg = 'binary' if unique_labels == 2 else 'weighted'
    return f1_score(y_true, y_pred, average=avg)


# -------------------- Training Function --------------------

def train_advanced(data_path='dataset/feature_engineered_diabetes.csv'):
    # ✅ Step 1: Load and prepare
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if 'readmitted' not in df.columns:
        raise ValueError("❌ 'readmitted' column not found!")

    print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Combine readmitted (1 & 2) → better separation, less imbalance
    df['readmitted'] = df['readmitted'].replace({1: 1, 2: 1, 0: 0})

    # Feature Engineering + Encoding
    df = add_engineered_features(df)
    df = encode_non_numeric(df)

    X = df.drop('readmitted', axis=1)
    y = df['readmitted']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ✅ Step 2: Balance classes
    print("⚖️ Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"✅ After SMOTE: {X_train_res.shape[0]} samples")

    # ✅ Step 3: Scaling for Logistic Regression
    scaler = StandardScaler()
    num_cols = X_train_res.select_dtypes(include=np.number).columns
    X_train_res[num_cols] = scaler.fit_transform(X_train_res[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # ---------------------- LOGISTIC REGRESSION ----------------------
    print("\n🚀 Training Logistic Regression...")
    log_model = LogisticRegression(max_iter=2000, C=1.5, class_weight='balanced')
    log_model.fit(X_train_res, y_train_res)
    y_pred_log = log_model.predict(X_test)
    acc_log = accuracy_score(y_test, y_pred_log)
    f1_log = f1_score(y_test, y_pred_log, average='weighted')
    print(f"✅ Logistic Regression Accuracy: {acc_log:.3f} | F1: {f1_log:.3f}")

    # ---------------------- RANDOM FOREST ----------------------
    print("\n🌲 Training Random Forest (tuned)...")
    rf_model = RandomForestClassifier(
        n_estimators=600,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_res, y_train_res)
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
    print(f"✅ Random Forest Accuracy: {acc_rf:.3f} | F1: {f1_rf:.3f}")

    # ---------------------- XGBOOST ----------------------
    print("\n⚡ Training XGBoost with tuning...")
    param_grid = {
        'n_estimators': [300, 400, 500],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_lambda': [1.0, 1.5],
        'reg_alpha': [0.3, 0.5]
    }

    xgb = XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )


    weighted_f1 = make_scorer(f1_score, average='weighted')

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=10,
        scoring=weighted_f1,
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    search.fit(X_train_res, y_train_res)
    best_params = search.best_params_
    print(f"🏆 Best XGBoost Params: {best_params}")

    best_xgb = XGBClassifier(
        **best_params,
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    best_xgb.fit(X_train_res, y_train_res)

    y_pred_xgb = best_xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')
    print(f"✅ XGBoost Accuracy: {acc_xgb:.3f} | F1: {f1_xgb:.3f}")
    print(classification_report(y_test, y_pred_xgb))

    # ---------------------- CROSS-VALIDATION ----------------------
    print("\n📈 Running 5-fold cross-validation for XGBoost...")
    scores = cross_val_score(best_xgb, X_train_res, y_train_res, cv=5, scoring='f1')
    print(f"Cross-val F1: {scores.mean():.3f} ± {scores.std():.3f}")

    # ---------------------- SAVE BEST MODEL ----------------------
    # ---------------------- SAVE BEST MODEL ----------------------

    results = {
        "Logistic Regression": (acc_log, f1_log),
        "Random Forest": (acc_rf, f1_rf),
        "XGBoost": (acc_xgb, f1_xgb)
    }

    print("\n📊 Model Comparison:")
    for model, (acc, f1) in results.items():
        print(f"{model:20s} → Accuracy: {acc:.3f}, F1: {f1:.3f}")

    best_model_name = max(results, key=lambda x: results[x][1])
    best_acc, best_f1 = results[best_model_name]
    print(f"\n🏆 Best Model: {best_model_name} (Acc={best_acc:.3f}, F1={best_f1:.3f})")

    # Mapping models to their objects
    models_map = {
        "Logistic Regression": log_model,
        "Random Forest": rf_model,
        "XGBoost": best_xgb
    }

    # ---------------------- SAVE MODELS + FEATURE FILES ----------------------
    for model_name, model_obj in models_map.items():

        clean_name = model_name.replace(" ", "_").lower()

        model_path = f"models/{clean_name}.pkl"
        feature_path = f"models/{clean_name}_features.json"

        # Get train metrics
        acc, f1 = results[model_name]

        # Save model
        save_model(model_obj, model_path)

        # Save features + metrics
        with open(feature_path, "w") as f:
            json.dump({
                "features": list(X.columns),
                "train_accuracy": float(round(acc, 4)),
                "train_f1": float(round(f1, 4))
            }, f, indent=4)

        print(f"📄 Saved model: {model_path}")
        print(f"📄 Saved feature+metrics JSON: {feature_path}")

    print("\n🎉 All models & metadata saved successfully!")


    # ---------------------- FEATURE IMPORTANCE ----------------------
    if best_model_name == "XGBoost":
        importances = best_xgb.feature_importances_
        feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        plt.barh(feat_df['Feature'], feat_df['Importance'])
        plt.gca().invert_yaxis()
        plt.title("Top 15 Important Features (XGBoost)")
        plt.tight_layout()
        plt.show()

# -------------------- Run --------------------
if __name__ == "__main__":
    train_advanced()
