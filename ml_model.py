"""Beginner-friendly Phase 1 predictive-maintenance ML code.

Only Logistic Regression and K-Nearest Neighbors (KNN) are implemented.
The official AI4I CSV must be placed at data/ai4i2020.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
TARGET = "Machine failure"


def load_and_clean_data(csv_path=DATA_PATH):
    """Load the official CSV, remove duplicates, and check required columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\nDownload the official UCI AI4I 2020 dataset and place its CSV at data/ai4i2020.csv."
        )
    df = pd.read_csv(csv_path)
    # Some copies of the dataset have spaces around column names.
    df.columns = [column.strip() for column in df.columns]
    required = FEATURES + [TARGET]
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError("The CSV is missing required columns: " + ", ".join(missing_columns))
    missing_before = int(df[required].isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    # Keep rows with a known target; fill numeric gaps with each column's median.
    df = df.dropna(subset=[TARGET])
    for column in FEATURES:
        if df[column].isna().any():
            if column == "Type":
                df[column] = df[column].fillna(df[column].mode().iloc[0])
            else:
                df[column] = df[column].fillna(df[column].median())
    # Target is binary; convert values to integers after validation.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df = df[df[TARGET].isin([0, 1])].copy()
    return df, {"missing_values_before_cleaning": missing_before,
                "duplicates_removed": duplicate_count}


def train_models(csv_path=DATA_PATH):
    """Train the Phase 1 models and return data for the dashboard."""
    df, cleaning = load_and_clean_data(csv_path)
    # UDI and Product ID are intentionally not used as predictive features.
    X = pd.get_dummies(df[FEATURES], columns=["Type"], drop_first=True, dtype=int)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Fit scaling on training data only to avoid information leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    trained = {}
    results = {}
    matrices = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
        }
        matrices[name] = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
        trained[name] = model

    return {
        "data": df, "features_encoded": list(X.columns), "feature_columns": list(X.columns),
        "scaler": scaler, "models": trained, "metrics": results,
        "confusion_matrices": matrices, "X_test": X_test, "y_test": y_test,
        "cleaning": cleaning,
    }


def predict_machine(model_bundle, machine_input, model_name="Logistic Regression"):
    """Predict failure for one set of sensor readings."""
    if model_name not in model_bundle["models"]:
        raise ValueError("Choose a model shown in the dashboard.")
    # Build a one-row DataFrame and apply the same Type encoding used for training.
    row = pd.DataFrame([machine_input], columns=FEATURES)
    row = pd.get_dummies(row, columns=["Type"], dtype=int)
    row = row.reindex(columns=model_bundle["feature_columns"], fill_value=0)
    scaled_row = model_bundle["scaler"].transform(row)
    model = model_bundle["models"][model_name]
    prediction = int(model.predict(scaled_row)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(scaled_row)[0][1])
    return {"prediction": prediction, "estimated_probability": probability}


def dataset_summary(df):
    """Return simple dataset and descriptive-statistics values for the UI."""
    numeric = df[["Air temperature [K]", "Process temperature [K]",
                  "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]]
    failure_count = int((df[TARGET] == 1).sum())
    return {
        "rows": int(df.shape[0]), "columns": int(df.shape[1]),
        "failure_count": failure_count,
        "failure_rate": float(df[TARGET].mean() * 100),
        "no_failure_count": int((df[TARGET] == 0).sum()),
        "stats": numeric.describe().round(2).to_dict(),
        "mean": numeric.mean().round(2).to_dict(),
        "median": numeric.median().round(2).to_dict(),
        "std": numeric.std().round(2).to_dict(),
        "variance": numeric.var().round(2).to_dict(),
        "minimum": numeric.min().round(2).to_dict(),
        "maximum": numeric.max().round(2).to_dict(),
    }
