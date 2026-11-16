from pathlib import Path
import sys
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from ml.ml_model import DATA_PATH, train_models, predict_machine, dataset_summary, FEATURES
from ml.phase2_models import train_phase2_models
import generate_charts

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "local-predict-maintenance-session"
UPLOAD_DIR = PROJECT_ROOT / "data" / "user_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_BUNDLE = None
LOAD_ERROR = None
ACTIVE_DATASET_NAME = "Default AI4I dataset"
ACTIVE_DATASET_PATH = DATA_PATH


def train_all_models(csv_path):
    """Train Phase 1 and Phase 2 models on the selected CSV."""
    bundle = train_models(csv_path)  # Train the existing Logistic Regression and KNN models.
    phase2 = train_phase2_models(bundle)  # Train the new classifiers and unsupervised examples.
    bundle["models"].update(phase2["models"])  # Add all Phase 2 classifiers to the prediction choices.
    bundle["metrics"].update(phase2["metrics"])  # Add Phase 2 scores to the dashboard comparison.
    bundle["confusion_matrices"].update(phase2["confusion_matrices"])  # Add Phase 2 confusion matrices.
    bundle["clustering"] = phase2["clustering"]  # Keep the K-Means summary for display.
    bundle["pca"] = phase2["pca"]  # Keep the PCA summary for display.
    return bundle  # Give the combined model results to the app.


def get_bundle():
    global MODEL_BUNDLE, LOAD_ERROR
    if MODEL_BUNDLE is None and LOAD_ERROR is None:
        try:
            MODEL_BUNDLE = train_all_models(DATA_PATH)
        except Exception as error:
            LOAD_ERROR = str(error)
    return MODEL_BUNDLE


def render_dashboard():
    bundle = get_bundle()
    summary = dataset_summary(bundle["data"]) if bundle else None
    return render_template("index.html", summary=summary, error=LOAD_ERROR,
                           metrics=bundle["metrics"] if bundle else None,
                           matrices=bundle["confusion_matrices"] if bundle else None,
                           model_names=list(bundle["metrics"].keys()) if bundle else [],
                           clustering=bundle.get("clustering") if bundle else None,
                           pca=bundle.get("pca") if bundle else None,
                           dataset_name=ACTIVE_DATASET_NAME)


@app.route("/")
def index():
    return render_dashboard()


@app.route("/upload-dataset", methods=["POST"])
def upload_dataset():
    """Train all project models on an uploaded CSV without replacing the default CSV."""
    global MODEL_BUNDLE, LOAD_ERROR, ACTIVE_DATASET_NAME, ACTIVE_DATASET_PATH
    uploaded = request.files.get("dataset")
    if not uploaded or not uploaded.filename:
        flash("Choose a CSV file first.", "error")
        return redirect(url_for("index"))
    if Path(uploaded.filename).suffix.lower() != ".csv":
        flash("Please upload a .csv file.", "error")
        return redirect(url_for("index"))

    safe_name = secure_filename(uploaded.filename) or "uploaded_dataset.csv"
    # Save under data/user_uploads; never overwrite data/ai4i2020.csv.
    destination = UPLOAD_DIR / safe_name
    if destination == DATA_PATH:
        destination = UPLOAD_DIR / ("uploaded_" + safe_name)
    uploaded.save(destination)
    try:
        new_bundle = train_all_models(destination)  # validates the CSV, trains every model, and checks Phase 2 too
        # Charts are generated from the selected CSV and replace only display images, not the default data.
        generate_charts.main(destination, PROJECT_ROOT / "ui" / "static" / "charts")
        MODEL_BUNDLE = new_bundle
        LOAD_ERROR = None
        ACTIVE_DATASET_NAME = safe_name + " (uploaded)"
        ACTIVE_DATASET_PATH = destination
        flash(f"Uploaded dataset activated: {safe_name}. The original AI4I CSV remains unchanged.", "success")
    except Exception as error:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        flash("Dataset was not activated: " + str(error), "error")
    return redirect(url_for("index"))


@app.route("/use-default-dataset", methods=["POST"])
def use_default_dataset():
    """Switch back to the official default CSV."""
    global MODEL_BUNDLE, LOAD_ERROR, ACTIVE_DATASET_NAME, ACTIVE_DATASET_PATH
    try:
        MODEL_BUNDLE = train_all_models(DATA_PATH)
        LOAD_ERROR = None
        ACTIVE_DATASET_NAME = "Default AI4I dataset"
        ACTIVE_DATASET_PATH = DATA_PATH
        generate_charts.main(DATA_PATH, PROJECT_ROOT / "ui" / "static" / "charts")
    except Exception as error:
        MODEL_BUNDLE = None
        LOAD_ERROR = str(error)
    return redirect(url_for("index"))


@app.route("/api/status")
def status():
    bundle = get_bundle()
    if not bundle:
        return jsonify({"ready": False, "error": LOAD_ERROR}), 503
    return jsonify({"ready": True, "dataset": ACTIVE_DATASET_NAME, "metrics": bundle["metrics"], "clustering": bundle["clustering"], "pca": bundle["pca"]})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    bundle = get_bundle()
    if not bundle:
        return jsonify({"error": LOAD_ERROR or "Dataset is not available."}), 503
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        machine_type = str(payload.get("type", "M"))
        if machine_type not in ["L", "M", "H"]:
            raise ValueError("Type must be L, M, or H.")
        inputs = {
            "Type": machine_type,
            "Air temperature [K]": float(payload["air_temperature"]),
            "Process temperature [K]": float(payload["process_temperature"]),
            "Rotational speed [rpm]": float(payload["rotational_speed"]),
            "Torque [Nm]": float(payload["torque"]),
            "Tool wear [min]": float(payload["tool_wear"]),
        }
        limits = [("Air temperature [K]", 250, 350), ("Process temperature [K]", 250, 400),
                  ("Rotational speed [rpm]", 0, 10000), ("Torque [Nm]", 0, 500),
                  ("Tool wear [min]", 0, 1000)]
        for key, low, high in limits:
            if not low <= inputs[key] <= high:
                raise ValueError(f"{key} must be between {low} and {high} for this demo.")
        model_name = str(payload.get("model", "Logistic Regression"))
        result = predict_machine(bundle, inputs, model_name)
        return jsonify({**result, "model": model_name,
                        "message": "NO FAILURE DETECTED" if result["prediction"] == 0 else "MACHINE FAILURE RISK",
                        "dataset": ACTIVE_DATASET_NAME})
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": str(error) or "Please enter valid sensor values."}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
