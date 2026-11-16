"""Phase 2 models for the beginner predictive-maintenance project."""

# Import the model classes used in the Phase 2 comparisons.
from sklearn.naive_bayes import GaussianNB  # Import the probability-based classifier.
from sklearn.svm import SVC  # Import Support Vector Machine classification.
from sklearn.tree import DecisionTreeClassifier  # Import the decision-tree model.
from sklearn.ensemble import RandomForestClassifier  # Import the combined decision-tree model.
# Import clustering, dimensionality reduction, and evaluation tools.
from sklearn.cluster import KMeans  # Import the unsupervised clustering algorithm.
from sklearn.decomposition import PCA  # Import Principal Component Analysis.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score  # Import the evaluation measures.
# Import tools that repeat the Phase 1 split and preprocessing clearly.
from sklearn.model_selection import train_test_split  # Import the repeatable train/test split.
from sklearn.preprocessing import StandardScaler  # Import feature standardization.
import pandas as pd  # Import Pandas to arrange the input columns.
from ml.ml_model import FEATURES, TARGET  # Reuse the project input and target names.


def train_phase2_models(model_bundle):  # Define the Phase 2 training function.
    """Train Phase 2 classifiers and run simple K-Means and PCA examples."""
    # Select the predictive input columns and the target from the cleaned data.
    data = model_bundle["data"]  # Get the cleaned records from Phase 1.
    X = pd.get_dummies(data[FEATURES], columns=["Type"], drop_first=True, dtype=int)  # Convert machine type into numeric columns.
    y = data[TARGET]  # Store the correct failure labels.
    # Keep the split identical to Phase 1 so model metrics are comparable.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)  # Use the same balanced split as Phase 1.
    # Fit scaling only on training rows to prevent information leakage.
    scaler = StandardScaler()  # Create a feature-scaling tool.
    X_train = scaler.fit_transform(X_train_raw)  # Learn scale values from training data.
    X_test = scaler.transform(X_test_raw)  # Apply training scale values to test data.
    X_all = scaler.transform(X)  # Scale every row for unsupervised examples.
    # Create the four additional classifiers required for Phase 2.
    models = {  # Store each additional classifier by its dashboard name.
        "Naive Bayes": GaussianNB(),  # Estimate probabilities from numeric sensor readings.
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),  # Use the RBF kernel for curved boundaries.
        "Decision Tree": DecisionTreeClassifier(criterion="gini", max_depth=8, random_state=42),  # Choose branches using Gini impurity.
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),  # Combine 100 limited-depth trees.
    }
    # Prepare dictionaries to keep trained models and test results.
    trained_models = {}  # Keep fitted models for later predictions.
    metrics = {}  # Keep the four classification scores.
    matrices = {}  # Keep each model confusion matrix.
    # Train each model and calculate its classification scores.
    for name, model in models.items():  # Train and measure each classifier.
        model.fit(X_train, y_train)  # Learn patterns from training examples.
        predictions = model.predict(X_test)  # Predict labels for held-out examples.
        metrics[name] = {  # Calculate this model performance measures.
            "accuracy": float(accuracy_score(y_test, predictions)),  # Fraction of predictions that are correct.
            "precision": float(precision_score(y_test, predictions, zero_division=0)),  # Correctness of predicted failures.
            "recall": float(recall_score(y_test, predictions, zero_division=0)),  # Fraction of failures detected.
            "f1": float(f1_score(y_test, predictions, zero_division=0)),  # Balance precision with recall.
        }
        matrices[name] = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()  # Count correct and incorrect class predictions.
        trained_models[name] = model  # Save this fitted model for the prediction form.
    # Run K-Means with two groups as a simple unsupervised learning example.
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)  # Ask K-Means to discover two groups.
    cluster_labels = kmeans.fit_predict(X_all)  # Assign each record to its nearest cluster center.
    # Calculate silhouette on a manageable sample of rows for speed.
    sample_size = min(2000, len(X_all))  # Use at most 2,000 rows when scoring clusters.
    silhouette = float(silhouette_score(X_all, cluster_labels, sample_size=sample_size, random_state=42)) if len(set(cluster_labels)) > 1 else 0.0  # Measure separation when there are two groups.
    # Count how many rows K-Means assigned to each cluster.
    cluster_counts = {str(cluster): int((cluster_labels == cluster).sum()) for cluster in sorted(set(cluster_labels))}  # Count records in each group.
    # Reduce the sensor features to two principal components for the PCA example.
    pca = PCA(n_components=2, random_state=42)  # Keep the two strongest principal components.
    pca.fit(X_train)  # Learn component directions from training inputs only.
    explained_variance = [float(value) for value in pca.explained_variance_ratio_]  # Record variance kept by each component.
    # Return models and results to the existing dashboard.
    return {"models": trained_models, "metrics": metrics, "confusion_matrices": matrices,  # Return classifier results to Flask.
            "clustering": {"cluster_counts": cluster_counts, "silhouette": silhouette},  # Return the K-Means summary.
            "pca": {"explained_variance": explained_variance, "total_explained_variance": float(sum(explained_variance))}}  # Return the PCA summary.
