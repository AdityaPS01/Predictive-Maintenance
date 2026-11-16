# Predictive Maintenance Using Machine Learning — Phase 2

A beginner-friendly B.Tech CSE / Statistical Machine Learning project that uses the official **UCI AI4I 2020 Predictive Maintenance Dataset** to classify whether a machine failure is present (`0 = No Failure`, `1 = Failure`). It runs locally with Flask and does not require a cloud service or paid API.

> **Scope statement:** This project predicts machine failure classification and does not estimate true Remaining Useful Life (RUL).

## 1. Project objective

Explore machine sensor data, summarize it statistically, clean and encode the data, split it into training and testing sets, standardize features, train and compare six classifiers, explore K-Means clustering and PCA, evaluate the models, and expose a local prediction form.

## 2. Dataset

Official source: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The project expects `data/ai4i2020.csv` with these columns:

- `UDI` and `Product ID` — identifiers; not used as predictive features.
- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `Machine failure` — binary target.

**Independent variables:** Type and the five sensor measurements. **Dependent variable:** Machine failure. Linear Regression and Polynomial Regression are not used as the main models because the target is categorical/binary.

The official AI4I dataset CSV is included at `data/ai4i2020.csv`. The application does not substitute fake data. The dashboard also supports uploading a compatible CSV; uploaded files are saved separately in `data/user_uploads/`, and the default `data/ai4i2020.csv` is never overwritten.

## 3. Phase 1 scope — implemented

- Pandas data loading, duplicate checks, missing-value handling
- Mean, median, standard deviation, variance, minimum and maximum
- EDA graphs using Matplotlib and Seaborn
- One-hot encoding for `Type` using `pandas.get_dummies()`
- 80/20 train/test split (`random_state=42`, stratified target)
- `StandardScaler`, fitted on the training set only
- Logistic Regression
- K-Nearest Neighbors (`n_neighbors=5`)
- Accuracy, precision, recall, F1-score and confusion matrices
- Local Flask prediction endpoint and interactive dashboard
- UI CSV upload: train on a compatible user CSV and switch back to the default AI4I dataset at any time

## 4. Phase 2 scope — implemented

- Gaussian Naive Bayes, Support Vector Machine with an RBF kernel, Decision Tree using Gini impurity, and Random Forest
- Side-by-side accuracy, precision, recall, F1-score, and confusion matrices for all six classifiers
- Interactive single-record prediction with any of the six classifiers
- K-Means clustering with two clusters and a silhouette score
- Principal Component Analysis (PCA) with two components and explained-variance reporting
- Same cleaned dataset and reproducible 80/20 stratified split for fair classifier comparison

The dashboard remains one page. The existing visual style and separate UI/model folders are retained. K-Means groups are exploratory and are not treated as the actual machine-failure labels. PCA reports variance captured by the first two components; it is not used to replace the original feature pipeline.

## 5. Folder structure

```text
Predictive_Maintenance_Phase1/
├── ml/
│   ├── ml_model.py
│   └── phase2_models.py
├── ui/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       ├── app.js
│       └── charts/                 # Generated from the real CSV
├── data/
│   └── PUT_AI4I2020_CSV_HERE.txt
├── architecture/
│   └── Predictive_Maintenance_Phase1_Architecture.png
├── generate_charts.py
├── requirements.txt
├── README.md
└── run_mac.sh
```

## 6. Installation (macOS)

Install Python 3.10 or newer if it is not already installed. Open Terminal and go to the extracted project folder:

```bash
cd Predictive_Maintenance_Phase1
python3 -m pip install -r requirements.txt
```

If your system requires a virtual environment, create one first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 7. Add the dataset

1. Visit the official UCI link above.
2. Download and extract the dataset if needed.
3. Copy `ai4i2020.csv` into the project's `data/` folder.
4. Confirm the exact path is `data/ai4i2020.csv`.

If the file is missing, the dashboard displays a clear setup message instead of crashing with an unexplained traceback.

## 8. Generate the EDA charts

From the project root:

```bash
python3 generate_charts.py
```

This creates eight PNGs in `ui/static/charts/`: machine-failure distribution; five sensor histograms; numeric correlation heatmap; and torque-vs-tool-wear scatter plot colored by failure class. These charts are calculated from the actual CSV. If the CSV is missing, the script prints setup instructions and exits without creating fake charts.

## 9. Run the Flask application

```bash
python3 ui/app.py
```

Open either URL in your browser:

- http://127.0.0.1:5050
- http://localhost:5050

Or use the convenience script (it installs dependencies, tries chart generation, then starts Flask):

```bash
chmod +x run_mac.sh
./run_mac.sh
```

Once dependencies and the dataset are present, the app runs locally. It does not need an external database, hosted ML service, OpenAI API, or internet access during normal use. Internet is needed initially to download dependencies and the official dataset.

## 10. Upload your own CSV from the dashboard

1. Start the Flask app and open the dashboard.
2. Find the **Bring your own dataset** panel.
3. Select a `.csv` file and click **Upload & train models**.
4. The CSV must include `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, and binary `Machine failure` columns.
5. The app validates the columns, cleans the data, retrains all six classifiers plus K-Means and PCA, recalculates metrics, and refreshes the EDA chart images from the uploaded file.
6. Click **Switch back to default AI4I** to restore the official default dataset.

Uploaded files are saved under `data/user_uploads/`. The default `data/ai4i2020.csv` is not replaced. Uploading a compatible CSV is intended for local experiments; values and model quality depend on the uploaded data.

## 11. Pipeline explanation for viva

1. **Load data:** Pandas reads the official CSV.
2. **Clean data:** Count missing values and duplicates; remove duplicate rows; remove rows with an unknown target; fill missing numeric sensor values with their median and missing `Type` with its mode.
3. **Statistical analysis:** Summarize mean, median, standard deviation, variance, minimum and maximum; count failures and calculate failure percentage.
4. **EDA:** Plot class balance, sensor distributions, correlations, and torque against tool wear.
5. **Encode category:** `Type` is categorical, so `pd.get_dummies()` turns it into numeric indicator columns. `drop_first=True` avoids redundant dummy columns.
6. **Split:** Use 80% of rows for training and 20% for testing. `stratify=y` helps preserve the failure/no-failure proportion in each split. `random_state=42` makes the split repeatable.
7. **Scale:** `StandardScaler` standardizes features. It is particularly important for KNN because distance calculations can otherwise be dominated by variables with larger numeric ranges. The scaler is fitted on training data only.
8. **Train models:** Logistic Regression, KNN, Naive Bayes, SVM, Decision Tree and Random Forest learn from the training data. Test predictions are summarized with accuracy, precision, recall, F1-score and a confusion matrix. Failure is the positive class (`1`).
9. **Explore unsupervised learning:** K-Means forms two groups without using the target labels; the silhouette score gives a simple grouping-quality measure. PCA combines the standardized input features into two components and reports how much variance they retain.
10. **Predict:** The web form uses the same encoding columns and fitted scaler as training, then returns a class and, where supported, an estimated probability.

## 12. Evaluation metrics

- **Accuracy:** fraction of all predictions that are correct.
- **Precision:** among predicted failures, fraction that were actual failures.
- **Recall:** among actual failures, fraction detected by the model.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** counts true negatives, false positives, false negatives and true positives (labels ordered `0`, `1`).

Failure examples are relatively uncommon in this dataset, so accuracy alone can be misleading; look at recall, precision and the confusion matrix too. Results shown in the dashboard are calculated at runtime and are not hard-coded.

## 13. Limitations

- AI4I 2020 is a synthetic predictive-maintenance dataset, not a complete record of all real factory conditions.
- Model quality depends on data quality and the chosen train/test split.
- This project predicts failure classification, not exact failure time.
- **No true RUL prediction is performed.**
- Real industrial machines may contain additional sensor variables and different failure patterns.
- This project focuses on classification, clustering and PCA topics from the course; it does not implement every algorithm listed in the full theory syllabus.
- The dashboard is a learning/demo tool, not a certified industrial safety system.
- A predicted probability is a model estimate, not certainty.

## 14. Code map

- `ml/ml_model.py`: cleaning, encoding, split, scaling, Phase 1 training, metrics and prediction logic.
- `ml/phase2_models.py`: Phase 2 classifiers, K-Means clustering and PCA.
- `generate_charts.py`: creates all EDA PNGs from the CSV.
- `ui/app.py`: Flask routes and request validation; imports the ML module.
- `ui/templates/index.html`: dashboard structure and server-rendered metric values.
- `ui/static/style.css`: responsive futuristic industrial styling and CSS animations.
- `ui/static/app.js`: prediction form interaction and result display.

## 15. Troubleshooting

- **Dataset not found:** place the official CSV at `data/ai4i2020.csv`.
- **Missing columns:** make sure you downloaded the AI4I 2020 CSV, not another predictive-maintenance dataset.
- **Charts do not show:** run `python3 generate_charts.py` from the project root after placing the CSV.
- **Package errors:** activate the same Python environment in which you installed `requirements.txt`.
- **Port 5050 busy:** stop the other local app using port 5050 before starting this app. The dashboard uses port 5050 to avoid conflicts with macOS services on port 5000.
