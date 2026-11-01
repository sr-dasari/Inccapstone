# Churn Prediction MLOps Incremental Capstone Project

An end-to-end Machine Learning pipeline for training, tracking, and serving churn prediction classification models using Python, MLflow, and FastAPI.

---

## 📌 Features

* **Data Preprocessing**: Cleaning, feature engineering, and one-hot encoding.
* **Experiment Tracking**: Training multiple models (Random Forest, Logistic Regression) with MLflow logging parameters, metrics, and serialized model artifacts.
* **Model Registry & UI**: SQLite-backed MLflow tracking server, UI dashboard, and model registration for deployment tracking.
* **Model Serving**: FastAPI application serving real-time predictions via a REST API.
* **Drift Monitoring**: Evidently AI reports detecting feature drift (covariate shift) and data quality issues.

---

## 📁 Project Structure

```text
mlops/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   └── utils.py                # Model loading and feature engineering helpers
├── data/
│   ├── Churn_Modeling.csv      # Raw dataset
│   └── processed_Churn_Modeling.csv  # Preprocessed dataset (generated)
├── reports/                    # Generated HTML reports (quality & drift)
├── samples/
│   ├── exited.json             # Sample input — predicts churn
│   └── nonexited.json          # Sample input — predicts stays
├── src/
│   ├── data_preprocessing.py   # Dataset cleaning, encoding & feature creation
│   ├── model_tracking_mlflow.py # Model training, evaluation & MLflow logging
│   └── generate_report.py      # Data quality & drift monitoring reports
├── venv/                       # Virtual environment
├── mlflow.db                   # SQLite database for MLflow (generated)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Setup Instructions

All commands below assume you are inside the `mlops/` directory.

### 1. Create and Activate Virtual Environment

```bash
python -m venv venv
```

*Activate on Windows (Git Bash):*
```bash
source venv/Scripts/activate
```

*Activate on Windows (PowerShell):*
```powershell
venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Pipeline Execution

### Step 1: Preprocess the Dataset

Clean raw data, encode categorical variables, and write the result to `data/processed_Churn_Modeling.csv`.

```bash
python src/data_preprocessing.py
```

### Step 2: Train & Track Models with MLflow

Train classification models (`RandomForestClassifier`, `LogisticRegression`) on the processed dataset. Each model is a scikit-learn `Pipeline` that standardizes numerical values before classifying. The script evaluates Accuracy, Precision, Recall, and F1-score, logs all metrics and artifacts to MLflow, and registers the best model (by F1-score) in the Model Registry.

```bash
python src/model_tracking_mlflow.py
```

### Step 3: Launch MLflow Tracking UI

Start the local MLflow dashboard to view experiment runs, compare metrics, and inspect saved artifacts.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5001
```

Access the UI in your web browser at:
👉 **`http://127.0.0.1:5001`**

### Step 4: Run the Prediction Application

In a separate terminal window (ensure you are inside `mlops/` with the `venv` activated), launch the application:

```bash
python app/main.py
```

The API automatically loads the latest registered model from the MLflow Model Registry. To choose a model interactively instead:

```bash
python app/main.py --select
```

### Step 5: Test Model Predictions (Interactive API Docs)

1. Open **`http://127.0.0.1:8000/docs`** in your web browser to access the Swagger UI.
2. Click on the **`POST /predict`** endpoint and click **Try it out**.
3. Copy the sample JSON payloads from the `samples/` directory and paste them into the request body:
   * **Exited** (`samples/exited.json`): Predicts an **exited** (churn) outcome.
   * **Non-Exited** (`samples/nonexited.json`): Predicts a **non-exited** (stays) outcome.
4. Click **Execute** to view the live inference output.

### Step 6: Automated Data & Prediction Drift Monitoring

To ensure the model performs reliably over time, we use **Evidently AI** to monitor incoming inference data for covariate shift (feature drift) and concept shift (prediction drift) against our baseline dataset.

**Run the Monitoring Pipeline:**

```bash
python src/generate_report.py
```

Two HTML reports are generated in the `reports/` directory:

* **`reports/quality_report.html`**: Dataset summary with column-level statistics and data quality details.
* **`reports/monitoring_report.html`**: Feature drift analysis comparing the training baseline against simulated production data.

---

## 🔄 Clean Slate Repave

To wipe all generated artifacts and re-run the full pipeline from scratch:

```bash
bash repave.sh
```

This single script performs the following:

1. **Removes** `venv/`, all `mlflow.db` files, `mlruns/`, `data/processed_Churn_Modeling.csv`, reports, and `__pycache__/`
2. **Creates** a fresh virtual environment and installs dependencies
3. **Runs** Step 1 (preprocess), Step 2 (train & register), and Step 6 (reports)

After it completes, launch the MLflow UI and API manually:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5001
python app/main.py
```

---

## ✅ Capstone Requirements Checklist

### Task 1: Working with Churn Data

- [x] Import essential Python libraries such as pandas, numpy, and matplotlib.
- [x] Load the `Churn_Modeling.csv` dataset into a Pandas DataFrame.
- [x] Inspect data types and identify categorical and numerical features.
- [x] Explore data distribution and identify missing or inconsistent values.
- [x] Perform grouping and pivot analysis on key variables such as Geography and Gender.
- [x] Generate summary statistics to understand customer churn patterns.
- [x] Prepare insights for model training and reporting.

### Task 2: Building Predictive Models

- [x] Train multiple machine learning models to predict customer churn.
- [x] Evaluate model performance using Accuracy, Precision, Recall, and F1-score.
- [x] Select the best-performing model based on F1-score.
- [x] Save trained model artifacts for further experimentation and deployment.

### Task 3: Managing and Comparing Experiments

- [x] Use MLflow to log model parameters, metrics, and model versions.
- [x] Automatically record details for each experiment run.
- [x] Compare model performance visually using the MLflow UI.
- [x] Register the best-performing model for deployment tracking and version control.

### Task 4: Operationalizing the Model

- [x] Deploy the trained churn prediction model using FastAPI.
- [ ] Containerize the application using Docker for scalability. *(OPTIONAL)*
- [x] Expose a REST API endpoint for real-time churn prediction.
- [x] Simulate model monitoring to track performance and potential data drift.
- [ ] Schedule model retraining when accuracy drops over time. *(OPTIONAL)*
