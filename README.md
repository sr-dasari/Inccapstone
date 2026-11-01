# Churn Prediction MLOps Project

A bank customer churn prediction platform built as an end-to-end MLOps pipeline. It trains and compares classification models, tracks experiments with MLflow, serves predictions through a REST API, and monitors for data drift.

## What It Does

- **Preprocesses** 10,000 customer records (credit score, geography, age, balance, etc.) with one-hot encoding and feature engineering
- **Trains** Random Forest and Logistic Regression models, evaluates each on Accuracy, Precision, Recall, and F1-score
- **Selects** the best model by F1-score and registers it in the MLflow Model Registry
- **Serves** real-time churn predictions via a FastAPI REST API
- **Monitors** for feature drift using Evidently AI, comparing production data against the training baseline

## Tech Stack

| Component | Tool |
|---|---|
| ML Models | scikit-learn (RandomForest, LogisticRegression) |
| Experiment Tracking | MLflow (SQLite backend) |
| API | FastAPI + Uvicorn |
| Drift Monitoring | Evidently AI |
| Language | Python 3.13 |

## Quick Start

```bash
cd mlops
python -m venv venv
source venv/Scripts/activate    # Git Bash on Windows
pip install -r requirements.txt
bash repave.sh                  # or run steps manually (see mlops/README.md)
```

See [mlops/README.md](mlops/README.md) for detailed setup, step-by-step pipeline execution, and the full capstone requirements checklist.
