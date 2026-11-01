import sys
import mlflow
import mlflow.sklearn
import pandas as pd

def select_model(experiment_name: str = "churn_experiment"):
    """Loads the latest registered model, or falls back to interactive selection."""
    client = mlflow.tracking.MlflowClient()

    # Try loading the latest version of the registered model first
    model_name = "churn_prediction_model"
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            run_id = latest.run_id
            print(f"\nLoading registered model '{model_name}' version {latest.version} (run {run_id[:8]})...")
            model_uri = f"models:/{model_name}/{latest.version}"
            loaded_model = mlflow.sklearn.load_model(model_uri)
            return loaded_model, run_id
    except Exception:
        pass

    # Fall back to interactive selection if no registered model exists
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found in mlflow.db.")
        sys.exit(1)

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    if not runs:
        print("No logged model runs found in experiment.")
        sys.exit(1)

    print(f"\n--- Available Models in MLFlow ({experiment_name}) ---")
    for idx, run in enumerate(runs):
        run_id = run.info.run_id
        model_type = run.data.params.get("model_type", "Unknown")
        accuracy = run.data.metrics.get("accuracy", "N/A")
        f1 = run.data.metrics.get("f1_score", "N/A")
        print(f"[{idx}] Run ID: {run_id} | Type: {model_type} | Accuracy: {accuracy} | F1: {f1}")

    choice = int(input("\nSelect model index to load into app: "))
    selected_run = runs[choice]
    run_id = selected_run.info.run_id

    print(f"\nLoading model from Run ID: {run_id}...")
    model_uri = f"runs:/{run_id}/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    return loaded_model, run_id


def engineer_features(df: pd.DataFrame, loaded_model) -> pd.DataFrame:
    """Derive the one-hot columns created by data_preprocessing.py from raw categorical fields."""
    expected_features = getattr(loaded_model, "feature_names_in_", [])

    for col in ["Geography", "Gender"]:
        if col in df.columns:
            for feature in expected_features:
                if feature.startswith(f"{col}_"):
                    df[feature] = df[col] == feature[len(col) + 1:]

    return df


def align_features(df: pd.DataFrame, loaded_model) -> pd.DataFrame:
    """Ensure exact feature presence and order matching the model schema."""
    if hasattr(loaded_model, "feature_names_in_"):
        expected_cols = list(loaded_model.feature_names_in_)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]
    return df