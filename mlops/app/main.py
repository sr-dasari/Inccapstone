import sys
import mlflow
import pandas as pd
from fastapi import FastAPI
import uvicorn

# Import helpers from utils
from utils import select_model, engineer_features, align_features

# Point to the SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Use --select flag to pick a model interactively; otherwise auto-loads the registered model
interactive = "--select" in sys.argv
model, selected_run_id = select_model("churn_experiment", interactive=interactive)

app = FastAPI(title=f"Churn Prediction API (Model: {selected_run_id[:8]})")

@app.get("/")
def root():
    return {"status": "active", "loaded_run_id": selected_run_id}

@app.post("/predict")
def predict(data: dict):
    # Step 1: Keep keys as sent, since the model's feature names are case-sensitive
    df = pd.DataFrame([data])

    # Step 2 & 3: Feature engineering and schema alignment
    df = engineer_features(df, model)
    df = align_features(df, model)

    prediction = model.predict(df)[0]
    
    # Return prediction as string (0 = non-exited, 1 = exited)
    return {"prediction": str(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)