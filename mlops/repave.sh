#!/usr/bin/env bash
# repave.sh — Wipe all generated artifacts and re-run the full pipeline from scratch.
# Run from the mlops/ directory:  bash repave.sh
set -e

echo "=== 1. Clean generated files ==="

# Virtual environment
rm -rf venv
echo "   Removed venv/"

# MLflow databases (local + stale copies)
rm -f mlflow.db
rm -f src/mlflow.db
rm -f ../mlflow.db
echo "   Removed mlflow.db files"

# MLflow run artifacts
rm -rf mlruns
rm -rf ../mlruns
mkdir -p mlruns
echo "   Removed mlruns/"

# Processed data
rm -f data/processed_Churn_Modeling.csv
echo "   Removed data/processed_Churn_Modeling.csv"

# Reports
rm -f reports/monitoring_report.html reports/quality_report.html
echo "   Removed reports/"

# Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
echo "   Removed __pycache__/"

echo ""
echo "=== 2. Create virtual environment ==="
python -m venv venv
source venv/Scripts/activate
echo "   Created and activated venv/"

echo ""
echo "=== 3. Install dependencies ==="
pip install -q -r requirements.txt
echo "   Installed requirements"

echo ""
echo "=== 4. Step 1 — Preprocess dataset ==="
python src/data_preprocessing.py

echo ""
echo "=== 5. Step 2 — Train models & register best ==="
python src/model_tracking_mlflow.py

echo ""
echo "=== 6. Step 6 — Generate monitoring reports ==="
python src/generate_report.py

echo ""
echo "=== Done ==="
echo "All steps completed. Next:"
echo "  • MLflow UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5001"
echo "  • API:        python app/main.py"
echo "  • Swagger:    http://127.0.0.1:8000/docs"
