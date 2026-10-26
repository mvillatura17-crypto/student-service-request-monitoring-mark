"""Train the midterm Decision Tree pipeline and save it to models/.

Usage:
    python train_model.py                          # uses data/sample_requests.csv (synthetic)
    python train_model.py my_midterm_dataset.csv   # uses YOUR midterm CSV

Run this with your real midterm CSV, then commit the new files in models/.
"""
import sys
from pathlib import Path

import sklearn

from modules.model_training import train_and_save

BASE = Path(__file__).resolve().parent

if len(sys.argv) > 1:
    csv_path = Path(sys.argv[1])
    source = f"User-provided dataset ({csv_path.name})"
else:
    csv_path = BASE / "data" / "sample_requests.csv"
    source = "Synthetic sample data (data/sample_requests.csv)"

info = train_and_save(csv_path, data_source=source)

print("Model saved to models/trained_model.joblib")
print("Data source :", info["data_source"])
print("Rows        :", info["n_rows_raw"], "| modeled:", info["n_rows_model"])
print("Train/Test  :", info["n_train"], "/", info["n_test"])
for k, v in info["metrics"].items():
    print(f"{k:20s}: {v:.3f}")
print("Confusion matrix:", info["confusion_matrix"])
print(f"\nTrained with scikit-learn {sklearn.__version__}. "
      f"Keep requirements.txt on scikit-learn=={sklearn.__version__}.")
