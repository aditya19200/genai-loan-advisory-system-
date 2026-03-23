"""Training script updated for German Credit Dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.services import ml_services
from app.services.process_data import PROCESSED_DATA_PATH, RAW_DATA_PATH, process_german_data
from app.utils.processor import FEATURE_ORDER

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PROCESSED_DATA_PATH), help="CSV file with training data")
    parser.add_argument("--target", default="Risk")
    parser.add_argument("--out_dir", default=str(ROOT_DIR / "app" / "models"))
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if not Path(args.data).exists():
        print(f"Training dataset not found at {args.data}. Building it from {RAW_DATA_PATH}...")
        process_german_data(RAW_DATA_PATH, args.data)
    
    model_out = str(Path(args.out_dir) / "xgboost_model.pkl")
    prep_out = str(Path(args.out_dir) / "preprocessor.pkl")

    print(f"Starting training with target: {args.target}...")

    result = ml_services.train_xgboost(
        data_csv=args.data,
        feature_cols=FEATURE_ORDER,
        target_col=args.target,
        model_out_path=model_out,
        preprocessor_out_path=prep_out,
    )

    print("✅ Training finished.")
    print(f"Validation AUC: {result['metrics']['val_auc']:.4f}")
    print(f"Validation accuracy: {result['metrics']['val_accuracy']:.4f}")
    print(f"Model type: {result['model_type']}")
    print(f"Model saved to: {model_out}")

if __name__ == "__main__":
    main()
