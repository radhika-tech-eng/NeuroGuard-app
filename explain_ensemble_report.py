
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.explain_prediction import explain_eeg_window

DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

PREDICTION_FILE = REPORT_DIR / "ensemble_temporal_predictions.csv"
EEG_FILE = DATA_DIR / "eeg_windows_new.npy"


def main():
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Prediction report not found: {PREDICTION_FILE}"
        )

    if not EEG_FILE.exists():
        raise FileNotFoundError(f"EEG dataset not found: {EEG_FILE}")

    predictions = pd.read_csv(PREDICTION_FILE)
    windows = np.load(EEG_FILE, mmap_mode="r")

    if len(predictions) != len(windows):
        raise ValueError(
            "Prediction rows and EEG windows do not match."
        )

    rows = []

    print("Generating explanations for actual EEG windows...")

    for i, row in predictions.iterrows():
        window = np.asarray(windows[i], dtype=np.float32)

        scores = {
            "xgboost": float(row["xgboost_probability"]),
            "cnn_1d": float(row["cnn_1d_probability"]),
            "cnn_lstm": float(row["cnn_lstm_probability"]),
        }

        explanation = explain_eeg_window(window, scores)

        rows.append({
            "window_index": int(row["window_index"]),
            "patient_id": row["patient_id"],
            "actual_label": int(row["actual_label"]),
            "ensemble_probability": float(
                row["ensemble_probability"]
            ),
            "temporal_decision": row["temporal_decision"],
            "explanation_status": explanation["status"],
            "explanation_summary": explanation["summary"],
            "probability_spread": explanation.get(
                "probability_spread", np.nan
            ),
            "signal_findings": " | ".join(
                explanation["signal_findings"]
            ),
            "model_findings": " | ".join(
                explanation["model_findings"]
            ),
        })

        if (i + 1) % 250 == 0:
            print(f"Explained {i + 1}/{len(predictions)} windows")

    output = REPORT_DIR / "explainable_ensemble_report.csv"
    pd.DataFrame(rows).to_csv(output, index=False)

    print("\nExplanation report created successfully.")
    print(f"Total windows: {len(rows)}")
    print(f"Saved to: {output}")

    print("\nFirst 5 explanations:")
    print(pd.DataFrame(rows).head().to_string(index=False))

    print("\nResearch prototype only; not for clinical diagnosis.")


if __name__ == "__main__":
    main()
