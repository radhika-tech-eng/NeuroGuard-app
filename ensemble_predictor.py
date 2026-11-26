
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.cnn_1d import EEG_CNN
from src.models.cnn_lstm import EEG_CNN_LSTM
from src.features.temporal_decision import TemporalDecisionEngine
from src.features.model_uncertainty import analyze_model_predictions


DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def extract_features(data):
    return np.concatenate(
        [
            data.mean(axis=2),
            data.std(axis=2),
            np.median(np.abs(data), axis=2),
            data.max(axis=2),
            data.min(axis=2),
        ],
        axis=1,
    )


def load_torch_checkpoint(path, model_class, device):
    try:
        checkpoint = torch.load(
            path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    model = model_class(
        n_channels=int(checkpoint["channels"])
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = np.asarray(
        checkpoint["normalization_mean"], dtype=np.float32
    )
    std = np.asarray(
        checkpoint["normalization_std"], dtype=np.float32
    )
    std = np.where(std < 1e-6, 1.0, std)

    return model, mean, std


def predict_cnn_batch(model, windows, mean, std, device):
    normalized = (windows - mean) / std
    normalized = normalized.astype(np.float32)

    tensor = torch.from_numpy(normalized).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)

    return probabilities[:, 1].cpu().numpy()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    eeg_path = DATA_DIR / "eeg_windows_new.npy"
    labels_path = DATA_DIR / "eeg_labels_new.npy"
    patients_path = DATA_DIR / "patient_ids_new.npy"

    xgb_path = MODEL_DIR / "xgboost_baseline.joblib"
    cnn_path = MODEL_DIR / "cnn_1d_patientwise.pt"
    lstm_path = MODEL_DIR / "cnn_lstm_patientwise.pt"

    required_files = [
        eeg_path, labels_path, patients_path,
        xgb_path, cnn_path, lstm_path
    ]
    missing = [str(p) for p in required_files if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    device = torch.device("cpu")

    print("Loading trained models...")
    xgb_model = joblib.load(xgb_path)

    cnn, cnn_mean, cnn_std = load_torch_checkpoint(
        cnn_path, EEG_CNN, device
    )
    lstm, lstm_mean, lstm_std = load_torch_checkpoint(
        lstm_path, EEG_CNN_LSTM, device
    )

    windows = np.load(eeg_path).astype(np.float32)
    labels = np.load(labels_path).astype(int)
    patients = np.load(patients_path)

    if len(windows) != len(labels) or len(windows) != len(patients):
        raise ValueError("Windows, labels and patient IDs must align.")

    if windows.shape[1:] != (cnn_mean.shape[-2], windows.shape[2]):
        raise ValueError("EEG channels do not match CNN checkpoint.")

    print(f"Processing {len(windows)} EEG windows...")

    # XGBoost features must match the original training procedure.
    xgb_features = extract_features(windows)
    xgb_probs = xgb_model.predict_proba(xgb_features)[:, 1]

    # Batch inference for the two neural networks.
    batch_size = 32
    cnn_probs = []
    lstm_probs = []

    for start in range(0, len(windows), batch_size):
        batch = windows[start:start + batch_size]

        cnn_probs.extend(
            predict_cnn_batch(
                cnn, batch, cnn_mean, cnn_std, device
            )
        )
        lstm_probs.extend(
            predict_cnn_batch(
                lstm, batch, lstm_mean, lstm_std, device
            )
        )

        if (start // batch_size) % 10 == 0:
            print(f"Processed {min(start + batch_size, len(windows))}"
                  f"/{len(windows)} windows")

    cnn_probs = np.asarray(cnn_probs)
    lstm_probs = np.asarray(lstm_probs)

    ensemble_probs = np.mean(
        np.vstack([xgb_probs, cnn_probs, lstm_probs]), axis=0
    )

    # Use a fresh temporal history for each patient.
    # This prevents histories from different patients mixing.
    temporal_engines = {}
    temporal_results = []

    for i, probability in enumerate(ensemble_probs):
        patient = str(patients[i])

        if patient not in temporal_engines:
            temporal_engines[patient] = TemporalDecisionEngine()

        result = temporal_engines[patient].update(
            float(probability)
        )

        temporal_results.append(result)

    # Uncertainty analysis for each window.
    uncertainty_results = []
    for i in range(len(windows)):
        scores = {
            "xgboost": float(xgb_probs[i]),
            "cnn_1d": float(cnn_probs[i]),
            "cnn_lstm": float(lstm_probs[i]),
        }
        uncertainty_results.append(analyze_model_predictions(scores))

    def result_field(result, field, default=""):
        if isinstance(result, dict):
            return result.get(field, default)
        return getattr(result, field, default)

    rows = []
    for i in range(len(windows)):
        temporal = temporal_results[i]
        uncertainty = uncertainty_results[i]

        rows.append({
            "window_index": i,
            "patient_id": str(patients[i]),
            "actual_label": int(labels[i]),
            "xgboost_probability": float(xgb_probs[i]),
            "cnn_1d_probability": float(cnn_probs[i]),
            "cnn_lstm_probability": float(lstm_probs[i]),
            "ensemble_probability": float(ensemble_probs[i]),
            "temporal_decision": str(
                result_field(temporal, "decision", temporal)
            ),
            "uncertainty_status": str(
                result_field(uncertainty, "status", "")
            ),
            "models_disagree": result_field(
                uncertainty, "models_disagree", ""
            ),
        })

    output_path = REPORT_DIR / "ensemble_temporal_predictions.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print("\nAll windows processed successfully.")
    print(f"Windows processed: {len(windows)}")
    print(f"Patients: {sorted(set(map(str, patients)))}")
    print(f"Report saved to: {output_path}")

    print("\nFirst 10 results:")
    print(pd.DataFrame(rows).head(10).to_string(index=False))

    print("\nResearch prototype only; not for clinical diagnosis.")


if __name__ == "__main__":
    main()
