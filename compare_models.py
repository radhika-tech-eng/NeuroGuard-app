
import os
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.models.cnn_1d import EEG_CNN
from src.models.cnn_lstm import EEG_CNN_LSTM


def extract_features(X):
    """Extract features in the same order as model training."""
    return np.concatenate(
        [
            X.mean(axis=2),
            X.std(axis=2),
            np.median(np.abs(X), axis=2),
            X.max(axis=2),
            X.min(axis=2),
        ],
        axis=1,
    )


def normalize(X, checkpoint):
    mean = np.asarray(checkpoint["normalization_mean"])
    std = np.asarray(checkpoint["normalization_std"])
    std = np.where(std == 0, 1, std)
    return ((X - mean) / std).astype(np.float32)


def main():
    # Load the processed EEG dataset
    X = np.load("data/processed/eeg_windows_new.npy").astype(np.float32)
    y = np.load("data/processed/eeg_labels_new.npy").astype(np.int64)
    patients = np.load("data/processed/patient_ids_new.npy")

    # Evaluate on the held-out patient
    test_mask = patients == "chb03"

    if not np.any(test_mask):
        raise ValueError(
            "No chb03 samples found in patient_ids_new.npy"
        )

    X_test = X[test_mask]
    y_test = y[test_mask]

    print("Held-out patient: chb03")
    print("Test windows:", len(y_test))
    print(
        "Test label counts:",
        dict(zip(*np.unique(y_test, return_counts=True))),
    )

    device = torch.device("cpu")

    # -------------------- 1D CNN --------------------
    cnn_checkpoint = torch.load(
        "models/cnn_1d_patientwise.pt",
        map_location=device,
        weights_only=False,
    )

    X_cnn = normalize(X_test, cnn_checkpoint)

    cnn = EEG_CNN(n_channels=X.shape[1]).to(device)
    cnn.load_state_dict(cnn_checkpoint["model_state_dict"])
    cnn.eval()

    with torch.no_grad():
        cnn_pred = (
            cnn(torch.from_numpy(X_cnn))
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

    # -------------------- CNN-LSTM --------------------
    lstm_checkpoint = torch.load(
        "models/cnn_lstm_patientwise.pt",
        map_location=device,
        weights_only=False,
    )

    X_lstm = normalize(X_test, lstm_checkpoint)

    lstm = EEG_CNN_LSTM(n_channels=X.shape[1]).to(device)
    lstm.load_state_dict(lstm_checkpoint["model_state_dict"])
    lstm.eval()

    with torch.no_grad():
        lstm_pred = (
            lstm(torch.from_numpy(X_lstm))
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

    # -------------------- XGBoost --------------------
    xgb = joblib.load("models/xgboost_baseline.joblib")
    xgb_pred = xgb.predict(extract_features(X_test))

    # All model predictions are collected here
    predictions = {
        "XGBoost": xgb_pred,
        "1D-CNN": cnn_pred,
        "CNN-LSTM": lstm_pred,
    }

    # -------------------- Compare Models --------------------
    rows = []
    os.makedirs("reports", exist_ok=True)

    for name, pred in predictions.items():
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Seizure Precision": precision_score(
                    y_test, pred, pos_label=1, zero_division=0
                ),
                "Seizure Recall": recall_score(
                    y_test, pred, pos_label=1, zero_division=0
                ),
                "Seizure F1": f1_score(
                    y_test, pred, pos_label=1, zero_division=0
                ),
                "True Negatives": int(tn),
                "False Positives": int(fp),
                "False Negatives": int(fn),
                "True Positives": int(tp),
            }
        )

        print(
            f"\n{name} confusion matrix "
            "(actual rows, predicted columns):"
        )
        print(cm)

    # Save metrics to CSV
    df = pd.DataFrame(rows)
    csv_path = "reports/model_comparison.csv"
    df.to_csv(csv_path, index=False)

    print("\nModel comparison (held-out patient chb03):")
    print(df.round(4).to_string(index=False))
    print(f"\nSaved: {csv_path}")

    # -------------------- Create Comparison Graph --------------------
    metric_columns = [
        "Accuracy",
        "Seizure Precision",
        "Seizure Recall",
        "Seizure F1",
    ]

    ax = df.set_index("Model")[metric_columns].plot(
        kind="bar",
        figsize=(10, 6),
        ylim=(0, 1),
    )

    ax.set_ylabel("Score")
    ax.set_title(
        "NeuroGuard Model Comparison — Held-out Patient chb03"
    )
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()

    plot_path = "reports/model_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Saved: {plot_path}")
    print("\nResearch prototype only; not for clinical diagnosis.")


if __name__ == "__main__":
    main()