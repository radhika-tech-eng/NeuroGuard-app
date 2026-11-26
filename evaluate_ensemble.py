
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "ensemble_temporal_predictions.csv"


def main():
    df = pd.read_csv(REPORT)

    y_true = df["actual_label"].astype(int).to_numpy()
    probabilities = df["ensemble_probability"].to_numpy()

    # Preliminary threshold: 0.5, not clinically validated.
    y_pred = (probabilities >= 0.5).astype(int)

    print("=== OVERALL ENSEMBLE EVALUATION ===")
    print(f"Total windows: {len(df)}")
    print(f"Actual seizure-labelled windows: {int(y_true.sum())}")
    print(f"Predicted positive windows: {int(y_pred.sum())}")

    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=["Non-seizure", "Seizure"],
        zero_division=0,
    ))

    if len(np.unique(y_true)) == 2:
        print("ROC-AUC:", round(roc_auc_score(y_true, probabilities), 4))
        print(
            "Average precision:",
            round(average_precision_score(y_true, probabilities), 4),
        )
    else:
        print("ROC-AUC/AP unavailable: only one actual class found.")

    print("\n=== PER-PATIENT RESULTS ===")
    for patient, group in df.groupby("patient_id"):
        actual = group["actual_label"].astype(int).to_numpy()
        pred = (
            group["ensemble_probability"].to_numpy() >= 0.5
        ).astype(int)

        cm = confusion_matrix(actual, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0

        print(f"\nPatient: {patient}")
        print(f"Windows: {len(group)}")
        print(f"Actual positive windows: {int(actual.sum())}")
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")

    print("\nResearch evaluation only; not for clinical diagnosis.")


if __name__ == "__main__":
    main()
