
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from xgboost import XGBClassifier


def extract_features(X):
    return np.concatenate([
        X.mean(axis=2),
        X.std(axis=2),
        np.median(np.abs(X), axis=2),
        X.max(axis=2),
        X.min(axis=2)
    ], axis=1)


def main():
    X = np.load("data/processed/eeg_windows_new.npy")
    y = np.load("data/processed/eeg_labels_new.npy").astype(int)
    patients = np.load(
        "data/processed/patient_ids_new.npy"
    ).astype(str)

    os.makedirs("reports", exist_ok=True)
    results = []

    for patient in np.unique(patients):
        print(f"\nTesting patient: {patient}")

        test = patients == patient
        train = ~test

        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        if len(np.unique(y_train)) < 2:
            print("Skipping: training data needs both classes.")
            continue

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=42
            )
        )

        model.fit(extract_features(X_train), y_train)
        pred = model.predict(extract_features(X_test))

        tn, fp, fn, tp = confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).ravel()

        row = {
            "Test Patient": patient,
            "Model": "XGBoost",
            "Test Windows": len(y_test),
            "Seizure Windows": int(y_test.sum()),
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "F1": f1_score(y_test, pred, zero_division=0),
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp)
        }

        results.append(row)
        print("Confusion matrix:", [[tn, fp], [fn, tp]])
        print("Accuracy:", round(row["Accuracy"], 4))
        print("Precision:", round(row["Precision"], 4))
        print("Recall:", round(row["Recall"], 4))
        print("F1:", round(row["F1"], 4))

    df = pd.DataFrame(results)
    output = "reports/patient_wise_evaluation.csv"
    df.to_csv(output, index=False)

    print("\nFINAL RESULTS")
    print(df.round(4).to_string(index=False))
    print("\nSaved to:", output)


if __name__ == "__main__":
    main()


