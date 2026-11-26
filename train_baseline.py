
import os
import numpy as np
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

DATA_DIR = "data/processed"
TEST_PATIENT = "chb03"

X = np.load(os.path.join(DATA_DIR, "eeg_windows_new.npy"))
y = np.load(os.path.join(DATA_DIR, "eeg_labels_new.npy"))
patients = np.load(os.path.join(DATA_DIR, "patient_ids_new.npy"))

print("Dataset:", X.shape)
print("Labels:", np.unique(y, return_counts=True))
print("Patients:", np.unique(patients))

# Hold out the complete chb03 patient
train_mask = patients != TEST_PATIENT
test_mask = patients == TEST_PATIENT

if not train_mask.any() or not test_mask.any():
    raise ValueError("Training or test set is empty.")

X_train_raw, X_test_raw = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

if len(np.unique(y_train)) < 2:
    raise ValueError("Training data must contain both classes.")

def extract_features(data):
    return np.concatenate([
        data.mean(axis=2),
        data.std(axis=2),
        np.median(np.abs(data), axis=2),
        data.max(axis=2),
        data.min(axis=2)
    ], axis=1)

X_train = extract_features(X_train_raw)
X_test = extract_features(X_test_raw)

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

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nTraining patients:", np.unique(patients[train_mask]))
print("Test patient:", TEST_PATIENT)
print("Training windows:", len(y_train))
print("Test windows:", len(y_test))
print("Test labels:", np.unique(y_test, return_counts=True))

print("\nClassification Report:")
print(classification_report(
    y_test,
    pred,
    labels=[0, 1],
    target_names=["Non-seizure", "Seizure"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred, labels=[0, 1]))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgboost_baseline.joblib")

print("\nModel saved: models/xgboost_baseline.joblib")
print("Research prototype only; not for clinical diagnosis.")
