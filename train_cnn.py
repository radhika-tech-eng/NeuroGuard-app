
import os
import copy
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.models.cnn_1d import EEG_CNN

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "data/processed"
MODEL_DIR = "models"
TEST_PATIENT = "chb03"
EPOCHS = 10
BATCH_SIZE = 16

X = np.load(os.path.join(DATA_DIR, "eeg_windows_new.npy")).astype(np.float32)
y = np.load(os.path.join(DATA_DIR, "eeg_labels_new.npy")).astype(np.int64)
patients = np.load(os.path.join(DATA_DIR, "patient_ids_new.npy"))

assert len(X) == len(y) == len(patients)

# Keep the final test patient completely separate.
train_pool = patients != TEST_PATIENT
test_mask = patients == TEST_PATIENT

if not train_pool.any() or not test_mask.any():
    raise ValueError("Training or test set is empty.")

X_pool, y_pool = X[train_pool], y[train_pool]
X_test_raw, y_test = X[test_mask], y[test_mask]

# Validation comes only from training patients.
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_pool,
    y_pool,
    test_size=0.2,
    random_state=42,
    stratify=y_pool
)

if len(np.unique(y_train)) < 2:
    raise ValueError("Training split must contain both classes.")

# Normalize using training data only.
mean = X_train_raw.mean(axis=(0, 2), keepdims=True)
std = X_train_raw.std(axis=(0, 2), keepdims=True)
std[std < 1e-6] = 1.0

def normalize(data):
    return ((data - mean) / std).astype(np.float32)

X_train = normalize(X_train_raw)
X_val = normalize(X_val_raw)
X_test = normalize(X_test_raw)

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
    batch_size=32,
    shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Training patients:", np.unique(patients[train_pool]))
print("Test patient:", TEST_PATIENT)
print("Train/validation/test:", len(y_train), len(y_val), len(y_test))
print("Test labels:", np.unique(y_test, return_counts=True))

model = EEG_CNN(n_channels=X.shape[1]).to(device)

counts = np.bincount(y_train, minlength=2)
weights = len(y_train) / (2 * np.maximum(counts, 1))

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(weights, dtype=torch.float32, device=device)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float("inf")
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss_sum += loss.item() * yb.size(0)
            val_correct += (logits.argmax(dim=1) == yb).sum().item()
            val_total += yb.size(0)

    val_loss = val_loss_sum / max(val_total, 1)
    val_accuracy = val_correct / max(val_total, 1)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train loss: {total_loss / len(train_loader):.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.4f}",
        flush=True
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())

if best_state is None:
    raise RuntimeError("No best model state was saved.")

model.load_state_dict(best_state)
model.eval()

def evaluate(data, labels, split_name):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data), torch.from_numpy(labels)),
        batch_size=32,
        shuffle=False
    )

    predictions = []

    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().numpy())

    print(f"\n{split_name} classification report:")
    print(classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["Non-seizure", "Seizure"],
        zero_division=0
    ))

    print(f"{split_name} confusion matrix:")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))

    return np.asarray(predictions)

# Validation results are for model selection, not final performance.
evaluate(X_val, y_val, "Validation")

# Evaluate the unseen patient once after training/model selection.
test_predictions = evaluate(X_test, y_test, "Held-out patient test")

os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, "cnn_1d_patientwise.pt")

torch.save({
    "model_state_dict": model.state_dict(),
    "channels": X.shape[1],
    "window_samples": X.shape[2],
    "normalization_mean": mean,
    "normalization_std": std,
    "test_patient": TEST_PATIENT
}, model_path)

print("\nSaved:", model_path)
print("Research prototype only; not for clinical diagnosis.")
