
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

EXPECTED_CHANNELS = 23
EXPECTED_SAMPLES = 1280
SEIZURE_THRESHOLD = 0.5


def extract_features(windows: np.ndarray) -> np.ndarray:
    """Extract five statistical features for every EEG channel."""

    return np.concatenate(
        [
            windows.mean(axis=2),
            windows.std(axis=2),
            np.median(np.abs(windows), axis=2),
            windows.max(axis=2),
            windows.min(axis=2),
        ],
        axis=1,
    ).astype(np.float32)


def load_checkpoint(path: Path, model_class):
    """Load a saved PyTorch model and its normalization statistics."""

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    model = model_class(
        n_channels=int(checkpoint["channels"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = np.asarray(
        checkpoint["normalization_mean"],
        dtype=np.float32,
    ).reshape(-1)

    std = np.asarray(
        checkpoint["normalization_std"],
        dtype=np.float32,
    ).reshape(-1)

    std = np.where(std < 1e-6, 1.0, std)

    if len(mean) != EXPECTED_CHANNELS:
        raise ValueError(
            f"Expected {EXPECTED_CHANNELS} normalization means, "
            f"received {len(mean)}."
        )

    if len(std) != EXPECTED_CHANNELS:
        raise ValueError(
            f"Expected {EXPECTED_CHANNELS} normalization standard "
            f"deviations, received {len(std)}."
        )

    return model, mean, std


def normalize_windows(
    windows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Normalize each channel using training-set statistics."""

    if len(mean) != windows.shape[1]:
        raise ValueError("Normalization mean does not match EEG channels.")

    if len(std) != windows.shape[1]:
        raise ValueError("Normalization std does not match EEG channels.")

    return (
        (windows - mean[None, :, None])
        / std[None, :, None]
    ).astype(np.float32)


@lru_cache(maxsize=1)
def load_models():
    """Load the three trained models once per backend process."""

    from src.models.cnn_1d import EEG_CNN
    from src.models.cnn_lstm import EEG_CNN_LSTM

    xgb_path = MODEL_DIR / "xgboost_baseline.joblib"
    cnn_path = MODEL_DIR / "cnn_1d_patientwise.pt"
    lstm_path = MODEL_DIR / "cnn_lstm_patientwise.pt"

    required = [xgb_path, cnn_path, lstm_path]
    missing = [str(path) for path in required if not path.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing required model file(s): " + ", ".join(missing)
        )

    xgb = joblib.load(xgb_path)

    # The current project assumes neural-network output index 1
    # represents the seizure class. Verify this against training labels.
    classes = getattr(xgb, "classes_", None)

    if classes is None or 1 not in classes:
        raise ValueError(
            "XGBoost must expose class label 1 for seizure. "
            "Verify the training labels and model classes."
        )

    xgb_seizure_index = int(np.flatnonzero(classes == 1)[0])

    cnn, cnn_mean, cnn_std = load_checkpoint(
        cnn_path,
        EEG_CNN,
    )

    lstm, lstm_mean, lstm_std = load_checkpoint(
        lstm_path,
        EEG_CNN_LSTM,
    )

    return (
        xgb,
        xgb_seizure_index,
        cnn,
        cnn_mean,
        cnn_std,
        lstm,
        lstm_mean,
        lstm_std,
    )


def predict_windows(windows: np.ndarray) -> list[dict]:
    """
    Predict EEG windows shaped (N, 23, 1280).

    Scores are exploratory model scores, not clinical probabilities.
    """

    windows = np.asarray(windows, dtype=np.float32)

    if windows.ndim != 3:
        raise ValueError(
            "Expected EEG shape (windows, 23, 1280)."
        )

    if windows.shape[1:] != (
        EXPECTED_CHANNELS,
        EXPECTED_SAMPLES,
    ):
        raise ValueError(
            "Expected EEG shape (windows, 23, 1280)."
        )

    if windows.shape[0] == 0:
        raise ValueError("No EEG windows were provided.")

    if not np.isfinite(windows).all():
        raise ValueError("EEG windows contain NaN or infinite values.")

    (
        xgb,
        xgb_seizure_index,
        cnn,
        cnn_mean,
        cnn_std,
        lstm,
        lstm_mean,
        lstm_std,
    ) = load_models()

    features = extract_features(windows)

    xgb_scores = xgb.predict_proba(features)[:, xgb_seizure_index]

    cnn_input = torch.from_numpy(
        normalize_windows(windows, cnn_mean, cnn_std)
    )

    lstm_input = torch.from_numpy(
        normalize_windows(windows, lstm_mean, lstm_std)
    )

    with torch.no_grad():
        cnn_outputs = cnn(cnn_input)
        lstm_outputs = lstm(lstm_input)

        if cnn_outputs.ndim != 2 or cnn_outputs.shape[1] != 2:
            raise ValueError("CNN must return two class scores per window.")

        if lstm_outputs.ndim != 2 or lstm_outputs.shape[1] != 2:
            raise ValueError("CNN-LSTM must return two class scores per window.")

        cnn_probabilities = torch.softmax(cnn_outputs.float(), dim=1)
        cnn_scores = cnn_probabilities[:, 1].detach().cpu().numpy().astype(float)
        print("CNN output shape:", cnn_outputs.shape)
        print("CNN probabilities (first 5):",
        torch.softmax(cnn_outputs, dim=1)[:5].cpu().numpy())

        lstm_scores = torch.softmax(
            lstm_outputs, dim=1
        )[:, 1].cpu().numpy()

    # Equal-weight average of the three model scores.
    ensemble_scores = (
        xgb_scores + cnn_scores + lstm_scores
    ) / 3.0

    results = []

    for i, ensemble_score in enumerate(ensemble_scores):
        score = float(ensemble_score)
        flagged = score >= SEIZURE_THRESHOLD

        model_scores = {
            "xgboost": float(xgb_scores[i]),
            "cnn_1d": float(cnn_scores[i]),
            "cnn_lstm": float(lstm_scores[i]),
            "ensemble": score,
        }

        results.append(
            {
                "window_index": i,
                "model_scores": model_scores,
                "threshold": SEIZURE_THRESHOLD,
                "research_flag": flagged,
                "prediction_label": (
                    "Seizure Detected"
                    if flagged
                    else "Non-Seizure"
                ),
                "interpretation": (
                    "The research model flagged this window for a "
                    "possible seizure pattern."
                    if flagged
                    else
                    "The research model did not flag this window "
                    "at the configured threshold."
                ),
            }
        )

    return results
