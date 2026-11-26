
from pathlib import Path

import numpy as np


def explain_eeg_window(window, model_probabilities):
    """
    Creates a transparent, rule-based summary of one EEG window.

    window shape: (channels, samples)
    model_probabilities: dictionary containing model seizure probabilities.

    This is a signal-summary explanation, not a causal explanation
    of the neural networks.
    """

    window = np.asarray(window, dtype=np.float64)

    if window.ndim != 2:
        raise ValueError("Expected EEG window shape: (channels, samples).")

    if not np.isfinite(window).all():
        return {
            "status": "REVIEW_SIGNAL",
            "summary": "Window contains non-finite values.",
            "signal_findings": [],
            "model_findings": [],
        }

    channel_std = np.std(window, axis=1)
    channel_ptp = np.ptp(window, axis=1)
    channel_rms = np.sqrt(np.mean(window ** 2, axis=1))

    findings = []

    flat_channels = np.where(channel_std < 1e-6)[0]
    if len(flat_channels):
        findings.append(
            f"{len(flat_channels)} channel(s) have nearly flat signals."
        )

    median_std = float(np.median(channel_std))
    median_rms = float(np.median(channel_rms))
    median_range = float(np.median(channel_ptp))

    if median_std > 0:
        unusual_variability = np.where(
            channel_std > 5 * median_std
        )[0]
        if len(unusual_variability):
            findings.append(
                f"{len(unusual_variability)} channel(s) have "
                "much higher variability than the channel median."
            )

    if median_range > 0:
        unusual_range = np.where(
            channel_ptp > 5 * median_range
        )[0]
        if len(unusual_range):
            findings.append(
                f"{len(unusual_range)} channel(s) have "
                "much larger amplitude ranges than the median."
            )

    if not findings:
        findings.append(
            "No extreme channel-level differences were found "
            "by these simple checks."
        )

    valid_probabilities = {
        name: float(probability)
        for name, probability in model_probabilities.items()
        if np.isfinite(probability) and 0 <= probability <= 1
    }

    if not valid_probabilities:
        return {
            "status": "REVIEW_MODEL_OUTPUT",
            "summary": "No valid model probabilities were provided.",
            "signal_findings": findings,
            "model_findings": [],
        }

    model_findings = [
        f"{name}: seizure-class probability = {probability:.4f}"
        for name, probability in valid_probabilities.items()
    ]

    values = list(valid_probabilities.values())
    spread = max(values) - min(values)

    if spread >= 0.5:
        model_findings.append(
            "Models differ substantially; treat the prediction as uncertain."
        )
    else:
        model_findings.append(
            "The probability spread is below the configured 0.5 review rule."
        )

    mean_probability = float(np.mean(values))

    if mean_probability >= 0.5:
        summary = (
            "The mean model probability is at least 0.5. "
            "Review the EEG and model outputs; this is not a diagnosis."
        )
    else:
        summary = (
            "The mean model probability is below 0.5. "
            "This does not rule out a seizure."
        )

    return {
        "status": (
    "REVIEW"
    if spread >= 0.35
    else "SUMMARY_AVAILABLE"
),
        "summary": summary,
        "mean_probability": mean_probability,
        "probability_spread": float(spread),
        "median_channel_std": median_std,
        "median_channel_rms": median_rms,
        "median_channel_range": median_range,
        "signal_findings": findings,
        "model_findings": model_findings,
    }


if __name__ == "__main__":
    # Small software test only; these are invented example values.
    rng = np.random.default_rng(42)
    demo_window = rng.normal(size=(23, 1280))

    demo_probabilities = {
        "xgboost": 0.62,
        "cnn_1d": 0.78,
        "cnn_lstm": 0.31,
    }

    explanation = explain_eeg_window(
        demo_window, demo_probabilities
    )

    for key, value in explanation.items():
        print(f"{key}: {value}")

    print("\nSoftware demo only; not a clinical explanation.")
