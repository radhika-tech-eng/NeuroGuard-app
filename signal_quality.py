
import numpy as np


def assess_signal_quality(
    eeg_window,
    flat_std_threshold=1e-6,
    amplitude_limit=None,
):
    """
    Basic EEG signal-quality screening.

    Parameters
    ----------
    eeg_window : numpy array
        Shape: (channels, samples), for example (23, 1280).
    flat_std_threshold : float
        Flags channels with almost no variation.
    amplitude_limit : float or None
        Optional amplitude limit in the dataset's existing units.
        Leave None until the signal units and scale are verified.

    Returns
    -------
    dict
        Signal-quality metrics and a screening result.
    """

    eeg = np.asarray(eeg_window, dtype=np.float64)

    if eeg.ndim != 2:
        raise ValueError(
            "Expected a 2D EEG window: (channels, samples)."
        )

    if eeg.shape[0] == 0 or eeg.shape[1] == 0:
        raise ValueError("EEG window cannot be empty.")

    finite_mask = np.isfinite(eeg)
    non_finite_count = int(eeg.size - finite_mask.sum())

    if non_finite_count > 0:
        return {
            "quality_status": "REVIEW",
            "is_usable": False,
            "reason": "Signal contains NaN or infinite values.",
            "non_finite_count": non_finite_count,
            "flat_channels": [],
            "high_amplitude_channels": [],
            "channel_count": int(eeg.shape[0]),
            "sample_count": int(eeg.shape[1]),
        }

    channel_std = np.std(eeg, axis=1)
    channel_peak_to_peak = np.ptp(eeg, axis=1)

    flat_channels = np.flatnonzero(
        channel_std <= flat_std_threshold
    ).tolist()

    high_amplitude_channels = []

    if amplitude_limit is not None:
        if amplitude_limit <= 0:
            raise ValueError("amplitude_limit must be positive.")

        high_amplitude_channels = np.flatnonzero(
            np.max(np.abs(eeg), axis=1) > amplitude_limit
        ).tolist()

    reasons = []

    if flat_channels:
        reasons.append("Flat or nearly flat channels detected.")

    if high_amplitude_channels:
        reasons.append("Amplitude limit exceeded.")

    if reasons:
        status = "REVIEW"
    else:
        status = "PASS"

    return {
        "quality_status": status,
        "is_usable": status == "PASS",
        "reason": (
            " ".join(reasons)
            if reasons
            else "No configured quality flags detected."
        ),
        "non_finite_count": non_finite_count,
        "flat_channels": flat_channels,
        "high_amplitude_channels": high_amplitude_channels,
        "channel_count": int(eeg.shape[0]),
        "sample_count": int(eeg.shape[1]),
        "mean_channel_std": float(np.mean(channel_std)),
        "max_channel_peak_to_peak": float(
            np.max(channel_peak_to_peak)
        ),
    }


def assess_dataset(eeg_windows):
    """Assess multiple windows and return one result per window."""

    eeg_windows = np.asarray(eeg_windows)

    if eeg_windows.ndim != 3:
        raise ValueError(
            "Expected shape: (windows, channels, samples)."
        )

    return [
        assess_signal_quality(window)
        for window in eeg_windows
    ]


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    data_path = (
        project_root
        / "data"
        / "processed"
        / "eeg_windows_new.npy"
    )

    eeg_windows = np.load(data_path, mmap_mode="r")

    print("Dataset shape:", eeg_windows.shape)

    # Test the first window before assessing the full dataset.
    result = assess_signal_quality(eeg_windows[0])

    print("\nFirst-window quality report:")
    for key, value in result.items():
        print(f"{key}: {value}")

    results = assess_dataset(eeg_windows)

    passed = sum(
        item["quality_status"] == "PASS"
        for item in results
    )
    review = len(results) - passed

    print("\nDataset quality summary")
    print("Total windows:", len(results))
    print("Passed screening:", passed)
    print("Flagged for review:", review)
