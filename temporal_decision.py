
from collections import deque
import numpy as np


class TemporalDecisionEngine:
    """
    Experimental decision smoothing for consecutive EEG windows.

    This is a research/demo component, not a clinical alarm system.
    """

    def __init__(
        self,
        probability_threshold=0.5,
        required_positive_windows=3,
        history_size=5,
    ):
        if not 0 <= probability_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")

        if required_positive_windows < 1:
            raise ValueError("required_positive_windows must be >= 1.")

        if history_size < required_positive_windows:
            raise ValueError(
                "history_size must be >= required_positive_windows."
            )

        self.threshold = probability_threshold
        self.required_positive_windows = required_positive_windows
        self.history = deque(maxlen=history_size)

    def update(self, probability):
        probability = float(probability)

        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        self.history.append(probability)

        positive_count = sum(
            p >= self.threshold for p in self.history
        )

        enough_history = (
            len(self.history) >= self.required_positive_windows
        )

        sustained_pattern = (
            enough_history
            and positive_count >= self.required_positive_windows
        )

        if sustained_pattern:
            decision = "SUSTAINED_HIGH_PATTERN"
            message = (
                "Repeated high probabilities detected; "
                "flag for research review."
            )
        elif probability >= self.threshold:
            decision = "ISOLATED_HIGH_WINDOW"
            message = (
                "One high-probability window; wait for more data "
                "and review the pattern."
            )
        else:
            decision = "NO_SUSTAINED_HIGH_PATTERN"
            message = (
                "The configured sustained-high rule was not met."
            )

        return {
            "current_probability": probability,
            "windows_observed": len(self.history),
            "positive_windows_in_history": positive_count,
            "decision": decision,
            "message": message,
        }


if __name__ == "__main__":
    # Demonstration probabilities only, not model predictions.
    demo_probabilities = [0.12, 0.78, 0.25, 0.82, 0.91, 0.88]

    engine = TemporalDecisionEngine(
        probability_threshold=0.5,
        required_positive_windows=3,
        history_size=5,
    )

    for index, probability in enumerate(demo_probabilities):
        result = engine.update(probability)

        print(f"\nWindow {index}:")
        for key, value in result.items():
            print(f"{key}: {value}")

    print("\nDemo complete. Not for clinical use.")
