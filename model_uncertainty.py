
import numpy as np


def analyze_model_predictions(
    probabilities,
    uncertainty_threshold=0.35,
    disagreement_threshold=0.50,
):
    """
    Compare seizure probabilities from multiple models.

    probabilities:
        Dictionary such as:
        {
            "xgboost": 0.70,
            "cnn_1d": 0.85,
            "cnn_lstm": 0.20
        }

    Each value must be the model's probability for class 1
    (seizure-positive), not its predicted class label.
    """

    if not isinstance(probabilities, dict) or len(probabilities) < 2:
        raise ValueError(
            "Provide probabilities from at least two models."
        )

    for name, value in probabilities.items():
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(
                f"Invalid probability from {name}: {value}"
            )

    names = list(probabilities.keys())
    values = np.array(
        [probabilities[name] for name in names],
        dtype=float,
    )

    # Each model votes seizure-positive at probability >= 0.5.
    predictions = (values >= 0.5).astype(int)

    disagreement = float(
        np.mean(predictions != predictions[0])
    )

    # Probability spread measures how far apart model scores are.
    probability_spread = float(np.max(values) - np.min(values))

    # Binary entropy: highest near 0.5, lowest near 0 or 1.
    eps = 1e-12
    clipped = np.clip(values, eps, 1 - eps)

    entropy = -(
        clipped * np.log2(clipped)
        + (1 - clipped) * np.log2(1 - clipped)
    )

    mean_entropy = float(np.mean(entropy))

    uncertain = mean_entropy >= uncertainty_threshold
    models_disagree = disagreement > 0 or (
        probability_spread >= disagreement_threshold
    )

    if uncertain or models_disagree:
        status = "REVIEW"
    else:
        status = "CONSENSUS"

    return {
        "model_probabilities": {
            name: float(probabilities[name])
            for name in names
        },
        "model_predictions": {
            name: int(prediction)
            for name, prediction in zip(names, predictions)
        },
        "mean_probability": float(np.mean(values)),
        "probability_spread": probability_spread,
        "mean_entropy": mean_entropy,
        "models_disagree": bool(models_disagree),
        "uncertain": bool(uncertain),
        "status": status,
        "message": (
            "Models disagree or show uncertainty; review this result."
            if status == "REVIEW"
            else "Models are broadly consistent under configured rules."
        ),
    }


if __name__ == "__main__":
    # Demonstration only; these are sample probabilities,
    # not predictions from your trained models.
    example = {
        "xgboost": 0.70,
        "cnn_1d": 0.85,
        "cnn_lstm": 0.20,
    }

    result = analyze_model_predictions(example)

    print("Demo model uncertainty report")
    for key, value in result.items():
        print(f"{key}: {value}")
