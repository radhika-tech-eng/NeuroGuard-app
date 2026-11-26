
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports"

INPUT_FILE = REPORT_DIR / "explainable_ensemble_report.csv"
OUTPUT_FILE = REPORT_DIR / "event_timeline.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input report not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "window_index",
        "patient_id",
        "actual_label",
        "ensemble_probability",
        "temporal_decision",
        "explanation_status",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Preserve the available dataset order within each patient.
    df = df.sort_values(
        ["patient_id", "window_index"]
    ).copy()

    df["review_priority"] = "ROUTINE"

    df.loc[
        df["ensemble_probability"] >= 0.5,
        "review_priority",
    ] = "REVIEW"

    df.loc[
        df["temporal_decision"] == "SUSTAINED_HIGH_PATTERN",
        "review_priority",
    ] = "HIGH_REVIEW"

    df.loc[
        df["explanation_status"] == "REVIEW",
        "review_priority",
    ] = "REVIEW"

    timeline = df[
        [
            "patient_id",
            "window_index",
            "actual_label",
            "ensemble_probability",
            "temporal_decision",
            "explanation_status",
            "review_priority",
        ]
    ].copy()

    timeline.to_csv(OUTPUT_FILE, index=False)

    print("Event timeline created.")
    print(f"Total windows: {len(timeline)}")
    print(f"Patients: {timeline['patient_id'].nunique()}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nReview-priority counts:")
    print(timeline["review_priority"].value_counts())

    print("\nFirst 10 rows:")
    print(timeline.head(10).to_string(index=False))

    print(
        "\nResearch timeline only; not for clinical diagnosis."
    )


if __name__ == "__main__":
    main()
