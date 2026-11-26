
import csv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports"
FEEDBACK_FILE = REPORT_DIR / "human_feedback.csv"

VALID_LABELS = {
    "seizure",
    "non_seizure",
    "uncertain",
}


def record_feedback(
    window_index,
    reviewer_label,
    reviewer_note="",
    reviewer_id="reviewer_1",
):
    if reviewer_label not in VALID_LABELS:
        raise ValueError(
            f"reviewer_label must be one of {sorted(VALID_LABELS)}"
        )

    if int(window_index) < 0:
        raise ValueError("window_index cannot be negative.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = FEEDBACK_FILE.exists()

    with FEEDBACK_FILE.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp_utc",
                "window_index",
                "reviewer_id",
                "reviewer_label",
                "reviewer_note",
            ],
        )

        if not file_exists or FEEDBACK_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow({
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "window_index": int(window_index),
            "reviewer_id": reviewer_id,
            "reviewer_label": reviewer_label,
            "reviewer_note": reviewer_note,
        })

    print(f"Feedback saved for window {window_index}.")
    print(f"File: {FEEDBACK_FILE}")


if __name__ == "__main__":
    # Demonstration entry: change these values to test another entry.
    record_feedback(
        window_index=0,
        reviewer_label="uncertain",
        reviewer_note="Demo entry; not a clinical review.",
    )
