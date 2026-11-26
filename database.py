from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parent / "reviews.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS human_reviews (
                review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT NOT NULL,
                reviewer_name TEXT NOT NULL,
                annotation TEXT NOT NULL CHECK(annotation IN ('Possible seizure', 'Noise detected', 'Needs review')),
                notes TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'Reviewed' CHECK(status IN ('Reviewed', 'Needs follow-up')),
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                updated_at TEXT
            )
        """)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_reviews_prediction ON human_reviews(prediction_id)")
        connection.commit()
