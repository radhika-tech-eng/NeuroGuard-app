from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_expected_directories_exist():
    expected = [
        "src",
        "src/preprocessing",
        "src/models",
        "src/evaluation",
        "data/raw",
        "data/processed",
        "tests",
    ]
    for relative in expected:
        assert (ROOT / relative).is_dir(), f"Missing directory: {relative}"
