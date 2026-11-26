
from pathlib import Path
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import predict_windows
from backend.app.reviews.router import router as reviews_router


app = FastAPI(
    title="NeuroGuard API",
    description="EEG seizure-pattern research prototype",
    version="1.1.0",
)

# Human Review API routes
app.include_router(reviews_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 50 * 1024 * 1024
EXPECTED_CHANNELS = 23
EXPECTED_SAMPLES = 1280


def read_and_validate_npy(
    filename: str,
    content: bytes,
) -> np.ndarray:
    """Validate an uploaded NumPy EEG array."""

    if Path(filename or "").suffix.lower() != ".npy":
        raise HTTPException(
            status_code=400,
            detail="Please upload a .npy EEG file.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 50 MB upload limit.",
        )

    try:
        eeg = np.load(
            BytesIO(content),
            allow_pickle=False,
        )
    except (ValueError, OSError, EOFError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported .npy file.",
        )

    if eeg.ndim == 2 and eeg.shape == (
        EXPECTED_CHANNELS,
        EXPECTED_SAMPLES,
    ):
        eeg = eeg[np.newaxis, ...]

    if eeg.ndim != 3 or eeg.shape[1:] != (
        EXPECTED_CHANNELS,
        EXPECTED_SAMPLES,
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Expected (windows, 23, 1280), "
                "or a single window shaped (23, 1280)."
            ),
        )

    if eeg.shape[0] == 0:
        raise HTTPException(
            status_code=400,
            detail="EEG file contains no windows.",
        )

    if not np.issubdtype(eeg.dtype, np.number):
        raise HTTPException(
            status_code=400,
            detail="EEG data must contain numeric values.",
        )

    if not np.isfinite(eeg).all():
        raise HTTPException(
            status_code=400,
            detail="EEG data contains NaN or infinite values.",
        )

    return eeg.astype(np.float32, copy=False)


async def read_upload(file: UploadFile) -> np.ndarray:
    """Read a bounded upload and validate its EEG shape."""

    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 50 MB upload limit.",
        )

    return read_and_validate_npy(
        file.filename or "",
        content,
    )


@app.get("/")
def home():
    return {
        "project": "NeuroGuard",
        "status": "running",
        "message": "NeuroGuard API is ready",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "neuroguard-backend",
    }


@app.post("/upload-npy")
async def upload_npy(file: UploadFile = File(...)):
    eeg = await read_upload(file)

    return {
        "status": "validated",
        "filename": Path(file.filename or "upload.npy").name,
        "windows": int(eeg.shape[0]),
        "channels": int(eeg.shape[1]),
        "samples_per_channel": int(eeg.shape[2]),
        "message": "File validated. No inference was run.",
    }


@app.post("/waveform")
async def get_waveform(
    file: UploadFile = File(...),
    window_index: int = 0,
):
    eeg = await read_upload(file)

    if not 0 <= window_index < eeg.shape[0]:
        raise HTTPException(
            status_code=400,
            detail="Invalid window index.",
        )

    window = eeg[window_index]

    stride = max(1, window.shape[1] // 640)
    indices = np.arange(0, window.shape[1], stride)

    return {
        "window_index": window_index,
        "channels": int(window.shape[0]),
        "samples": int(window.shape[1]),
        "sample_indices": indices.tolist(),
        "signals": window[:, indices].tolist(),
        "message": (
            "EEG samples for visualization only; "
            "not a medical interpretation."
        ),
    }


@app.post("/predict")
async def predict_eeg(file: UploadFile = File(...)):
    eeg = await read_upload(file)

    try:
        results = predict_windows(eeg)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model inference failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    flagged_windows = sum(
        1 for result in results
        if result["research_flag"]
    )

    total_windows = len(results)

    # This is a recording-level summary based on the configured
    # per-window research threshold, not a clinical diagnosis.
    recording_flagged = flagged_windows > 0

    return {
        "status": "research_prediction_complete",
        "filename": Path(file.filename or "upload.npy").name,
        "windows_processed": total_windows,
        "flagged_windows": flagged_windows,
        "non_flagged_windows": total_windows - flagged_windows,
        "recording_prediction_label": (
            "Seizure Detected"
            if recording_flagged
            else "Non-Seizure"
        ),
        "results": results,
        "models": ["XGBoost", "CNN-1D", "CNN-LSTM"],
        "threshold": 0.5,
        "disclaimer": (
            "Research prototype only. Not a medical diagnosis. "
            "Predictions are not clinically validated."
        ),
    }
