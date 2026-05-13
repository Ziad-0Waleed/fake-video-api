import io
import os
import numpy as np
import cv2
import httpx
import imageio
from pathlib import Path
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel, Field

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = 380
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1792
THRESHOLD = 0.5


# ── 1. Strict Pydantic Schemas (The API Contract) ─────────────────────────────
class VideoPredictionRequest(BaseModel):
    video_url: str = Field(..., description="Public or S3 pre-signed URL to the video file.", example="https://storage.example.com/video.mp4")


class VideoPredictionResponse(BaseModel):
    verdict: str = Field(..., description="The classification result. Either 'real' or 'fake'.", example="real")
    confidenceScore: int = Field(..., description="The model's confidence score as an integer (0-100).", example=95)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Deepfake Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Mount static files if you are serving a frontend from the same server
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Global model holders ──────────────────────────────────────────────────────
sequence_model = None
feature_extractor = None


# ── Startup: load models ──────────────────────────────────────────────────────
@app.on_event("startup")
async def load_models():
    global sequence_model, feature_extractor

    model_path = Path("deepfake_video_model.h5")
    if not model_path.exists():
        print("⚠  deepfake_video_model.h5 not found — place it next to app.py")
        return

    sequence_model = keras.models.load_model(str(model_path))
    print("✓  Sequence (GRU) model loaded")

    base = keras.applications.EfficientNetB4(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    outputs = base(inputs)
    feature_extractor = keras.Model(inputs, outputs, name="feature_extractor")
    print("✓  EfficientNetB4 feature extractor ready")


# ── 2. In-Memory Video Helpers ────────────────────────────────────────────────
def crop_center_square(frame: np.ndarray) -> np.ndarray:
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]


def load_video(path: str, max_frames: int = MAX_SEQ_LENGTH) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame[:, :, [2, 1, 0]]  # BGR → RGB
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def prepare_single_video(frames: np.ndarray):
    frames = frames[None, ...]
    frame_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor(batch[None, j, :], training=False)
        frame_mask[i, :length] = 1

    return frame_features, frame_mask


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Deepfake Detection API is live. Visit /docs for documentation."}

@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": sequence_model is not None and feature_extractor is not None,
    }


# ── 3. The Refactored Endpoint ────────────────────────────────────────────────
@app.post(
    "/api/v1/predict",
    response_model=VideoPredictionResponse,
    summary="Analyze Video URL for Deepfakes",
    tags=["Forensics"]
)
async def predict(request: VideoPredictionRequest):
    if sequence_model is None or feature_extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # A. Async Download from S3 / URL
    try:
        async with httpx.AsyncClient() as client:
            # changed request.url to request.video_url here:
            response = await client.get(request.video_url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
            video_bytes = response.content
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch video from URL: {str(e)}")

    if not video_bytes:
        raise HTTPException(status_code=400, detail="The provided URL returned an empty file.")

    # B. Safe Temp File Processing & Inference
    tmp_path = ""
    try:
        # Create a hidden temp file (delete=False is required for Windows compatibility)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        # Read using rock-solid OpenCV
        frames = load_video(tmp_path)
        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="Could not extract frames from video.")

        # Run Inference
        frame_features, frame_mask = prepare_single_video(frames)
        raw_prob = float(sequence_model([frame_features, frame_mask], training=False)[0][0])

        # C. Format Strict Response
        verdict = "fake" if raw_prob >= THRESHOLD else "real"
        confidence_float = raw_prob if verdict == "fake" else (1.0 - raw_prob)
        confidence_int = int(round(confidence_float * 100))

        return VideoPredictionResponse(
            verdict=verdict,
            confidenceScore=confidence_int
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error during inference: {str(e)}")

    finally:
        # D. GUARANTEED CLEANUP: Always delete the temp file from the server
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)