"""
FastAPI application for Cats vs Dogs classification inference service.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.schemas import PredictionResponse, HealthResponse, ErrorResponse
from src.utils.inference import load_model, predict_from_bytes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="Binary image classification API for cats and dogs",
    version="1.0.0"
)

# Global model variable
MODEL: Optional[tf.keras.Model] = None
MODEL_PATH = os.getenv("MODEL_PATH", "models/cats_dogs_classifier.h5")
IMG_SIZE = (224, 224)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global MODEL

    logger.info("Starting up API...")
    logger.info(f"Loading model from: {MODEL_PATH}")

    try:
        if Path(MODEL_PATH).exists():
            MODEL = load_model(MODEL_PATH)
            logger.info("Model loaded successfully!")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")
            logger.warning("API will run but predictions will fail until model is available")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.warning("API will run but predictions will fail")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Cats vs Dogs Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse with API status and model availability
    """
    model_loaded = MODEL is not None

    logger.info(f"Health check - Model loaded: {model_loaded}")

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Make prediction on uploaded image.

    Args:
        file: Uploaded image file (JPEG, PNG)

    Returns:
        PredictionResponse with class prediction and confidence scores

    Raises:
        HTTPException: If model is not loaded or image processing fails
    """
    # Check if model is loaded
    if MODEL is None:
        logger.error("Prediction attempted but model is not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please contact administrator."
        )

    # Validate file type
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )

    try:
        # Read image bytes
        image_bytes = await file.read()
        logger.info(f"Processing image: {file.filename} ({len(image_bytes)} bytes)")

        # Make prediction
        start_time = datetime.now()
        result = predict_from_bytes(MODEL, image_bytes, IMG_SIZE)
        latency = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"Prediction completed - "
            f"File: {file.filename}, "
            f"Class: {result['class']}, "
            f"Confidence: {result['confidence']:.4f}, "
            f"Latency: {latency:.3f}s"
        )

        return PredictionResponse(
            class_name=result['class'],
            confidence=result['confidence'],
            dog_probability=result['dog_probability'],
            cat_probability=result['cat_probability']
        )

    except tf.errors.InvalidArgumentError as e:
        logger.error(f"Invalid image format: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Please upload a valid JPEG or PNG image."
        )
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during prediction: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Cats vs Dogs Classifier API...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
