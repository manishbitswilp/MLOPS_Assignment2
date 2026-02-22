"""
Pydantic models for API request and response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    class_name: str = Field(
        ...,
        alias='class',
        description="Predicted class: 'cat' or 'dog'"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the predicted class (0-1)"
    )
    dog_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of dog class"
    )
    cat_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of cat class"
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "class": "dog",
                "confidence": 0.9523,
                "dog_probability": 0.9523,
                "cat_probability": 0.0477
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(
        ...,
        description="API health status"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded and ready"
    )
    version: str = Field(
        default="1.0.0",
        description="API version"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Response model for error messages."""

    error: str = Field(
        ...,
        description="Error message"
    )
    detail: Optional[str] = Field(
        None,
        description="Detailed error information"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid image format",
                "detail": "The uploaded file is not a valid image"
            }
        }
