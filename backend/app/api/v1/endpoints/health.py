"""
Health check and status endpoint.
"""

from fastapi import APIRouter
from app.config import settings
from app.api.v1.schemas.battery import HealthCheckResponse
from app.ml.model_registry import model_registry

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Returns the operational status and loaded model availability."""
    return HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION,
        models_loaded=model_registry.is_loaded,
    )
