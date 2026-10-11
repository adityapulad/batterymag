"""
FastAPI application entry point.

Battery Management System — Hybrid AI + Physics API.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.v1.router import api_router
from app.ml.model_registry import model_registry
from app.db.database import init_db
from app.utils.logging import RequestLoggingMiddleware


# ---------------------------------------------------------------------------
# Lifespan: load models & initialize DB on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models into memory and initialize database on startup."""
    await init_db()
    model_registry.load_all()
    yield
    model_registry.unload_all()


# ---------------------------------------------------------------------------
# Create the FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI/ML-Based Battery Management System combining "
        "Physics-based ECM models with Deep Learning (LSTM/CNN) "
        "for SoH estimation, SoC estimation, RUL prediction, "
        "and fault detection."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware: CORS and Request Logging
# ---------------------------------------------------------------------------
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Include API routes
# ---------------------------------------------------------------------------
app.include_router(api_router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Battery BMS API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }
