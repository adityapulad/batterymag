"""
Application configuration via environment variables.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # ── App ──
    APP_NAME: str = "Battery BMS API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Paths ──
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    UPLOADS_DIR: Path = BASE_DIR / "uploads"

    # ── Database ──
    DATABASE_URL: str = "sqlite+aiosqlite:///./bms.db"

    # ── CORS ──
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000", "*"]

    # ── Model defaults ──
    DEFAULT_BATTERY_ID: str = "B0005"
    SOH_MODEL_PATH: str = "soh/hybrid_soh_v1.keras"
    SOC_MODEL_PATH: str = "soc/hybrid_soc_v1.keras"
    RUL_MODEL_PATH: str = "rul/hybrid_rul_v1.keras"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()

# Ensure directories exist
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
