"""
API v1 Router aggregation.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import (
    health,
    soh,
    soc,
    rul,
    ecm,
    fault,
    upload,
    history,
)

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(soh.router, prefix="/soh", tags=["State of Health (SoH)"])
api_router.include_router(soc.router, prefix="/soc", tags=["State of Charge (SoC)"])
api_router.include_router(rul.router, prefix="/rul", tags=["Remaining Useful Life (RUL)"])
api_router.include_router(ecm.router, prefix="/ecm", tags=["Equivalent Circuit Model (ECM)"])
api_router.include_router(fault.router, prefix="/fault", tags=["Fault Detection"])
api_router.include_router(upload.router, prefix="/upload", tags=["Dataset Ingestion"])
api_router.include_router(history.router, prefix="/audit", tags=["Audit & History"])
