"""
State of Health (SoH) prediction endpoint.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.battery import SoHRequest, SoHResponse
from app.ml.model_registry import model_registry
from app.db.database import get_db, PredictionLog

router = APIRouter()


@router.post("/predict", response_model=SoHResponse)
async def predict_soh(req: SoHRequest, db: AsyncSession = Depends(get_db)):
    """
    Predict battery State of Health (SoH) from time-series operational data.
    """
    if len(req.voltage) < 2 or len(req.current) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 data points are required for time-series estimation.",
        )
    if len(req.voltage) != len(req.current) or len(req.voltage) != len(req.temperature):
        raise HTTPException(
            status_code=400,
            detail="Voltage, current, and temperature arrays must have identical length.",
        )

    try:
        result = model_registry.predict_soh(
            voltage=req.voltage,
            current=req.current,
            temperature=req.temperature,
            time=req.time,
            battery_id=req.battery_id,
        )

        # Log prediction to DB for audit history
        log_entry = PredictionLog(
            battery_id=req.battery_id,
            endpoint="soh",
            soh_percent=result.get("soh_percent"),
            prediction_source=result.get("prediction_source"),
            raw_prediction=result.get("raw_prediction"),
            fallback_reason=result.get("fallback_reason"),
            confidence=result.get("confidence"),
            ecm_parameters=result.get("ecm_params"),
            model_version=result.get("model_version"),
        )
        db.add(log_entry)
        await db.commit()

        return SoHResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
