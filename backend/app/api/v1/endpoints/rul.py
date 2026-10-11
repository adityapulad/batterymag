"""
Remaining Useful Life (RUL) prediction endpoint.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.battery import RULRequest, RULResponse
from app.ml.model_registry import model_registry
from app.db.database import get_db, PredictionLog

router = APIRouter()


@router.post("/predict", response_model=RULResponse)
async def predict_rul(req: RULRequest, db: AsyncSession = Depends(get_db)):
    """
    Predict battery Remaining Useful Life (RUL) in cycle counts.
    """
    if len(req.voltage) < 2 or len(req.current) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 data points are required for time-series estimation.",
        )

    try:
        result = model_registry.predict_rul(
            voltage=req.voltage,
            current=req.current,
            temperature=req.temperature,
            current_cycle=req.current_cycle,
            capacity=req.capacity,
            nominal_capacity=req.nominal_capacity,
            battery_id=req.battery_id,
            time=req.time,
        )

        log_entry = PredictionLog(
            battery_id=req.battery_id,
            endpoint="rul",
            rul_cycles=result.get("rul_cycles"),
            prediction_source=result.get("prediction_source"),
            raw_prediction=result.get("raw_prediction"),
            fallback_reason=result.get("fallback_reason"),
            confidence=0.95,
            model_version=result.get("model_version"),
        )
        db.add(log_entry)
        await db.commit()

        return RULResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
