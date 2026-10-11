"""
State of Charge (SoC) prediction endpoint.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.battery import SoCRequest, SoCResponse
from app.ml.model_registry import model_registry
from app.db.database import get_db, PredictionLog

router = APIRouter()


@router.post("/predict", response_model=SoCResponse)
async def predict_soc(req: SoCRequest, db: AsyncSession = Depends(get_db)):
    """
    Predict battery State of Charge (SoC) profile over time.
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
        result = model_registry.predict_soc(
            voltage=req.voltage,
            current=req.current,
            temperature=req.temperature,
            time=req.time,
            battery_id=req.battery_id,
            initial_soc=req.initial_soc,
        )

        log_entry = PredictionLog(
            battery_id=req.battery_id,
            endpoint="soc",
            current_soc=result.get("current_soc"),
            prediction_source=result.get("prediction_source"),
            confidence=result.get("confidence"),
            model_version=result.get("model_version"),
        )
        db.add(log_entry)
        await db.commit()

        return SoCResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
