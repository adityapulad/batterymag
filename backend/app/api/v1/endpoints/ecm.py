"""
Equivalent Circuit Model (ECM) parameter analysis endpoint.
"""

from fastapi import APIRouter, HTTPException
from app.api.v1.schemas.battery import ECMAnalyzeRequest, ECMAnalyzeResponse
from app.ml.model_registry import model_registry

router = APIRouter()


@router.post("/analyze", response_model=ECMAnalyzeResponse)
async def analyze_ecm(req: ECMAnalyzeRequest):
    """
    Fit and evaluate Equivalent Circuit Model (Rint, RC Thevenin, 2RC PNGV).
    """
    if len(req.voltage) < 2 or len(req.current) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 data points are required for ECM fitting.",
        )

    try:
        result = model_registry.analyze_ecm(
            voltage=req.voltage,
            current=req.current,
            model_type=req.model_type,
            soc=req.soc,
            time=req.time,
            battery_id=req.battery_id,
        )
        return ECMAnalyzeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ECM fitting error: {str(e)}")
