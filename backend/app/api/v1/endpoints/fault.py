"""
Fault detection endpoint based on ECM degradation thresholds.
"""

from fastapi import APIRouter, HTTPException
from app.api.v1.schemas.battery import FaultDetectRequest, FaultDetectResponse, FaultItem
from app.core.fault_detector import FaultDetector

router = APIRouter()
detector = FaultDetector()

# Default healthy reference values for 18650 Li-ion cell
DEFAULT_BASELINE = {
    "R0": 0.045,
    "R1": 0.020,
    "C1": 1500.0,
    "capacity": 2.0,
}


@router.post("/detect", response_model=FaultDetectResponse)
async def detect_faults(req: FaultDetectRequest):
    """
    Detect battery faults and parameter anomalies relative to baseline healthy cell.
    """
    try:
        current_dict = {
            "R0": req.current_params.R0,
            "R1": req.current_params.R1,
            "C1": req.current_params.C1,
            "capacity": req.current_params.capacity,
        }
        # Filter None
        current_dict = {k: v for k, v in current_dict.items() if v is not None}

        if req.baseline_params:
            baseline_dict = {
                "R0": req.baseline_params.R0,
                "R1": req.baseline_params.R1,
                "C1": req.baseline_params.C1,
                "capacity": req.baseline_params.capacity or DEFAULT_BASELINE.get("capacity", 2.0),
            }
            baseline_dict = {k: v for k, v in baseline_dict.items() if v is not None}
        else:
            baseline_dict = DEFAULT_BASELINE

        alerts = detector.detect(current_dict, baseline_dict)
        overall = detector.overall_status(alerts)

        fault_items = [
            FaultItem(
                type=a.type,
                parameter=a.parameter,
                current_value=a.current_value,
                baseline_value=a.baseline_value,
                delta_percent=a.change_percent,
                severity=a.severity,
                recommendation=a.message,
            )
            for a in alerts
        ]

        return FaultDetectResponse(
            battery_id=req.battery_id,
            faults_detected=fault_items,
            overall_health=overall,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fault detection error: {str(e)}")
