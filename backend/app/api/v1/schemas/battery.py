"""
Pydantic schemas for BMS API request and response payloads.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


# ============================================================================
# Common / Base
# ============================================================================

class TimeSeriesInput(BaseModel):
    battery_id: str = Field(default="B0005", description="Battery identifier")
    voltage: List[float] = Field(..., description="Voltage readings in Volts")
    current: List[float] = Field(..., description="Current readings in Amperes")
    temperature: List[float] = Field(..., description="Temperature readings in Celsius")
    time: Optional[List[float]] = Field(default=None, description="Time in seconds")


# ============================================================================
# SoH (State of Health)
# ============================================================================

class SoHRequest(TimeSeriesInput):
    pass


class ECMParams(BaseModel):
    R0: float = Field(..., description="Ohmic resistance in Ohms")
    R1: Optional[float] = Field(None, description="Charge-transfer resistance in Ohms")
    C1: Optional[float] = Field(None, description="Polarization capacitance in Farads")
    tau1: Optional[float] = Field(None, description="Time constant R1*C1 in seconds")
    R2: Optional[float] = Field(None, description="Diffusion resistance in Ohms")
    C2: Optional[float] = Field(None, description="Diffusion capacitance in Farads")
    tau2: Optional[float] = Field(None, description="Time constant R2*C2 in seconds")
    capacity: Optional[float] = Field(None, description="Current cell capacity in Ah")


class SoHResponse(BaseModel):
    battery_id: str
    soh_percent: float = Field(..., description="Estimated State of Health percentage (0-100)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model prediction confidence score")
    prediction_source: str = Field(
        default="lstm_model",
        description="Provenance of SoH value: 'lstm_model', 'physics_fallback', or 'error'",
    )
    raw_prediction: Optional[float] = Field(
        default=None,
        description="Raw continuous value output directly by the neural network prior to bounding",
    )
    fallback_reason: Optional[str] = Field(
        default=None,
        description="Explicit explanation if physics surrogate fallback was triggered",
    )
    ecm_params: Optional[ECMParams] = None
    degradation_status: str = Field(..., description="'normal', 'moderate', or 'severe'")
    model_version: str = "hybrid_soh_v1"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# SoC (State of Charge)
# ============================================================================

class SoCRequest(TimeSeriesInput):
    initial_soc: Optional[float] = Field(default=None, description="Optional starting SoC percentage (0-100)")


class SoCResponse(BaseModel):
    battery_id: str
    soc_trajectory: List[float] = Field(..., description="Estimated SoC trajectory (%) over time")
    current_soc: float = Field(..., description="Latest estimated SoC (%)")
    confidence: float = Field(..., ge=0.0, le=1.0)
    prediction_source: str = Field(
        default="xgboost_model",
        description="Provenance: 'xgboost_model', 'coulomb_counting_fallback', or 'error'",
    )
    model_version: str = "hybrid_soc_v1"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# RUL (Remaining Useful Life)
# ============================================================================

class RULRequest(TimeSeriesInput):
    current_cycle: int = Field(..., ge=0, description="Current cycle index")
    capacity: Optional[float] = Field(None, description="Current capacity in Ah")
    nominal_capacity: float = Field(default=2.0, description="Nominal fresh capacity in Ah")


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float


class RULResponse(BaseModel):
    battery_id: str
    rul_cycles: int = Field(..., description="Estimated remaining cycles until 80% EOL")
    rul_days_estimated: Optional[int] = Field(None, description="Estimated remaining days under standard usage")
    confidence_interval: ConfidenceInterval
    eol_threshold: float = Field(default=0.8, description="End-of-life threshold fraction of nominal")
    current_soh: float = Field(..., description="Current SoH fraction (0.0 - 1.0)")
    risk_level: str = Field(..., description="'low', 'medium', or 'high'")
    prediction_source: str = Field(
        default="lstm_model",
        description="Provenance: 'lstm_model', 'empirical_fallback', or 'error'",
    )
    raw_prediction: Optional[float] = Field(
        default=None,
        description="Raw output directly from the LSTM prior to clipping/rounding",
    )
    fallback_reason: Optional[str] = Field(
        default=None,
        description="Reason if fallback was triggered",
    )
    model_version: str = "hybrid_rul_v1"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# ECM (Equivalent Circuit Model)
# ============================================================================

class ECMAnalyzeRequest(BaseModel):
    battery_id: str = Field(default="B0005")
    model_type: str = Field(default="RC", description="'Rint', 'RC' (Thevenin), or '2RC' (PNGV)")
    voltage: List[float]
    current: List[float]
    soc: Optional[List[float]] = None
    time: Optional[List[float]] = None


class ECMMetrics(BaseModel):
    RMSE_mV: float
    R2: Optional[float] = Field(None, description="Coefficient of determination; null if excitation insufficient / undefined variance")
    MAE_mV: float


class ECMAnalyzeResponse(BaseModel):
    battery_id: str
    model_type: str
    estimation_status: str = Field(
        default="converged",
        description="'converged', 'unidentifiable_insufficient_excitation', 'boundary_constrained', or 'failed'"
    )
    boundary_hit: Optional[bool] = Field(
        default=False,
        description="True if one or more parameters pinned to physical optimization boundaries"
    )
    status_message: Optional[str] = Field(
        default=None,
        description="Diagnostic details regarding excitation level or parameter identifiability"
    )
    parameters: ECMParams
    metrics: ECMMetrics
    voltage_predicted: List[float]


# ============================================================================
# Fault Detection
# ============================================================================

class FaultItem(BaseModel):
    type: str
    parameter: str
    current_value: float
    baseline_value: float
    delta_percent: float
    severity: str
    recommendation: str


class FaultDetectRequest(BaseModel):
    battery_id: str = Field(default="B0005")
    current_params: ECMParams
    baseline_params: Optional[ECMParams] = None


class FaultDetectResponse(BaseModel):
    battery_id: str
    faults_detected: List[FaultItem]
    overall_health: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# Health Check
# ============================================================================

class HealthCheckResponse(BaseModel):
    status: str
    version: str
    models_loaded: Dict[str, bool]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
