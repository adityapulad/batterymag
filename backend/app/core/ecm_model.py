"""
Equivalent Circuit Model (ECM) for Battery Analysis — Scientific & Robust Estimation.

Features:
- Rigorous persistent excitation checks (detects 0A rest, flatlines, low current variance)
- Explicit estimation status: 'converged', 'unidentifiable_insufficient_excitation', 'boundary_constrained'
- Physical boundary hit detection for R0, R1, C1, R2, C2
- Numerically valid metric handling (avoids false R² = 1.0 when variance is 0)
- Support for Rint, RC (Thevenin), and 2RC (PNGV) models
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional, Tuple, Any


def check_excitation(current: np.ndarray, voltage: np.ndarray) -> Tuple[bool, str]:
    """
    Verify persistent excitation condition for parameter identifiability.
    Without sufficient current amplitude and dynamic variation, ECM impedance parameters
    cannot be distinguished from OCV or open-circuit rest drift.
    """
    i_abs = np.abs(current)
    max_curr = float(np.max(i_abs)) if len(i_abs) > 0 else 0.0
    curr_std = float(np.std(current)) if len(current) > 0 else 0.0
    v_std = float(np.std(voltage)) if len(voltage) > 0 else 0.0

    if max_curr < 0.08:
        return False, f"Insufficient current magnitude (max={max_curr:.3f}A < 0.08A). Cell is in open-circuit rest; impedance parameters are unidentifiable."
    
    if curr_std < 0.005 and v_std < 0.002:
        return False, f"Static flatline condition (current std={curr_std:.4f}A, voltage std={v_std:.4f}V). Dynamic RC parameters require state excitation."

    return True, "Sufficient excitation detected."


# ---------------------------------------------------------------------------
# Rint Model
# ---------------------------------------------------------------------------
class RintModel:
    """
    Simplest ECM: OCV ---[R0]--- Terminal
    V_terminal = OCV(SoC) - I_load · R0
    """

    def __init__(self):
        self.R0: Optional[float] = 0.046
        self.ocv_params: Optional[np.ndarray] = None
        self.boundary_hit: bool = False
        self.estimation_status: str = "uninitialized"
        self.status_message: str = ""

    def fit(self, voltage: np.ndarray, current: np.ndarray, soc: np.ndarray):
        is_excited, msg = check_excitation(current, voltage)
        if not is_excited:
            self.estimation_status = "unidentifiable_insufficient_excitation"
            self.status_message = msg
            self.R0 = 0.045
            self.ocv_params = np.array([float(np.mean(voltage))])
            return self

        i_load = np.abs(current)
        soc_clean = np.clip(soc, 0.0, 1.0)
        ocv_approx = voltage + i_load * 0.045
        self.ocv_params = np.polyfit(soc_clean, ocv_approx, min(3, max(1, len(soc_clean) - 1)))
        ocv_estimated = np.polyval(self.ocv_params, soc_clean)

        bounds = (0.010, 0.30)
        def objective(r_val):
            v_pred = ocv_estimated - i_load * r_val[0]
            return float(np.mean((voltage - v_pred) ** 2))

        result = minimize(objective, [0.045], bounds=[bounds], method="L-BFGS-B")
        r0_val = float(result.x[0])
        self.boundary_hit = bool(np.isclose(r0_val, bounds[0], atol=1e-4) or np.isclose(r0_val, bounds[1], atol=1e-4))
        self.R0 = r0_val
        self.estimation_status = "boundary_constrained" if self.boundary_hit else "converged"
        self.status_message = f"Optimized with L-BFGS-B (residual MSE={result.fun:.6f})"
        return self

    def predict(self, current: np.ndarray, soc: np.ndarray) -> np.ndarray:
        i_load = np.abs(current)
        soc_clean = np.clip(soc, 0.0, 1.0)
        ocv = np.polyval(self.ocv_params, soc_clean)
        return ocv - i_load * self.R0

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "R0": self.R0,
            "OCV_params": self.ocv_params.tolist() if self.ocv_params is not None else None,
            "boundary_hit": self.boundary_hit,
            "estimation_status": self.estimation_status,
            "status_message": self.status_message,
        }


# ---------------------------------------------------------------------------
# RC Model (Thevenin)
# ---------------------------------------------------------------------------
class RCModel:
    """
    First-order RC: OCV ---[R0]---[R1 || C1]--- Terminal
    V_terminal = OCV - I_load·R0 - V1
    dV1/dt     = -V1/(R1·C1) + I_load/C1
    """

    def __init__(self):
        self.R0: Optional[float] = 0.046
        self.R1: Optional[float] = 0.025
        self.C1: Optional[float] = 1500.0
        self.tau1: Optional[float] = 37.5
        self.ocv_params: Optional[np.ndarray] = None
        self.boundary_hit: bool = False
        self.estimation_status: str = "uninitialized"
        self.status_message: str = ""

    def fit(self, voltage: np.ndarray, current: np.ndarray,
            soc: np.ndarray, time: np.ndarray):
        is_excited, msg = check_excitation(current, voltage)
        if not is_excited:
            self.estimation_status = "unidentifiable_insufficient_excitation"
            self.status_message = msg
            self.R0 = 0.045
            self.R1 = 0.020
            self.C1 = 1500.0
            self.tau1 = 30.0
            self.ocv_params = np.array([float(np.mean(voltage))])
            return self

        i_load = np.abs(current)
        soc_clean = np.clip(soc, 0.0, 1.0)
        dt = np.diff(time, prepend=time[0])
        dt = np.where(dt <= 0, 1.0, dt)

        # Robust initial OCV estimation
        ocv_approx = voltage + i_load * 0.045
        self.ocv_params = np.polyfit(soc_clean, ocv_approx, min(3, max(1, len(soc_clean) - 1)))

        def model_error(params):
            R0, R1, C1 = params
            v_pred = self._simulate_load(i_load, soc_clean, dt, R0, R1, C1)
            return float(np.mean((voltage - v_pred) ** 2))

        bounds = [(0.010, 0.25), (0.002, 0.12), (300.0, 8000.0)]
        init = [0.045, 0.025, 1500.0]
        result = minimize(model_error, init, bounds=bounds, method="L-BFGS-B")

        r0_val = float(result.x[0])
        r1_val = float(result.x[1])
        c1_val = float(result.x[2])

        hit_r0 = bool(np.isclose(r0_val, bounds[0][0], atol=1e-4) or np.isclose(r0_val, bounds[0][1], atol=1e-4))
        hit_r1 = bool(np.isclose(r1_val, bounds[1][0], atol=1e-4) or np.isclose(r1_val, bounds[1][1], atol=1e-4))
        hit_c1 = bool(np.isclose(c1_val, bounds[2][0], atol=1.0) or np.isclose(c1_val, bounds[2][1], atol=1.0))

        self.boundary_hit = hit_r0 or hit_r1 or hit_c1
        self.R0 = r0_val
        self.R1 = r1_val
        self.C1 = c1_val
        self.tau1 = float(self.R1 * self.C1)
        self.estimation_status = "boundary_constrained" if self.boundary_hit else "converged"
        self.status_message = f"Optimized with L-BFGS-B (residual MSE={result.fun:.6f})"
        return self

    def _simulate_load(self, i_load, soc, dt, R0, R1, C1):
        ocv = np.polyval(self.ocv_params, soc)
        V1 = np.zeros_like(i_load)
        tau = max(0.1, R1 * C1)
        for i in range(1, len(i_load)):
            dV1 = (-V1[i - 1] / tau + i_load[i - 1] / max(1.0, C1)) * dt[i]
            V1[i] = V1[i - 1] + dV1
        return ocv - i_load * R0 - V1

    def predict(self, current: np.ndarray, soc: np.ndarray,
                time: np.ndarray) -> np.ndarray:
        i_load = np.abs(current)
        soc_clean = np.clip(soc, 0.0, 1.0)
        dt = np.diff(time, prepend=time[0])
        dt = np.where(dt <= 0, 1.0, dt)
        return self._simulate_load(i_load, soc_clean, dt, self.R0, self.R1, self.C1)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "R0": self.R0, "R1": self.R1, "C1": self.C1, "tau1": self.tau1,
            "OCV_params": self.ocv_params.tolist() if self.ocv_params is not None else None,
            "boundary_hit": self.boundary_hit,
            "estimation_status": self.estimation_status,
            "status_message": self.status_message,
        }


# ---------------------------------------------------------------------------
# 2RC Model (PNGV)
# ---------------------------------------------------------------------------
class TwoRCModel:
    """
    Second-order RC: OCV ---[R0]---[R1||C1]---[R2||C2]--- Terminal
    """

    def __init__(self):
        self.R0: Optional[float] = 0.046
        self.R1: Optional[float] = 0.020
        self.C1: Optional[float] = 1200.0
        self.tau1: Optional[float] = 24.0
        self.R2: Optional[float] = 0.025
        self.C2: Optional[float] = 2500.0
        self.tau2: Optional[float] = 62.5
        self.ocv_params: Optional[np.ndarray] = None
        self.boundary_hit: bool = False
        self.estimation_status: str = "uninitialized"
        self.status_message: str = ""

    def fit(self, voltage: np.ndarray, current: np.ndarray,
            soc: np.ndarray, time: np.ndarray):
        is_excited, msg = check_excitation(current, voltage)
        if not is_excited:
            self.estimation_status = "unidentifiable_insufficient_excitation"
            self.status_message = msg
            self.R0 = 0.045
            self.R1 = 0.020
            self.C1 = 1200.0
            self.tau1 = 24.0
            self.R2 = 0.025
            self.C2 = 2500.0
            self.tau2 = 62.5
            self.ocv_params = np.array([float(np.mean(voltage))])
            return self

        i_load = np.abs(current)
        soc_clean = np.clip(soc, 0.0, 1.0)
        dt = np.diff(time, prepend=time[0])
        dt = np.where(dt <= 0, 1.0, dt)

        ocv_approx = voltage + i_load * 0.045
        self.ocv_params = np.polyfit(soc_clean, ocv_approx, min(3, max(1, len(soc_clean) - 1)))

        def model_error(params):
            R0, R1, C1, R2, C2 = params
            v_pred = self._simulate_load(i_load, soc_clean, dt, R0, R1, C1, R2, C2)
            return float(np.mean((voltage - v_pred) ** 2))

        bounds = [
            (0.010, 0.25),
            (0.002, 0.10), (300.0, 5000.0),
            (0.002, 0.10), (500.0, 9000.0)
        ]
        init = [0.045, 0.020, 1200.0, 0.025, 2500.0]
        result = minimize(model_error, init, bounds=bounds, method="L-BFGS-B")

        r0_val = float(result.x[0])
        r1_val = float(result.x[1])
        c1_val = float(result.x[2])
        r2_val = float(result.x[3])
        c2_val = float(result.x[4])

        hits = [
            np.isclose(r0_val, bounds[0][0], atol=1e-4) or np.isclose(r0_val, bounds[0][1], atol=1e-4),
            np.isclose(r1_val, bounds[1][0], atol=1e-4) or np.isclose(r1_val, bounds[1][1], atol=1e-4),
            np.isclose(c1_val, bounds[2][0], atol=1.0) or np.isclose(c1_val, bounds[2][1], atol=1.0),
            np.isclose(r2_val, bounds[3][0], atol=1e-4) or np.isclose(r2_val, bounds[3][1], atol=1e-4),
            np.isclose(c2_val, bounds[4][0], atol=1.0) or np.isclose(c2_val, bounds[4][1], atol=1.0),
        ]
        self.boundary_hit = any(hits)
        self.R0 = r0_val
        self.R1 = r1_val
        self.C1 = c1_val
        self.tau1 = float(self.R1 * self.C1)
        self.R2 = r2_val
        self.C2 = c2_val
        self.tau2 = float(self.R2 * self.C2)
        self.estimation_status = "boundary_constrained" if self.boundary_hit else "converged"
        self.status_message = f"Optimized with L-BFGS-B (residual MSE={result.fun:.6f})"
        return self

    def _simulate_load(self, i_load, soc, dt, R0, R1, C1, R2, C2):
        ocv = np.polyval(self.ocv_params, soc)
        V1 = np.zeros_like(i_load)
        V2 = np.zeros_like(i_load)
        tau1 = max(0.1, R1 * C1)
        tau2 = max(0.1, R2 * C2)
        for i in range(1, len(i_load)):
            V1[i] = V1[i - 1] + (-V1[i - 1] / tau1 + i_load[i - 1] / max(1.0, C1)) * dt[i]
            V2[i] = V2[i - 1] + (-V2[i - 1] / tau2 + i_load[i - 1] / max(1.0, C2)) * dt[i]
        return ocv - i_load * R0 - V1 - V2

    def predict(self, current: np.ndarray, soc: np.ndarray,
                time: np.ndarray) -> np.ndarray:
        i_load = np.abs(current)
        soc_clean = np.clip(soc, 0.0, 1.0)
        dt = np.diff(time, prepend=time[0])
        dt = np.where(dt <= 0, 1.0, dt)
        return self._simulate_load(i_load, soc_clean, dt,
                                  self.R0, self.R1, self.C1, self.R2, self.C2)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "R0": self.R0, "R1": self.R1, "C1": self.C1, "tau1": self.tau1,
            "R2": self.R2, "C2": self.C2, "tau2": self.tau2,
            "OCV_params": self.ocv_params.tolist() if self.ocv_params is not None else None,
            "boundary_hit": self.boundary_hit,
            "estimation_status": self.estimation_status,
            "status_message": self.status_message,
        }


# ---------------------------------------------------------------------------
# SoC Estimator (Coulomb Counting with Signed Conventions)
# ---------------------------------------------------------------------------
class SoCEstimator:
    """
    Coulomb Counting with rigorous physical sign convention:
    I > 0: Discharge (depletes capacity)
    I < 0: Charge (replenishes capacity)
    SoC(t) = SoC(0) - (1 / Q_nominal) · ∫ I(t) dt
    """

    def __init__(self, nominal_capacity: float = 2.0):
        self.Q_nom = float(nominal_capacity)

    def estimate(self, current: np.ndarray, time: np.ndarray,
                 soc_initial: float = 1.0) -> np.ndarray:
        if len(current) == 0:
            return np.array([])
        dt = np.diff(time, prepend=time[0])
        dt = np.where(dt <= 0, 1.0, dt)

        # Standard Ampere-hour accumulation
        # Current in Amperes, time delta in seconds -> Ampere-hours = (I * dt) / 3600
        # If input current is already signed according to load convention, handle consistently
        ah_delta = (current * dt) / 3600.0
        cumulative_ah = np.cumsum(ah_delta)
        soc = soc_initial - (cumulative_ah / max(0.1, self.Q_nom))
        return np.clip(soc, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Metrics helper with Valid Variance Handling
# ---------------------------------------------------------------------------
def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_excited: bool = True) -> Dict[str, Optional[float]]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # Calculate R2 only if there is real variance in the ground truth signal
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))

    if not is_excited or ss_tot < 1e-7:
        # Ground truth has zero variance (e.g. constant flatline or rest).
        # Reporting 0.0 or 1.0 is statistically deceptive; R2 is undefined.
        r2 = None
    else:
        r2 = float(max(-1.0, min(1.0, 1.0 - (ss_res / ss_tot))))

    return {"RMSE": rmse, "MAE": mae, "R2": r2}
