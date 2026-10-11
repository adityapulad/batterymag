"""
Model registry and inference manager for BMS.

Loads trained Keras/TensorFlow LSTM models and scikit-learn/XGBoost models,
or gracefully falls back to physics-informed analytical estimation (ECM + Coulomb counting + degradation law)
if weights are not present.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import joblib

from app.config import settings
from app.core.ecm_model import RCModel, RintModel, TwoRCModel, SoCEstimator, calculate_model_metrics
from app.ml.preprocessing import (
    prepare_soh_features,
    prepare_soc_features,
    prepare_rul_features,
)

logger = logging.getLogger("bms.ml.registry")


class ModelRegistry:
    """Central registry and inference orchestrator for BMS models."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.is_loaded: Dict[str, bool] = {
            "soh": False,
            "soc": False,
            "rul": False,
        }
        self.model_versions: Dict[str, str] = {
            "soh": "hybrid_soh_lstm_v1",
            "soc": "xgboost_soc_v1",
            "rul": "hybrid_rul_lstm_v1",
        }

    def load_all(self):
        """Attempt to load saved models from disk."""
        logger.info("Initializing Model Registry...")
        self._load_soh_model()
        self._load_soc_model()
        self._load_rul_model()

    def unload_all(self):
        """Clean up loaded models."""
        self.models.clear()
        self.scalers.clear()
        for k in self.is_loaded:
            self.is_loaded[k] = False
        logger.info("Model Registry unloaded.")

    def _load_soh_model(self):
        soh_path = settings.MODELS_DIR / settings.SOH_MODEL_PATH
        scaler_path = settings.MODELS_DIR / "soh" / "scaler_soh.joblib"
        if soh_path.exists():
            try:
                from tensorflow.keras.models import load_model
                self.models["soh"] = load_model(str(soh_path), compile=False)
                if scaler_path.exists():
                    self.scalers["soh"] = joblib.load(str(scaler_path))
                self.is_loaded["soh"] = True
                logger.info(f"Loaded trained SoH LSTM model from {soh_path}")
            except Exception as e:
                logger.warning(f"Could not load SoH LSTM model ({e}). Using physics surrogate.")
                self.is_loaded["soh"] = False
        else:
            self.is_loaded["soh"] = False

    def _load_soc_model(self):
        xgb_path = settings.MODELS_DIR / "soc" / "xgboost_soc_v1.joblib"
        scaler_path = settings.MODELS_DIR / "soc" / "scaler_soc.joblib"
        if xgb_path.exists() and scaler_path.exists():
            try:
                self.models["soc"] = joblib.load(str(xgb_path))
                self.scalers["soc"] = joblib.load(str(scaler_path))
                self.is_loaded["soc"] = True
                logger.info(f"Loaded trained XGBoost SoC model from {xgb_path}")
            except Exception as e:
                logger.warning(f"Could not load XGBoost SoC model ({e}). Using Coulomb counting.")
                self.is_loaded["soc"] = False
        else:
            self.is_loaded["soc"] = False

    def _load_rul_model(self):
        rul_path = settings.MODELS_DIR / settings.RUL_MODEL_PATH
        scaler_path = settings.MODELS_DIR / "rul" / "scaler_rul.joblib"
        if rul_path.exists():
            try:
                from tensorflow.keras.models import load_model
                self.models["rul"] = load_model(str(rul_path), compile=False)
                if scaler_path.exists():
                    self.scalers["rul"] = joblib.load(str(scaler_path))
                self.is_loaded["rul"] = True
                logger.info(f"Loaded trained RUL LSTM model from {rul_path}")
            except Exception as e:
                logger.warning(f"Could not load RUL LSTM model ({e}). Using physics surrogate.")
                self.is_loaded["rul"] = False
        else:
            self.is_loaded["rul"] = False

    # ------------------------------------------------------------------------
    # SoH Inference (Trained LSTM Model)
    # ------------------------------------------------------------------------
    def predict_soh(
        self,
        voltage: List[float],
        current: List[float],
        temperature: List[float],
        time: Optional[List[float]] = None,
        battery_id: str = "B0005",
    ) -> Dict[str, Any]:
        """
        Estimate State of Health (SoH) and extract ECM equivalent circuit parameters.
        """
        v_arr = np.array(voltage, dtype=float)
        i_arr = np.array(current, dtype=float)
        t_arr = np.array(temperature, dtype=float)
        time_arr = np.array(time, dtype=float) if time else np.arange(len(v_arr), dtype=float)

        # Strict physical validation: check for NaN, Inf, and length mismatches
        if np.any(np.isnan(v_arr)) or np.any(np.isinf(v_arr)):
            raise ValueError("Telemetry contains invalid NaN or Infinite voltage values.")
        if np.any(np.isnan(i_arr)) or np.any(np.isinf(i_arr)):
            raise ValueError("Telemetry contains invalid NaN or Infinite current values.")
        if np.any(np.isnan(t_arr)) or np.any(np.isinf(t_arr)):
            raise ValueError("Telemetry contains invalid NaN or Infinite temperature values.")
        if len(v_arr) != len(i_arr) or len(v_arr) != len(t_arr) or len(v_arr) != len(time_arr):
            raise ValueError(f"Array length mismatch: voltage({len(v_arr)}), current({len(i_arr)}), temperature({len(t_arr)}), time({len(time_arr)}).")
        if len(time_arr) > 1 and np.any(np.diff(time_arr) < 0):
            raise ValueError("Time vector must be monotonically non-decreasing.")

        # Check physical domain for supported lithium-ion cell chemistry (1.5V to 5.0V, -30C to 85C)
        is_voltage_outlier = bool(np.any(v_arr < 1.5) or np.any(v_arr > 5.0))
        is_temp_outlier = bool(np.any(t_arr < -30.0) or np.any(t_arr > 90.0))

        # 1. Fit ECM parameters from operational profile
        soc_est = SoCEstimator().estimate(i_arr, time_arr)
        delta_v = np.abs(v_arr[0] - v_arr[1]) if len(v_arr) > 1 else 0.05
        delta_i = np.abs(i_arr[1]) if len(i_arr) > 1 and np.abs(i_arr[1]) > 0.1 else 2.0
        r0_step = float(np.clip(delta_v / delta_i, 0.03, 0.15))

        rc_model = RCModel().fit(v_arr, i_arr, soc_est, time_arr)
        ecm_params = rc_model.get_parameters()

        ecm_dict = {
            "R0": round(r0_step, 5),
            "R1": 0.022,
            "C1": 1500.0,
            "tau1": 33.0,
        }
        ecm_fitted = {
            "R0": round(float(ecm_params.get("R0", r0_step)), 5),
            "R1": round(float(ecm_params.get("R1", 0.022)), 5),
            "C1": round(float(ecm_params.get("C1", 1500.0)), 2),
            "tau1": round(float(ecm_params.get("tau1", 33.0)), 2),
        }

        # 2. LSTM Neural Network Inference
        phys_soh, phys_conf, phys_ver = self._physics_soh(r0_step, v_arr, i_arr, t_arr)
        prediction_source = "physics_fallback"
        raw_prediction = None
        fallback_reason = None

        if is_voltage_outlier or is_temp_outlier:
            fallback_reason = f"Extreme sensor outlier detected: Voltage range [{np.min(v_arr):.2f}V, {np.max(v_arr):.2f}V], Temp range [{np.min(t_arr):.1f}C, {np.max(t_arr):.1f}C] exceeds physical cell boundary."
            logger.warning(fallback_reason)
            soh_percent, confidence, version = phys_soh, 0.50, "physics_sensor_alert_v1"
            prediction_source = "sensor_anomaly_fallback"
        elif self.is_loaded.get("soh") and "soh" in self.models:
            try:
                X = prepare_soh_features(v_arr, i_arr, t_arr, ecm_dict, time=time_arr)
                if self.scalers.get("soh") is not None:
                    n_s, s_len, n_f = X.shape
                    X_scaled = self.scalers["soh"].transform(X.reshape(-1, n_f)).reshape(n_s, s_len, n_f)
                else:
                    X_scaled = X
                pred = self.models["soh"].predict(X_scaled, verbose=0)
                raw_pred = float(pred[0][0])
                raw_prediction = round(raw_pred, 4)

                # Strict validation of raw model prediction
                if 0.4 <= raw_pred <= 1.2:
                    soh_percent = float(np.clip(raw_pred * 100.0, 40.0, 100.0))
                    confidence = 0.96
                    version = self.model_versions["soh"]
                    prediction_source = "lstm_model"
                    fallback_reason = None
                elif 40.0 <= raw_pred <= 100.0:
                    soh_percent = float(np.clip(raw_pred, 40.0, 100.0))
                    confidence = 0.96
                    version = self.model_versions["soh"]
                    prediction_source = "lstm_model"
                    fallback_reason = None
                else:
                    # Model produced an out-of-range value
                    fallback_reason = f"LSTM raw prediction ({raw_pred:.3f}) was outside physical domain [0.4, 1.2]."
                    logger.warning(f"SoH model out-of-bounds: {fallback_reason}. Explicitly reporting physics fallback.")
                    soh_percent, confidence, version = phys_soh, phys_conf, phys_ver
                    prediction_source = "physics_fallback"
            except ValueError as ve:
                fallback_reason = str(ve)
                logger.info(f"SoH input history insufficient: {fallback_reason}. Using physics surrogate.")
                soh_percent, confidence, version = phys_soh, phys_conf, phys_ver
                prediction_source = "physics_fallback"
            except Exception as ex:
                fallback_reason = f"Exception evaluating LSTM model: {str(ex)}"
                logger.warning(f"Error evaluating loaded SoH model ({ex}), explicitly activating physics fallback.")
                soh_percent, confidence, version = phys_soh, phys_conf, phys_ver
                prediction_source = "physics_fallback"
        else:
            fallback_reason = "SoH LSTM model is not loaded in registry."
            soh_percent, confidence, version = phys_soh, phys_conf, phys_ver
            prediction_source = "physics_fallback"

        if soh_percent >= 85.0:
            status = "normal"
        elif soh_percent >= 75.0:
            status = "moderate"
        else:
            status = "severe"

        return {
            "battery_id": battery_id,
            "soh_percent": round(soh_percent, 2),
            "confidence": round(confidence, 2),
            "prediction_source": prediction_source,
            "raw_prediction": raw_prediction,
            "fallback_reason": fallback_reason,
            "ecm_params": ecm_fitted,
            "degradation_status": status,
            "model_version": version,
        }

    def _physics_soh(self, R0: float, v: np.ndarray, i: np.ndarray, t: np.ndarray) -> Tuple[float, float, str]:
        R0_fresh = 0.045
        R0_eol = 0.095
        ratio = (R0 - R0_fresh) / (R0_eol - R0_fresh)
        soh_fraction = 1.0 - 0.25 * np.clip(ratio, 0.0, 1.2)
        soh_percent = float(np.clip(soh_fraction * 100, 50.0, 100.0))
        return soh_percent, 0.91, "physics_surrogate_v1"

    # ------------------------------------------------------------------------
    # SoC Inference (Trained XGBoost Model)
    # ------------------------------------------------------------------------
    def predict_soc(
        self,
        voltage: List[float],
        current: List[float],
        temperature: List[float],
        time: Optional[List[float]] = None,
        battery_id: str = "B0005",
        initial_soc: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Estimate SoC trajectory and current state of charge using trained XGBoost regressor.
        """
        v_arr = np.array(voltage, dtype=float)
        i_arr = np.array(current, dtype=float)
        t_arr = np.array(temperature, dtype=float)
        time_arr = np.array(time, dtype=float) if time else np.arange(len(v_arr), dtype=float)

        # Strict physical validation
        if np.any(np.isnan(v_arr)) or np.any(np.isinf(v_arr)):
            raise ValueError("Telemetry contains invalid NaN or Infinite voltage values.")
        if np.any(np.isnan(i_arr)) or np.any(np.isinf(i_arr)):
            raise ValueError("Telemetry contains invalid NaN or Infinite current values.")
        if np.any(np.isnan(t_arr)) or np.any(np.isinf(t_arr)):
            raise ValueError("Telemetry contains invalid NaN or Infinite temperature values.")
        if len(v_arr) != len(i_arr) or len(v_arr) != len(t_arr) or len(v_arr) != len(time_arr):
            raise ValueError(f"Array length mismatch: voltage({len(v_arr)}), current({len(i_arr)}), temperature({len(t_arr)}), time({len(time_arr)}).")
        if len(time_arr) > 1 and np.any(np.diff(time_arr) < 0):
            raise ValueError("Time vector must be monotonically non-decreasing.")

        delta_v = np.abs(v_arr[0] - v_arr[1]) if len(v_arr) > 1 else 0.05
        delta_i = np.abs(i_arr[1]) if len(i_arr) > 1 and np.abs(i_arr[1]) > 0.1 else 2.0
        r0 = float(np.clip(delta_v / delta_i, 0.03, 0.15))

        # Determine flow direction:
        # Standard convention: positive current = discharge, negative current = charge.
        # If all currents are negative, user is using IEEE load sign convention (negative = discharge).
        is_ieee_negative_discharge = bool(np.all(i_arr <= 0.05) and np.mean(i_arr) < -0.1)
        if is_ieee_negative_discharge:
            # Convert IEEE load convention (-2A discharge) to physical discharge current (+2A)
            i_physical = -i_arr
        else:
            i_physical = i_arr

        # 1. Base Physical Coulomb Counting Tracking with explicit Ah accumulation
        coulomb_traj, coulomb_last, coulomb_conf, coulomb_ver = self._coulomb_soc(
            v_arr, i_physical, time_arr, initial_soc
        )

        prediction_source = "coulomb_counting_baseline"
        if self.is_loaded.get("soc") and "soc" in self.models and "soc" in self.scalers:
            try:
                # Prepare features for all time points: [V, I, Temp, Time, R0]
                # Using physical negative discharge convention for the ML model
                features = np.column_stack([
                    v_arr,
                    -np.abs(i_arr),
                    t_arr,
                    time_arr,
                    np.full(len(v_arr), r0),
                ])
                features_scaled = self.scalers["soc"].transform(features)
                preds = self.models["soc"].predict(features_scaled)
                xgb_raw = [float(np.clip(p, 0.0, 100.0)) for p in preds]

                # Complementary Fusion:
                # Coulomb counting tracks high-frequency dynamics with zero point-wise jitter.
                # XGBoost acts as a drift-correction observer.
                # If current is predominantly unidirectional discharge, enforce monotonic non-increasing trajectory.
                blended = []
                for idx in range(len(coulomb_traj)):
                    c_val = coulomb_traj[idx]
                    x_val = xgb_raw[idx]
                    # Complementary filter: 60% Coulomb counting + 40% XGBoost voltage observer
                    val = 0.60 * c_val + 0.40 * x_val
                    blended.append(val)

                # Monotonicity enforcement during continuous discharge:
                if np.all(i_physical >= -0.05): # Continuous discharge / rest
                    for k in range(1, len(blended)):
                        if blended[k] > blended[k - 1]:
                            blended[k] = blended[k - 1]
                elif np.all(i_physical <= 0.05): # Continuous charge
                    for k in range(1, len(blended)):
                        if blended[k] < blended[k - 1]:
                            blended[k] = blended[k - 1]

                soc_trajectory = [round(float(b), 2) for b in blended]
                current_soc = soc_trajectory[-1]
                confidence = 0.96
                version = "hybrid_coulomb_xgboost_v1"
                prediction_source = "hybrid_coulomb_xgboost"
            except Exception as ex:
                logger.warning(f"Error evaluating XGBoost SoC model ({ex}), utilizing Coulomb counting directly.")
                soc_trajectory, current_soc, confidence, version = coulomb_traj, coulomb_last, coulomb_conf, coulomb_ver
                prediction_source = "coulomb_counting_fallback"
        else:
            soc_trajectory, current_soc, confidence, version = coulomb_traj, coulomb_last, coulomb_conf, coulomb_ver
            prediction_source = "coulomb_counting_baseline"

        return {
            "battery_id": battery_id,
            "soc_trajectory": soc_trajectory,
            "current_soc": current_soc,
            "confidence": confidence,
            "prediction_source": prediction_source,
            "model_version": version,
        }

    def _coulomb_soc(self, v_arr, i_physical, time_arr, initial_soc):
        if initial_soc is not None:
            init_soc_fraction = float(np.clip(initial_soc / 100.0, 0.0, 1.0))
        else:
            v_start = float(v_arr[0])
            if v_start >= 4.15:
                init_soc_fraction = 1.0
            elif v_start <= 3.3:
                init_soc_fraction = 0.05
            else:
                init_soc_fraction = float(np.clip((v_start - 3.3) / (4.15 - 3.3), 0.05, 1.0))

        estimator = SoCEstimator(nominal_capacity=2.0)
        trajectory = estimator.estimate(i_physical, time_arr, soc_initial=init_soc_fraction)
        soc_list = [round(float(s * 100), 2) for s in trajectory]
        return soc_list, soc_list[-1], 0.94, "coulomb_counting_v1"

    # ------------------------------------------------------------------------
    # RUL Inference (Trained LSTM Model)
    # ------------------------------------------------------------------------
    def predict_rul(
        self,
        voltage: List[float],
        current: List[float],
        temperature: List[float],
        current_cycle: int,
        capacity: Optional[float] = None,
        nominal_capacity: float = 2.0,
        battery_id: str = "B0005",
        time: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate Remaining Useful Life (RUL) in cycles using trained LSTM network.
        """
        v_arr = np.array(voltage, dtype=float)
        i_arr = np.array(current, dtype=float)
        t_arr = np.array(temperature, dtype=float)
        time_arr = np.array(time, dtype=float) if time else None

        # Strict physical telemetry validation
        if np.any(np.isnan(v_arr)) or np.any(np.isinf(v_arr)):
            raise ValueError("Telemetry contains invalid NaN or Infinite voltage values.")
        if np.any(np.isnan(i_arr)) or np.any(np.isinf(i_arr)):
            raise ValueError("Telemetry contains invalid NaN or Infinite current values.")
        if np.any(np.isnan(t_arr)) or np.any(np.isinf(t_arr)):
            raise ValueError("Telemetry contains invalid NaN or Infinite temperature values.")
        if len(v_arr) != len(i_arr) or len(v_arr) != len(t_arr):
            raise ValueError(f"Array length mismatch: voltage({len(v_arr)}), current({len(i_arr)}), temperature({len(t_arr)}).")
        if time_arr is not None:
            if len(time_arr) != len(v_arr):
                raise ValueError(f"Time array length ({len(time_arr)}) does not match voltage length ({len(v_arr)}).")
            if len(time_arr) > 1 and np.any(np.diff(time_arr) < 0):
                raise ValueError("Time vector must be monotonically non-decreasing.")

        if nominal_capacity <= 0:
            raise ValueError("Nominal capacity must be strictly positive.")

        if capacity is not None and capacity > 0:
            curr_cap = float(capacity)
        else:
            soh_info = self.predict_soh(voltage, current, temperature, battery_id=battery_id)
            curr_cap = (soh_info.get("soh_percent", 90.0) / 100.0) * nominal_capacity
        init_cap = max(curr_cap, 1.856 if str(battery_id).startswith("B00") else nominal_capacity)
        current_soh = min(1.0, curr_cap / init_cap)
        eol_threshold = 0.8
        eol_capacity = init_cap * eol_threshold

        delta_v = np.abs(v_arr[0] - v_arr[1]) if len(v_arr) > 1 else 0.05
        delta_i = np.abs(i_arr[1]) if len(i_arr) > 1 and np.abs(i_arr[1]) > 0.1 else 2.0
        r0 = float(np.clip(delta_v / delta_i, 0.03, 0.15))
        ecm_dict = {"R0": r0, "R1": 0.022, "C1": 1500.0, "tau1": 33.0}

        prediction_source = "empirical_fallback"
        raw_prediction = None
        fallback_reason = None

        if self.is_loaded.get("rul") and "rul" in self.models:
            try:
                X = prepare_rul_features(
                    v_arr, i_arr, t_arr, ecm_dict,
                    current_capacity=curr_cap,
                    current_soh=current_soh,
                    initial_capacity=init_cap,
                    current_cycle=current_cycle,
                    time=time_arr,
                    sequence_length=30,
                )
                if self.scalers.get("rul") is not None:
                    n_s, s_len, n_f = X.shape
                    X_scaled = self.scalers["rul"].transform(X.reshape(-1, n_f)).reshape(n_s, s_len, n_f)
                else:
                    X_scaled = X
                pred = self.models["rul"].predict(X_scaled, verbose=0)
                pred_rul = float(pred[0][0])
                raw_prediction = round(pred_rul, 2)

                if -10.0 <= pred_rul <= 300.0:
                    rul_cycles = int(max(0, round(pred_rul)))
                    margin = max(4, int(round(rul_cycles * 0.12)))
                    lower = max(0, rul_cycles - margin)
                    upper = rul_cycles + margin
                    confidence = 0.95
                    version = self.model_versions["rul"]
                    prediction_source = "lstm_model"
                    fallback_reason = None
                else:
                    fallback_reason = f"LSTM raw prediction ({pred_rul:.1f}) was out of bounds [-10, 300]."
                    rul_cycles, lower, upper, version = self._empirical_rul(curr_cap, eol_capacity, current_cycle, nominal_capacity)
                    prediction_source = "empirical_fallback"
            except Exception as ex:
                fallback_reason = f"Exception in LSTM RUL inference: {str(ex)}"
                logger.warning(f"Error evaluating LSTM RUL model ({ex}), falling back to empirical curve.")
                rul_cycles, lower, upper, version = self._empirical_rul(curr_cap, eol_capacity, current_cycle, nominal_capacity)
                prediction_source = "empirical_fallback"
        else:
            fallback_reason = "RUL LSTM model is not loaded in registry."
            rul_cycles, lower, upper, version = self._empirical_rul(curr_cap, eol_capacity, current_cycle, nominal_capacity)
            prediction_source = "empirical_fallback"

        if current_soh < 0.83:
            risk = "high"
        elif current_soh < 0.90:
            risk = "medium"
        else:
            risk = "low"

        rul_days = int(round(rul_cycles / 2.0))

        return {
            "battery_id": battery_id,
            "rul_cycles": rul_cycles,
            "rul_days_estimated": rul_days,
            "confidence_interval": {"lower": lower, "upper": upper},
            "eol_threshold": eol_threshold,
            "current_soh": round(current_soh, 4),
            "risk_level": risk,
            "prediction_source": prediction_source,
            "raw_prediction": raw_prediction,
            "fallback_reason": fallback_reason,
            "model_version": version,
        }

    def _empirical_rul(self, curr_cap, eol_capacity, current_cycle, nominal_capacity):
        if curr_cap <= eol_capacity:
            return 0, 0, 3, "empirical_rul_v1"
        delta_cap = curr_cap - eol_capacity
        fade_rate = (nominal_capacity - curr_cap) / max(current_cycle, 1) if current_cycle > 10 else 0.0025
        fade_rate = max(0.0015, min(fade_rate, 0.0045))
        rul_cycles = int(round(delta_cap / fade_rate))
        margin = max(5, int(round(rul_cycles * 0.18)))
        return rul_cycles, max(0, rul_cycles - margin), rul_cycles + margin, "empirical_rul_v1"

    # ------------------------------------------------------------------------
    # ECM Analysis
    # ------------------------------------------------------------------------
    def analyze_ecm(
        self,
        voltage: List[float],
        current: List[float],
        model_type: str = "RC",
        soc: Optional[List[float]] = None,
        time: Optional[List[float]] = None,
        battery_id: str = "B0005",
    ) -> Dict[str, Any]:
        v_arr = np.array(voltage, dtype=float)
        i_arr = np.array(current, dtype=float)
        time_arr = np.array(time, dtype=float) if time else np.arange(len(v_arr), dtype=float)

        if soc is not None and len(soc) == len(v_arr):
            soc_arr = np.array(soc, dtype=float)
            if np.max(soc_arr) > 1.0:
                soc_arr = soc_arr / 100.0
        else:
            soc_arr = SoCEstimator().estimate(i_arr, time_arr)

        if model_type.upper() == "RINT":
            model = RintModel().fit(v_arr, i_arr, soc_arr)
            v_pred = model.predict(i_arr, soc_arr)
            raw_params = model.get_parameters()
            params = {
                "R0": round(float(raw_params["R0"]), 5),
                "R1": None, "C1": None, "tau1": None,
                "R2": None, "C2": None, "tau2": None,
            }
        elif model_type.upper() == "2RC":
            model = TwoRCModel().fit(v_arr, i_arr, soc_arr, time_arr)
            v_pred = model.predict(i_arr, soc_arr, time_arr)
            raw_params = model.get_parameters()
            params = {
                "R0": round(float(raw_params["R0"]), 5),
                "R1": round(float(raw_params["R1"]), 5),
                "C1": round(float(raw_params["C1"]), 2),
                "tau1": round(float(raw_params["tau1"]), 2),
                "R2": round(float(raw_params["R2"]), 5),
                "C2": round(float(raw_params["C2"]), 2),
                "tau2": round(float(raw_params["tau2"]), 2),
            }
        else:
            model = RCModel().fit(v_arr, i_arr, soc_arr, time_arr)
            v_pred = model.predict(i_arr, soc_arr, time_arr)
            raw_params = model.get_parameters()
            params = {
                "R0": round(float(raw_params["R0"]), 5),
                "R1": round(float(raw_params["R1"]), 5),
                "C1": round(float(raw_params["C1"]), 2),
                "tau1": round(float(raw_params["tau1"]), 2),
                "R2": None, "C2": None, "tau2": None,
            }

        estimation_status = raw_params.get("estimation_status", "converged")
        boundary_hit = raw_params.get("boundary_hit", False)
        status_message = raw_params.get("status_message", "")
        is_excited = estimation_status != "unidentifiable_insufficient_excitation"

        metrics = calculate_model_metrics(v_arr, v_pred, is_excited=is_excited)

        r2_val = round(max(0.0, min(metrics["R2"], 1.0)), 4) if metrics["R2"] is not None else None

        return {
            "battery_id": battery_id,
            "model_type": model_type,
            "estimation_status": estimation_status,
            "boundary_hit": boundary_hit,
            "status_message": status_message,
            "parameters": params,
            "metrics": {
                "RMSE_mV": round(metrics["RMSE"] * 1000.0, 2),
                "R2": r2_val,
                "MAE_mV": round(metrics["MAE"] * 1000.0, 2),
            },
            "voltage_predicted": [round(float(v), 4) for v in v_pred],
        }


# Singleton instance
model_registry = ModelRegistry()
