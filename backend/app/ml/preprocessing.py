"""
Preprocessing pipeline for battery data — feature engineering for inference.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, Any


def prepare_soh_features(
    voltage: np.ndarray,
    current: np.ndarray,
    temperature: np.ndarray,
    ecm_params: Optional[dict] = None,
    time: Optional[np.ndarray] = None,
    sequence_length: int = 50,
) -> np.ndarray:
    """
    Build the scientifically rigorous 5-feature input tensor for the Hybrid SoH model.

    Features:
    0: Voltage V(t) in Volts
    1: Normalized discharge current I(t) in Amperes (negative discharge convention)
    2: Measured temperature T(t) in Celsius
    3: Voltage rate-of-change dV/dt(t) in V/s (polarization / phase transition dynamics)
    4: Ohmic internal resistance R0 per cycle in Ohms

    Returns shape: (1, sequence_length, 5)
    """
    n = len(voltage)
    min_required = 20  # Minimum samples needed to compute legitimate time-derivatives dV/dt and degradation patterns
    if n < min_required:
        raise ValueError(
            f"Insufficient time-series history: received {n} sample points, but at least {min_required} consecutive points "
            f"are required to evaluate dynamic polarization derivatives (dV/dt) and LSTM degradation state. "
            f"Please accumulate telemetry points before querying trained LSTM inference."
        )

    # If sequence is between min_required and sequence_length, interpolate linearly across physical time
    if n < sequence_length:
        t_orig = time if (time is not None and len(time) == n) else np.linspace(0, float(n - 1), n)
        t_interp = np.linspace(t_orig[0], t_orig[-1], sequence_length)
        voltage = np.interp(t_interp, t_orig, voltage)
        current = np.interp(t_interp, t_orig, current)
        temperature = np.interp(t_interp, t_orig, temperature)
        time = t_interp
        n = sequence_length

    indices = np.linspace(0, n - 1, sequence_length, dtype=int)
    seq_v = voltage[indices]
    seq_i = -np.abs(current[indices])
    seq_t = temperature[indices]

    if time is not None and len(time) == len(voltage):
        t_seq = time[indices]
        dt = np.diff(t_seq, prepend=t_seq[0])
        dt = np.where(dt <= 0, 1.0, dt)
        dv_dt = np.diff(seq_v, prepend=seq_v[0]) / dt
    else:
        # Default 1.0s interval proxy if explicit time vector is not provided
        dv_dt = np.diff(seq_v, prepend=seq_v[0])

    if ecm_params and "R0" in ecm_params:
        r0_val = float(ecm_params["R0"])
    else:
        delta_v = np.abs(voltage[0] - voltage[1]) if len(voltage) > 1 else 0.05
        delta_i = np.abs(current[1]) if len(current) > 1 and np.abs(current[1]) > 0.1 else 2.0
        r0_val = float(np.clip(delta_v / delta_i, 0.03, 0.15))

    seq_r0 = np.full(sequence_length, r0_val)

    features = np.column_stack([
        seq_v,
        seq_i,
        seq_t,
        dv_dt,
        seq_r0,
    ])  # Shape: (sequence_length, 5)

    return features[np.newaxis, ...]  # Shape: (1, sequence_length, 5)


def prepare_soc_features(
    voltage: np.ndarray,
    current: np.ndarray,
    temperature: np.ndarray,
    ecm_params: dict,
    sequence_length: int = 100,
) -> np.ndarray:
    """
    Build input tensor for Hybrid SoC model.

    Returns shape: (1, sequence_length, 9)
    """
    n = len(voltage)
    if n < sequence_length:
        pad = sequence_length - n
        voltage = np.concatenate([voltage, np.full(pad, voltage[-1])])
        current = np.concatenate([current, np.full(pad, current[-1])])
        temperature = np.concatenate([temperature, np.full(pad, temperature[-1])])
        n = sequence_length

    # Use a window of exactly sequence_length
    start = max(0, n - sequence_length)
    i_window = -np.abs(current[start:start + sequence_length])
    raw = np.column_stack([
        voltage[start:start + sequence_length],
        i_window,
        temperature[start:start + sequence_length],
        voltage[start:start + sequence_length],
        i_window,
    ])

    c1_scaled = float(ecm_params.get("C1", 1500.0)) / 1000.0
    ecm = np.array([[
        float(ecm_params.get("R0", 0.05)),
        float(ecm_params.get("R1", 0.02)),
        c1_scaled,
        float(ecm_params.get("tau1", 30.0)),
    ]] * sequence_length)

    combined = np.hstack([raw, ecm])
    return combined[np.newaxis, ...]


def prepare_rul_features(
    voltage: np.ndarray,
    current: np.ndarray,
    temperature: np.ndarray,
    ecm_params: Optional[dict],
    current_capacity: float,
    current_soh: float,
    initial_capacity: float,
    current_cycle: int,
    time: Optional[np.ndarray] = None,
    sequence_length: int = 30,
) -> np.ndarray:
    """
    Build 8-feature input tensor for Hybrid RUL model without static placeholders.

    Features:
    0: Voltage V(t)
    1: Current I(t) [normalized signed discharge]
    2: Temperature T(t)
    3: Voltage rate-of-change dV/dt
    4: Ohmic drop R0
    5: Current capacity in Ah
    6: Current State of Health (SoH fraction)
    7: Cycle number

    Returns shape: (1, sequence_length, 8)
    """
    n = len(voltage)
    if n < sequence_length:
        pad = sequence_length - n
        voltage = np.concatenate([voltage, np.full(pad, voltage[-1])])
        current = np.concatenate([current, np.full(pad, current[-1])])
        temperature = np.concatenate([temperature, np.full(pad, temperature[-1])])
        if time is not None:
            time = np.concatenate([time, np.linspace(time[-1] + 1, time[-1] + pad, pad)])
        n = sequence_length

    indices = np.linspace(0, n - 1, sequence_length, dtype=int)
    seq_v = voltage[indices]
    seq_i = -np.abs(current[indices])
    seq_t = temperature[indices]

    if time is not None and len(time) == len(voltage):
        t_seq = time[indices]
        dt = np.diff(t_seq, prepend=t_seq[0])
        dt = np.where(dt <= 0, 1.0, dt)
        dv_dt = np.diff(seq_v, prepend=seq_v[0]) / dt
    else:
        dv_dt = np.diff(seq_v, prepend=seq_v[0])

    if ecm_params and "R0" in ecm_params:
        r0_val = float(ecm_params["R0"])
    else:
        delta_v = np.abs(voltage[0] - voltage[1]) if len(voltage) > 1 else 0.05
        delta_i = np.abs(current[1]) if len(current) > 1 and np.abs(current[1]) > 0.1 else 2.0
        r0_val = float(np.clip(delta_v / delta_i, 0.03, 0.15))

    seq_r0 = np.full(sequence_length, r0_val)
    seq_cap = np.full(sequence_length, current_capacity)
    seq_soh = np.full(sequence_length, current_soh)
    seq_cycle = np.full(sequence_length, float(current_cycle))

    features = np.column_stack([
        seq_v,
        seq_i,
        seq_t,
        dv_dt,
        seq_r0,
        seq_cap,
        seq_soh,
        seq_cycle,
    ])  # (seq_len, 8)

    return features[np.newaxis, ...]  # (1, seq_len, 8)


def scale_features(
    X: np.ndarray,
    scaler = None,
) -> Tuple[np.ndarray, Any]:
    """
    Scale a 3-D tensor (samples, steps, features) using a fitted scaler.

    If scaler is None a new one is fit. Returns (X_scaled, scaler).
    """
    n_samples, n_steps, n_features = X.shape
    X_flat = X.reshape(-1, n_features)

    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_flat)
    else:
        X_scaled = scaler.transform(X_flat)

    return X_scaled.reshape(n_samples, n_steps, n_features), scaler
