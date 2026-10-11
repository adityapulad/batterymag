"""
Rigorous Battery-Wise (Leave-One-Battery-Out) Validation for NASA Dataset.

Excludes held-out test batteries from both:
1. Feature Scaler fitting (StandardScaler fitted strictly on training batteries)
2. Model training (LSTM weights optimized strictly on training batteries)

Evaluates out-of-sample generalization against ground-truth capacity SoH per cycle.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = BASE_DIR.parent / "datasets" / "battery_data"
RESULTS_DIR = BASE_DIR / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_battery_cycles(battery_id: str):
    """
    Extract verified constant-current discharge cycles for a given battery ID.
    Features:
    0: Voltage V(t)
    1: Current I(t) [normalized signed discharge]
    2: Temperature T(t)
    3: Voltage Gradient dV/dt(t)
    4: Ohmic resistance R0 per cycle
    """
    mat_path = DATA_DIR / f"{battery_id}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"{mat_path} not found")

    mat = loadmat(str(mat_path))
    raw_cycles = mat[battery_id][0, 0]["cycle"][0]

    valid_cycles = []
    for c in raw_cycles:
        if str(c["type"][0]) == "discharge":
            d = c["data"]
            if "Capacity" in d.dtype.names and d["Capacity"][0, 0].size > 0:
                cap = float(d["Capacity"][0, 0].flatten()[0])
                v = d["Voltage_measured"][0, 0].flatten()
                curr = d["Current_measured"][0, 0].flatten()
                temp = d["Temperature_measured"][0, 0].flatten()
                t = d["Time"][0, 0].flatten()
                if len(v) >= 50 and cap > 0.5:
                    valid_cycles.append({
                        "cap": cap, "v": v, "curr": curr, "temp": temp, "t": t
                    })

    if not valid_cycles:
        return []

    c0 = valid_cycles[0]["cap"]
    eol_idx = len(valid_cycles)
    for idx, vc in enumerate(valid_cycles):
        if vc["cap"] <= c0 * 0.80:
            eol_idx = idx
            break

    records = []
    for idx, vc in enumerate(valid_cycles):
        cap = vc["cap"]
        true_soh = cap / c0
        true_rul = max(0, eol_idx - idx)

        v = vc["v"]
        curr = -np.abs(vc["curr"])
        temp = vc["temp"]
        t = vc["t"]

        # 50 uniform sampling points
        indices = np.linspace(0, len(v) - 1, 50, dtype=int)
        seq_v = v[indices]
        seq_curr = curr[indices]
        seq_temp = temp[indices]
        seq_t = t[indices]

        # Calculate voltage rate-of-change dV/dt
        dt = np.diff(seq_t, prepend=seq_t[0])
        dt = np.where(dt <= 0, 1.0, dt)
        dv_dt = np.diff(seq_v, prepend=seq_v[0]) / dt

        # Initial Ohmic drop R0
        delta_v = np.abs(v[0] - v[1]) if len(v) > 1 else 0.05
        delta_i = np.abs(curr[1]) if len(curr) > 1 and np.abs(curr[1]) > 0.1 else 2.0
        r0 = float(np.clip(delta_v / delta_i, 0.03, 0.15))

        # Scientific feature matrix: (50, 5) without duplicates or fake constants
        # [V, I, Temp, dV/dt, R0]
        feature_matrix = np.column_stack([
            seq_v,
            seq_curr,
            seq_temp,
            dv_dt,
            np.full(50, r0),
        ])

        records.append({
            "battery_id": battery_id,
            "cycle_num": idx + 1,
            "initial_cap": c0,
            "capacity": cap,
            "true_soh": true_soh,
            "true_rul": true_rul,
            "X": feature_matrix,
        })

    return records


def build_lstm_model(input_shape=(50, 5)):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def run_leave_one_battery_out_experiment(benchmark_batteries=["B0005", "B0006", "B0007", "B0018"]):
    print("=" * 80)
    print("RUNNING BATTERY-WISE LEAVE-ONE-BATTERY-OUT (LOBO) EVALUATION")
    print("=" * 80)

    # 1. Load data separated strictly by battery
    battery_data = {}
    for b_id in benchmark_batteries:
        recs = extract_battery_cycles(b_id)
        battery_data[b_id] = recs
        print(f"Loaded {b_id}: {len(recs)} cycles | C0 = {recs[0]['initial_cap']:.3f} Ah")

    results = []

    for test_bid in benchmark_batteries:
        train_bids = [b for b in benchmark_batteries if b != test_bid]
        print(f"\n" + "-" * 70)
        print(f"Fold: Train on {train_bids} ---> TEST ON UNSEEN {test_bid}")
        print("-" * 70)

        # Build train tensors
        train_records = [r for b in train_bids for r in battery_data[b]]
        test_records = battery_data[test_bid]

        X_train_raw = np.array([r["X"] for r in train_records])  # (N_train, 50, 5)
        y_train = np.array([r["true_soh"] for r in train_records])

        X_test_raw = np.array([r["X"] for r in test_records])    # (N_test, 50, 5)
        y_test = np.array([r["true_soh"] for r in test_records])

        # FIT SCALER STRICTLY ON TRAINING BATTERIES ONLY
        scaler = StandardScaler()
        n_tr, s_len, n_feat = X_train_raw.shape
        X_train_flat = X_train_raw.reshape(-1, n_feat)
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_tr, s_len, n_feat)

        # TRANSFORM HELD-OUT TEST BATTERY USING TRAINING SCALER
        n_te = X_test_raw.shape[0]
        X_test_flat = X_test_raw.reshape(-1, n_feat)
        X_test_scaled = scaler.transform(X_test_flat).reshape(n_te, s_len, n_feat)

        # Train LSTM strictly on train_scaled
        model = build_lstm_model(input_shape=(50, 5))
        early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        model.fit(
            X_train_scaled, y_train,
            epochs=30,
            batch_size=16,
            verbose=0,
            callbacks=[early_stop]
        )

        # Evaluate on unseen test battery
        y_pred = model.predict(X_test_scaled, verbose=0).flatten()

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"RESULTS FOR UNSEEN {test_bid}:")
        print(f"  * Out-of-Sample MAE:  {mae:.4f} ({mae*100:.2f}% SoH)")
        print(f"  * Out-of-Sample RMSE: {rmse:.4f} ({rmse*100:.2f}% SoH)")
        print(f"  * Out-of-Sample R^2:  {r2:.4f}")

        # Check key cycles
        sample_indices = [0, len(test_records)//2, -1]
        for idx in sample_indices:
            cyc = test_records[idx]["cycle_num"]
            t_soh = y_test[idx] * 100.0
            p_soh = y_pred[idx] * 100.0
            print(f"    Cycle {cyc:3d}: Ground Truth = {t_soh:.2f}%, LSTM Prediction = {p_soh:.2f}% (Err = {abs(t_soh - p_soh):.2f}%)")

        results.append({
            "test_battery": test_bid,
            "train_batteries": ", ".join(train_bids),
            "test_cycles": len(test_records),
            "mae_soh": round(float(mae), 4),
            "mae_percent": round(float(mae * 100.0), 2),
            "rmse_soh": round(float(rmse), 4),
            "rmse_percent": round(float(rmse * 100.0), 2),
            "r2_score": round(float(r2), 4),
        })

    # Summary table
    df_res = pd.DataFrame(results)
    out_csv = RESULTS_DIR / "leave_one_battery_out_results.csv"
    df_res.to_csv(out_csv, index=False)
    print("\n" + "=" * 80)
    print("LOBO EVALUATION SUMMARY TABLE:")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print(f"\nSaved detailed evaluation metrics to: {out_csv}")
    return df_res


if __name__ == "__main__":
    run_leave_one_battery_out_experiment()
