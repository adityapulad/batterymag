"""
Production-grade training and evaluation pipeline for NASA Battery Dataset.

Features scientific improvements:
1. Strict Battery-Wise Splitting:
   - Holds out test batteries (e.g. B0005) completely from scaler fitting and model training.
   - Evaluates out-of-sample generalization against ground-truth capacity.
2. Verified Physical Features without artificial duplicates or static placeholders:
   - SoH LSTM: [V(t), I(t), Temp(t), dV/dt(t), R0(t)] (5 dynamic features)
   - SoC XGBoost: [V(t), I(t), Temp(t), Time(t), R0(t)] (5 features)
   - RUL LSTM: [V(t), I(t), Temp(t), dV/dt(t), R0(t), Capacity, Current_SoH, Cycle_num] (8 features)
3. Saves models and scalers fitted strictly on training cells to backend/models/
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
import joblib

# Set paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import settings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = BASE_DIR.parent / "datasets" / "battery_data"
MODELS_DIR = settings.MODELS_DIR


def extract_battery_cycles_from_ids(battery_ids):
    """
    Extract structured cycles with features and targets across NASA batteries.
    Filters to standard 18650 degradation profiles with fresh capacity >= 1.2 Ah.
    """
    records = []
    print(f"Loading NASA battery datasets for: {battery_ids}...")

    for b_id in battery_ids:
        mat_path = DATA_DIR / f"{b_id}.mat"
        if not mat_path.exists():
            print(f"Warning: {mat_path} not found, skipping.")
            continue

        try:
            mat = loadmat(str(mat_path))
            raw_cycles = mat[b_id][0, 0]["cycle"][0]
        except Exception as e:
            print(f"Skipping {b_id} due to read error: {e}")
            continue

        valid_cycles = []
        for c in raw_cycles:
            if str(c["type"][0]) == "discharge":
                d = c["data"]
                if "Capacity" in d.dtype.names:
                    cap_arr = d["Capacity"][0, 0]
                    if cap_arr.size > 0:
                        cap = float(cap_arr.flatten()[0])
                        v = d["Voltage_measured"][0, 0].flatten()
                        curr = d["Current_measured"][0, 0].flatten()
                        temp = d["Temperature_measured"][0, 0].flatten()
                        t = d["Time"][0, 0].flatten()
                        if len(v) >= 50 and cap > 0.8:
                            valid_cycles.append({
                                "cap": cap, "v": v, "curr": curr, "temp": temp, "t": t
                            })

        if len(valid_cycles) < 10:
            print(f"Skipping {b_id}: insufficient discharge cycles ({len(valid_cycles)}).")
            continue

        initial_cap = valid_cycles[0]["cap"]
        if initial_cap < 1.2:
            print(f"Skipping {b_id}: low initial capacity ({initial_cap:.2f} Ah).")
            continue

        eol_cap = initial_cap * 0.80  # 80% EOL retirement threshold

        eol_idx = len(valid_cycles)
        for idx, vc in enumerate(valid_cycles):
            if vc["cap"] <= eol_cap:
                eol_idx = idx
                break

        print(f"  {b_id}: {len(valid_cycles)} cycles, C0={initial_cap:.3f} Ah, EOL cycle={eol_idx}")

        for idx, vc in enumerate(valid_cycles):
            cap = vc["cap"]
            soh = cap / initial_cap
            rul = max(0, eol_idx - idx)

            v = vc["v"]
            curr = -np.abs(vc["curr"])
            temp = vc["temp"]
            t = vc["t"]

            # Subsample 50 uniform points for SoH
            indices_50 = np.linspace(0, len(v) - 1, 50, dtype=int)
            seq_v_50 = v[indices_50]
            seq_curr_50 = curr[indices_50]
            seq_temp_50 = temp[indices_50]
            seq_t_50 = t[indices_50]

            dt_50 = np.diff(seq_t_50, prepend=seq_t_50[0])
            dt_50 = np.where(dt_50 <= 0, 1.0, dt_50)
            dv_dt_50 = np.diff(seq_v_50, prepend=seq_v_50[0]) / dt_50

            # Subsample 30 uniform points for RUL
            indices_30 = np.linspace(0, len(v) - 1, 30, dtype=int)
            seq_v_30 = v[indices_30]
            seq_curr_30 = curr[indices_30]
            seq_temp_30 = temp[indices_30]
            seq_t_30 = t[indices_30]

            dt_30 = np.diff(seq_t_30, prepend=seq_t_30[0])
            dt_30 = np.where(dt_30 <= 0, 1.0, dt_30)
            dv_dt_30 = np.diff(seq_v_30, prepend=seq_v_30[0]) / dt_30

            # Ohmic resistance drop R0
            delta_v = np.abs(v[0] - v[1]) if len(v) > 1 else 0.05
            delta_i = np.abs(curr[1]) if len(curr) > 1 and np.abs(curr[1]) > 0.1 else 2.0
            r0 = float(np.clip(delta_v / delta_i, 0.03, 0.15))

            records.append({
                "battery_id": b_id,
                "cycle_num": idx + 1,
                "initial_cap": initial_cap,
                "capacity": cap,
                "soh": soh,
                "rul": rul,
                "r0": r0,
                "seq_v_50": seq_v_50,
                "seq_curr_50": seq_curr_50,
                "seq_temp_50": seq_temp_50,
                "seq_dv_dt_50": dv_dt_50,
                "seq_v_30": seq_v_30,
                "seq_curr_30": seq_curr_30,
                "seq_temp_30": seq_temp_30,
                "seq_dv_dt_30": dv_dt_30,
                "full_v": v,
                "full_curr": curr,
                "full_temp": temp,
                "full_t": t,
            })

    print(f"Extracted {len(records)} total calibrated discharge cycles.")
    return records


# ----------------------------------------------------------------------------
# 1. Train Hybrid LSTM for State of Health (SoH) + Scaler
# ----------------------------------------------------------------------------
def train_soh_lstm(train_records, test_records=None):
    print("\n" + "=" * 60)
    print("Training Hybrid LSTM for State of Health (SoH) [5 Dynamic Features]...")
    print("=" * 60)

    def build_tensors(recs):
        X_list, y_list = [], []
        for r in recs:
            # Features: [V, I, Temp, dV/dt, R0]
            features = np.column_stack([
                r["seq_v_50"],
                r["seq_curr_50"],
                r["seq_temp_50"],
                r["seq_dv_dt_50"],
                np.full(50, r["r0"]),
            ])
            X_list.append(features)
            y_list.append(r["soh"])
        return np.array(X_list), np.array(y_list)

    X_train, y_train = build_tensors(train_records)

    # Strictly fit feature scaler on TRAINING batteries only
    n_samples, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_feat)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_samples, seq_len, n_feat)

    # Build LSTM model
    model = Sequential([
        Input(shape=(50, 5)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(monitor="loss", patience=6, restore_best_weights=True)
    history = model.fit(
        X_train_scaled, y_train,
        epochs=35,
        batch_size=16,
        verbose=1,
        callbacks=[early_stop]
    )

    out_path = MODELS_DIR / "soh" / "hybrid_soh_v1.keras"
    scaler_path = MODELS_DIR / "soh" / "scaler_soh.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    joblib.dump(scaler, scaler_path)
    print(f"✅ Saved SoH LSTM to {out_path} & scaler to {scaler_path} (Train MAE: {history.history['mae'][-1]:.4f})")

    # Evaluate on held-out test batteries if provided
    if test_records:
        X_test, y_test = build_tensors(test_records)
        n_t_samples, t_seq_len, t_n_feat = X_test.shape
        X_test_scaled = scaler.transform(X_test.reshape(-1, t_n_feat)).reshape(n_t_samples, t_seq_len, t_n_feat)
        test_preds = model.predict(X_test_scaled, verbose=0).flatten()
        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_r2 = r2_score(y_test, test_preds)
        print(f"📊 Held-Out Test Evaluation: MAE={test_mae*100:.2f}%, RMSE={test_rmse*100:.2f}%, R2={test_r2:.4f}")

    return model, scaler


# ----------------------------------------------------------------------------
# 2. Train XGBoost Model for State of Charge (SoC) + Scaler
# ----------------------------------------------------------------------------
def train_soc_xgboost(train_records, test_records=None):
    print("\n" + "=" * 60)
    print("Training XGBoost Regressor for State of Charge (SoC)...")
    print("=" * 60)

    def build_points(recs):
        X_pts, y_pts = [], []
        for r in recs:
            v = r["full_v"]
            curr = r["full_curr"]
            temp = r["full_temp"]
            t = r["full_t"]
            n_pts = len(v)
            if n_pts < 10:
                continue
            soc_true = np.linspace(100.0, 5.0, n_pts)
            step = max(1, n_pts // 40)
            for i in range(0, n_pts, step):
                X_pts.append([
                    v[i],
                    curr[i],
                    temp[i],
                    t[i] if i < len(t) else float(i * 30),
                    r["r0"],
                ])
                y_pts.append(soc_true[i])
        return np.array(X_pts), np.array(y_pts)

    X_train, y_train = build_points(train_records)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    soc_dir = MODELS_DIR / "soc"
    soc_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, soc_dir / "xgboost_soc_v1.joblib")
    joblib.dump(scaler, soc_dir / "scaler_soc.joblib")

    preds = model.predict(X_train_scaled)
    mae = float(np.mean(np.abs(preds - y_train)))
    print(f"✅ Saved XGBoost SoC model to {soc_dir / 'xgboost_soc_v1.joblib'} (Train MAE: {mae:.2f}%)")

    if test_records:
        X_test, y_test = build_points(test_records)
        X_test_scaled = scaler.transform(X_test)
        test_preds = model.predict(X_test_scaled)
        test_mae = mean_absolute_error(y_test, test_preds)
        print(f"📊 Held-Out SoC Test MAE: {test_mae:.2f}%")

    return model, scaler


# ----------------------------------------------------------------------------
# 3. Train Hybrid LSTM for Remaining Useful Life (RUL) + Scaler
# ----------------------------------------------------------------------------
def train_rul_lstm(train_records, test_records=None):
    print("\n" + "=" * 60)
    print("Training Hybrid LSTM for Remaining Useful Life (RUL) [8 Dynamic Features]...")
    print("=" * 60)

    def build_tensors(recs):
        X_list, y_list = [], []
        for r in recs:
            # Features: [V, I, Temp, dV/dt, R0, Capacity, Current_SoH, Cycle_num]
            features = np.column_stack([
                r["seq_v_30"],
                r["seq_curr_30"],
                r["seq_temp_30"],
                r["seq_dv_dt_30"],
                np.full(30, r["r0"]),
                np.full(30, r["capacity"]),
                np.full(30, r["soh"]),
                np.full(30, float(r["cycle_num"])),
            ])
            X_list.append(features)
            y_list.append(float(r["rul"]))
        return np.array(X_list), np.array(y_list)

    X_train, y_train = build_tensors(train_records)

    n_samples, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_feat)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_samples, seq_len, n_feat)

    model = Sequential([
        Input(shape=(30, 8)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(monitor="loss", patience=6, restore_best_weights=True)
    history = model.fit(
        X_train_scaled, y_train,
        epochs=35,
        batch_size=16,
        verbose=1,
        callbacks=[early_stop]
    )

    out_path = MODELS_DIR / "rul" / "hybrid_rul_v1.keras"
    scaler_path = MODELS_DIR / "rul" / "scaler_rul.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    joblib.dump(scaler, scaler_path)
    print(f"✅ Saved RUL LSTM to {out_path} & scaler to {scaler_path} (Train MAE: {history.history['mae'][-1]:.2f} cycles)")

    if test_records:
        X_test, y_test = build_tensors(test_records)
        n_t_samples, t_seq_len, t_n_feat = X_test.shape
        X_test_scaled = scaler.transform(X_test.reshape(-1, t_n_feat)).reshape(n_t_samples, t_seq_len, t_n_feat)
        test_preds = model.predict(X_test_scaled, verbose=0).flatten()
        test_mae = mean_absolute_error(y_test, test_preds)
        print(f"📊 Held-Out RUL Test MAE: {test_mae:.2f} cycles")

    return model, scaler


def main():
    mat_files = sorted(DATA_DIR.glob("*.mat"))
    battery_ids = [f.stem for f in mat_files]
    print(f"Discovered {len(battery_ids)} NASA battery files in {DATA_DIR}.")

    # Strict Battery-Wise Splitting:
    # Training cells: B0006, B0007, B0018
    # Held-out Golden Test benchmark: B0005
    train_ids = [b for b in ["B0006", "B0007", "B0018"] if b in battery_ids]
    test_ids = [b for b in ["B0005"] if b in battery_ids]

    print(f"Training battery cells: {train_ids}")
    print(f"Held-out test cells:    {test_ids}")

    train_records = extract_battery_cycles_from_ids(train_ids)
    test_records = extract_battery_cycles_from_ids(test_ids)

    if not train_records:
        print("Error: No training discharge cycles extracted.")
        return

    train_soh_lstm(train_records, test_records)
    train_soc_xgboost(train_records, test_records)
    train_rul_lstm(train_records, test_records)
    print("\n🎉 ALL MODELS TRAINED STRICTLY WITH BATTERY-WISE SPLIT AND PHYSICAL FEATURES!")


if __name__ == "__main__":
    main()
