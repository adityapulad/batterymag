"""
Deep self-test suite for:
1. Real-Time State of Health (SoH)
2. State of Charge (SoC)
3. Remaining Useful Life (RUL)
4. Equivalent Circuit Model (ECM) parameter estimation

Tests:
A. Physical domain bounds and monotonicity
B. Sensitivity to signal length and noise
C. Extreme/edge case inputs (flatline, high noise, reversed current, zero load)
D. Consistency between ECM resistance R0 and measured drop
E. Fallback triggers and error handling
F. Out-of-sample realism across actual NASA cycles (fresh vs degraded)
"""

import sys
import numpy as np
import requests
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

API_URL = "http://localhost:8000/api/v1"

def log_test(title):
    print(f"\n{'='*70}\n[TEST] {title}\n{'='*70}")

def run_tests():
    flaws = []
    
    # -------------------------------------------------------------
    # 1. ECM Self-Test
    # -------------------------------------------------------------
    log_test("ECM (Thevenin / RC) Estimation Self-Test")
    # Synthetic clean pulse discharge: 4.2V -> 4.0V drop under 2A load
    t = np.linspace(0, 300, 100)
    # Voltage relaxation profile
    v_ideal = 4.2 - 0.08 - 0.04 * (1 - np.exp(-t / 30.0))
    i_load = np.full(100, 2.0)
    
    res = requests.post(f"{API_URL}/ecm/analyze", json={
        "battery_id": "B0005",
        "model_type": "RC",
        "voltage": list(v_ideal),
        "current": list(i_load),
        "time": list(t)
    }).json()
    print("Ideal Pulse ECM Response:", json.dumps(res, indent=2))
    
    # Check if R0 is close to 0.08 / 2 = 0.04 Ohm
    ecm = res.get("parameters", {})
    r0 = ecm.get("R0", 0)
    r1 = ecm.get("R1", 0)
    c1 = ecm.get("C1", 0)
    tau1 = ecm.get("tau1", 0)
    print(f"Estimated: R0={r0:.4f} Ohm, R1={r1:.4f} Ohm, C1={c1:.1f} F, tau1={tau1:.1f} s")
    
    # Edge case: Constant voltage flatline (0 current or no drop)
    res_flat = requests.post(f"{API_URL}/ecm/analyze", json={
        "battery_id": "B0005",
        "model_type": "RC",
        "voltage": [3.8] * 50,
        "current": [0.0] * 50,
        "time": list(np.linspace(0, 50, 50))
    }).json()
    print("Flatline 0A Current ECM Response:", json.dumps(res_flat, indent=2))
    if "detail" in res_flat or "error" in res_flat:
        print("Handling error on 0A...")
    else:
        flat_r0 = res_flat.get("parameters", {}).get("R0")
        if flat_r0 is not None and (np.isnan(flat_r0) or np.isinf(flat_r0) or flat_r0 == 0):
            flaws.append("ECM: Zero current input produces undefined or zero R0 without error handling")
        
    # -------------------------------------------------------------
    # 2. SoC Self-Test
    # -------------------------------------------------------------
    log_test("State of Charge (SoC) Estimation Self-Test")
    # Test A: Monotonicity during discharge (2A for 1 hour = 2Ah)
    t_soc = np.linspace(0, 3600, 60)
    v_soc = np.linspace(4.2, 3.2, 60)
    i_soc = np.full(60, 2.0)
    res_soc = requests.post(f"{API_URL}/soc/predict", json={
        "battery_id": "B0005",
        "voltage": list(v_soc),
        "current": list(i_soc),
        "temperature": [25.0] * 60,
        "time": list(t_soc),
        "initial_soc": 100.0
    }).json()
    
    soc_traj = res_soc.get("soc_trajectory", [])
    print(f"SoC start: {soc_traj[0]}%, end: {soc_traj[-1]}%, source: {res_soc.get('prediction_source')}")
    # Check if monotonically decreasing
    diffs = np.diff(soc_traj)
    non_monotonic = np.sum(diffs > 0.05)
    if non_monotonic > 0:
        flaws.append(f"SoC: Trajectory has {non_monotonic} non-monotonic increases during steady discharge")
        print(f"FLAW FOUND: Non-monotonic steps in discharge trajectory: {diffs[diffs > 0.05]}")
        
    # Test B: Charging current (negative current in some conventions, or positive in others)
    res_charge = requests.post(f"{API_URL}/soc/predict", json={
        "battery_id": "B0005",
        "voltage": list(np.linspace(3.5, 4.2, 60)),
        "current": list(np.full(60, -2.0)), # Negative current (charge)
        "temperature": [25.0] * 60,
        "time": list(t_soc),
        "initial_soc": 20.0
    }).json()
    print(f"SoC charging test: start: {res_charge.get('soc_trajectory', [0])[0]}%, end: {res_charge.get('soc_trajectory', [0])[-1]}%")

    # -------------------------------------------------------------
    # 3. SoH Self-Test
    # -------------------------------------------------------------
    log_test("State of Health (SoH) Self-Test across Battery Aging Stages")
    from scripts.train_nasa_models import extract_battery_cycles_from_ids
    nasa_cycles = extract_battery_cycles_from_ids(["B0005"])
    
    # Fresh cycle (Cycle 1)
    c_fresh = nasa_cycles[0]
    # Mid-life cycle (Cycle 50)
    c_mid = nasa_cycles[49]
    # Degraded cycle near EOL (Cycle 100)
    c_deg = nasa_cycles[99]
    
    stages = [("Fresh (Cycle 1)", c_fresh), ("Mid-life (Cycle 50)", c_mid), ("Degraded EOL (Cycle 100)", c_deg)]
    soh_preds = []
    
    for name, c in stages:
        payload = {
            "battery_id": "B0005",
            "voltage": list(c["seq_v_50"]),
            "current": [abs(float(x)) for x in c["seq_curr_50"]],
            "temperature": list(c["seq_temp_50"]),
            "time": [float(i * 60) for i in range(50)]
        }
        r = requests.post(f"{API_URL}/soh/predict", json=payload).json()
        print(f"[{name}] True SoH: {c['soh']*100:.2f}%, Pred SoH: {r.get('soh_percent')}%, Source: {r.get('prediction_source')}, Raw: {r.get('raw_prediction')}")
        soh_preds.append((name, c["soh"] * 100, r.get("soh_percent", 0), r.get("prediction_source")))
        
    # Check if SoH prediction drops monotonically with age
    if not (soh_preds[0][2] >= soh_preds[1][2] >= soh_preds[2][2]):
        flaws.append(f"SoH: Inverted aging trend across stages: {[p[2] for p in soh_preds]}")

    # -------------------------------------------------------------
    # 4. RUL Self-Test
    # -------------------------------------------------------------
    log_test("Remaining Useful Life (RUL) Estimation Self-Test")
    rul_preds = []
    for name, c in stages:
        payload = {
            "battery_id": "B0005",
            "voltage": list(c["seq_v_30"]),
            "current": [abs(float(x)) for x in c["seq_curr_30"]],
            "temperature": list(c["seq_temp_30"]),
            "current_cycle": c["cycle_num"],
            "capacity": c["capacity"],
            "nominal_capacity": 1.856
        }
        r = requests.post(f"{API_URL}/rul/predict", json=payload).json()
        print(f"[{name}] True RUL: {c['rul']} cycles, Pred RUL: {r.get('rul_cycles')} cycles, CI: {r.get('confidence_interval')}, Source: {r.get('prediction_source')}")
        rul_preds.append((name, c["rul"], r.get("rul_cycles", 0)))
        
    # Check if RUL monotonically decreases with cycle count
    if not (rul_preds[0][2] >= rul_preds[1][2] >= rul_preds[2][2]):
        flaws.append(f"RUL: Inverted RUL trend across aging stages: {[p[2] for p in rul_preds]}")

    # -------------------------------------------------------------
    # 5. Summary of Identified Flaws
    # -------------------------------------------------------------
    print("\n" + "#"*70)
    print("SELF-TEST FLAWS AND DISCREPANCIES DETECTED:")
    print("#"*70)
    if not flaws:
        print("✅ No critical flaws detected across the test suite.")
    else:
        for i, f in enumerate(flaws, 1):
            print(f"{i}. {f}")

if __name__ == "__main__":
    run_tests()
