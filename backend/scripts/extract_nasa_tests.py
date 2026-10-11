"""
Extract clean testing datasets from NASA Battery .mat files into CSV and JSON formats
for direct API and Frontend testing.
"""

import os
import json
from pathlib import Path
import pandas as pd
from scipy.io import loadmat

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "datasets" / "battery_data"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def extract_battery_cycle(battery_id: str, cycle_num: int = 1):
    mat_path = DATA_DIR / f"{battery_id}.mat"
    if not mat_path.exists():
        print(f"File {mat_path} not found.")
        return None

    mat = loadmat(str(mat_path))
    raw_cycles = mat[battery_id][0, 0]["cycle"][0]

    discharge_count = 0
    target_data = None
    capacity = None

    for i, c in enumerate(raw_cycles):
        if str(c["type"][0]) == "discharge":
            discharge_count += 1
            if discharge_count == cycle_num or (cycle_num == -1 and i == len(raw_cycles) - 1):
                target_data = c["data"]
                if "Capacity" in target_data.dtype.names:
                    capacity = float(target_data["Capacity"][0, 0][0, 0])
                break

    if target_data is None:
        print(f"Cycle {cycle_num} not found for {battery_id}.")
        return None

    v = target_data["Voltage_measured"][0, 0].flatten().tolist()
    curr = target_data["Current_measured"][0, 0].flatten().tolist()
    temp = target_data["Temperature_measured"][0, 0].flatten().tolist()
    time = target_data["Time"][0, 0].flatten().tolist()

    return {
        "battery_id": battery_id,
        "cycle_index": cycle_num,
        "capacity": capacity,
        "total_points": len(v),
        "voltage": v,
        "current": curr,
        "temperature": temp,
        "time": time,
    }


def main():
    test_specs = [
        # Fresh / Healthy early cycle
        ("B0005", 2, "B0005_cycle2_healthy"),
        # Moderate degradation mid cycle
        ("B0006", 50, "B0006_cycle50_moderate"),
        # High degradation / near end of life
        ("B0007", 130, "B0007_cycle130_degraded"),
        # Additional cell B0018
        ("B0018", 30, "B0018_cycle30"),
    ]

    for battery_id, cycle_idx, out_name in test_specs:
        data = extract_battery_cycle(battery_id, cycle_idx)
        if not data:
            continue

        # Save as JSON payload ready for /api/v1/soh/predict or /soc/predict
        json_path = FIXTURES_DIR / f"{out_name}.json"
        with open(json_path, "w") as f:
            json.dump({
                "battery_id": battery_id,
                "current_cycle": cycle_idx,
                "capacity": data["capacity"],
                "voltage": [round(x, 4) for x in data["voltage"]],
                "current": [round(x, 4) for x in data["current"]],
                "temperature": [round(x, 2) for x in data["temperature"]],
                "time": [round(x, 1) for x in data["time"]],
            }, f, indent=2)

        # Save as CSV ready for Drag-and-Drop file upload in UI
        csv_path = FIXTURES_DIR / f"{out_name}.csv"
        df = pd.DataFrame({
            "Time": data["time"],
            "Voltage_measured": data["voltage"],
            "Current_measured": data["current"],
            "Temperature_measured": data["temperature"],
        })
        df.to_csv(csv_path, index=False)

        print(f"✅ Generated {out_name}: {len(df)} data points (Capacity: {data['capacity']} Ah)")


if __name__ == "__main__":
    main()
