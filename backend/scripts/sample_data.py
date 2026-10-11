"""
Generate realistic battery telemetry samples for testing and frontend demonstrations.
"""

import json
import numpy as np
from pathlib import Path


def generate_sample_profile(cycles: int = 50, points: int = 120, state: str = "healthy"):
    """
    Generate discharge curves with physics-realistic current, voltage, temperature.
    """
    t = np.linspace(0, 3600, points)  # 1 hour discharge

    if state == "healthy":
        r0 = 0.045
        cap = 1.95
    elif state == "moderate":
        r0 = 0.065
        cap = 1.65
    else:  # degraded / fault
        r0 = 0.092
        cap = 1.35

    # Nominal 2A CC discharge
    i = np.full(points, 2.0)
    # Voltage drop from OCV minus IR drop and polarization
    soc = np.linspace(1.0, 0.1, points)
    # Polynomial OCV curve
    ocv = 3.4 + 0.8 * soc - 0.2 * (soc ** 2) + 0.15 * (soc ** 3)
    v = ocv - i * r0 - 0.03 * (1 - np.exp(-t / 30.0)) + np.random.normal(0, 0.003, points)
    temp = 24.0 + 5.0 * (1 - np.exp(-t / 1200.0)) + np.random.normal(0, 0.1, points)

    return {
        "battery_id": "B0005",
        "voltage": [round(float(x), 4) for x in v],
        "current": [round(float(x), 4) for x in i],
        "temperature": [round(float(x), 2) for x in temp],
        "time": [round(float(x), 1) for x in t],
        "current_cycle": cycles,
        "capacity": cap,
    }


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "fixtures"
    out_dir.mkdir(exist_ok=True)

    for state in ["healthy", "moderate", "degraded"]:
        data = generate_sample_profile(
            cycles=20 if state == "healthy" else (80 if state == "moderate" else 150),
            state=state,
        )
        with open(out_dir / f"{state}_sample.json", "w") as f:
            json.dump(data, f, indent=2)
    print(f"Sample test profiles generated in {out_dir}")
