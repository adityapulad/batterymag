"""
Refactored, production-ready NASA Battery .mat and .csv data loader.

Fixes hardcoded relative paths from legacy battery_loader.py, uses configurable
settings.BASE_DIR, handles missing files gracefully, and returns structured dictionaries
or DataFrames ready for ECM and AI modeling.
"""

import os
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from typing import Dict, Any, Tuple, Optional, List

from app.config import settings


class BatteryDataLoader:
    """Robust loader for NASA Prognostics Center battery aging datasets."""

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            # Check configured uploads dir or project root datasets
            root = settings.BASE_DIR.parent
            candidates = [
                root / "datasets" / "battery_data",
                root / "datasets" / "archive" / "cleaned_dataset",
                settings.UPLOADS_DIR,
            ]
            self.data_dir = next((p for p in candidates if p.exists()), root / "datasets")

    def load_mat(self, battery_id: str) -> Dict[str, Any]:
        """
        Load NASA battery .mat structure by battery identifier (e.g. 'B0005').
        """
        mat_path = self.data_dir / f"{battery_id}.mat"
        if not mat_path.exists():
            # Search recursively in data directory
            found = list(self.data_dir.rglob(f"{battery_id}.mat"))
            if found:
                mat_path = found[0]
            else:
                raise FileNotFoundError(
                    f"Battery dataset '{battery_id}.mat' was not found in {self.data_dir}."
                )

        mat = loadmat(str(mat_path))
        raw_cycles = mat[battery_id][0, 0]["cycle"][0]

        discharge_cycles = []
        for i, cycle in enumerate(raw_cycles):
            cycle_type = str(cycle["type"][0])
            if cycle_type == "discharge":
                amb_temp = float(cycle["ambient_temperature"][0][0][0, 0])
                time_tuple = cycle["time"][0][0]
                dt = datetime.datetime(
                    int(time_tuple[0, 0]),
                    int(time_tuple[0, 1]),
                    int(time_tuple[0, 2]),
                    int(time_tuple[0, 3]),
                    int(time_tuple[0, 4]),
                ) + datetime.timedelta(seconds=float(time_tuple[0, 5]))

                data = cycle["data"]
                discharge_cycles.append({
                    "cycle_index": len(discharge_cycles) + 1,
                    "datetime": dt,
                    "ambient_temp": amb_temp,
                    "voltage_measured": data["Voltage_measured"][0, 0].flatten().tolist(),
                    "current_measured": data["Current_measured"][0, 0].flatten().tolist(),
                    "temperature_measured": data["Temperature_measured"][0, 0].flatten().tolist(),
                    "time": data["Time"][0, 0].flatten().tolist(),
                    "capacity": float(data["Capacity"][0, 0][0, 0]) if "Capacity" in data.dtype.names else None,
                })

        return {
            "battery_id": battery_id,
            "total_cycles": len(raw_cycles),
            "discharge_cycles_count": len(discharge_cycles),
            "cycles": discharge_cycles,
        }

    def load_csv(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse battery cycle CSV into normalized lists of voltage, current, temp, and time.
        """
        df = pd.read_csv(file_path)
        col_map = {c: c.lower().strip() for c in df.columns}
        df.rename(columns=col_map, inplace=True)

        v_col = next((c for c in df.columns if "volt" in c or c in ["v", "voltage_measured"]), None)
        i_col = next((c for c in df.columns if "curr" in c or c in ["i", "current_measured"]), None)
        t_col = next((c for c in df.columns if "temp" in c or c in ["temperature_measured"]), None)
        time_col = next((c for c in df.columns if "time" in c or c in ["t", "seconds"]), None)

        if not v_col or not i_col:
            raise ValueError(f"CSV must contain voltage and current columns. Found: {list(df.columns)}")

        return {
            "battery_id": file_path.stem,
            "voltage": df[v_col].astype(float).tolist(),
            "current": df[i_col].astype(float).tolist(),
            "temperature": df[t_col].astype(float).tolist() if t_col else [25.0] * len(df),
            "time": df[time_col].astype(float).tolist() if time_col else list(range(len(df))),
        }
