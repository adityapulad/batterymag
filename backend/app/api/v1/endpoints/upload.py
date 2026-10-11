"""
File upload and battery dataset ingestion endpoint.
Supports CSV, TXT, and MATLAB (.mat) files.
"""

import io
import tempfile
from pathlib import Path
import pandas as pd
from scipy.io import loadmat
from typing import Dict, Any, List
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.core.battery_loader import BatteryDataLoader

router = APIRouter()


class UploadSummaryResponse(BaseModel):
    filename: str
    total_records: int
    columns_found: List[str]
    voltage_sample: List[float]
    current_sample: List[float]
    temperature_sample: List[float]
    time_sample: List[float]


@router.post("/dataset", response_model=UploadSummaryResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a battery cycle CSV or MATLAB .mat dataset, validate, and return a parsed sample.
    """
    lower_name = file.filename.lower()
    if not lower_name.endswith((".csv", ".txt", ".mat")):
        raise HTTPException(
            status_code=400,
            detail="Supported formats: CSV (.csv), TXT (.txt), or MATLAB (.mat).",
        )

    try:
        content = await file.read()

        # Handle MATLAB .mat files
        if lower_name.endswith(".mat"):
            with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            try:
                mat = loadmat(str(tmp_path))
                # Detect top-level battery key (e.g. 'B0005')
                valid_keys = [k for k in mat.keys() if not k.startswith("__")]
                if not valid_keys:
                    raise ValueError("No valid battery structure found in .mat file.")
                b_key = valid_keys[0]
                raw_cycles = mat[b_key][0, 0]["cycle"][0]

                # Find first discharge cycle
                target_data = None
                for c in raw_cycles:
                    if str(c["type"][0]) == "discharge":
                        target_data = c["data"]
                        break

                if target_data is None:
                    raise ValueError("No discharge cycle found in .mat file.")

                v_arr = target_data["Voltage_measured"][0, 0].flatten()
                i_arr = target_data["Current_measured"][0, 0].flatten()
                t_arr = target_data["Temperature_measured"][0, 0].flatten()
                time_arr = target_data["Time"][0, 0].flatten()

                sample_limit = min(200, len(v_arr))
                return UploadSummaryResponse(
                    filename=file.filename,
                    total_records=len(v_arr),
                    columns_found=["Voltage_measured", "Current_measured", "Temperature_measured", "Time"],
                    voltage_sample=[round(float(x), 4) for x in v_arr[:sample_limit]],
                    current_sample=[round(float(x), 4) for x in i_arr[:sample_limit]],
                    temperature_sample=[round(float(x), 2) for x in t_arr[:sample_limit]],
                    time_sample=[round(float(x), 1) for x in time_arr[:sample_limit]],
                )
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        # Handle CSV / TXT files
        df = pd.read_csv(io.BytesIO(content))
        col_map = {c: c.lower().strip() for c in df.columns}
        df.rename(columns=col_map, inplace=True)

        v_col = next((c for c in df.columns if "volt" in c or c in ["v", "voltage_measured"]), None)
        i_col = next((c for c in df.columns if "curr" in c or c in ["i", "current_measured"]), None)
        t_col = next((c for c in df.columns if "temp" in c or c in ["temperature_measured"]), None)
        time_col = next((c for c in df.columns if "time" in c or c in ["t", "seconds"]), None)

        if not v_col or not i_col:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded CSV must contain columns for voltage and current. Found: {list(df.columns)}",
            )

        sample_df = df.head(200)
        v_sample = sample_df[v_col].astype(float).tolist()
        i_sample = sample_df[i_col].astype(float).tolist()
        t_sample = sample_df[t_col].astype(float).tolist() if t_col else [25.0] * len(v_sample)
        time_sample = sample_df[time_col].astype(float).tolist() if time_col else list(range(len(v_sample)))

        return UploadSummaryResponse(
            filename=file.filename,
            total_records=len(df),
            columns_found=list(df.columns),
            voltage_sample=v_sample,
            current_sample=i_sample,
            temperature_sample=t_sample,
            time_sample=time_sample,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse dataset: {str(e)}")
