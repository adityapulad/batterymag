# Dataset Structure for Battery Management System

## Current Dataset Organization

### 1. State of Charge (SoC) Data
**Location**: `datasets/Dataset_Li-ion/`
**Status**: ✅ Ready to use
**Used by**: SoC notebooks (CNN, LSTM, DNN, XGBoost in `soc/` folder)
**Format**: CSV files organized by temperature folders (0degC, 10degC, 25degC, etc.)

### 2. State of Health (SoH) Data
**Location**: `datasets/archive/cleaned_dataset/`
**Status**: ⚠️ Needs conversion
**Used by**: SoH notebooks (CNN, LSTM, DNN in `soh/` folder)
**Current Format**: CSV files (7565 files)
**Expected Format**: MATLAB .mat files in `battery_data/` folder

### 3. Additional Battery Dataset
**Location**: `datasets/archive (1)/Battery_dataset.csv`
**Status**: Available for use
**Format**: Single CSV file with battery data

## Required Actions for Full Project Functionality

### Option 1: Convert CSV to MAT format (Recommended for original notebooks)
The SoH notebooks expect MATLAB .mat files. You would need to:
1. Create a `battery_data/` folder in the project root
2. Convert the NASA CSV files to .mat format
3. Name them according to battery IDs (e.g., B0005.mat, B0006.mat, etc.)

### Option 2: Modify SoH notebooks to use CSV (Easier approach)
Modify the SoH notebooks to read CSV files directly instead of .mat files:
1. Keep data in `datasets/archive/cleaned_dataset/`
2. Update the `load_data()` function in each SoH notebook
3. Read CSV files using pandas instead of scipy.io.loadmat

## Dataset Details

### NASA Battery Dataset (cleaned_dataset)
- **Total Files**: 7565 CSV files
- **Batteries**: B0045, B0047 (and possibly others)
- **Data Types**: discharge, charge, impedance
- **Columns**: Voltage_measured, Current_measured, Temperature_measured, Current_load, Voltage_load, Time
- **Metadata**: Available in `metadata.csv` with capacity, resistance, and test information

### Li-ion Dataset (Dataset_Li-ion)
- **Temperature Folders**: 0degC, 10degC, 25degC, n10degC
- **File Types**: Various test profiles (HPPC, UDDS, US06, Mixed, Charge, etc.)
- **Format**: CSV with time-series battery measurements

## Recommended Next Steps

1. **Run SoC notebooks immediately** - They are ready to use with Dataset_Li-ion
2. **Choose conversion approach** for SoH notebooks:
   - Convert CSV to MAT (maintains original code)
   - Modify notebooks to read CSV (simpler, no conversion needed)
3. **Clean up empty folders** - Remove `datasets/archive (2)/`

## File Paths Reference

```
datasets/
├── Dataset_Li-ion/          # ✅ Ready for SoC notebooks
│   ├── 0degC/
│   ├── 10degC/
│   ├── 25degC/
│   └── n10degC/
├── archive/
│   └── cleaned_dataset/     # NASA battery data (CSV format)
│       ├── data/            # 7565 CSV files
│       ├── extra_infos/     # README files
│       └── metadata.csv     # Battery test metadata
└── archive (1)/
    └── Battery_dataset.csv  # Additional battery data
```

## Notes
- The project expects `.mat` files for SoH estimation but has CSV files
- All SoC functionality is ready to use immediately
- SoH notebooks need either data conversion or code modification
