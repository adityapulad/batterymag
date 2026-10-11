"""
Copy each section below into separate cells in your Jupyter notebook
Run them in order from top to bottom
"""

# ============================================================================
# CELL 1: Imports
# ============================================================================
from battery_loader import load_data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

print("✓ Imports successful")


# ============================================================================
# CELL 2: Load Data
# ============================================================================
dataset, capacity_data = load_data('B0005')
print(f"✓ Dataset loaded: {dataset.shape}")


# ============================================================================
# CELL 3: Create SOH
# ============================================================================
C = dataset['capacity'].iloc[0]
soh_values = (dataset['capacity'] / C).values
soh = pd.DataFrame(data=soh_values, columns=['SoH'])
print(f"✓ SOH created: {soh.shape}")


# ============================================================================
# CELL 4: Extract Features
# ============================================================================
attribs = ['capacity', 'voltage_measured', 'current_measured',
           'temperature_measured', 'current_load', 'voltage_load', 'time']
train_dataset = dataset[attribs].values
print(f"✓ Features extracted: {train_dataset.shape}")


# ============================================================================
# CELL 5: Scale Features
# ============================================================================
sc = MinMaxScaler(feature_range=(0, 1))
train_dataset_scaled = sc.fit_transform(train_dataset)
print(f"✓ Features scaled: {train_dataset_scaled.shape}")
print(f"✓ Range: [{train_dataset_scaled.min():.4f}, {train_dataset_scaled.max():.4f}]")


# ============================================================================
# CELL 6: Summary
# ============================================================================
print("\n" + "="*70)
print("DATA PREPARATION COMPLETE")
print("="*70)
print(f"Scaled features: {train_dataset_scaled.shape}")
print(f"SOH targets: {soh.shape}")
print(f"Ready for model training!")
