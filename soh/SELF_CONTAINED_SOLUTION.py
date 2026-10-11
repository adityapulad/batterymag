"""
SELF-CONTAINED SOLUTION - Copy this entire code into ONE notebook cell
No external imports needed - everything is included here
"""

import datetime
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# STEP 1: Define load_data function (no external file needed)
# ============================================================================

def load_data(battery):
    """Load battery data from .mat file"""
    mat = loadmat('../datasets/battery_data/' + battery + '.mat')
    print('Total data in dataset: ', len(mat[battery][0, 0]['cycle'][0]))
    counter = 0
    dataset = []
    capacity_data = []
    
    for i in range(len(mat[battery][0, 0]['cycle'][0])):
        row = mat[battery][0, 0]['cycle'][0, i]
        if row['type'][0] == 'discharge':
            ambient_temperature = float(row['ambient_temperature'][0][0][0,0])
            date_time = datetime.datetime(int(row['time'][0][0][0,0]),
                                     int(row['time'][0][0][0,1]),
                                     int(row['time'][0][0][0,2]),
                                     int(row['time'][0][0][0,3]),
                                     int(row['time'][0][0][0,4])) + datetime.timedelta(seconds=int(row['time'][0][0][0,5]))
            data = row['data']
            capacity = float(data[0][0]['Capacity'][0][0][0,0])
            for j in range(len(data[0][0]['Voltage_measured'][0][0][0])):
                voltage_measured = float(data[0][0]['Voltage_measured'][0][0][0, j])
                current_measured = float(data[0][0]['Current_measured'][0][0][0, j])
                temperature_measured = float(data[0][0]['Temperature_measured'][0][0][0, j])
                current_load = float(data[0][0]['Current_load'][0][0][0, j])
                voltage_load = float(data[0][0]['Voltage_load'][0][0][0, j])
                time = float(data[0][0]['Time'][0][0][0, j])
                dataset.append([counter + 1, ambient_temperature, date_time, capacity,
                              voltage_measured, current_measured,
                              temperature_measured, current_load,
                              voltage_load, time])
            capacity_data.append([counter + 1, ambient_temperature, date_time, capacity])
            counter = counter + 1
    
    return [pd.DataFrame(data=dataset,
                         columns=['cycle', 'ambient_temperature', 'datetime',
                                  'capacity', 'voltage_measured',
                                  'current_measured', 'temperature_measured',
                                  'current_load', 'voltage_load', 'time']),
            pd.DataFrame(data=capacity_data,
                         columns=['cycle', 'ambient_temperature', 'datetime',
                                  'capacity'])]

# ============================================================================
# STEP 2: Load the data
# ============================================================================

print("Loading battery data...")
dataset, capacity_data = load_data('B0005')

# ============================================================================
# STEP 3: Create SOH (State of Health)
# ============================================================================

print("\nCreating SOH...")
C = dataset['capacity'].iloc[0]  # Use .iloc[0] instead of [0]
soh_values = (dataset['capacity'] / C).values  # Vectorized operation
soh = pd.DataFrame(data=soh_values, columns=['SoH'])

# ============================================================================
# STEP 4: Extract and scale features
# ============================================================================

print("\nExtracting features...")
attribs = ['capacity', 'voltage_measured', 'current_measured',
           'temperature_measured', 'current_load', 'voltage_load', 'time']

# Extract as numpy array directly
train_dataset = dataset[attribs].values

print(f"Dataset shape: {train_dataset.shape}")
print(f"Dataset dtype: {train_dataset.dtype}")

# ============================================================================
# STEP 5: Apply MinMaxScaler
# ============================================================================

print("\nApplying MinMaxScaler...")
sc = MinMaxScaler(feature_range=(0, 1))
train_dataset_scaled = sc.fit_transform(train_dataset)

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*70)
print("✓ SUCCESS!")
print("="*70)
print(f"Scaled dataset shape: {train_dataset_scaled.shape}")
print(f"SOH shape: {soh.shape}")
print(f"\nScaled data range: [{train_dataset_scaled.min():.4f}, {train_dataset_scaled.max():.4f}]")
print(f"SOH range: [{soh['SoH'].min():.4f}, {soh['SoH'].max():.4f}]")
print("\nFirst 5 rows of scaled data:")
print(train_dataset_scaled[:5])
print("\nFirst 5 rows of SOH:")
print(soh.head())
