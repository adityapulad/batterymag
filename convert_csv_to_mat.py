"""
Convert NASA Battery CSV Dataset to MATLAB .mat format for SoH notebooks

This script converts the cleaned NASA battery dataset from CSV format
to MATLAB .mat format that the SoH notebooks expect.

Usage:
    python convert_csv_to_mat.py
"""

import os
import pandas as pd
import numpy as np
from scipy.io import savemat
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def parse_start_time(time_str):
    """Parse the start_time array string from metadata"""
    try:
        # Remove brackets and split
        time_str = time_str.strip('[]')
        parts = [float(x.strip()) for x in time_str.split()]
        return parts
    except:
        return None

def load_metadata():
    """Load and parse the metadata CSV"""
    print("Loading metadata...")
    metadata_path = 'datasets/archive/cleaned_dataset/metadata.csv'
    metadata = pd.read_csv(metadata_path)
    return metadata

def group_by_battery(metadata):
    """Group files by battery ID"""
    batteries = {}
    for _, row in metadata.iterrows():
        battery_id = row['battery_id']
        if battery_id not in batteries:
            batteries[battery_id] = []
        batteries[battery_id].append(row)
    return batteries

def load_csv_data(filename):
    """Load a single CSV file"""
    filepath = f'datasets/archive/cleaned_dataset/data/{filename}'
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def create_mat_structure(battery_id, battery_data):
    """
    Create MATLAB structure matching the expected format:
    battery[0,0]['cycle'][0] contains array of discharge cycles
    Each cycle has: type, ambient_temperature, time, data
    """
    print(f"\nProcessing battery {battery_id}...")
    
    cycles = []
    cycle_num = 0
    
    for idx, row in enumerate(battery_data):
        # Only process discharge cycles for SoH estimation
        if row['type'] != 'discharge':
            continue
            
        cycle_num += 1
        
        # Load the CSV data
        csv_data = load_csv_data(row['filename'])
        if csv_data is None:
            continue
        
        # Parse time
        time_parts = parse_start_time(row['start_time'])
        if time_parts is None:
            continue
        
        # Create cycle structure
        cycle = {
            'type': 'discharge',
            'ambient_temperature': np.array([[row['ambient_temperature']]]),
            'time': np.array([time_parts]),
            'data': {
                'Voltage_measured': csv_data['Voltage_measured'].values,
                'Current_measured': csv_data['Current_measured'].values,
                'Temperature_measured': csv_data['Temperature_measured'].values,
                'Current_load': csv_data['Current_load'].values,
                'Voltage_load': csv_data['Voltage_load'].values,
                'Time': csv_data['Time'].values,
                'Capacity': np.array([[row['Capacity']]]) if pd.notna(row['Capacity']) else np.array([[0.0]])
            }
        }
        
        cycles.append(cycle)
        
        if cycle_num % 10 == 0:
            print(f"  Processed {cycle_num} discharge cycles...")
    
    print(f"  Total discharge cycles: {cycle_num}")
    
    # Create the main structure
    mat_structure = {
        battery_id: {
            'cycle': np.array(cycles, dtype=object)
        }
    }
    
    return mat_structure

def convert_battery_to_mat(battery_id, battery_data):
    """Convert a single battery's data to .mat format"""
    try:
        # Create the structure
        mat_structure = create_mat_structure(battery_id, battery_data)
        
        # Create output directory
        os.makedirs('battery_data', exist_ok=True)
        
        # Save to .mat file
        output_file = f'battery_data/{battery_id}.mat'
        savemat(output_file, mat_structure, oned_as='row')
        
        print(f"✅ Saved {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error converting {battery_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main conversion process"""
    print("="*60)
    print("NASA Battery Dataset CSV to MAT Converter")
    print("="*60)
    
    # Load metadata
    metadata = load_metadata()
    print(f"Loaded metadata with {len(metadata)} entries")
    
    # Group by battery
    batteries = group_by_battery(metadata)
    print(f"\nFound {len(batteries)} unique batteries:")
    for battery_id in batteries.keys():
        print(f"  - {battery_id}: {len(batteries[battery_id])} test files")
    
    # Convert each battery
    print("\n" + "="*60)
    print("Starting conversion...")
    print("="*60)
    
    success_count = 0
    for battery_id, battery_data in batteries.items():
        if convert_battery_to_mat(battery_id, battery_data):
            success_count += 1
    
    print("\n" + "="*60)
    print(f"Conversion complete!")
    print(f"Successfully converted {success_count}/{len(batteries)} batteries")
    print(f"Output directory: battery_data/")
    print("="*60)
    
    # List created files
    if os.path.exists('battery_data'):
        files = os.listdir('battery_data')
        print(f"\nCreated files:")
        for f in sorted(files):
            size = os.path.getsize(f'battery_data/{f}') / (1024*1024)
            print(f"  - {f} ({size:.2f} MB)")

if __name__ == '__main__':
    main()
