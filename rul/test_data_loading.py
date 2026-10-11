"""
Test script to verify the RUL dataset can be loaded correctly
"""
import pandas as pd
import os

print("=" * 60)
print("TESTING RUL DATASET LOADING")
print("=" * 60)

# Get current directory
current_dir = os.getcwd()
print(f"\nCurrent directory: {current_dir}")

# Test the path from the rul directory
os.chdir('rul')
print(f"Changed to: {os.getcwd()}")

# Try to load the dataset using the notebook's path
dataset_path = "../datasets/rul_dataset/Battery_RUL.csv"
print(f"\nAttempting to load: {dataset_path}")

try:
    df = pd.read_csv(dataset_path)
    print(f"✅ SUCCESS! Dataset loaded successfully")
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset info:")
    print(df.info())
    print(f"\n✅ The notebook should work perfectly now!")
except Exception as e:
    print(f"❌ ERROR: {e}")
    print(f"\nTrying absolute path...")
    os.chdir('..')
    abs_path = "datasets/rul_dataset/Battery_RUL.csv"
    try:
        df = pd.read_csv(abs_path)
        print(f"✅ Dataset loaded with absolute path: {abs_path}")
        print(f"Dataset shape: {df.shape}")
    except Exception as e2:
        print(f"❌ ERROR with absolute path: {e2}")

print("\n" + "=" * 60)
