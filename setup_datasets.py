"""
Setup and organize datasets for the Battery Management System project

This script:
1. Verifies the dataset structure
2. Provides options to convert data or modify notebooks
3. Cleans up unnecessary folders

Usage:
    python setup_datasets.py
"""

import os
import shutil

def check_directory_exists(path):
    """Check if a directory exists"""
    return os.path.exists(path) and os.path.isdir(path)

def check_file_exists(path):
    """Check if a file exists"""
    return os.path.exists(path) and os.path.isfile(path)

def count_files_in_directory(path, extension=None):
    """Count files in a directory"""
    if not check_directory_exists(path):
        return 0
    
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if extension is None or file.endswith(extension):
                count += 1
    return count

def verify_datasets():
    """Verify the current dataset structure"""
    print("="*70)
    print("BATTERY MANAGEMENT SYSTEM - Dataset Verification")
    print("="*70)
    
    results = {}
    
    # Check SoC dataset (Dataset_Li-ion)
    print("\n1. State of Charge (SoC) Dataset")
    print("-" * 70)
    soc_path = 'datasets/Dataset_Li-ion'
    if check_directory_exists(soc_path):
        csv_count = count_files_in_directory(soc_path, '.csv')
        print(f"   ✅ Found: {soc_path}")
        print(f"   📊 Files: {csv_count} CSV files")
        print(f"   📁 Status: READY FOR USE")
        results['soc'] = True
    else:
        print(f"   ❌ Missing: {soc_path}")
        print(f"   📁 Status: NOT FOUND")
        results['soc'] = False
    
    # Check SoH dataset (NASA cleaned_dataset)
    print("\n2. State of Health (SoH) Dataset")
    print("-" * 70)
    soh_csv_path = 'datasets/archive/cleaned_dataset'
    if check_directory_exists(soh_csv_path):
        csv_count = count_files_in_directory(f'{soh_csv_path}/data', '.csv')
        print(f"   ✅ Found: {soh_csv_path}")
        print(f"   📊 Files: {csv_count} CSV files")
        print(f"   📁 Format: CSV (notebooks expect .mat)")
        print(f"   ⚠️  Status: NEEDS CONVERSION OR NOTEBOOK MODIFICATION")
        results['soh_csv'] = True
    else:
        print(f"   ❌ Missing: {soh_csv_path}")
        results['soh_csv'] = False
    
    # Check if battery_data folder exists (expected by SoH notebooks)
    battery_data_path = 'battery_data'
    if check_directory_exists(battery_data_path):
        mat_count = count_files_in_directory(battery_data_path, '.mat')
        print(f"   ✅ Found: {battery_data_path}")
        print(f"   📊 Files: {mat_count} .mat files")
        print(f"   📁 Status: READY FOR SOH NOTEBOOKS")
        results['soh_mat'] = True
    else:
        print(f"   ℹ️  Not found: {battery_data_path} (will be created if needed)")
        results['soh_mat'] = False
    
    # Check additional datasets
    print("\n3. Additional Datasets")
    print("-" * 70)
    
    archive1_path = 'datasets/archive (1)/Battery_dataset.csv'
    if check_file_exists(archive1_path):
        size = os.path.getsize(archive1_path) / (1024*1024)
        print(f"   ✅ Found: {archive1_path} ({size:.2f} MB)")
        results['archive1'] = True
    else:
        print(f"   ℹ️  Not found: {archive1_path}")
        results['archive1'] = False
    
    archive2_path = 'datasets/archive (2)'
    if check_directory_exists(archive2_path):
        file_count = count_files_in_directory(archive2_path)
        if file_count == 0:
            print(f"   ⚠️  Empty folder: {archive2_path} (can be deleted)")
            results['archive2_empty'] = True
        else:
            print(f"   ✅ Found: {archive2_path} ({file_count} files)")
            results['archive2_empty'] = False
    
    return results

def show_recommendations(results):
    """Show recommendations based on verification results"""
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if results.get('soc'):
        print("\n✅ SoC Notebooks (soc/ folder) - READY TO RUN")
        print("   You can immediately run the SoC estimation notebooks:")
        print("   - soc/CNN.ipynb")
        print("   - soc/LSTM.ipynb")
        print("   - soc/DNN.ipynb")
        print("   - soc/XGBoost.ipynb")
    
    if results.get('soh_csv') and not results.get('soh_mat'):
        print("\n⚠️  SoH Notebooks (soh/ folder) - NEEDS SETUP")
        print("   Choose one of these options:")
        print()
        print("   Option 1: Convert CSV to MAT format (Recommended)")
        print("   ---------------------------------------------------------")
        print("   Run: python convert_csv_to_mat.py")
        print("   This will create battery_data/ folder with .mat files")
        print("   Then you can run SoH notebooks without modifications")
        print()
        print("   Option 2: Modify notebooks to use CSV directly")
        print("   ---------------------------------------------------------")
        print("   Manually update the load_data() function in each SoH notebook")
        print("   to read CSV files instead of .mat files")
    
    if results.get('soh_mat'):
        print("\n✅ SoH Notebooks (soh/ folder) - READY TO RUN")
        print("   You can run the SoH estimation notebooks:")
        print("   - soh/CNN.ipynb")
        print("   - soh/LSTM.ipynb (if available)")
        print("   - soh/DNN.ipynb (if available)")
    
    if results.get('archive2_empty'):
        print("\n🧹 Cleanup Recommendation")
        print("   Remove empty folder: datasets/archive (2)/")
        print("   Run: python setup_datasets.py --cleanup")

def cleanup_empty_folders():
    """Remove empty archive folders"""
    print("\n" + "="*70)
    print("CLEANUP")
    print("="*70)
    
    archive2_path = 'datasets/archive (2)'
    if check_directory_exists(archive2_path):
        file_count = count_files_in_directory(archive2_path)
        if file_count == 0:
            try:
                shutil.rmtree(archive2_path)
                print(f"✅ Removed empty folder: {archive2_path}")
            except Exception as e:
                print(f"❌ Error removing {archive2_path}: {e}")
        else:
            print(f"⚠️  Folder not empty: {archive2_path} ({file_count} files)")
    else:
        print(f"ℹ️  Folder doesn't exist: {archive2_path}")

def main():
    """Main function"""
    import sys
    
    # Check for cleanup flag
    if '--cleanup' in sys.argv:
        cleanup_empty_folders()
        return
    
    # Verify datasets
    results = verify_datasets()
    
    # Show recommendations
    show_recommendations(results)
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Open Jupyter Lab (already running):")
    print("   http://localhost:8888/lab")
    print()
    print("2. Start with SoC notebooks (ready to use)")
    print()
    print("3. For SoH notebooks, run:")
    print("   python convert_csv_to_mat.py")
    print()
    print("4. To cleanup empty folders:")
    print("   python setup_datasets.py --cleanup")
    print()

if __name__ == '__main__':
    main()
