"""
Battery RUL Setup Verification Script
This script verifies that all required libraries are installed and the dataset is accessible.
"""

import sys

def check_imports():
    """Check if all required libraries can be imported."""
    print("=" * 60)
    print("CHECKING REQUIRED LIBRARIES")
    print("=" * 60)
    
    libraries = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib.pyplot',
        'seaborn': 'seaborn',
        'sklearn': 'sklearn',
        'tensorflow': 'tensorflow',
        'keras': 'keras',
        'xgboost': 'xgboost',
        'jupyter': 'jupyter',
        'plotly': 'plotly'
    }
    
    failed = []
    
    for name, import_path in libraries.items():
        try:
            __import__(import_path)
            print(f"✅ {name:20s} - OK")
        except ImportError as e:
            print(f"❌ {name:20s} - FAILED: {e}")
            failed.append(name)
    
    print()
    
    if failed:
        print(f"⚠️  {len(failed)} library(ies) failed to import: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All libraries imported successfully!")
        return True

def check_sklearn_modules():
    """Check specific sklearn modules used in the notebook."""
    print("\n" + "=" * 60)
    print("CHECKING SCIKIT-LEARN MODULES")
    print("=" * 60)
    
    modules = [
        ('RandomForestRegressor', 'sklearn.ensemble', 'RandomForestRegressor'),
        ('AdaBoostRegressor', 'sklearn.ensemble', 'AdaBoostRegressor'),
        ('GradientBoostingRegressor', 'sklearn.ensemble', 'GradientBoostingRegressor'),
        ('BaggingRegressor', 'sklearn.ensemble', 'BaggingRegressor'),
        ('SVR', 'sklearn.svm', 'SVR'),
        ('DecisionTreeRegressor', 'sklearn.tree', 'DecisionTreeRegressor'),
        ('ExtraTreeRegressor', 'sklearn.tree', 'ExtraTreeRegressor'),
        ('LinearRegression', 'sklearn.linear_model', 'LinearRegression'),
        ('SGDRegressor', 'sklearn.linear_model', 'SGDRegressor'),
        ('KNeighborsRegressor', 'sklearn.neighbors', 'KNeighborsRegressor'),
        ('StandardScaler', 'sklearn.preprocessing', 'StandardScaler'),
        ('train_test_split', 'sklearn.model_selection', 'train_test_split'),
        ('mean_squared_error', 'sklearn.metrics', 'mean_squared_error'),
        ('mean_absolute_error', 'sklearn.metrics', 'mean_absolute_error'),
    ]
    
    failed = []
    
    for name, module, attr in modules:
        try:
            mod = __import__(module, fromlist=[attr])
            getattr(mod, attr)
            print(f"✅ {name:30s} - OK")
        except (ImportError, AttributeError) as e:
            print(f"❌ {name:30s} - FAILED: {e}")
            failed.append(name)
    
    print()
    
    if failed:
        print(f"⚠️  {len(failed)} module(s) failed: {', '.join(failed)}")
        return False
    else:
        print("✅ All sklearn modules available!")
        return True

def check_dataset():
    """Check if the dataset file exists and is readable."""
    print("\n" + "=" * 60)
    print("CHECKING DATASET")
    print("=" * 60)
    
    import os
    import pandas as pd
    
    dataset_paths = [
        "../datasets/archive (1)/Battery_dataset.csv",
        "datasets/archive (1)/Battery_dataset.csv",
        "../datasets/archive (1)/Battery_RUL.csv",
        "datasets/archive (1)/Battery_RUL.csv",
    ]
    
    dataset_found = False
    
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"✅ Dataset found: {path}")
            try:
                df = pd.read_csv(path)
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                print(f"   First few rows:")
                print(df.head(3))
                dataset_found = True
                break
            except Exception as e:
                print(f"❌ Error reading dataset: {e}")
                return False
    
    if not dataset_found:
        print("❌ Dataset not found in expected locations:")
        for path in dataset_paths:
            print(f"   - {path}")
        print("\n💡 Please ensure the dataset is in the correct location.")
        return False
    
    print("\n✅ Dataset is accessible and readable!")
    return True

def check_versions():
    """Display versions of key libraries."""
    print("\n" + "=" * 60)
    print("LIBRARY VERSIONS")
    print("=" * 60)
    
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import tensorflow as tf
        import keras
        import xgboost as xgb
        
        print(f"NumPy:        {np.__version__}")
        print(f"Pandas:       {pd.__version__}")
        print(f"Scikit-learn: {sklearn.__version__}")
        print(f"TensorFlow:   {tf.__version__}")
        print(f"Keras:        {keras.__version__}")
        print(f"XGBoost:      {xgb.__version__}")
        print(f"Python:       {sys.version.split()[0]}")
        
    except Exception as e:
        print(f"⚠️  Could not retrieve all versions: {e}")

def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("BATTERY RUL SETUP VERIFICATION")
    print("=" * 60)
    print()
    
    checks = [
        ("Library Imports", check_imports),
        ("Sklearn Modules", check_sklearn_modules),
        ("Dataset Access", check_dataset),
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error during {name} check: {e}")
            results.append((name, False))
    
    # Display versions
    check_versions()
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:20s}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 SUCCESS! All checks passed. You're ready to run the RUL notebook!")
        print("\nNext steps:")
        print("1. Start Jupyter: jupyter notebook")
        print("2. Open: rul/battery_remaining_life_prediction.ipynb")
        print("3. Update dataset path in the notebook if needed")
        print("4. Run all cells")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("1. Install missing libraries: pip install -r requirements.txt")
        print("2. Ensure dataset is in: datasets/archive (1)/Battery_dataset.csv")
        print("3. Check the SETUP_GUIDE.md for more help")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
