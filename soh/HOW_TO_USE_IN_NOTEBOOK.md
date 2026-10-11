# ✅ How to Use battery_loader in Your Notebook

## ✨ Good News!

The `battery_loader.py` file has been **fixed** and is now in your `soh/` directory with the correct path.

---

## 📋 Method 1: Separate Cells (Recommended)

Copy each code block below into **separate cells** in your notebook:

### Cell 1: Imports
```python
from battery_loader import load_data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
```

### Cell 2: Load Data
```python
dataset, capacity_data = load_data('B0005')
```

### Cell 3: Create SOH
```python
C = dataset['capacity'].iloc[0]
soh_values = (dataset['capacity'] / C).values
soh = pd.DataFrame(data=soh_values, columns=['SoH'])
```

### Cell 4: Extract Features
```python
attribs = ['capacity', 'voltage_measured', 'current_measured',
           'temperature_measured', 'current_load', 'voltage_load', 'time']
train_dataset = dataset[attribs].values
```

### Cell 5: Scale Features
```python
sc = MinMaxScaler(feature_range=(0, 1))
train_dataset_scaled = sc.fit_transform(train_dataset)

print(f"Scaled dataset shape: {train_dataset_scaled.shape}")
print(f"SOH shape: {soh.shape}")
```

---

## 📋 Method 2: Quick Start (All in One Cell)

If you prefer, put this in one cell:

```python
from battery_loader import load_data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Load data
dataset, capacity_data = load_data('B0005')

# Create SOH
C = dataset['capacity'].iloc[0]
soh = pd.DataFrame(data=(dataset['capacity'] / C).values, columns=['SoH'])

# Extract and scale features
attribs = ['capacity', 'voltage_measured', 'current_measured',
           'temperature_measured', 'current_load', 'voltage_load', 'time']
train_dataset = dataset[attribs].values
sc = MinMaxScaler(feature_range=(0, 1))
train_dataset_scaled = sc.fit_transform(train_dataset)

print(f"Scaled: {train_dataset_scaled.shape}, SOH: {soh.shape}")
```

---

## ✅ Expected Output

```
Total data in dataset:  168
[1, 24.0, datetime.datetime(2008, 4, 2, 15, 25, 41), 1.856487...]
Scaled dataset shape: (50285, 7)
SOH shape: (50285, 1)
```

---

## 🎯 What's Fixed

1. ✅ **battery_loader.py** - Now in `soh/` directory with correct path (`../datasets/`)
2. ✅ **Import works** - No more `ModuleNotFoundError`
3. ✅ **MinMaxScaler works** - Using `.values` to extract numpy array
4. ✅ **DataFrame works** - Using `data=` parameter correctly

---

## 📁 Files in soh/ Directory

- **`battery_loader.py`** - The fixed module (ready to import)
- **`HOW_TO_USE_IN_NOTEBOOK.md`** - This guide
- **`CLEAN_NOTEBOOK_TEMPLATE.md`** - Detailed template with examples
- **`notebook_cells.py`** - Code organized by cells (for reference)

---

## 🚀 Next Steps

After running the cells above, you'll have:

- `train_dataset_scaled` - Your scaled features (50285, 7)
- `soh` - Your target values (50285, 1)
- `dataset` - Full original data
- `sc` - The fitted scaler

Now you can build and train your model!

---

## 💡 Pro Tips

1. **Run cells in order** - Each cell depends on previous ones
2. **Restart kernel if needed** - Kernel → Restart Kernel
3. **Check your working directory** - Should be in `soh/` folder
4. **Use separate cells** - Easier to debug and modify

---

## ❓ Troubleshooting

If you still get `ModuleNotFoundError`:

```python
# Add this at the very top of your notebook
import sys
import os
print(f"Current directory: {os.getcwd()}")
print(f"battery_loader.py exists: {os.path.exists('battery_loader.py')}")
```

Should show:
```
Current directory: .../soh
battery_loader.py exists: True
```

---

**You're all set! Start with Cell 1 and work your way down.** 🎉
