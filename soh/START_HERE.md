# 🎯 START HERE - Use battery_loader in Separate Cells

## ✅ Everything is Fixed and Ready!

The `battery_loader.py` module is now working correctly in your `soh/` directory.

---

## 🚀 Quick Start - Copy These 5 Cells

Open your Jupyter notebook and create **5 separate cells** with this code:

### 📦 Cell 1: Imports
```python
from battery_loader import load_data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
```

### 📂 Cell 2: Load Data
```python
dataset, capacity_data = load_data('B0005')
```

### 🔋 Cell 3: Create SOH
```python
C = dataset['capacity'].iloc[0]
soh = pd.DataFrame(data=(dataset['capacity'] / C).values, columns=['SoH'])
```

### 📊 Cell 4: Extract Features
```python
attribs = ['capacity', 'voltage_measured', 'current_measured',
           'temperature_measured', 'current_load', 'voltage_load', 'time']
train_dataset = dataset[attribs].values
```

### ⚖️ Cell 5: Scale Features
```python
sc = MinMaxScaler(feature_range=(0, 1))
train_dataset_scaled = sc.fit_transform(train_dataset)

print(f"✓ Scaled: {train_dataset_scaled.shape}")
print(f"✓ SOH: {soh.shape}")
```

---

## ✨ Run the Cells

Press `Shift + Enter` on each cell in order.

**Expected output from Cell 5:**
```
Total data in dataset:  168
✓ Scaled: (50285, 7)
✓ SOH: (50285, 1)
```

---

## 🎉 Done!

Now you have:
- ✅ `train_dataset_scaled` - Scaled features ready for training
- ✅ `soh` - Target values (State of Health)
- ✅ `dataset` - Full original data
- ✅ `sc` - Fitted scaler

Continue with your model in the next cells!

---

## 📚 More Help

- **`HOW_TO_USE_IN_NOTEBOOK.md`** - Detailed guide
- **`CLEAN_NOTEBOOK_TEMPLATE.md`** - Full template with model example
- **`battery_loader.py`** - The working module

---

**That's it! Just copy the 5 cells above and you're ready to go!** 🚀
