# Clean Notebook Template - Separate Cells

Use this structure in your notebook with separate cells for better organization.

---

## Cell 1: Imports

```python
from battery_loader import load_data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
```

---

## Cell 2: Load Data

```python
# Load battery data
dataset, capacity_data = load_data('B0005')

print(f"Dataset shape: {dataset.shape}")
print(f"Capacity data shape: {capacity_data.shape}")
```

---

## Cell 3: Create State of Health (SOH)

```python
# Calculate SOH
C = dataset['capacity'].iloc[0]
soh_values = (dataset['capacity'] / C).values
soh = pd.DataFrame(data=soh_values, columns=['SoH'])

print(f"SOH shape: {soh.shape}")
print(f"SOH range: [{soh['SoH'].min():.4f}, {soh['SoH'].max():.4f}]")
```

---

## Cell 4: Extract Features

```python
# Define features to use
attribs = ['capacity', 'voltage_measured', 'current_measured',
           'temperature_measured', 'current_load', 'voltage_load', 'time']

# Extract features as numpy array
train_dataset = dataset[attribs].values

print(f"Training dataset shape: {train_dataset.shape}")
print(f"Features: {attribs}")
```

---

## Cell 5: Scale Features

```python
# Apply MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
train_dataset_scaled = sc.fit_transform(train_dataset)

print(f"Scaled dataset shape: {train_dataset_scaled.shape}")
print(f"Scaled range: [{train_dataset_scaled.min():.4f}, {train_dataset_scaled.max():.4f}]")
```

---

## Cell 6: Verify Data

```python
# Display summary
print("="*70)
print("DATA PREPARATION COMPLETE")
print("="*70)
print(f"✓ Scaled features: {train_dataset_scaled.shape}")
print(f"✓ SOH targets: {soh.shape}")
print(f"✓ Feature names: {attribs}")
print(f"✓ Data ready for model training!")
```

---

## Cell 7: Split Data (Optional)

```python
from sklearn.model_selection import train_test_split

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_dataset_scaled, 
    soh, 
    test_size=0.2, 
    random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
```

---

## Cell 8: Build Model (Example)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(7,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()
```

---

## Cell 9: Train Model (Example)

```python
history = model.fit(
    X_train, 
    y_train, 
    epochs=50, 
    batch_size=32, 
    validation_data=(X_val, y_val),
    verbose=1
)
```

---

## Cell 10: Evaluate Model (Example)

```python
import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('Model MAE')

plt.tight_layout()
plt.show()
```

---

## Variables Available After Running Cells 1-6

| Variable | Type | Shape | Description |
|----------|------|-------|-------------|
| `dataset` | DataFrame | (50285, 10) | Full original dataset |
| `capacity_data` | DataFrame | (168, 4) | Capacity per cycle |
| `soh` | DataFrame | (50285, 1) | State of Health values |
| `train_dataset` | ndarray | (50285, 7) | Unscaled features |
| `train_dataset_scaled` | ndarray | (50285, 7) | Scaled features [0,1] |
| `sc` | MinMaxScaler | - | Fitted scaler object |
| `attribs` | list | 7 items | Feature names |

---

## Tips

1. **Run cells in order** from top to bottom
2. **Don't skip cells** - each depends on the previous ones
3. **Restart kernel** if you get errors (Kernel → Restart Kernel)
4. **Save often** (Ctrl+S or Cmd+S)
