# ECM (Equivalent Circuit Model) Implementation Guide

## 🎯 What is ECM?

**ECM (Equivalent Circuit Model)** is a physics-based approach to battery modeling that represents battery behavior using electrical circuit elements:

- **Resistors (R)**: Internal resistance, polarization effects
- **Capacitors (C)**: Charge storage, transient dynamics
- **Voltage Source (OCV)**: Open Circuit Voltage (function of SoC)

Unlike pure data-driven ML approaches, ECM provides **physically interpretable parameters** that directly relate to battery degradation mechanisms.

---

## 🔋 ECM Models Implemented

### 1. **Rint Model** (Simplest)

```
Circuit: OCV ---[R0]--- Terminal

Equation: V_terminal = OCV - I × R0
```

**Parameters:**
- `OCV`: Open Circuit Voltage (function of SoC)
- `R0`: Internal resistance (Ω)

**Use Case:** Quick estimation, real-time applications

**Pros:** Fast, simple, low computational cost  
**Cons:** Low accuracy, doesn't capture transient dynamics

---

### 2. **RC Model** (Thevenin Model)

```
Circuit: OCV ---[R0]---[R1-C1]--- Terminal

Equations:
  V_terminal = OCV - I×R0 - V1
  dV1/dt = -V1/(R1×C1) + I/C1
```

**Parameters:**
- `R0`: Ohmic resistance (Ω)
- `R1`: Polarization resistance (Ω)
- `C1`: Polarization capacitance (F)
- `τ1 = R1×C1`: Time constant (s)

**Use Case:** Good balance between accuracy and complexity

**Pros:** Captures transient response, physically meaningful  
**Cons:** May not capture all dynamics

---

### 3. **2RC Model** (PNGV Model)

```
Circuit: OCV ---[R0]---[R1-C1]---[R2-C2]--- Terminal

Equations:
  V_terminal = OCV - I×R0 - V1 - V2
  dV1/dt = -V1/(R1×C1) + I/C1  (fast dynamics)
  dV2/dt = -V2/(R2×C2) + I/C2  (slow dynamics)
```

**Parameters:**
- `R0`: Ohmic resistance (Ω)
- `R1, C1`: Fast polarization (charge transfer)
- `R2, C2`: Slow polarization (diffusion)
- `τ1 = R1×C1`: Fast time constant (s)
- `τ2 = R2×C2`: Slow time constant (s)

**Use Case:** High-accuracy applications

**Pros:** Captures both fast and slow dynamics, best accuracy  
**Cons:** More parameters to identify, higher computational cost

---

## 📊 Comparison: ECM vs ML Approach

| Aspect | ECM (This Implementation) | ML (Original Project) |
|--------|---------------------------|----------------------|
| **Approach** | Physics-based | Data-driven |
| **Model Type** | Differential equations | Neural networks |
| **Parameters** | R, C, OCV (physically meaningful) | Weights/biases (black box) |
| **Training Data** | Can work with limited data | Requires large datasets |
| **Interpretability** | ✅ High - parameters have physical meaning | ❌ Low - black box |
| **Real-time** | ✅ Fast computation | ⚠️ Depends on model size |
| **Accuracy** | ⚠️ Good with proper identification | ✅ High with enough data |
| **Generalization** | ✅ Better across conditions | ⚠️ Limited to training distribution |
| **Fault Detection** | ✅ Parameter changes indicate faults | ⚠️ Requires labeled fault data |

---

## 🚀 Quick Start

### 1. **Run the Jupyter Notebook**

```bash
cd ecm
jupyter notebook ecm_notebook.ipynb
```

The notebook includes:
- Model comparison on single cycle
- Parameter extraction over lifetime
- Degradation visualization
- Correlation analysis

### 2. **Use Python Scripts**

```python
from ecm_parameter_extraction import compare_ecm_models, ECMParameterExtractor

# Compare models on a single cycle
compare_ecm_models('B0005', cycle_num=50)

# Extract parameters over lifetime
extractor = ECMParameterExtractor('B0005')
extractor.load_battery_data()
params_df = extractor.extract_parameters_over_lifetime(model_type='RC', cycle_step=10)
extractor.plot_parameter_degradation(save_path='parameter_degradation.png')
extractor.save_parameters()
```

### 3. **Use Individual Models**

```python
from ecm_model import RCModel, SoCEstimator
import numpy as np

# Load your data
voltage = np.array([...])  # Terminal voltage (V)
current = np.array([...])  # Current (A)
time = np.array([...])     # Time (s)

# Estimate SoC
soc_estimator = SoCEstimator(nominal_capacity=2.0)
soc = soc_estimator.estimate(current, time, soc_initial=1.0)

# Fit RC model
model = RCModel()
model.fit(voltage, current, soc, time)

# Get parameters
params = model.get_parameters()
print(f"R0 = {params['R0']:.6f} Ω")
print(f"R1 = {params['R1']:.6f} Ω")
print(f"C1 = {params['C1']:.2f} F")

# Predict voltage
v_pred = model.predict(current, soc, time)
```

---

## 📈 What You Can Do with ECM

### 1. **State of Health (SoH) Estimation**

ECM parameters change with battery aging:

```python
# R0 increases with aging
SoH = 1 - (R0_current - R0_initial) / (R0_EOL - R0_initial)
```

**Physical Interpretation:**
- **R0 increase** → SEI layer growth, electrolyte degradation
- **R1 increase** → Active material loss, charge transfer resistance
- **C1 decrease** → Surface area reduction

### 2. **Remaining Useful Life (RUL) Prediction**

Model parameter degradation trends:

```python
# Fit degradation model (e.g., exponential)
R0(cycle) = R0_initial + a × exp(b × cycle)

# Predict when R0 reaches threshold
RUL = (R0_threshold - R0_current) / degradation_rate
```

### 3. **Fault Detection**

Abnormal parameter changes indicate faults:

```python
# Sudden R0 increase → Internal short circuit
# Sudden C1 decrease → Electrolyte leakage
# Abnormal τ1 change → Thermal issues
```

### 4. **Real-time SoC Estimation**

Use Extended Kalman Filter (EKF) with ECM:

```python
# State: [SoC, V1, V2]
# Measurement: V_terminal
# Process model: ECM equations
# Update: Kalman filter
```

---

## 🔬 Parameter Interpretation

### **R0 (Ohmic Resistance)**

**Physical Meaning:**
- Immediate voltage drop when current flows
- Represents: SEI layer, electrolyte, current collectors

**Degradation:**
- ⬆️ Increases with aging
- Causes: SEI growth, electrolyte decomposition, corrosion

**Typical Values:**
- New battery: 0.02-0.05 Ω
- Aged battery: 0.05-0.15 Ω

**Health Indicator:**
- Strong correlation with capacity fade
- Used for SoH estimation

---

### **R1 (Polarization Resistance)**

**Physical Meaning:**
- Charge transfer resistance at electrode-electrolyte interface
- Represents: Electrochemical reaction kinetics

**Degradation:**
- ⬆️ Increases with aging
- Causes: Active material loss, surface passivation

**Typical Values:**
- New battery: 0.01-0.03 Ω
- Aged battery: 0.03-0.10 Ω

**Health Indicator:**
- Related to power capability
- Indicates reaction kinetics degradation

---

### **C1 (Polarization Capacitance)**

**Physical Meaning:**
- Double-layer capacitance at electrode surface
- Represents: Charge storage capability

**Degradation:**
- ⬇️ Decreases with aging
- Causes: Surface area reduction, pore clogging

**Typical Values:**
- New battery: 1000-5000 F
- Aged battery: 500-2000 F

**Health Indicator:**
- Related to available surface area
- Indicates structural degradation

---

### **τ1 = R1 × C1 (Time Constant)**

**Physical Meaning:**
- Response time of polarization dynamics
- Represents: How fast battery responds to current changes

**Degradation:**
- Can increase or decrease depending on R1 and C1 changes
- Indicates overall dynamic behavior changes

**Typical Values:**
- New battery: 10-50 s
- Aged battery: 20-100 s

**Health Indicator:**
- Changes indicate degradation mechanisms
- Used for fault detection

---

## 📊 Expected Results

### **Model Accuracy**

| Model | RMSE (mV) | MAE (mV) | R² |
|-------|-----------|----------|-----|
| Rint  | 20-50     | 15-40    | 0.85-0.92 |
| RC    | 10-25     | 8-20     | 0.92-0.97 |
| 2RC   | 5-15      | 4-12     | 0.96-0.99 |

### **Parameter Degradation Trends**

For Battery B0005 (168 cycles):

- **R0**: +30% to +50% increase
- **R1**: +20% to +40% increase
- **C1**: -15% to -30% decrease
- **Capacity**: -15% to -20% decrease

### **Correlation with Capacity**

- **R0 vs Capacity**: -0.85 to -0.95 (strong negative)
- **R1 vs Capacity**: -0.70 to -0.85 (moderate negative)
- **C1 vs Capacity**: +0.60 to +0.80 (moderate positive)

---

## 🛠️ Advanced Usage

### **Hybrid ECM + ML Approach**

Combine ECM and ML for best results:

```python
# 1. Extract ECM parameters
extractor = ECMParameterExtractor('B0005')
extractor.load_battery_data()
params_df = extractor.extract_parameters_over_lifetime(model_type='RC')

# 2. Use ECM parameters as features for ML
from sklearn.ensemble import RandomForestRegressor

X = params_df[['R0', 'R1', 'C1', 'tau1']].values
y = params_df['capacity'].values

model = RandomForestRegressor()
model.fit(X, y)

# 3. Predict capacity from ECM parameters
capacity_pred = model.predict(X_new)
```

**Benefits:**
- ✅ Physically interpretable features
- ✅ Better generalization
- ✅ Requires less training data
- ✅ Fault detection capability

---

### **Multi-Battery Analysis**

Analyze multiple batteries:

```python
batteries = ['B0005', 'B0006', 'B0007', 'B0018']
all_params = []

for battery_id in batteries:
    extractor = ECMParameterExtractor(battery_id)
    extractor.load_battery_data()
    params = extractor.extract_parameters_over_lifetime(model_type='RC')
    params['battery_id'] = battery_id
    all_params.append(params)

# Combine and analyze
combined_df = pd.concat(all_params, ignore_index=True)

# Compare degradation rates
for battery_id in batteries:
    battery_data = combined_df[combined_df['battery_id'] == battery_id]
    R0_change = (battery_data['R0'].iloc[-1] - battery_data['R0'].iloc[0]) / battery_data['R0'].iloc[0]
    print(f"{battery_id}: R0 change = {R0_change*100:.2f}%")
```

---

## 📚 Files Created

1. **`ecm_model.py`** - Core ECM model implementations
   - `RintModel` class
   - `RCModel` class
   - `TwoRCModel` class
   - `SoCEstimator` class
   - Utility functions

2. **`ecm_parameter_extraction.py`** - Parameter extraction tools
   - `ECMParameterExtractor` class
   - `compare_ecm_models()` function
   - Visualization functions

3. **`ecm_notebook.ipynb`** - Interactive Jupyter notebook
   - Step-by-step tutorial
   - Visualization examples
   - Analysis workflows

4. **`ECM_GUIDE.md`** - This comprehensive guide

---

## ✅ Summary

You now have a complete ECM implementation that:

✅ **Three ECM models** (Rint, RC, 2RC)  
✅ **Parameter extraction** from battery data  
✅ **Degradation tracking** over lifetime  
✅ **Visualization tools** for analysis  
✅ **Physically interpretable** results  
✅ **Ready for hybrid ML+ECM** approaches  

**Key Advantage:** Unlike pure ML, ECM provides **physical insights** into battery degradation mechanisms!

---

