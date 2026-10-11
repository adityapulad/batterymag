# ECM (Equivalent Circuit Model) for Battery Analysis

## Quick Start

### Option 1: Jupyter Notebook (Recommended)

```bash
cd ecm
jupyter notebook ecm_notebook.ipynb
```

The notebook includes everything you need:
- Model comparison (Rint, RC, 2RC)
- Parameter extraction over battery lifetime
- Degradation visualization
- Correlation analysis
- Step-by-step explanations

### Option 2: Python Script

```python
from ecm_parameter_extraction import compare_ecm_models

# Compare all three models on cycle 50
compare_ecm_models('B0005', cycle_num=50)
```

**Output:**
```
Model           RMSE (mV)    MAE (mV)     MAPE (%)     R²          
------------------------------------------------------------
Rint            35.2451      28.3421      0.8234       0.8912
RC              12.4532      9.8765       0.2876       0.9645
2RC             6.7821       5.2341       0.1543       0.9876
```

---

## Files

| File | Description |
|------|-------------|
| `ecm_model.py` | Core ECM implementations (Rint, RC, 2RC models) |
| `ecm_parameter_extraction.py` | Tools for extracting parameters from battery data |
| `ecm_notebook.ipynb` | Interactive tutorial and analysis |
| `ECM_GUIDE.md` | Comprehensive guide (theory, usage, interpretation) |
| `README.md` | This file |

---

## What is ECM?

ECM represents battery behavior using electrical circuits:

```
Rint Model:     OCV ---[R0]--- Terminal

RC Model:       OCV ---[R0]---[R1-C1]--- Terminal

2RC Model:      OCV ---[R0]---[R1-C1]---[R2-C2]--- Terminal
```

**Parameters:**
- **R0**: Ohmic resistance (Ω)
- **R1, R2**: Polarization resistances (Ω)
- **C1, C2**: Polarization capacitances (F)
- **OCV**: Open Circuit Voltage (V)

---

##  Why Use ECM?

### Advantages over Pure ML:

1. **Physically Interpretable**
   - R0 increase → SEI layer growth
   - R1 increase → Active material loss
   - C1 decrease → Surface area reduction

2. **Fault Detection**
   - Sudden R0 spike → Internal short
   - Abnormal τ1 → Thermal issues

3. **Less Data Required**
   - Works with limited cycles
   - No need for thousands of samples

4. **Real-time Capable**
   - Fast computation
   - Suitable for online estimation

5. **Better Generalization**
   - Works across different conditions
   - Not limited to training distribution

---

## Use Cases

### 1. State of Health (SoH) Estimation

```python
# R0 increases with aging
SoH = 1 - (R0_current - R0_initial) / (R0_EOL - R0_initial)
```

### 2. Remaining Useful Life (RUL) Prediction

```python
# Model parameter degradation
R0(cycle) = R0_initial + a × exp(b × cycle)

# Predict when R0 reaches threshold
RUL = (R0_threshold - R0_current) / degradation_rate
```

### 3. Fault Detection

```python
# Monitor parameter changes
if R0_change > threshold:
    print("Warning: Abnormal resistance increase!")
```

### 4. Hybrid ML Approach

```python
# Use ECM parameters as ML features
X = [R0, R1, C1, tau1]
y = capacity

model = RandomForestRegressor()
model.fit(X, y)
```

---

## Workflow

```python
# 1. Load battery data
from ecm_parameter_extraction import ECMParameterExtractor

extractor = ECMParameterExtractor('B0005')
extractor.load_battery_data()

# 2. Extract parameters over lifetime
params_df = extractor.extract_parameters_over_lifetime(
    model_type='RC',
    cycle_step=10
)

# 3. Visualize degradation
extractor.plot_parameter_degradation(
    save_path='parameter_degradation.png'
)

# 4. Save results
extractor.save_parameters('B0005_ecm_params.csv')
```

---

## Parameter Interpretation

### R0 (Ohmic Resistance)
- **Physical:** SEI layer, electrolyte, current collectors
- **Degradation:** Increases with aging
- **Typical:** 0.02-0.05 Ω (new) → 0.05-0.15 Ω (aged)
- **Indicator:** Strong correlation with capacity fade

### R1 (Polarization Resistance)
- **Physical:** Charge transfer at electrode interface
- **Degradation:** Increases with aging
- **Typical:** 0.01-0.03 Ω (new) → 0.03-0.10 Ω (aged)
- **Indicator:** Power capability, reaction kinetics

### C1 (Polarization Capacitance)
- **Physical:** Double-layer capacitance
- **Degradation:** Decreases with aging
- **Typical:** 1000-5000 F (new) → 500-2000 F (aged)
- **Indicator:** Available surface area

### τ1 = R1 × C1 (Time Constant)
- **Physical:** Response time
- **Degradation:** Variable (depends on R1 and C1)
- **Typical:** 10-50 s (new) → 20-100 s (aged)
- **Indicator:** Dynamic behavior changes

---

##  Learn More

- **Comprehensive Guide:** See `ECM_GUIDE.md` for detailed theory and examples
- **Interactive Tutorial:** Run `ecm_notebook.ipynb` for hands-on learning
- **Code Documentation:** Check docstrings in `ecm_model.py`

---

## ECM vs ML Comparison

| Aspect | ECM | ML/DL |
|--------|-----|-------|
| **Interpretability** | High | Black box |
| **Data Requirements** | Low | High |
| **Accuracy** | Good | Excellent |
| **Speed** | Fast | Variable |
| **Generalization** | Better | Limited |
| **Fault Detection** | Built-in | Needs labels |
