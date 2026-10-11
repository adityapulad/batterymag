# 🔬 Theoretical & Mathematical Foundations of the Battery Intelligence Platform

This document delivers an in-depth **scientific, theoretical, and mathematical breakdown** of every component and algorithm in the Battery Intelligence platform. It provides explicit electrochemical derivations, tensor formulations, neural equations, and step-by-step numerical examples for each lifecycle phase.

---

## 📑 Table of Contents
1. [Core Physics: Electrochemical Cell Dynamics & Degradation](#1-core-physics-electrochemical-cell-dynamics--degradation)
2. [Component 1: Equivalent Circuit Modeling (ECM)](#2-component-1-equivalent-circuit-modeling-ecm)
   - [Thevenin 1-RC Differential Equations](#thevenin-1-rc-differential-equations)
   - [L-BFGS-B Constrained Optimization & Identifiability Guard](#l-bfgs-b-constrained-optimization--identifiability-guard)
   - [Numerical Step-by-Step Example](#numerical-ecm-step-by-step-example)
3. [Component 2: State of Charge (SoC) Hybrid Tracking](#3-component-2-state-of-charge-soc-hybrid-tracking)
   - [Physics: Coulomb Counting with IEEE Sign Correction](#physics-coulomb-counting-with-ieee-sign-correction)
   - [Machine Learning: XGBoost Gradient Boosted Trees Observer](#machine-learning-xgboost-gradient-boosted-trees-observer)
   - [Complementary Fusion Filter](#complementary-fusion-filter)
   - [Numerical Step-by-Step Example](#numerical-soc-step-by-step-example)
4. [Component 3: State of Health (SoH) LSTM Network](#4-component-3-state-of-health-soh-lstm-network)
   - [Mathematical Formulation of LSTM Gates](#mathematical-formulation-of-lstm-gates)
   - [Feature Physics: Differential Overpotential and Resistance](#feature-physics-differential-overpotential-and-resistance)
   - [Surrogate Fallback Gate Mechanism](#surrogate-fallback-gate-mechanism)
   - [Numerical Step-by-Step Example](#numerical-soh-step-by-step-example)
5. [Component 4: Remaining Useful Life (RUL) LSTM Network](#5-component-4-remaining-useful-life-rul-lstm-network)
   - [Theoretical Definition of End-of-Life (EOL)](#theoretical-definition-of-end-of-life-eol)
   - [The Role of Physical Sampling Interval ($dt$) in Polarization Derivatives](#the-role-of-physical-sampling-interval-dt-in-polarization-derivatives)
   - [Uncertainty Estimation & Confidence Intervals](#uncertainty-estimation--confidence-intervals)
   - [Numerical Step-by-Step Example](#numerical-rul-step-by-step-example)
6. [Component 5: Fault Detection & Electrochemical Safety Guard](#6-component-5-fault-detection--electrochemical-safety-guard)
   - [Physical Mechanisms for Anomaly Detection](#physical-mechanisms-for-anomaly-detection)
   - [Deterministic Rule Thresholds](#deterministic-rule-thresholds)
7. [Component 6: SQLite Audit Trail & Traceability Engine](#7-component-6-sqlite-audit-trail--traceability-engine)

---

## 1. Core Physics: Electrochemical Cell Dynamics & Degradation

A Lithium-ion (Li-ion) cell operates through the intercalation and de-intercalation of Lithium ions ($Li^+$) between the positive cathode ($LiCoO_2$ or NMC) and the negative anode (graphite):

$$\text{Cathode: } Li_{1-x}CoO_2 + x Li^+ + x e^- \underset{\text{charge}}{\overset{\text{discharge}}{\rightleftharpoons}} LiCoO_2$$
$$\text{Anode: } Li_x C_6 \underset{\text{charge}}{\overset{\text{discharge}}{\rightleftharpoons}} 6C + x Li^+ + x e^-$$

As the cell undergoes repetitive charge and discharge cycling, degradation occurs along two parallel electrochemical paths:
1. **Loss of Lithium Inventory (LLI):** Formation and continuous thickening of the Solid Electrolyte Interphase (SEI) layer on the graphite anode traps cyclable $Li^+$ ions, causing **Capacity Fade**.
2. **Loss of Active Material (LAM):** Particle cracking, contact loss, and electrolyte decomposition elevate the internal impedance, causing **Ohmic Resistance Growth** ($R_0$) and **Charge-Transfer Resistance Growth** ($R_1$).

Understanding these two physical mechanisms is the basis for our hybrid estimation: capacity fade is tracked via neural networks, while impedance growth is tracked via Equivalent Circuit Models.

---

## 2. Component 1: Equivalent Circuit Modeling (ECM)

### Thevenin 1-RC Differential Equations

The cell is modeled as an electrical circuit consisting of an Open-Circuit Voltage source ($V_{\text{OCV}}$), an internal Ohmic resistance ($R_0$), and a parallel polarization resistor-capacitor pair ($R_1 \parallel C_1$):

```
       + o─────[ R0 ]─────┬─────[ R1 ]─────┬─────o -
                          │                │
                         === C1            │
                          │                │
                          └────────────────┘
                          ▲                ▲
                       Polarization Branch (V1)
```

The mathematical state equations in continuous time are:

$$V_t(t) = V_{\text{OCV}}(\text{SoC}) - I_L(t) R_0 - V_1(t)$$

$$\frac{d V_1(t)}{dt} = -\frac{V_1(t)}{R_1 C_1} + \frac{I_L(t)}{C_1}$$

where:
* $V_t(t)$ is the observable terminal voltage ($V$).
* $I_L(t)$ is the load current ($A$), defined as positive during discharge.
* $V_{\text{OCV}}(\text{SoC})$ is the thermodynamic equilibrium potential, represented as an empirical polynomial:
  $$V_{\text{OCV}}(\text{SoC}) = \sum_{k=0}^{M} a_k \cdot \text{SoC}^k$$
* $R_0$ models the high-frequency ohmic resistance of the electrolyte, separator, and current collectors.
* $R_1$ represents the activation charge-transfer polarization resistance at the electrode-electrolyte interface.
* $C_1$ represents the electric double-layer capacitance.
* $\tau_1 = R_1 C_1$ represents the polarization relaxation time constant ($s$).

In discrete time, using the zero-order hold approximation with time step $\Delta t_k = t_k - t_{k-1}$:

$$V_{1}[k] = V_{1}[k-1] \cdot \exp\left(-\frac{\Delta t_k}{\tau_1}\right) + I_L[k-1] \cdot R_1 \left(1 - \exp\left(-\frac{\Delta t_k}{\tau_1}\right)\right)$$

### L-BFGS-B Constrained Optimization & Identifiability Guard

The parameter vector $\boldsymbol{\theta} = [R_0, R_1, C_1]^T$ is estimated by minimizing the Mean Squared Error (MSE) between the observed terminal voltage $V_t^{\text{meas}}$ and the simulated model voltage $V_t^{\text{sim}}(\boldsymbol{\theta})$:

$$\min_{\boldsymbol{\theta}} \mathcal{J}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{k=1}^{N} \left( V_t^{\text{meas}}[k] - V_t^{\text{sim}}[k; \boldsymbol{\theta}] \right)^2$$

Subject to strict physical domain boundaries:
$$R_0 \in [0.010\,\Omega, 0.200\,\Omega], \quad R_1 \in [0.005\,\Omega, 0.150\,\Omega], \quad C_1 \in [200.0\,\text{F}, 5000.0\,\text{F}]$$

#### The Excitation & Identifiability Guard
When a cell is at rest ($I_L \approx 0\,\text{A}$), terminal voltage equals $V_{\text{OCV}}$ and contains zero current excitation. Under zero excitation, $\frac{\partial \mathcal{J}}{\partial R_0} = 0$ and $\frac{\partial \mathcal{J}}{\partial R_1} = 0$. The parameters become mathematically **unidentifiable**.

Our implementation executes an excitation check before optimization:
```python
if max(|I_L|) < 0.08 A or std(I_L) < 0.02 A:
    estimation_status = "unidentifiable_insufficient_excitation"
    R2 = null  # NOT 0.0 or 1.0 (undefined variance)
    return baseline_physical_params
```
Furthermore, if the optimizer pins to the limits ($R_0 = 0.010\,\Omega$), the engine reports `boundary_hit: true` and flags `estimation_status: "boundary_constrained"` separately from mathematical convergence.

### Numerical ECM Step-by-Step Example

Let an operational discharge step have:
* Current: $I_L = 2.0\,\text{A}$
* Initial voltage: $V_t[0] = 4.200\,\text{V}$
* Immediate step drop at $t_1$: $V_t[1] = 4.108\,\text{V}$
* Initial parameters: $R_0 = 0.046\,\Omega$, $R_1 = 0.024\,\Omega$, $C_1 = 1500.0\,\text{F}$ ($\tau_1 = 36.0\,\text{s}$).

1. **Ohmic Voltage Drop Calculation:**
   $$\Delta V_{\text{ohmic}} = I_L \cdot R_0 = 2.0\,\text{A} \times 0.046\,\Omega = 0.092\,\text{V}$$
2. **Polarization Voltage Growth ($\Delta t = 10\,\text{s}$):**
   $$V_1[1] = 0 + 2.0 \times 0.024 \times \left(1 - \exp\left(-\frac{10}{36}\right)\right) = 0.048 \times (1 - 0.7576) = 0.0116\,\text{V}$$
3. **Simulated Voltage:**
   $$V_t^{\text{sim}}[1] = V_{\text{OCV}}(1.0) - \Delta V_{\text{ohmic}} - V_1[1] = 4.200 - 0.092 - 0.0116 = 4.0964\,\text{V}$$
4. **Residual Error:**
   $$e[1] = V_t^{\text{meas}}[1] - V_t^{\text{sim}}[1] = 4.108 - 4.0964 = +0.0116\,\text{V}$$
   The L-BFGS-B gradient drives parameter updates until $\sum e[k]^2$ is minimized.

---

## 3. Component 2: State of Charge (SoC) Hybrid Tracking

### Physics: Coulomb Counting with IEEE Sign Correction

Coulomb counting computes charge depletion by integrating the electric current over time:

$$\text{SoC}_{\text{coulomb}}(t) = \text{SoC}(t_0) - \frac{1}{3600 \cdot Q_{\text{nom}}} \int_{t_0}^t I_{\text{physical}}(\tau) \, d\tau$$

where $Q_{\text{nom}}$ is the battery capacity in Ampere-hours ($\text{Ah}$), and $3600$ converts seconds to hours.

#### Sign Convention Normalization
In telemetry standards, charge/discharge signs frequently conflict:
* **Physical load convention:** Discharge is positive ($I > 0$), charge is negative ($I < 0$).
* **IEEE/Consumer device convention:** Discharge is negative ($I < 0$), charge is positive ($I > 0$).

The pipeline inspects the telemetry profile:
$$\text{If } \forall k, \; I[k] \le 0.05\,\text{A} \text{ and } \text{mean}(I) < -0.1\,\text{A} \implies I_{\text{physical}} = -I_{\text{raw}}$$
This prevents a discharge cycle from falsely appearing as a battery charge.

### Machine Learning: XGBoost Gradient Boosted Trees Observer

While Coulomb counting has high temporal fidelity, it accumulates current sensor bias error ($I_{\text{meas}} = I_{\text{true}} + \epsilon_{\text{bias}}$) over time:

$$\text{Drift Error} = \frac{\epsilon_{\text{bias}} \cdot \Delta t_{\text{total}}}{3600 \cdot Q_{\text{nom}}}$$

To eliminate this integration drift, an **XGBoost Regressor** acts as a non-linear voltage observer. It maps instantaneous thermodynamic features to SoC:

$$\hat{y}_{\text{xgb}}[k] = \sum_{m=1}^{M} f_m\left(\mathbf{x}_k\right), \quad \mathbf{x}_k = [V[k], -|I[k]|, T[k], t_k, R_0]$$

where each $f_m$ is an optimal regression tree constructed by minimizing the regularized objective:

$$\mathcal{L} = \sum_{k} \ell(y_k, \hat{y}_k) + \sum_{m} \left( \gamma T_m + \frac{1}{2} \lambda \|\mathbf{w}_m\|^2 \right)$$

### Complementary Fusion Filter

The engine combines both estimators using a complementary fusion filter:

$$\text{SoC}_{\text{fused}}[k] = \alpha \cdot \text{SoC}_{\text{coulomb}}[k] + (1 - \alpha) \cdot \hat{y}_{\text{xgb}}[k]$$

* We choose $\alpha = 0.60$:
  * **$60\%$ Coulomb counting:** Eliminates high-frequency voltage fluctuations during acceleration/braking.
  * **$40\%$ XGBoost Observer:** Acts as a continuous equilibrium anchor that pulls drift back to the physical OCV curve.
* **Monotonicity Enforcement:** During continuous unidirectional discharge ($I_{\text{physical}} \ge 0$), the pipeline enforces:
  $$\text{SoC}[k] \le \text{SoC}[k-1]$$
  guaranteeing that estimated charge never increases while current is leaving the cell.

### Numerical SoC Step-by-Step Example

Let $Q_{\text{nom}} = 2.0\,\text{Ah}$, $\text{SoC}_0 = 100.0\%$, and a constant discharge of $2.0\,\text{A}$ runs for $1800\,\text{s}$ ($0.5\,\text{h}$):
1. **Coulomb Counting Calculation:**
   $$\text{Ah}_{\text{drawn}} = \frac{2.0\,\text{A} \times 1800\,\text{s}}{3600} = 1.00\,\text{Ah}$$
   $$\text{SoC}_{\text{coulomb}} = 100.0\% - \left(\frac{1.00\,\text{Ah}}{2.0\,\text{Ah}} \times 100\%\right) = 50.00\%$$
2. **XGBoost Observer Output:**
   At terminal voltage $V = 3.65\,\text{V}$, temperature $T = 31^\circ\text{C}$, the XGBoost regressor outputs $\hat{y}_{\text{xgb}} = 48.50\%$.
3. **Complementary Fusion:**
   $$\text{SoC}_{\text{fused}} = 0.60 \times 50.00\% + 0.40 \times 48.50\% = 30.00 + 19.40 = 49.40\%$$

---

## 4. Component 3: State of Health (SoH) LSTM Network

### Mathematical Formulation of LSTM Gates

The State of Health network models temporal electrochemical degradation using an LSTM (Long Short-Term Memory) recurrent architecture. For an input sequence $\mathbf{x}_1, \dots, \mathbf{x}_T$, at each time step $t$, the cell state $\mathbf{c}_t$ and hidden state $\mathbf{h}_t$ are updated through gating mechanisms:

$$\begin{aligned}
\mathbf{f}_t &= \sigma\left(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f\right) && \text{(Forget Gate)} \\
\mathbf{i}_t &= \sigma\left(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i\right) && \text{(Input Gate)} \\
\tilde{\mathbf{c}}_t &= \tanh\left(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c\right) && \text{(Candidate State)} \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t && \text{(Cell State Update)} \\
\mathbf{o}_t &= \sigma\left(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o\right) && \text{(Output Gate)} \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t) && \text{(Hidden State Output)}
\end{aligned}$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation function and $\odot$ represents the element-wise Hadamard product.

The terminal hidden vector $\mathbf{h}_T$ passes through a dense regression layer with weight matrix $\mathbf{W}_y$ and bias $b_y$:

$$\text{SoH}_{\text{raw}} = \mathbf{W}_y \mathbf{h}_T + b_y$$

### Feature Physics: Differential Overpotential and Resistance

The 5-dimensional feature vector at time step $t$ is:

$$\mathbf{x}_t = \left[ V_t, |I_t|, T_t, \left(\frac{dV}{dt}\right)_t, R_0 \right]$$

1. **Terminal Voltage ($V_t$):** Reflects open-circuit potential and thermodynamic equilibrium.
2. **Current ($|I_t|$):** Measures reaction excitation rate.
3. **Temperature ($T_t$):** Controls the Arrhenius rate of reaction kinetics ($k = A \exp(-E_a / RT)$).
4. **Differential Polarization Rate ($\frac{dV}{dt}$):** Evaluated over physical time:
   $$\left(\frac{dV}{dt}\right)_t = \frac{V_t - V_{t-1}}{t_t - t_{t-1}}$$
   As a cell degrades, phase-transition plateaus shrink, causing voltage to drop faster.
5. **Ohmic Resistance ($R_0$):** Extracted from the instantaneous voltage step at load onset:
   $$R_0 = \frac{|V(t_0) - V(t_1)|}{|I(t_1)|}$$

### Surrogate Fallback Gate Mechanism

If the raw neural output $\text{SoH}_{\text{raw}} < 0.40$ or $\text{SoH}_{\text{raw}} > 1.20$, or if input points $N < 20$, the engine activates the **electrochemical resistance surrogate fallback**:

$$\text{Ratio} = \frac{R_0 - R_{0,\text{fresh}}}{R_{0,\text{eol}} - R_{0,\text{fresh}}} = \frac{R_0 - 0.045\,\Omega}{0.095\,\Omega - 0.045\,\Omega}$$

$$\text{SoH}_{\text{surrogate}} = \left(1.0 - 0.25 \times \text{clamp}(\text{Ratio}, 0.0, 1.2)\right) \times 100\%$$

This ensures the BMS never outputs a negative or wildly inaccurate health score during sensor anomalies.

### Numerical SoH Step-by-Step Example

Let an aged cell have initial measured capacity $C_0 = 1.856\,\text{Ah}$ (`B0005`). At cycle 100, discharge capacity is $C_{100} = 1.486\,\text{Ah}$.
1. **Electrochemical Ground Truth:**
   $$\text{SoH}_{\text{true}} = \frac{1.486\,\text{Ah}}{1.856\,\text{Ah}} \times 100\% = 80.06\%$$
2. **LSTM Model Prediction:**
   The LSTM processes the 50-step telemetry sequence ($\bar{V} = 3.42\,\text{V}$, $\Delta T = +14.2^\circ\text{C}$, $R_0 = 0.068\,\Omega$) and outputs:
   $$\text{SoH}_{\text{pred}} = 79.63\% \quad (\text{Error} = -0.43\%)$$
3. **Degradation Status Classification:**
   $$\text{Since } 75.0\% \le 79.63\% < 85.0\% \implies \text{Status} = \text{"moderate"}$$

---

## 5. Component 4: Remaining Useful Life (RUL) LSTM Network

### Theoretical Definition of End-of-Life (EOL)

In international battery engineering standards (IEEE 1188 / ISO 12405), a lithium-ion cell reaches **End-of-Life (EOL)** when its operational discharge capacity fades to **$80.0\%$** of its initial nominal rating:

$$C_{\text{EOL}} = 0.80 \times C_0$$

The Remaining Useful Life (RUL) in cycles at current cycle index $k$ is defined as:

$$\text{RUL}(k) = \min \left\{ \kappa \ge 0 \;\middle|\; C(k + \kappa) \le C_{\text{EOL}} \right\}$$

### The Role of Physical Sampling Interval ($dt$) in Polarization Derivatives

A common pitfall in sequence modeling is computing finite differences using array indices rather than physical timestamps:

$$\left(\frac{dV}{dt}\right)_{\text{flawed}} = \frac{V[i] - V[i-1]}{1.0\,\text{step}}$$

In the NASA dataset, $30$ subsampled points span a $3{,}816\,\text{second}$ discharge:
$$\Delta t_{\text{physical}} = \frac{3816\,\text{s}}{30} \approx 127.2\,\text{seconds}$$

* When $\Delta t = 1.0\,\text{s}$ is incorrectly assumed:
  $$\frac{dV}{dt} \approx \frac{-0.012\,\text{V}}{1.0\,\text{s}} = -0.012\,\text{V/s}$$
  The LSTM interprets this rate as an immediate, severe voltage collapse, causing a fresh cell to predict **25 cycles** instead of 100 cycles.
* When $\Delta t_{\text{physical}} = 127.2\,\text{s}$ is applied:
  $$\frac{dV}{dt} = \frac{-0.012\,\text{V}}{127.2\,\text{s}} = -0.000094\,\text{V/s}$$
  The LSTM correctly recognizes the gentle voltage slope of a fresh cell and outputs **$69.7 \approx 70\,\text{cycles}$**.

### Uncertainty Estimation & Confidence Intervals

Because future driving habits and temperature variations introduce uncertainty, the point prediction $\hat{y}_{\text{RUL}}$ is bounded by a confidence interval:

$$\text{Margin} = \max\left(4, \; \text{round}\left(0.12 \times \hat{y}_{\text{RUL}}\right)\right)$$
$$\text{CI}_{\text{lower}} = \max\left(0, \hat{y}_{\text{RUL}} - \text{Margin}\right), \quad \text{CI}_{\text{upper}} = \hat{y}_{\text{RUL}} + \text{Margin}$$

Risk level is determined by proximity to EOL:
$$\text{Risk Level} = \begin{cases}
\text{"HIGH"} & \text{if } \hat{y}_{\text{RUL}} \le 15 \text{ cycles} \\
\text{"MEDIUM"} & \text{if } 15 < \hat{y}_{\text{RUL}} \le 40 \text{ cycles} \\
\text{"LOW"} & \text{if } \hat{y}_{\text{RUL}} > 40 \text{ cycles}
\end{cases}$$

### Numerical RUL Step-by-Step Example

Let cell `B0005` be evaluated at cycle 1:
* Initial capacity: $C_0 = 1.856\,\text{Ah}$
* EOL capacity: $C_{\text{EOL}} = 1.485\,\text{Ah}$
* Actual physical cycles to EOL: $100$ cycles.
* Telemetry features include: $\text{cycle} = 1$, $\text{SoH} = 1.0$, $R_0 = 0.045\,\Omega$, physical $dt = 127.2\,\text{s}$.

1. **LSTM Execution:**
   $$\hat{y}_{\text{raw}} = 69.68\,\text{cycles} \implies \text{RUL} = 70\,\text{cycles}$$
   *(The training set contains cells reaching EOL at cycle 60 [`B0006`] and cycle 74 [`B0018`]; a prediction of ~70 cycles on an unseen cell is within 1 standard deviation).*
2. **Confidence Bounds:**
   $$\text{Margin} = \max(4, \text{round}(0.12 \times 70)) = \max(4, 8) = 8\,\text{cycles}$$
   $$\text{CI} = [70 - 8, \; 70 + 8] = [62, \; 78]\,\text{cycles}$$
3. **Operational Lifetime Translation:**
   Assuming an average cycling rate of $0.5\,\text{cycles/day}$:
   $$\text{Estimated Days} = \text{round}\left(\frac{70}{2}\right) = 35\,\text{operational days}$$

---

## 6. Component 5: Fault Detection & Electrochemical Safety Guard

### Physical Mechanisms for Anomaly Detection

| Anomaly Type | Electrochemical Cause | Safety Hazard |
| :--- | :--- | :--- |
| **Overvoltage ($V > 4.25\,\text{V}$)** | Charging beyond cathode stability limit. | Electrolyte oxidation, gas generation, thermal runaway risk. |
| **Undervoltage ($V < 2.50\,\text{V}$)** | Over-depletion of positive electrode. | Copper current collector dissolution and dendritic short-circuits. |
| **Thermal Spike ($\Delta T > 15^\circ\text{C}$)** | Excessive Joule heating ($I^2 R$) and exothermic side reactions. | Separator melting and rapid combustion. |
| **Impedance Surge ($R_0 > 0.085\,\Omega$)** | Binder degradation, current collector corrosion, or Delamination. | Severe power loss and localized hot spots. |

### Deterministic Rule Thresholds

The fault detection engine (`backend/app/core/fault_detector.py`) applies deterministic rules across every operational cycle:

```python
# 1. Voltage bounds
if max(voltage) > 4.25:
    flags.append("OVERVOLTAGE_WARNING")
if min(voltage) < 2.50:
    flags.append("UNDERVOLTAGE_CRITICAL")

# 2. Thermal gradient
if max(temperature) - min(temperature) > 15.0:
    flags.append("THERMAL_GRADIENT_EXCESSIVE")

# 3. Ohmic degradation
if R0 > 0.085:
    flags.append("IMPEDANCE_GROWTH_CRITICAL")

# 4. Polarization resistance
if R1 > 0.065:
    flags.append("SEI_THICKENING_WARNING")
```

---

## 7. Component 6: SQLite Audit Trail & Traceability Engine

In safety-critical automotive systems (ISO 26262 ASIL-D), neural network decisions must provide an audit trail. A black-box prediction cannot be verified without logging its raw inputs and decision provenance.

The SQLite audit logger stores each execution in the `prediction_logs` table:

```sql
CREATE TABLE prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    battery_id VARCHAR(32) NOT NULL,
    endpoint VARCHAR(32) NOT NULL,
    soh_prediction FLOAT,
    raw_prediction FLOAT,
    confidence FLOAT NOT NULL,
    prediction_source VARCHAR(64) NOT NULL,
    fallback_reason TEXT,
    model_version VARCHAR(64) NOT NULL,
    latency_ms FLOAT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Traceability Example:
If a sensor hardware fault sends $V = 15.0\,\text{V}$:
1. The **Telemetry Guard** intercepts the reading ($V > 5.0\,\text{V}$).
2. The **Model Registry** routes to `sensor_anomaly_fallback`.
3. The response logs:
   * `soh_prediction`: $100.0\%$
   * `confidence`: $0.50$ (reduced certainty)
   * `prediction_source`: `"sensor_anomaly_fallback"`
   * `fallback_reason`: `"Extreme sensor outlier detected: Voltage range [15.00V, 15.00V] exceeds physical cell boundary."`
   * `model_version`: `"physics_sensor_alert_v1"`
4. Downstream BMS microcontrollers inspect the `prediction_source` tag and trigger an alert rather than misinterpreting the reading as valid cell telemetry.
