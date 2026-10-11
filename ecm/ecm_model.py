"""
Equivalent Circuit Model (ECM) for Battery Analysis

This module implements various ECM architectures for battery modeling:
1. Rint Model (Simple resistance model)
2. RC Model (First-order RC model / Thevenin model)
3. 2RC Model (Second-order RC model / PNGV model)

Author: Battery Analysis Project
Date: May 2026
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


class RintModel:
    """
    Rint Model: Simplest ECM with only internal resistance
    
    Circuit: OCV ---[R0]--- Terminal
    
    Equation: V_terminal = OCV - I * R0
    
    Parameters:
        - OCV: Open Circuit Voltage (function of SoC)
        - R0: Internal resistance
    """
    
    def __init__(self):
        self.R0 = None
        self.ocv_params = None
        
    def fit(self, voltage: np.ndarray, current: np.ndarray, soc: np.ndarray):
        """
        Fit Rint model parameters from experimental data
        
        Args:
            voltage: Terminal voltage measurements (V)
            current: Current measurements (A), positive for discharge
            soc: State of Charge (0-1)
        """
        # Fit OCV-SoC relationship (polynomial)
        self.ocv_params = np.polyfit(soc, voltage + current * 0.1, 3)  # Initial guess
        
        # Estimate R0 from voltage drop
        ocv_estimated = np.polyval(self.ocv_params, soc)
        self.R0 = np.mean((ocv_estimated - voltage) / (current + 1e-6))
        
        # Refine R0 using optimization
        def objective(R0):
            v_pred = ocv_estimated - current * R0
            return np.sum((voltage - v_pred) ** 2)
        
        result = minimize(objective, self.R0, method='Nelder-Mead')
        self.R0 = result.x[0]
        
        return self
    
    def predict(self, current: np.ndarray, soc: np.ndarray) -> np.ndarray:
        """
        Predict terminal voltage given current and SoC
        
        Args:
            current: Current profile (A)
            soc: State of Charge (0-1)
            
        Returns:
            Terminal voltage (V)
        """
        ocv = np.polyval(self.ocv_params, soc)
        v_terminal = ocv - current * self.R0
        return v_terminal
    
    def get_parameters(self) -> Dict:
        """Return model parameters"""
        return {
            'R0': self.R0,
            'OCV_params': self.ocv_params
        }


class RCModel:
    """
    RC Model (Thevenin Model): First-order RC circuit
    
    Circuit: OCV ---[R0]---[R1-C1]--- Terminal
    
    Equations:
        V_terminal = OCV - I*R0 - V1
        dV1/dt = -V1/(R1*C1) + I/C1
    
    Parameters:
        - OCV: Open Circuit Voltage (function of SoC)
        - R0: Ohmic resistance
        - R1: Polarization resistance
        - C1: Polarization capacitance
    """
    
    def __init__(self):
        self.R0 = None
        self.R1 = None
        self.C1 = None
        self.tau1 = None  # Time constant R1*C1
        self.ocv_params = None
        
    def fit(self, voltage: np.ndarray, current: np.ndarray, 
            soc: np.ndarray, time: np.ndarray):
        """
        Fit RC model parameters using pulse discharge data
        
        Args:
            voltage: Terminal voltage (V)
            current: Current (A)
            soc: State of Charge (0-1)
            time: Time vector (s)
        """
        # Fit OCV-SoC relationship
        # Use rest periods (low current) for OCV estimation
        rest_mask = np.abs(current) < 0.1
        if np.sum(rest_mask) > 10:
            self.ocv_params = np.polyfit(soc[rest_mask], voltage[rest_mask], 3)
        else:
            self.ocv_params = np.polyfit(soc, voltage, 3)
        
        # Parameter identification using optimization
        def model_error(params):
            R0, R1, tau1 = params
            if R0 < 0 or R1 < 0 or tau1 < 0:
                return 1e10
            
            C1 = tau1 / (R1 + 1e-10)
            v_pred = self._simulate(current, soc, time, R0, R1, C1)
            return np.sum((voltage - v_pred) ** 2)
        
        # Initial guess
        x0 = [0.05, 0.02, 10.0]  # R0, R1, tau1
        
        result = minimize(model_error, x0, method='Nelder-Mead',
                         options={'maxiter': 1000})
        
        self.R0, self.R1, self.tau1 = result.x
        self.C1 = self.tau1 / (self.R1 + 1e-10)
        
        return self
    
    def _simulate(self, current: np.ndarray, soc: np.ndarray, 
                  time: np.ndarray, R0: float, R1: float, C1: float) -> np.ndarray:
        """Simulate voltage response"""
        ocv = np.polyval(self.ocv_params, soc)
        dt = np.diff(time, prepend=time[0])
        
        V1 = np.zeros_like(current)
        for i in range(1, len(current)):
            # Euler integration: dV1/dt = -V1/(R1*C1) + I/C1
            dV1 = (-V1[i-1]/(R1*C1 + 1e-10) + current[i-1]/C1) * dt[i]
            V1[i] = V1[i-1] + dV1
        
        v_terminal = ocv - current * R0 - V1
        return v_terminal
    
    def predict(self, current: np.ndarray, soc: np.ndarray, 
                time: np.ndarray) -> np.ndarray:
        """
        Predict terminal voltage
        
        Args:
            current: Current profile (A)
            soc: State of Charge (0-1)
            time: Time vector (s)
            
        Returns:
            Terminal voltage (V)
        """
        return self._simulate(current, soc, time, self.R0, self.R1, self.C1)
    
    def get_parameters(self) -> Dict:
        """Return model parameters"""
        return {
            'R0': self.R0,
            'R1': self.R1,
            'C1': self.C1,
            'tau1': self.tau1,
            'OCV_params': self.ocv_params
        }


class TwoRCModel:
    """
    2RC Model (PNGV Model): Second-order RC circuit
    
    Circuit: OCV ---[R0]---[R1-C1]---[R2-C2]--- Terminal
    
    Equations:
        V_terminal = OCV - I*R0 - V1 - V2
        dV1/dt = -V1/(R1*C1) + I/C1
        dV2/dt = -V2/(R2*C2) + I/C2
    
    Parameters:
        - OCV: Open Circuit Voltage
        - R0: Ohmic resistance
        - R1, C1: Fast polarization (charge transfer)
        - R2, C2: Slow polarization (diffusion)
    """
    
    def __init__(self):
        self.R0 = None
        self.R1 = None
        self.C1 = None
        self.R2 = None
        self.C2 = None
        self.tau1 = None  # Fast time constant
        self.tau2 = None  # Slow time constant
        self.ocv_params = None
        
    def fit(self, voltage: np.ndarray, current: np.ndarray,
            soc: np.ndarray, time: np.ndarray):
        """
        Fit 2RC model parameters
        
        Args:
            voltage: Terminal voltage (V)
            current: Current (A)
            soc: State of Charge (0-1)
            time: Time vector (s)
        """
        # Fit OCV-SoC relationship
        rest_mask = np.abs(current) < 0.1
        if np.sum(rest_mask) > 10:
            self.ocv_params = np.polyfit(soc[rest_mask], voltage[rest_mask], 3)
        else:
            self.ocv_params = np.polyfit(soc, voltage, 3)
        
        # Parameter identification
        def model_error(params):
            R0, R1, tau1, R2, tau2 = params
            if any(p < 0 for p in params):
                return 1e10
            
            C1 = tau1 / (R1 + 1e-10)
            C2 = tau2 / (R2 + 1e-10)
            v_pred = self._simulate(current, soc, time, R0, R1, C1, R2, C2)
            return np.sum((voltage - v_pred) ** 2)
        
        # Initial guess: R0, R1, tau1, R2, tau2
        x0 = [0.05, 0.02, 10.0, 0.03, 100.0]
        
        result = minimize(model_error, x0, method='Nelder-Mead',
                         options={'maxiter': 2000})
        
        self.R0, self.R1, self.tau1, self.R2, self.tau2 = result.x
        self.C1 = self.tau1 / (self.R1 + 1e-10)
        self.C2 = self.tau2 / (self.R2 + 1e-10)
        
        return self
    
    def _simulate(self, current: np.ndarray, soc: np.ndarray,
                  time: np.ndarray, R0: float, R1: float, C1: float,
                  R2: float, C2: float) -> np.ndarray:
        """Simulate voltage response"""
        ocv = np.polyval(self.ocv_params, soc)
        dt = np.diff(time, prepend=time[0])
        
        V1 = np.zeros_like(current)
        V2 = np.zeros_like(current)
        
        for i in range(1, len(current)):
            # Fast RC dynamics
            dV1 = (-V1[i-1]/(R1*C1 + 1e-10) + current[i-1]/C1) * dt[i]
            V1[i] = V1[i-1] + dV1
            
            # Slow RC dynamics
            dV2 = (-V2[i-1]/(R2*C2 + 1e-10) + current[i-1]/C2) * dt[i]
            V2[i] = V2[i-1] + dV2
        
        v_terminal = ocv - current * R0 - V1 - V2
        return v_terminal
    
    def predict(self, current: np.ndarray, soc: np.ndarray,
                time: np.ndarray) -> np.ndarray:
        """
        Predict terminal voltage
        
        Args:
            current: Current profile (A)
            soc: State of Charge (0-1)
            time: Time vector (s)
            
        Returns:
            Terminal voltage (V)
        """
        return self._simulate(current, soc, time, self.R0, self.R1, 
                            self.C1, self.R2, self.C2)
    
    def get_parameters(self) -> Dict:
        """Return model parameters"""
        return {
            'R0': self.R0,
            'R1': self.R1,
            'C1': self.C1,
            'tau1': self.tau1,
            'R2': self.R2,
            'C2': self.C2,
            'tau2': self.tau2,
            'OCV_params': self.ocv_params
        }


class SoCEstimator:
    """
    State of Charge (SoC) Estimator using Coulomb Counting
    
    SoC(t) = SoC(0) - (1/Q_nom) * integral(I(t) dt)
    
    where Q_nom is the nominal capacity in Ah
    """
    
    def __init__(self, nominal_capacity: float = 2.0):
        """
        Args:
            nominal_capacity: Battery nominal capacity in Ah
        """
        self.Q_nom = nominal_capacity
        
    def estimate(self, current: np.ndarray, time: np.ndarray,
                 soc_initial: float = 1.0) -> np.ndarray:
        """
        Estimate SoC using coulomb counting
        
        Args:
            current: Current measurements (A), positive for discharge
            time: Time vector (s)
            soc_initial: Initial SoC (0-1)
            
        Returns:
            SoC trajectory (0-1)
        """
        dt = np.diff(time, prepend=time[0])
        charge_throughput = np.cumsum(current * dt / 3600)  # Ah
        soc = soc_initial - charge_throughput / self.Q_nom
        return np.clip(soc, 0, 1)


def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculate model performance metrics
    
    Args:
        y_true: Actual voltage measurements
        y_pred: Predicted voltage
        
    Returns:
        Dictionary with RMSE, MAE, MAPE, R²
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }


def plot_ecm_results(time: np.ndarray, voltage_actual: np.ndarray,
                     voltage_pred: np.ndarray, current: np.ndarray,
                     model_name: str = "ECM"):
    """
    Plot ECM model results
    
    Args:
        time: Time vector (s)
        voltage_actual: Measured voltage (V)
        voltage_pred: Predicted voltage (V)
        current: Current profile (A)
        model_name: Name of the model
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Voltage comparison
    axes[0].plot(time, voltage_actual, 'b-', label='Measured', linewidth=1.5)
    axes[0].plot(time, voltage_pred, 'r--', label='Predicted', linewidth=1.5)
    axes[0].set_ylabel('Voltage (V)', fontsize=12)
    axes[0].set_title(f'{model_name} - Voltage Prediction', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Current profile
    axes[1].plot(time, current, 'g-', linewidth=1.5)
    axes[1].set_ylabel('Current (A)', fontsize=12)
    axes[1].set_title('Current Profile', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Error
    error = voltage_actual - voltage_pred
    axes[2].plot(time, error * 1000, 'k-', linewidth=1.5)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Error (mV)', fontsize=12)
    axes[2].set_title('Prediction Error', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage
    print("ECM Models Available:")
    print("1. Rint Model - Simple resistance model")
    print("2. RC Model - First-order RC (Thevenin)")
    print("3. 2RC Model - Second-order RC (PNGV)")
    print("\nUse these models with your battery data!")
