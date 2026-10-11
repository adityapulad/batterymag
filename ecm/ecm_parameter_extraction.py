"""
ECM Parameter Extraction from NASA Battery Dataset

This module extracts ECM parameters from the NASA battery dataset
and tracks parameter degradation over battery lifetime.

Author: Battery Analysis Project
Date: May 2026
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from battery_loader import load_data
from ecm.ecm_model import RintModel, RCModel, TwoRCModel, SoCEstimator, calculate_model_metrics


class ECMParameterExtractor:
    """
    Extract ECM parameters from battery cycling data
    """
    
    def __init__(self, battery_id: str = 'B0005'):
        """
        Args:
            battery_id: Battery identifier (e.g., 'B0005')
        """
        self.battery_id = battery_id
        self.dataset = None
        self.capacity_data = None
        self.parameters_history = []
        
    def load_battery_data(self):
        """Load battery data using battery_loader"""
        print(f"Loading battery {self.battery_id}...")
        self.dataset, self.capacity_data = load_data(self.battery_id)
        print(f"Loaded {len(self.capacity_data)} discharge cycles")
        return self
    
    def extract_cycle_data(self, cycle_num: int):
        """
        Extract data for a specific cycle
        
        Args:
            cycle_num: Cycle number (1-indexed)
            
        Returns:
            Dictionary with time, voltage, current, capacity
        """
        cycle_data = self.dataset[self.dataset['cycle'] == cycle_num].copy()
        
        if len(cycle_data) == 0:
            return None
        
        # Reset time to start from 0
        time = cycle_data['time'].values
        time = time - time[0]
        
        return {
            'time': time,
            'voltage': cycle_data['voltage_measured'].values,
            'current': cycle_data['current_measured'].values,
            'temperature': cycle_data['temperature_measured'].values,
            'capacity': cycle_data['capacity'].iloc[0]
        }
    
    def calculate_soc(self, cycle_data: dict, nominal_capacity: float = 2.0):
        """
        Calculate SoC for a discharge cycle
        
        Args:
            cycle_data: Dictionary with cycle data
            nominal_capacity: Nominal capacity in Ah
            
        Returns:
            SoC array (0-1)
        """
        estimator = SoCEstimator(nominal_capacity=nominal_capacity)
        soc = estimator.estimate(
            current=cycle_data['current'],
            time=cycle_data['time'],
            soc_initial=1.0
        )
        return soc
    
    def fit_rint_model(self, cycle_num: int):
        """
        Fit Rint model to a specific cycle
        
        Args:
            cycle_num: Cycle number
            
        Returns:
            Fitted RintModel and metrics
        """
        cycle_data = self.extract_cycle_data(cycle_num)
        if cycle_data is None:
            return None, None
        
        soc = self.calculate_soc(cycle_data)
        
        # Fit model
        model = RintModel()
        model.fit(
            voltage=cycle_data['voltage'],
            current=cycle_data['current'],
            soc=soc
        )
        
        # Predict and evaluate
        v_pred = model.predict(cycle_data['current'], soc)
        metrics = calculate_model_metrics(cycle_data['voltage'], v_pred)
        
        return model, metrics
    
    def fit_rc_model(self, cycle_num: int):
        """
        Fit RC model to a specific cycle
        
        Args:
            cycle_num: Cycle number
            
        Returns:
            Fitted RCModel and metrics
        """
        cycle_data = self.extract_cycle_data(cycle_num)
        if cycle_data is None:
            return None, None
        
        soc = self.calculate_soc(cycle_data)
        
        # Fit model
        model = RCModel()
        model.fit(
            voltage=cycle_data['voltage'],
            current=cycle_data['current'],
            soc=soc,
            time=cycle_data['time']
        )
        
        # Predict and evaluate
        v_pred = model.predict(cycle_data['current'], soc, cycle_data['time'])
        metrics = calculate_model_metrics(cycle_data['voltage'], v_pred)
        
        return model, metrics
    
    def fit_2rc_model(self, cycle_num: int):
        """
        Fit 2RC model to a specific cycle
        
        Args:
            cycle_num: Cycle number
            
        Returns:
            Fitted TwoRCModel and metrics
        """
        cycle_data = self.extract_cycle_data(cycle_num)
        if cycle_data is None:
            return None, None
        
        soc = self.calculate_soc(cycle_data)
        
        # Fit model
        model = TwoRCModel()
        model.fit(
            voltage=cycle_data['voltage'],
            current=cycle_data['current'],
            soc=soc,
            time=cycle_data['time']
        )
        
        # Predict and evaluate
        v_pred = model.predict(cycle_data['current'], soc, cycle_data['time'])
        metrics = calculate_model_metrics(cycle_data['voltage'], v_pred)
        
        return model, metrics
    
    def extract_parameters_over_lifetime(self, model_type: str = 'RC',
                                        cycle_step: int = 10):
        """
        Extract ECM parameters over battery lifetime
        
        Args:
            model_type: 'Rint', 'RC', or '2RC'
            cycle_step: Extract parameters every N cycles
            
        Returns:
            DataFrame with parameters vs cycle
        """
        print(f"\nExtracting {model_type} parameters over lifetime...")
        print(f"Analyzing every {cycle_step} cycles...")
        
        max_cycle = self.capacity_data['cycle'].max()
        cycles_to_analyze = range(1, max_cycle + 1, cycle_step)
        
        results = []
        
        for cycle_num in cycles_to_analyze:
            print(f"Processing cycle {cycle_num}/{max_cycle}...", end='\r')
            
            if model_type == 'Rint':
                model, metrics = self.fit_rint_model(cycle_num)
            elif model_type == 'RC':
                model, metrics = self.fit_rc_model(cycle_num)
            elif model_type == '2RC':
                model, metrics = self.fit_2rc_model(cycle_num)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            if model is None:
                continue
            
            params = model.get_parameters()
            capacity = self.capacity_data[
                self.capacity_data['cycle'] == cycle_num
            ]['capacity'].iloc[0]
            
            result = {
                'cycle': cycle_num,
                'capacity': capacity,
                **params,
                **metrics
            }
            results.append(result)
        
        print(f"\nCompleted! Analyzed {len(results)} cycles.")
        
        self.parameters_history = pd.DataFrame(results)
        return self.parameters_history
    
    def plot_parameter_degradation(self, save_path: str = None):
        """
        Plot ECM parameter degradation over cycles
        
        Args:
            save_path: Path to save figure (optional)
        """
        if len(self.parameters_history) == 0:
            print("No parameter history available. Run extract_parameters_over_lifetime first.")
            return
        
        df = self.parameters_history
        
        # Determine which parameters to plot
        param_cols = [col for col in df.columns 
                     if col not in ['cycle', 'capacity', 'RMSE', 'MAE', 'MAPE', 'R2', 'OCV_params']]
        
        n_params = len(param_cols)
        fig, axes = plt.subplots(n_params + 2, 1, figsize=(12, 4 * (n_params + 2)))
        
        # Plot capacity degradation
        axes[0].plot(df['cycle'], df['capacity'], 'b-o', linewidth=2, markersize=4)
        axes[0].set_ylabel('Capacity (Ah)', fontsize=12)
        axes[0].set_title(f'Battery {self.battery_id} - Capacity Degradation', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Plot each parameter
        for idx, param in enumerate(param_cols, start=1):
            axes[idx].plot(df['cycle'], df[param], 'r-o', linewidth=2, markersize=4)
            axes[idx].set_ylabel(param, fontsize=12)
            axes[idx].set_title(f'{param} vs Cycle', fontsize=14)
            axes[idx].grid(True, alpha=0.3)
        
        # Plot model accuracy (RMSE)
        axes[-1].plot(df['cycle'], df['RMSE'] * 1000, 'g-o', linewidth=2, markersize=4)
        axes[-1].set_xlabel('Cycle Number', fontsize=12)
        axes[-1].set_ylabel('RMSE (mV)', fontsize=12)
        axes[-1].set_title('Model Accuracy', fontsize=14)
        axes[-1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def save_parameters(self, filename: str = None):
        """
        Save extracted parameters to CSV
        
        Args:
            filename: Output filename (default: battery_id_ecm_params.csv)
        """
        if len(self.parameters_history) == 0:
            print("No parameters to save.")
            return
        
        if filename is None:
            filename = f'ecm/{self.battery_id}_ecm_params.csv'
        
        # Drop OCV_params column (numpy array, not CSV-friendly)
        df_to_save = self.parameters_history.drop(columns=['OCV_params'], errors='ignore')
        df_to_save.to_csv(filename, index=False)
        print(f"Parameters saved to {filename}")


def compare_ecm_models(battery_id: str = 'B0005', cycle_num: int = 50):
    """
    Compare different ECM models on the same cycle
    
    Args:
        battery_id: Battery identifier
        cycle_num: Cycle number to analyze
    """
    print(f"\n{'='*60}")
    print(f"Comparing ECM Models - Battery {battery_id}, Cycle {cycle_num}")
    print(f"{'='*60}\n")
    
    extractor = ECMParameterExtractor(battery_id)
    extractor.load_battery_data()
    
    # Fit all models
    print("Fitting Rint Model...")
    rint_model, rint_metrics = extractor.fit_rint_model(cycle_num)
    
    print("Fitting RC Model...")
    rc_model, rc_metrics = extractor.fit_rc_model(cycle_num)
    
    print("Fitting 2RC Model...")
    tworc_model, tworc_metrics = extractor.fit_2rc_model(cycle_num)
    
    # Print comparison
    print(f"\n{'Model':<15} {'RMSE (mV)':<12} {'MAE (mV)':<12} {'MAPE (%)':<12} {'R²':<12}")
    print("-" * 60)
    
    if rint_metrics:
        print(f"{'Rint':<15} {rint_metrics['RMSE']*1000:<12.4f} "
              f"{rint_metrics['MAE']*1000:<12.4f} {rint_metrics['MAPE']:<12.4f} "
              f"{rint_metrics['R2']:<12.4f}")
    
    if rc_metrics:
        print(f"{'RC':<15} {rc_metrics['RMSE']*1000:<12.4f} "
              f"{rc_metrics['MAE']*1000:<12.4f} {rc_metrics['MAPE']:<12.4f} "
              f"{rc_metrics['R2']:<12.4f}")
    
    if tworc_metrics:
        print(f"{'2RC':<15} {tworc_metrics['RMSE']*1000:<12.4f} "
              f"{tworc_metrics['MAE']*1000:<12.4f} {tworc_metrics['MAPE']:<12.4f} "
              f"{tworc_metrics['R2']:<12.4f}")
    
    print("\n" + "="*60)
    
    # Print parameters
    print("\nModel Parameters:")
    print("-" * 60)
    
    if rint_model:
        params = rint_model.get_parameters()
        print(f"\nRint Model:")
        print(f"  R0 = {params['R0']:.6f} Ω")
    
    if rc_model:
        params = rc_model.get_parameters()
        print(f"\nRC Model:")
        print(f"  R0 = {params['R0']:.6f} Ω")
        print(f"  R1 = {params['R1']:.6f} Ω")
        print(f"  C1 = {params['C1']:.2f} F")
        print(f"  τ1 = {params['tau1']:.2f} s")
    
    if tworc_model:
        params = tworc_model.get_parameters()
        print(f"\n2RC Model:")
        print(f"  R0 = {params['R0']:.6f} Ω")
        print(f"  R1 = {params['R1']:.6f} Ω")
        print(f"  C1 = {params['C1']:.2f} F")
        print(f"  τ1 = {params['tau1']:.2f} s (fast)")
        print(f"  R2 = {params['R2']:.6f} Ω")
        print(f"  C2 = {params['C2']:.2f} F")
        print(f"  τ2 = {params['tau2']:.2f} s (slow)")
    
    return extractor, rint_model, rc_model, tworc_model


if __name__ == "__main__":
    # Example: Compare models
    compare_ecm_models('B0005', cycle_num=50)
    
    # Example: Extract parameters over lifetime
    extractor = ECMParameterExtractor('B0005')
    extractor.load_battery_data()
    extractor.extract_parameters_over_lifetime(model_type='RC', cycle_step=10)
    extractor.plot_parameter_degradation(save_path='ecm/B0005_parameter_degradation.png')
    extractor.save_parameters()
