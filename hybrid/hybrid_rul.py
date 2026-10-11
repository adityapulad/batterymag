"""
Hybrid ECM + LSTM Model for Battery RUL (Remaining Useful Life) Prediction

This module combines physics-based ECM parameters with LSTM neural networks
for improved Remaining Useful Life prediction.

Approach:
1. Extract ECM parameters (R0, R1, C1, tau1) from battery data
2. Use ECM parameters as additional features for LSTM
3. Train LSTM on combined features (voltage, current, temp + ECM params)
4. Predict remaining cycles until battery reaches end-of-life (EOL)

EOL Criteria: Capacity drops below 80% of initial capacity (SOH < 80%)

Author: Battery Analysis Project
Date: May 2026
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Import our modules
from battery_loader import load_data
from ecm.ecm_parameter_extraction import ECMParameterExtractor


class HybridECMLSTM_RUL:
    """
    Hybrid model combining ECM parameters with LSTM for RUL prediction
    
    Architecture:
    1. ECM layer: Extract R0, R1, C1, tau1 for each cycle
    2. Feature layer: Combine ECM params with raw measurements + degradation features
    3. Bidirectional LSTM layer: Process time-series with enhanced features
    4. Output layer: Predict Remaining Useful Life (cycles until EOL)
    
    EOL Definition: SOH < 80% (capacity < 80% of initial)
    """
    
    def __init__(self, battery_id='B0005', sequence_length=30, eol_threshold=0.8):
        """
        Args:
            battery_id: Battery identifier
            sequence_length: Number of time steps for LSTM
            eol_threshold: End-of-life threshold (default: 0.8 = 80% SOH)
        """
        self.battery_id = battery_id
        self.sequence_length = sequence_length
        self.eol_threshold = eol_threshold
        self.model = None
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.ecm_params_df = None
        
    def extract_ecm_features(self, cycle_step=5):
        """
        Extract ECM parameters for all cycles
        
        Args:
            cycle_step: Extract parameters every N cycles
            
        Returns:
            DataFrame with ECM parameters per cycle
        """
        print("Step 1: Extracting ECM parameters for RUL...")
        extractor = ECMParameterExtractor(self.battery_id)
        extractor.load_battery_data()
        
        # Extract RC model parameters
        self.ecm_params_df = extractor.extract_parameters_over_lifetime(
            model_type='RC',
            cycle_step=cycle_step
        )
        
        print(f"✓ Extracted ECM parameters for {len(self.ecm_params_df)} cycles")
        return self.ecm_params_df
    
    def prepare_rul_features(self):
        """
        Prepare features for RUL prediction with ECM parameters
        
        Returns:
            X: Feature array (samples, sequence_length, features)
            y: Target array (RUL values in cycles)
        """
        print("\nStep 2: Preparing RUL features with hybrid approach...")
        
        # Load raw battery data
        dataset, capacity_data = load_data(self.battery_id)
        
        # Calculate SOH
        initial_capacity = dataset['capacity'].iloc[0]
        dataset['soh'] = dataset['capacity'] / initial_capacity
        
        # Find EOL cycle (when SOH drops below threshold)
        eol_cycles = dataset[dataset['soh'] < self.eol_threshold]['cycle'].values
        if len(eol_cycles) > 0:
            eol_cycle = eol_cycles[0]
        else:
            eol_cycle = dataset['cycle'].max()
            print(f"⚠️  Battery did not reach EOL threshold. Using max cycle: {eol_cycle}")
        
        print(f"✓ EOL cycle identified: {eol_cycle} (SOH < {self.eol_threshold*100}%)")
        
        # Get cycles that have ECM parameters
        available_cycles = self.ecm_params_df['cycle'].values
        available_cycles = available_cycles[available_cycles < eol_cycle]  # Only use cycles before EOL
        
        X_sequences = []
        y_rul = []
        
        for i, cycle_num in enumerate(available_cycles):
            # Calculate RUL for this cycle
            rul = eol_cycle - cycle_num
            
            if rul <= 0:
                continue
            
            # Get cycle data
            cycle_data = dataset[dataset['cycle'] == cycle_num]
            
            if len(cycle_data) < self.sequence_length:
                continue
            
            # Get ECM parameters for this cycle
            ecm_row = self.ecm_params_df[self.ecm_params_df['cycle'] == cycle_num].iloc[0]
            R0 = ecm_row['R0']
            R1 = ecm_row['R1']
            C1 = ecm_row['C1']
            tau1 = ecm_row['tau1']
            
            # Sample sequence_length points from cycle
            indices = np.linspace(0, len(cycle_data)-1, self.sequence_length, dtype=int)
            sampled_data = cycle_data.iloc[indices]
            
            # Raw features
            raw_features = sampled_data[[
                'voltage_measured',
                'current_measured',
                'temperature_measured',
                'voltage_load',
                'current_load'
            ]].values
            
            # Add degradation features
            current_capacity = sampled_data['capacity'].iloc[-1]
            current_soh = sampled_data['soh'].iloc[-1]
            capacity_fade = initial_capacity - current_capacity
            
            degradation_features = np.array([
                [current_capacity, current_soh, capacity_fade, cycle_num]
            ] * self.sequence_length)
            
            # Add ECM parameters as additional features
            ecm_features = np.array([[R0, R1, C1, tau1]] * self.sequence_length)
            
            # Combine all features: raw + degradation + ECM
            combined_features = np.hstack([raw_features, degradation_features, ecm_features])
            
            X_sequences.append(combined_features)
            y_rul.append(rul)
        
        X = np.array(X_sequences)
        y = np.array(y_rul)
        
        print(f"✓ Created {len(X)} sequences")
        print(f"  - Sequence shape: {X.shape}")
        print(f"  - Features: 5 raw + 4 degradation + 4 ECM = 13 total")
        print(f"  - Target shape: {y.shape}")
        print(f"  - RUL range: {y.min():.0f} to {y.max():.0f} cycles")
        
        return X, y
    
    def build_model(self, input_shape):
        """
        Build hybrid Bidirectional LSTM model for RUL
        
        Args:
            input_shape: (sequence_length, n_features)
        """
        print("\nStep 3: Building hybrid Bidirectional LSTM model for RUL...")
        
        model = Sequential([
            # First Bidirectional LSTM layer
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            
            # Second Bidirectional LSTM layer
            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.3),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            
            # Output layer (RUL prediction)
            Dense(1, activation='relu')  # ReLU to ensure positive RUL
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        print("✓ Model architecture:")
        model.summary()
        
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=200, batch_size=32):
        """
        Train the hybrid RUL model
        
        Args:
            X: Feature array
            y: Target array (RUL values)
            validation_split: Fraction for validation
            epochs: Training epochs
            batch_size: Batch size
        """
        print("\nStep 4: Training hybrid RUL model...")
        
        # Scale features
        n_samples, n_steps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler_features.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_steps, n_features)
        
        # Scale target
        y_scaled = self.scaler_target.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=validation_split, random_state=42
        )
        
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Validation set: {len(X_val)} samples")
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'hybrid/best_hybrid_rul_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=12,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint, reduce_lr],
            verbose=1
        )
        
        print("\n✓ Training complete!")
        
        return history
    
    def evaluate(self, X, y):
        """
        Evaluate RUL model performance
        
        Args:
            X: Feature array
            y: True RUL values
            
        Returns:
            Dictionary with metrics
        """
        # Scale features
        n_samples, n_steps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler_features.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_steps, n_features)
        
        # Predict
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_target.inverse_transform(y_pred_scaled).flatten()
        
        # Ensure non-negative predictions
        y_pred = np.maximum(y_pred, 0)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-10))) * 100
        
        # Additional RUL-specific metrics
        max_error = np.max(np.abs(y - y_pred))
        within_10_cycles = np.mean(np.abs(y - y_pred) <= 10) * 100
        within_20_cycles = np.mean(np.abs(y - y_pred) <= 20) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Max_Error': max_error,
            'Within_10_Cycles': within_10_cycles,
            'Within_20_Cycles': within_20_cycles
        }
        
        print("\n" + "="*60)
        print("RUL Prediction Performance:")
        print("="*60)
        print(f"RMSE:              {rmse:.2f} cycles")
        print(f"MAE:               {mae:.2f} cycles")
        print(f"R²:                {r2:.6f}")
        print(f"MAPE:              {mape:.2f}%")
        print(f"Max Error:         {max_error:.2f} cycles")
        print(f"Within ±10 cycles: {within_10_cycles:.1f}%")
        print(f"Within ±20 cycles: {within_20_cycles:.1f}%")
        print("="*60)
        
        return metrics, y_pred
    
    def plot_rul_results(self, y_true, y_pred, save_path=None):
        """
        Plot RUL prediction results
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Prediction vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=50, c='blue', edgecolors='black')
        axes[0, 0].plot([0, y_true.max()], [0, y_true.max()], 'r--', linewidth=2)
        
        # Add ±10 and ±20 cycle bands
        x_range = np.linspace(0, y_true.max(), 100)
        axes[0, 0].fill_between(x_range, x_range-10, x_range+10, alpha=0.2, color='green', label='±10 cycles')
        axes[0, 0].fill_between(x_range, x_range-20, x_range+20, alpha=0.1, color='yellow', label='±20 cycles')
        
        axes[0, 0].set_xlabel('True RUL (cycles)', fontsize=12)
        axes[0, 0].set_ylabel('Predicted RUL (cycles)', fontsize=12)
        axes[0, 0].set_title('Hybrid ECM-LSTM: RUL Prediction vs Actual', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. RUL degradation over time
        axes[0, 1].plot(y_true, 'b-o', label='True RUL', markersize=6, linewidth=2)
        axes[0, 1].plot(y_pred, 'r--s', label='Predicted RUL', markersize=6, linewidth=2)
        axes[0, 1].set_xlabel('Sample Index', fontsize=12)
        axes[0, 1].set_ylabel('RUL (cycles)', fontsize=12)
        axes[0, 1].set_title('RUL Degradation Over Time', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # 3. Error distribution
        error = y_true - y_pred
        axes[1, 0].hist(error, bins=30, edgecolor='black', alpha=0.7, color='purple')
        axes[1, 0].set_xlabel('Prediction Error (cycles)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('RUL Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].axvline(x=-10, color='g', linestyle=':', linewidth=1.5, alpha=0.7)
        axes[1, 0].axvline(x=10, color='g', linestyle=':', linewidth=1.5, alpha=0.7)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add statistics
        axes[1, 0].text(0.02, 0.98, 
                       f'Mean: {np.mean(error):.2f} cycles\nStd: {np.std(error):.2f} cycles',
                       transform=axes[1, 0].transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Absolute error over time
        abs_error = np.abs(error)
        axes[1, 1].plot(abs_error, 'g-o', markersize=6, linewidth=2)
        axes[1, 1].axhline(y=10, color='orange', linestyle='--', linewidth=2, label='10 cycles')
        axes[1, 1].axhline(y=20, color='red', linestyle='--', linewidth=2, label='20 cycles')
        axes[1, 1].set_xlabel('Sample Index', fontsize=12)
        axes[1, 1].set_ylabel('Absolute Error (cycles)', fontsize=12)
        axes[1, 1].set_title('Absolute RUL Error Over Time', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Figure saved to {save_path}")
        
        return fig
    
    def compare_with_baseline(self, X, y):
        """
        Compare hybrid model with baseline (LSTM without ECM)
        
        Args:
            X: Feature array (with ECM features)
            y: Target array
        """
        print("\n" + "="*60)
        print("Comparing Hybrid vs Baseline LSTM for RUL")
        print("="*60)
        
        # Hybrid model (already trained)
        metrics_hybrid, y_pred_hybrid = self.evaluate(X, y)
        
        # Baseline: LSTM without ECM features (only first 9 features: raw + degradation)
        print("\nTraining baseline LSTM (without ECM features)...")
        X_baseline = X[:, :, :9]  # Only raw + degradation features
        
        # Build baseline model
        baseline_model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_baseline.shape[1], X_baseline.shape[2])),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='relu')
        ])
        
        baseline_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Scale and train
        n_samples, n_steps, n_features = X_baseline.shape
        X_reshaped = X_baseline.reshape(-1, n_features)
        scaler_baseline = MinMaxScaler()
        X_scaled = scaler_baseline.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_steps, n_features)
        
        y_scaled = self.scaler_target.transform(y.reshape(-1, 1)).flatten()
        
        baseline_model.fit(X_scaled, y_scaled, epochs=100, batch_size=32, verbose=0)
        
        # Predict
        y_pred_baseline_scaled = baseline_model.predict(X_scaled)
        y_pred_baseline = self.scaler_target.inverse_transform(y_pred_baseline_scaled).flatten()
        y_pred_baseline = np.maximum(y_pred_baseline, 0)
        
        # Metrics
        rmse_baseline = np.sqrt(mean_squared_error(y, y_pred_baseline))
        mae_baseline = mean_absolute_error(y, y_pred_baseline)
        r2_baseline = r2_score(y, y_pred_baseline)
        
        # Comparison table
        print("\n" + "="*60)
        print(f"{'Model':<20} {'RMSE':<15} {'MAE':<15} {'R²':<12}")
        print("-"*60)
        print(f"{'Baseline LSTM':<20} {rmse_baseline:<15.2f} {mae_baseline:<15.2f} {r2_baseline:<12.6f}")
        print(f"{'Hybrid ECM-LSTM':<20} {metrics_hybrid['RMSE']:<15.2f} {metrics_hybrid['MAE']:<15.2f} {metrics_hybrid['R2']:<12.6f}")
        print("-"*60)
        
        improvement_rmse = ((rmse_baseline - metrics_hybrid['RMSE']) / rmse_baseline) * 100
        improvement_mae = ((mae_baseline - metrics_hybrid['MAE']) / mae_baseline) * 100
        improvement_r2 = ((metrics_hybrid['R2'] - r2_baseline) / (r2_baseline + 1e-10)) * 100
        
        print(f"{'Improvement':<20} {improvement_rmse:+.2f}%         {improvement_mae:+.2f}%         {improvement_r2:+.2f}%")
        print("="*60)
        
        return {
            'baseline': {'RMSE': rmse_baseline, 'MAE': mae_baseline, 'R2': r2_baseline},
            'hybrid': metrics_hybrid,
            'improvement': {'RMSE': improvement_rmse, 'MAE': improvement_mae, 'R2': improvement_r2}
        }


def run_hybrid_rul_analysis(battery_id='B0005'):
    """
    Complete hybrid ECM-LSTM RUL analysis workflow
    
    Args:
        battery_id: Battery to analyze
    """
    print("="*60)
    print(f"Hybrid ECM-LSTM RUL Analysis - Battery {battery_id}")
    print("="*60)
    
    # Initialize hybrid model
    hybrid = HybridECMLSTM_RUL(battery_id=battery_id, sequence_length=30, eol_threshold=0.8)
    
    # Extract ECM features
    ecm_params = hybrid.extract_ecm_features(cycle_step=5)
    
    # Prepare RUL features
    X, y = hybrid.prepare_rul_features()
    
    # Build model
    hybrid.build_model(input_shape=(X.shape[1], X.shape[2]))
    
    # Train model
    history = hybrid.train(X, y, epochs=200, batch_size=32)
    
    # Evaluate
    metrics, y_pred = hybrid.evaluate(X, y)
    
    # Plot results
    hybrid.plot_rul_results(y, y_pred, save_path=f'hybrid/results/{battery_id}_hybrid_rul_results.png')
    
    # Compare with baseline
    comparison = hybrid.compare_with_baseline(X, y)
    
    print("\n✅ Hybrid RUL analysis complete!")
    print(f"Results saved to: hybrid/results/{battery_id}_hybrid_rul_results.png")
    
    return hybrid, metrics, comparison


if __name__ == "__main__":
    # Run hybrid RUL analysis
    hybrid, metrics, comparison = run_hybrid_rul_analysis('B0005')
