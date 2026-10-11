"""
Hybrid ECM + LSTM Model for Battery SoH Estimation

This module combines physics-based ECM parameters with LSTM neural networks
for improved State of Health estimation.

Approach:
1. Extract ECM parameters (R0, R1, C1) from battery data
2. Use ECM parameters as additional features for LSTM
3. Train LSTM on combined features (voltage, current, temp + ECM params)
4. Achieve better accuracy and interpretability

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
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import our modules
from battery_loader import load_data
from ecm.ecm_parameter_extraction import ECMParameterExtractor


class HybridECMLSTM:
    """
    Hybrid model combining ECM parameters with LSTM for SoH estimation
    
    Architecture:
    1. ECM layer: Extract R0, R1, C1, tau1 for each cycle
    2. Feature layer: Combine ECM params with raw measurements
    3. LSTM layer: Process time-series with enhanced features
    4. Output layer: Predict State of Health
    """
    
    def __init__(self, battery_id='B0005', sequence_length=50):
        """
        Args:
            battery_id: Battery identifier
            sequence_length: Number of time steps for LSTM
        """
        self.battery_id = battery_id
        self.sequence_length = sequence_length
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
        print("Step 1: Extracting ECM parameters...")
        extractor = ECMParameterExtractor(self.battery_id)
        extractor.load_battery_data()
        
        # Extract RC model parameters
        self.ecm_params_df = extractor.extract_parameters_over_lifetime(
            model_type='RC',
            cycle_step=cycle_step
        )
        
        print(f"✓ Extracted ECM parameters for {len(self.ecm_params_df)} cycles")
        return self.ecm_params_df
    
    def prepare_hybrid_features(self):
        """
        Combine raw measurements with ECM parameters
        
        Returns:
            X: Feature array (samples, sequence_length, features)
            y: Target array (SoH values)
        """
        print("\nStep 2: Preparing hybrid features...")
        
        # Load raw battery data
        dataset, capacity_data = load_data(self.battery_id)
        
        # Calculate SoH
        initial_capacity = dataset['capacity'].iloc[0]
        dataset['soh'] = dataset['capacity'] / initial_capacity
        
        # Get cycles that have ECM parameters
        available_cycles = self.ecm_params_df['cycle'].values
        
        X_sequences = []
        y_soh = []
        
        for cycle_num in available_cycles:
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
            
            # Add ECM parameters as additional features (repeated for each time step)
            ecm_features = np.array([[R0, R1, C1, tau1]] * self.sequence_length)
            
            # Combine raw + ECM features
            combined_features = np.hstack([raw_features, ecm_features])
            
            X_sequences.append(combined_features)
            y_soh.append(sampled_data['soh'].iloc[-1])  # SoH at end of cycle
        
        X = np.array(X_sequences)
        y = np.array(y_soh)
        
        print(f"✓ Created {len(X)} sequences")
        print(f"  - Sequence shape: {X.shape}")
        print(f"  - Features: 5 raw + 4 ECM = 9 total")
        print(f"  - Target shape: {y.shape}")
        
        return X, y
    
    def build_model(self, input_shape):
        """
        Build hybrid LSTM model
        
        Args:
            input_shape: (sequence_length, n_features)
        """
        print("\nStep 3: Building hybrid LSTM model...")
        
        model = Sequential([
            # First LSTM layer
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            
            # Output layer (SoH prediction)
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        print("✓ Model architecture:")
        model.summary()
        
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the hybrid model
        
        Args:
            X: Feature array
            y: Target array
            validation_split: Fraction for validation
            epochs: Training epochs
            batch_size: Batch size
        """
        print("\nStep 4: Training hybrid model...")
        
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
            patience=15,
            restore_best_weights=True
        )
        
        checkpoint = ModelCheckpoint(
            'hybrid/best_hybrid_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        print("\n✓ Training complete!")
        
        return history
    
    def evaluate(self, X, y):
        """
        Evaluate model performance
        
        Args:
            X: Feature array
            y: True SoH values
            
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
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        print("\n" + "="*60)
        print("Model Performance:")
        print("="*60)
        print(f"RMSE:  {rmse:.6f}")
        print(f"MAE:   {mae:.6f}")
        print(f"R²:    {r2:.6f}")
        print(f"MAPE:  {mape:.2f}%")
        print("="*60)
        
        return metrics, y_pred
    
    def plot_results(self, y_true, y_pred, save_path=None):
        """
        Plot prediction results
        
        Args:
            y_true: True SoH values
            y_pred: Predicted SoH values
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Prediction vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], 
                        [y_true.min(), y_true.max()], 
                        'r--', linewidth=2)
        axes[0, 0].set_xlabel('True SoH', fontsize=12)
        axes[0, 0].set_ylabel('Predicted SoH', fontsize=12)
        axes[0, 0].set_title('Hybrid ECM-LSTM: Prediction vs Actual', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Time series
        axes[0, 1].plot(y_true, 'b-o', label='True', markersize=4)
        axes[0, 1].plot(y_pred, 'r--s', label='Predicted', markersize=4)
        axes[0, 1].set_xlabel('Sample Index', fontsize=12)
        axes[0, 1].set_ylabel('SoH', fontsize=12)
        axes[0, 1].set_title('SoH Over Time', fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error distribution
        error = y_true - y_pred
        axes[1, 0].hist(error, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Prediction Error', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Error Distribution', fontsize=14)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error over time
        axes[1, 1].plot(error, 'g-o', markersize=4)
        axes[1, 1].set_xlabel('Sample Index', fontsize=12)
        axes[1, 1].set_ylabel('Prediction Error', fontsize=12)
        axes[1, 1].set_title('Error Over Time', fontsize=14)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
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
        print("Comparing Hybrid vs Baseline LSTM")
        print("="*60)
        
        # Hybrid model (already trained)
        metrics_hybrid, y_pred_hybrid = self.evaluate(X, y)
        
        # Baseline: LSTM without ECM features (only first 5 features)
        print("\nTraining baseline LSTM (without ECM features)...")
        X_baseline = X[:, :, :5]  # Only raw features
        
        # Build baseline model
        baseline_model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X_baseline.shape[1], X_baseline.shape[2])),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        baseline_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Scale and train
        n_samples, n_steps, n_features = X_baseline.shape
        X_reshaped = X_baseline.reshape(-1, n_features)
        scaler_baseline = MinMaxScaler()
        X_scaled = scaler_baseline.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_steps, n_features)
        
        y_scaled = self.scaler_target.transform(y.reshape(-1, 1)).flatten()
        
        baseline_model.fit(X_scaled, y_scaled, epochs=50, batch_size=32, verbose=0)
        
        # Predict
        y_pred_baseline_scaled = baseline_model.predict(X_scaled)
        y_pred_baseline = self.scaler_target.inverse_transform(y_pred_baseline_scaled).flatten()
        
        # Metrics
        rmse_baseline = np.sqrt(mean_squared_error(y, y_pred_baseline))
        mae_baseline = mean_absolute_error(y, y_pred_baseline)
        r2_baseline = r2_score(y, y_pred_baseline)
        
        # Comparison table
        print("\n" + "="*60)
        print(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
        print("-"*60)
        print(f"{'Baseline LSTM':<20} {rmse_baseline:<12.6f} {mae_baseline:<12.6f} {r2_baseline:<12.6f}")
        print(f"{'Hybrid ECM-LSTM':<20} {metrics_hybrid['RMSE']:<12.6f} {metrics_hybrid['MAE']:<12.6f} {metrics_hybrid['R2']:<12.6f}")
        print("-"*60)
        
        improvement_rmse = ((rmse_baseline - metrics_hybrid['RMSE']) / rmse_baseline) * 100
        improvement_mae = ((mae_baseline - metrics_hybrid['MAE']) / mae_baseline) * 100
        improvement_r2 = ((metrics_hybrid['R2'] - r2_baseline) / r2_baseline) * 100
        
        print(f"{'Improvement':<20} {improvement_rmse:+.2f}%      {improvement_mae:+.2f}%      {improvement_r2:+.2f}%")
        print("="*60)
        
        return {
            'baseline': {'RMSE': rmse_baseline, 'MAE': mae_baseline, 'R2': r2_baseline},
            'hybrid': metrics_hybrid,
            'improvement': {'RMSE': improvement_rmse, 'MAE': improvement_mae, 'R2': improvement_r2}
        }


def run_hybrid_analysis(battery_id='B0005'):
    """
    Complete hybrid ECM-LSTM analysis workflow
    
    Args:
        battery_id: Battery to analyze
    """
    print("="*60)
    print(f"Hybrid ECM-LSTM Analysis - Battery {battery_id}")
    print("="*60)
    
    # Initialize hybrid model
    hybrid = HybridECMLSTM(battery_id=battery_id, sequence_length=50)
    
    # Extract ECM features
    ecm_params = hybrid.extract_ecm_features(cycle_step=5)
    
    # Prepare hybrid features
    X, y = hybrid.prepare_hybrid_features()
    
    # Build model
    hybrid.build_model(input_shape=(X.shape[1], X.shape[2]))
    
    # Train model
    history = hybrid.train(X, y, epochs=100, batch_size=16)
    
    # Evaluate
    metrics, y_pred = hybrid.evaluate(X, y)
    
    # Plot results
    hybrid.plot_results(y, y_pred, save_path=f'hybrid/{battery_id}_hybrid_results.png')
    
    # Compare with baseline
    comparison = hybrid.compare_with_baseline(X, y)
    
    print("\n✅ Hybrid analysis complete!")
    print(f"Results saved to: hybrid/{battery_id}_hybrid_results.png")
    
    return hybrid, metrics, comparison


if __name__ == "__main__":
    # Run hybrid analysis
    hybrid, metrics, comparison = run_hybrid_analysis('B0005')
