"""
Hybrid ECM + LSTM Model for Battery SOC (State of Charge) Estimation

This module combines physics-based ECM parameters with LSTM neural networks
for improved State of Charge estimation.

Approach:
1. Extract ECM parameters (R0, R1, C1, tau1) from battery data
2. Use ECM parameters as additional features for LSTM
3. Train LSTM on combined features (voltage, current, temp + ECM params)
4. Achieve better SOC accuracy and interpretability

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


class HybridECMLSTM_SOC:
    """
    Hybrid model combining ECM parameters with LSTM for SOC estimation
    
    Architecture:
    1. ECM layer: Extract R0, R1, C1, tau1 for each cycle
    2. Feature layer: Combine ECM params with raw measurements
    3. Bidirectional LSTM layer: Process time-series with enhanced features
    4. Output layer: Predict State of Charge (0-100%)
    """
    
    def __init__(self, battery_id='B0005', sequence_length=100):
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
        print("Step 1: Extracting ECM parameters for SOC...")
        extractor = ECMParameterExtractor(self.battery_id)
        extractor.load_battery_data()
        
        # Extract RC model parameters
        self.ecm_params_df = extractor.extract_parameters_over_lifetime(
            model_type='RC',
            cycle_step=cycle_step
        )
        
        print(f"✓ Extracted ECM parameters for {len(self.ecm_params_df)} cycles")
        return self.ecm_params_df
    
    def prepare_soc_sequences(self):
        """
        Prepare sequences for SOC estimation with ECM features
        
        Returns:
            X: Feature array (samples, sequence_length, features)
            y: Target array (SOC values)
        """
        print("\nStep 2: Preparing SOC sequences with hybrid features...")
        
        # Load raw battery data
        dataset, capacity_data = load_data(self.battery_id)
        
        # Calculate SOC (State of Charge) from capacity
        # SOC = remaining_capacity / total_capacity * 100
        dataset['soc'] = (dataset['capacity'] / dataset['capacity'].max()) * 100
        
        # Get cycles that have ECM parameters
        available_cycles = self.ecm_params_df['cycle'].values
        
        X_sequences = []
        y_soc = []
        
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
            
            # Create sliding windows within the cycle
            for i in range(0, len(cycle_data) - self.sequence_length, self.sequence_length // 2):
                window_data = cycle_data.iloc[i:i+self.sequence_length]
                
                if len(window_data) < self.sequence_length:
                    continue
                
                # Raw features
                raw_features = window_data[[
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
                y_soc.append(window_data['soc'].values)  # SOC for each time step
        
        X = np.array(X_sequences)
        y = np.array(y_soc)
        
        print(f"✓ Created {len(X)} sequences")
        print(f"  - Sequence shape: {X.shape}")
        print(f"  - Features: 5 raw + 4 ECM = 9 total")
        print(f"  - Target shape: {y.shape}")
        
        return X, y
    
    def build_model(self, input_shape):
        """
        Build hybrid Bidirectional LSTM model for SOC
        
        Args:
            input_shape: (sequence_length, n_features)
        """
        print("\nStep 3: Building hybrid Bidirectional LSTM model for SOC...")
        
        model = Sequential([
            # First Bidirectional LSTM layer
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            
            # Second Bidirectional LSTM layer
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            
            # Third LSTM layer
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            
            # Dense layers for each time step
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # SOC prediction for each time step
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
    
    def train(self, X, y, validation_split=0.2, epochs=150, batch_size=32):
        """
        Train the hybrid SOC model
        
        Args:
            X: Feature array
            y: Target array (SOC values)
            validation_split: Fraction for validation
            epochs: Training epochs
            batch_size: Batch size
        """
        print("\nStep 4: Training hybrid SOC model...")
        
        # Scale features
        n_samples, n_steps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler_features.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_steps, n_features)
        
        # Scale target (SOC)
        y_reshaped = y.reshape(-1, 1)
        y_scaled = self.scaler_target.fit_transform(y_reshaped)
        y_scaled = y_scaled.reshape(n_samples, n_steps, 1)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=validation_split, random_state=42
        )
        
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Validation set: {len(X_val)} samples")
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'hybrid/best_hybrid_soc_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
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
        Evaluate SOC model performance
        
        Args:
            X: Feature array
            y: True SOC values
            
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
        y_pred = self.scaler_target.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).reshape(n_samples, n_steps)
        
        # Flatten for metrics
        y_flat = y.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_flat, y_pred_flat))
        mae = mean_absolute_error(y_flat, y_pred_flat)
        r2 = r2_score(y_flat, y_pred_flat)
        mape = np.mean(np.abs((y_flat - y_pred_flat) / (y_flat + 1e-10))) * 100
        
        # Max error
        max_error = np.max(np.abs(y_flat - y_pred_flat))
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Max_Error': max_error
        }
        
        print("\n" + "="*60)
        print("SOC Estimation Performance:")
        print("="*60)
        print(f"RMSE:       {rmse:.4f}%")
        print(f"MAE:        {mae:.4f}%")
        print(f"R²:         {r2:.6f}")
        print(f"MAPE:       {mape:.2f}%")
        print(f"Max Error:  {max_error:.4f}%")
        print("="*60)
        
        return metrics, y_pred
    
    def plot_soc_results(self, y_true, y_pred, save_path=None):
        """
        Plot SOC prediction results
        
        Args:
            y_true: True SOC values
            y_pred: Predicted SOC values
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Flatten arrays for plotting
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # 1. Prediction vs Actual
        axes[0, 0].scatter(y_true_flat, y_pred_flat, alpha=0.3, s=10)
        axes[0, 0].plot([0, 100], [0, 100], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('True SOC (%)', fontsize=12)
        axes[0, 0].set_ylabel('Predicted SOC (%)', fontsize=12)
        axes[0, 0].set_title('Hybrid ECM-LSTM: SOC Prediction vs Actual', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim([0, 100])
        axes[0, 0].set_ylim([0, 100])
        
        # 2. Time series (sample sequence)
        sample_idx = 0
        axes[0, 1].plot(y_true[sample_idx], 'b-', label='True SOC', linewidth=2)
        axes[0, 1].plot(y_pred[sample_idx], 'r--', label='Predicted SOC', linewidth=2)
        axes[0, 1].set_xlabel('Time Step', fontsize=12)
        axes[0, 1].set_ylabel('SOC (%)', fontsize=12)
        axes[0, 1].set_title('SOC Trajectory (Sample Sequence)', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error distribution
        error = y_true_flat - y_pred_flat
        axes[1, 0].hist(error, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Prediction Error (%)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('SOC Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add statistics
        axes[1, 0].text(0.02, 0.98, 
                       f'Mean: {np.mean(error):.3f}%\nStd: {np.std(error):.3f}%',
                       transform=axes[1, 0].transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Error over time
        axes[1, 1].plot(error[:1000], 'g-', alpha=0.6, linewidth=1)
        axes[1, 1].set_xlabel('Sample Index', fontsize=12)
        axes[1, 1].set_ylabel('Prediction Error (%)', fontsize=12)
        axes[1, 1].set_title('SOC Error Over Time', fontsize=14, fontweight='bold')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Figure saved to {save_path}")
        
        return fig


def run_hybrid_soc_analysis(battery_id='B0005'):
    """
    Complete hybrid ECM-LSTM SOC analysis workflow
    
    Args:
        battery_id: Battery to analyze
    """
    print("="*60)
    print(f"Hybrid ECM-LSTM SOC Analysis - Battery {battery_id}")
    print("="*60)
    
    # Initialize hybrid model
    hybrid = HybridECMLSTM_SOC(battery_id=battery_id, sequence_length=100)
    
    # Extract ECM features
    ecm_params = hybrid.extract_ecm_features(cycle_step=5)
    
    # Prepare SOC sequences
    X, y = hybrid.prepare_soc_sequences()
    
    # Build model
    hybrid.build_model(input_shape=(X.shape[1], X.shape[2]))
    
    # Train model
    history = hybrid.train(X, y, epochs=150, batch_size=32)
    
    # Evaluate
    metrics, y_pred = hybrid.evaluate(X, y)
    
    # Plot results
    hybrid.plot_soc_results(y, y_pred, save_path=f'hybrid/results/{battery_id}_hybrid_soc_results.png')
    
    print("\n✅ Hybrid SOC analysis complete!")
    print(f"Results saved to: hybrid/results/{battery_id}_hybrid_soc_results.png")
    
    return hybrid, metrics


if __name__ == "__main__":
    # Run hybrid SOC analysis
    hybrid, metrics = run_hybrid_soc_analysis('B0005')
