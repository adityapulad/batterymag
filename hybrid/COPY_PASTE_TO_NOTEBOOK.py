"""
🎯 COPY THIS ENTIRE FILE TO YOUR JUPYTER NOTEBOOK
This will generate ALL required graphs for your presentation

Just run this in a single cell after training your models!
"""

# ============================================================================
# COMPLETE VISUALIZATION CODE - COPY TO JUPYTER NOTEBOOK
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# Create output directory
import os
os.makedirs('hybrid/results/comprehensive', exist_ok=True)

print("="*70)
print("🎨 GENERATING ALL REQUIRED GRAPHS FOR PRESENTATION")
print("="*70)

# ============================================================================
# GRAPH 1: Actual vs Predicted SoC (All Models) - VERY IMPORTANT
# ============================================================================

def generate_graph_1(hybrid_soc, X_soc, y_soc_true, battery_id='B0005'):
    print("\n📊 Graph 1: Actual vs Predicted SoC (All Models)...")
    
    # Get hybrid predictions
    X_scaled = hybrid_soc.scaler_features.transform(X_soc.reshape(-1, X_soc.shape[2])).reshape(X_soc.shape)
    y_hybrid = hybrid_soc.model.predict(X_scaled, verbose=0)
    y_hybrid = hybrid_soc.scaler_target.inverse_transform(y_hybrid.reshape(-1, 1)).reshape(y_soc_true.shape)
    
    # Train ECM-only model
    X_ecm = X_soc[:, :, :5]
    scaler_ecm = MinMaxScaler()
    X_ecm_scaled = scaler_ecm.fit_transform(X_ecm.reshape(-1, X_ecm.shape[2])).reshape(X_ecm.shape)
    y_scaled = hybrid_soc.scaler_target.transform(y_soc_true.reshape(-1, 1)).reshape(y_soc_true.shape[0], y_soc_true.shape[1], 1)
    
    ecm_model = Sequential([LSTM(64, input_shape=(X_ecm.shape[1], X_ecm.shape[2]), return_sequences=True), Dense(1)])
    ecm_model.compile(optimizer='adam', loss='mse')
    ecm_model.fit(X_ecm_scaled, y_scaled, epochs=30, verbose=0, batch_size=32)
    y_ecm = ecm_model.predict(X_ecm_scaled, verbose=0)
    y_ecm = hybrid_soc.scaler_target.inverse_transform(y_ecm.reshape(-1, 1)).reshape(y_soc_true.shape)
    
    # Train LSTM-only model
    lstm_model = Sequential([
        LSTM(128, input_shape=(X_ecm.shape[1], X_ecm.shape[2]), return_sequences=True),
        Dropout(0.2), LSTM(64, return_sequences=True), Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_ecm_scaled, y_scaled, epochs=50, verbose=0, batch_size=32)
    y_lstm = lstm_model.predict(X_ecm_scaled, verbose=0)
    y_lstm = hybrid_soc.scaler_target.inverse_transform(y_lstm.reshape(-1, 1)).reshape(y_soc_true.shape)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    sample_idx = 5
    time_steps = np.arange(len(y_soc_true[sample_idx]))
    
    # All models comparison
    axes[0, 0].plot(time_steps, y_soc_true[sample_idx], 'k-', linewidth=3, label='Actual SoC', marker='o', markersize=4)
    axes[0, 0].plot(time_steps, y_ecm[sample_idx], 'b--', linewidth=2, label='ECM SoC', marker='s', markersize=3)
    axes[0, 0].plot(time_steps, y_lstm[sample_idx], 'g--', linewidth=2, label='LSTM SoC', marker='^', markersize=3)
    axes[0, 0].plot(time_steps, y_hybrid[sample_idx], 'r-', linewidth=2, label='Hybrid ECM-LSTM SoC', marker='d', markersize=3)
    axes[0, 0].set_xlabel('Time Step', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('SoC (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_title('SoC Prediction: All Models Comparison', fontsize=16, fontweight='bold')
    axes[0, 0].legend(fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 105])
    
    # Scatter plot
    axes[0, 1].scatter(y_soc_true.flatten(), y_hybrid.flatten(), alpha=0.3, s=10, c='blue')
    axes[0, 1].plot([0, 100], [0, 100], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('Actual SoC (%)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Predicted SoC (%)', fontsize=14, fontweight='bold')
    axes[0, 1].set_title('Hybrid: Actual vs Predicted', fontsize=16, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    r2 = r2_score(y_soc_true.flatten(), y_hybrid.flatten())
    axes[0, 1].text(5, 90, f'R² = {r2:.4f}', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Multiple sequences
    for i, idx in enumerate([0, 10, 20]):
        alpha = 0.7 - i*0.2
        axes[1, 0].plot(y_soc_true[idx], 'k-', linewidth=2, alpha=alpha, label='Actual' if i==0 else '')
        axes[1, 0].plot(y_hybrid[idx], 'r--', linewidth=2, alpha=alpha, label='Hybrid' if i==0 else '')
    axes[1, 0].set_xlabel('Time Step', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('SoC (%)', fontsize=14, fontweight='bold')
    axes[1, 0].set_title('SoC Trajectories (Multiple Cycles)', fontsize=16, fontweight='bold')
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error distribution
    error = y_soc_true.flatten() - y_hybrid.flatten()
    axes[1, 1].hist(error, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Prediction Error (%)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontsize=14, fontweight='bold')
    axes[1, 1].set_title('Error Distribution', fontsize=16, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0.02, 0.98, f'Mean: {np.mean(error):.3f}%\\nStd: {np.std(error):.3f}%',
                   transform=axes[1, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat'), fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'hybrid/results/comprehensive/{battery_id}_01_soc_all_models.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Graph 1 saved!")
    
    return y_ecm, y_lstm, y_hybrid


# ============================================================================
# GRAPH 2: SoC Error Comparison - MOST IMPORTANT GRAPH
# ============================================================================

def generate_graph_2(y_true, y_ecm, y_lstm, y_hybrid, battery_id='B0005'):
    print("\n📊 Graph 2: SoC Error Comparison (MOST IMPORTANT)...")
    
    y_true_flat = y_true.flatten()
    y_ecm_flat = y_ecm.flatten()
    y_lstm_flat = y_lstm.flatten()
    y_hybrid_flat = y_hybrid.flatten()
    
    # Calculate metrics
    metrics = {
        'ECM': {
            'MAE': mean_absolute_error(y_true_flat, y_ecm_flat),
            'RMSE': np.sqrt(mean_squared_error(y_true_flat, y_ecm_flat)),
            'MAPE': np.mean(np.abs((y_true_flat - y_ecm_flat) / (y_true_flat + 1e-10))) * 100
        },
        'LSTM': {
            'MAE': mean_absolute_error(y_true_flat, y_lstm_flat),
            'RMSE': np.sqrt(mean_squared_error(y_true_flat, y_lstm_flat)),
            'MAPE': np.mean(np.abs((y_true_flat - y_lstm_flat) / (y_true_flat + 1e-10))) * 100
        },
        'Hybrid': {
            'MAE': mean_absolute_error(y_true_flat, y_hybrid_flat),
            'RMSE': np.sqrt(mean_squared_error(y_true_flat, y_hybrid_flat)),
            'MAPE': np.mean(np.abs((y_true_flat - y_hybrid_flat) / (y_true_flat + 1e-10))) * 100
        }
    }
    
    # Create bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    models = ['ECM', 'LSTM', 'Hybrid']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # MAE
    mae_values = [metrics[m]['MAE'] for m in models]
    bars1 = axes[0].bar(models, mae_values, color=colors, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('MAE (%)', fontsize=14, fontweight='bold')
    axes[0].set_title('Mean Absolute Error', fontsize=16, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # RMSE
    rmse_values = [metrics[m]['RMSE'] for m in models]
    bars2 = axes[1].bar(models, rmse_values, color=colors, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('RMSE (%)', fontsize=14, fontweight='bold')
    axes[1].set_title('Root Mean Square Error', fontsize=16, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # MAPE
    mape_values = [metrics[m]['MAPE'] for m in models]
    bars3 = axes[2].bar(models, mape_values, color=colors, edgecolor='black', linewidth=2)
    axes[2].set_ylabel('MAPE (%)', fontsize=14, fontweight='bold')
    axes[2].set_title('Mean Absolute Percentage Error', fontsize=16, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.suptitle('SoC Error Comparison: ECM vs LSTM vs Hybrid ECM-LSTM', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'hybrid/results/comprehensive/{battery_id}_02_soc_error_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print improvements
    print("\n" + "="*70)
    print("📊 HYBRID IMPROVEMENTS:")
    print("="*70)
    print(f"vs ECM:  MAE {((metrics['ECM']['MAE'] - metrics['Hybrid']['MAE']) / metrics['ECM']['MAE'] * 100):+.2f}%")
    print(f"vs LSTM: MAE {((metrics['LSTM']['MAE'] - metrics['Hybrid']['MAE']) / metrics['LSTM']['MAE'] * 100):+.2f}%")
    print("="*70)
    print("✅ Graph 2 saved (MOST IMPORTANT)!")
    
    return metrics


# ============================================================================
# RUN ALL GRAPHS
# ============================================================================

print("\n🚀 Starting graph generation...")
print("This will take a few minutes...\n")

# Prepare data
X_soc, y_soc = hybrid_soc.prepare_soc_sequences()

# Generate graphs
y_ecm, y_lstm, y_hybrid = generate_graph_1(hybrid_soc, X_soc, y_soc, 'B0005')
soc_metrics = generate_graph_2(y_soc, y_ecm, y_lstm, y_hybrid, 'B0005')

print("\n" + "="*70)
print("🎉 CORE GRAPHS GENERATED SUCCESSFULLY!")
print("="*70)
print("\nSaved to: hybrid/results/comprehensive/")
print("\n✅ Graph 1: Actual vs Predicted SoC (All Models)")
print("✅ Graph 2: SoC Error Comparison (MOST IMPORTANT)")
print("\n📝 For remaining graphs, see QUICK_VISUALIZATION_GUIDE.md")
print("="*70)
