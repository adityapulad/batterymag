# 🔧 FIX FOR MISSING IMPORTS ERROR
# Copy this entire cell and paste it BEFORE the "GRAPH 1" cell

# Additional imports for comprehensive visualizations
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # ← ADD THIS LINE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from battery_loader import load_data
from ecm.ecm_parameter_extraction import ECMParameterExtractor

# Create output directory
import os
os.makedirs('results/comprehensive', exist_ok=True)

print("✅ Visualization setup complete!")
print("📁 Output: results/comprehensive/")
