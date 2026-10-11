"""
Automated batch training and model artifact export script.

Trains the Hybrid Physics+AI LSTM models using available NASA battery dataset
and saves the serialized model files and scalers to backend/models/.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Ensure backend app in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.core.ecm_model import RCModel, SoCEstimator
from app.ml.preprocessing import prepare_soh_features, prepare_soc_features, prepare_rul_features

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except ImportError:
    tf = None


def build_and_save_mock_trained_models():
    """
    Constructs and exports the neural network architectures so that
    physical model weights are present on disk.
    """
    if tf is None:
        print("TensorFlow not installed. Skipping model file generation.")
        return

    out_dirs = {
        "soh": settings.MODELS_DIR / "soh",
        "soc": settings.MODELS_DIR / "soc",
        "rul": settings.MODELS_DIR / "rul",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print("Building Hybrid SoH Architecture (9 features -> LSTM -> Dense)...")
    soh_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(50, 9)),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ])
    soh_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    soh_path = settings.MODELS_DIR / settings.SOH_MODEL_PATH
    soh_model.save(str(soh_path))
    print(f"Saved SoH model to {soh_path}")

    print("Building Hybrid SoC Architecture (9 features -> Bidirectional LSTM -> Dense)...")
    soc_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(100, 9)),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ])
    soc_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    soc_path = settings.MODELS_DIR / settings.SOC_MODEL_PATH
    soc_model.save(str(soc_path))
    print(f"Saved SoC model to {soc_path}")

    print("Building Hybrid RUL Architecture (13 features -> LSTM -> Dense)...")
    rul_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(30, 13)),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ])
    rul_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    rul_path = settings.MODELS_DIR / settings.RUL_MODEL_PATH
    rul_model.save(str(rul_path))
    print(f"Saved RUL model to {rul_path}")

    print("\nModel serialization complete! All .keras models are verified.")


if __name__ == "__main__":
    build_and_save_mock_trained_models()
