"""
Train PI-LSTM model and save as artifact for streamlit dashboard
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")
    exit(1)

from physics_informed_lstm import PhysicsInformedLSTM

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
APP_DIR = Path(__file__).resolve().parent


def train_pilstm_model(dataset_path: str | Path | None = None):
    dataset_file = APP_DIR / "heat_exchanger_dataset.csv" if dataset_path is None else Path(dataset_path)

    print("Loading dataset...")
    df = pd.read_csv(dataset_file)
    print(f"Dataset shape: {df.shape}")
    
    print("\nInitializing Physics-Informed LSTM...")
    pi_lstm = PhysicsInformedLSTM(
        sequence_length=10,
        lstm_units=64,
        learning_rate=0.001
    )
    
    print("Preparing data...")
    X, y = pi_lstm.prepare_data(df)
    X_seq, y_seq = pi_lstm.create_sequences(X, y)
    print(f"Sequence input shape: {X_seq.shape}")
    print(f"Sequence output shape: {y_seq.shape}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_seq, y_seq, test_size=0.15, random_state=42, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, shuffle=False
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print("\nBuilding model...")
    model = pi_lstm.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    print("\nTraining Physics-Informed LSTM...")
    history = pi_lstm.train(
        X_train, y_train,
        X_val, y_val,
        epochs=1000,
        batch_size=32,
        verbose=1
    )
    
    print("\nEvaluating on test set...")
    y_pred = pi_lstm.predict(X_test)
    
    # Calculate metrics
    mae_hot = np.mean(np.abs(y_test[:, 0] - y_pred[:, 0]))
    mae_cold = np.mean(np.abs(y_test[:, 1] - y_pred[:, 1]))
    rmse_hot = np.sqrt(np.mean((y_test[:, 0] - y_pred[:, 0])**2))
    rmse_cold = np.sqrt(np.mean((y_test[:, 1] - y_pred[:, 1])**2))
    mape_hot = np.mean(np.abs((y_test[:, 0] - y_pred[:, 0]) / y_test[:, 0])) * 100
    mape_cold = np.mean(np.abs((y_test[:, 1] - y_pred[:, 1]) / y_test[:, 1])) * 100
    
    print("\nTest Set Performance:")
    print("-" * 50)
    print("Hot Outlet Temperature:")
    print(f"  MAE:  {mae_hot:.4f} K")
    print(f"  RMSE: {rmse_hot:.4f} K")
    print(f"  MAPE: {mape_hot:.4f} %")
    print("\nCold Outlet Temperature:")
    print(f"  MAE:  {mae_cold:.4f} K")
    print(f"  RMSE: {rmse_cold:.4f} K")
    print(f"  MAPE: {mape_cold:.4f} %")
    
    # Save the model weights only
    print("\nSaving PI-LSTM model weights...")
    weights_path = APP_DIR / "pilstm_model.weights.h5"
    pi_lstm.model.save_weights(str(weights_path))
    
    # Save the artifact without the compiled model
    artifact = {
        'sequence_length': pi_lstm.sequence_length,
        'lstm_units': pi_lstm.lstm_units,
        'learning_rate': pi_lstm.learning_rate,
        'scaler_X_mean': pi_lstm.scaler_X.mean_,
        'scaler_X_scale': pi_lstm.scaler_X.scale_,
        'scaler_y_mean': pi_lstm.scaler_y.mean_,
        'scaler_y_scale': pi_lstm.scaler_y.scale_,
        'input_features': [
            'hot_inlet_temperature_k',
            'cold_inlet_mass_flow_kg_s',
            'hx_1_heat_load_kw',
            'hot_outlet_pressure_pa',
            'cold_outlet_pressure_pa',
            'hot_outlet_mass_flow_kg_s',
            'cold_outlet_mass_flow_kg_s',
            'hx_1_logarithmic_mean_temperature_difference_lmtd_k'
        ],
        'metrics': {
            'hot_outlet': {
                'MAE': mae_hot,
                'RMSE': rmse_hot,
                'MAPE': mape_hot,
                'R2': 1 - (rmse_hot**2 / np.var(y_test[:, 0]))
            },
            'cold_outlet': {
                'MAE': mae_cold,
                'RMSE': rmse_cold,
                'MAPE': mape_cold,
                'R2': 1 - (rmse_cold**2 / np.var(y_test[:, 1]))
            }
        }
    }
    
    artifact_path = APP_DIR / "pilstm_artifact.joblib"
    joblib.dump(artifact, artifact_path)
    print(f"PI-LSTM artifact saved as '{artifact_path}'")
    print(f"PI-LSTM weights saved as '{weights_path}'")
    
    return artifact


if __name__ == "__main__":
    artifact = train_pilstm_model()
    print("\nTraining complete!")
