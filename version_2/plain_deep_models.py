from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


INPUT_FEATURES = [
    "hot_inlet_temperature_k",
    "cold_inlet_mass_flow_kg_s",
    "hx_1_heat_load_kw",
    "hot_outlet_pressure_pa",
    "cold_outlet_pressure_pa",
    "hot_outlet_mass_flow_kg_s",
    "cold_outlet_mass_flow_kg_s",
    "hx_1_logarithmic_mean_temperature_difference_lmtd_k",
]

HOT_OUTLET_TARGET = "hot_outlet_temperature_k"
COLD_INLET_TEMPERATURE_K = 293.15
HOT_OUTLET_PRESSURE_PA = 500000.0
COLD_OUTLET_PRESSURE_PA = 100000.0
HOT_MASS_FLOW_KG_S = 1.0


def create_sequence_dataset(
    df: pd.DataFrame,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X = df[INPUT_FEATURES].to_numpy(dtype=float)
    y = df[HOT_OUTLET_TARGET].to_numpy(dtype=float)
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i : i + sequence_length])
        y_seq.append(y[i + sequence_length])
    eval_frame = df.iloc[sequence_length:].reset_index(drop=True)
    return np.array(X_seq, dtype=float), np.array(y_seq, dtype=float), eval_frame


def scale_sequence_dataset(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, object]:
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    return {
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
    }


def build_mlp_sequence_model(sequence_length: int, n_features: int) -> "keras.Model":
    model = keras.Sequential(
        [
            layers.Input(shape=(sequence_length * n_features,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def build_vanilla_lstm_model(sequence_length: int, n_features: int) -> "keras.Model":
    inputs = layers.Input(shape=(sequence_length, n_features))
    x = layers.LSTM(32, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(16)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def save_deep_artifact(
    *,
    artifact_path: Path,
    weights_path: Path,
    model_type: str,
    model: "keras.Model",
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    sequence_length: int,
    metrics: dict[str, float],
    notes: str,
    architecture: dict[str, object],
) -> None:
    model.save_weights(weights_path)
    artifact = {
        "model_type": model_type,
        "sequence_length": sequence_length,
        "input_features": list(INPUT_FEATURES),
        "scaler_X_mean": scaler_X.mean_,
        "scaler_X_scale": scaler_X.scale_,
        "scaler_y_mean": scaler_y.mean_,
        "scaler_y_scale": scaler_y.scale_,
        "weights_file": weights_path.name,
        "metrics": metrics,
        "notes": notes,
        "architecture": architecture,
    }
    joblib.dump(artifact, artifact_path)


def restore_deep_model(artifact_path: Path, weights_path: Path) -> dict | None:
    if not TENSORFLOW_AVAILABLE:
        return None
    if not artifact_path.exists() or not weights_path.exists():
        return None

    artifact = joblib.load(artifact_path)
    sequence_length = int(artifact["sequence_length"])
    n_features = len(artifact["input_features"])

    if artifact["model_type"] == "mlp_sequence":
        model = build_mlp_sequence_model(sequence_length, n_features)
    elif artifact["model_type"] == "vanilla_lstm":
        model = build_vanilla_lstm_model(sequence_length, n_features)
    else:
        return None

    model.load_weights(str(weights_path))

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.mean_ = np.array(artifact["scaler_X_mean"], dtype=float)
    scaler_X.scale_ = np.array(artifact["scaler_X_scale"], dtype=float)
    scaler_X.var_ = scaler_X.scale_ ** 2
    scaler_X.n_features_in_ = len(scaler_X.mean_)
    scaler_y.mean_ = np.array(artifact["scaler_y_mean"], dtype=float)
    scaler_y.scale_ = np.array(artifact["scaler_y_scale"], dtype=float)
    scaler_y.var_ = scaler_y.scale_ ** 2
    scaler_y.n_features_in_ = len(scaler_y.mean_)

    artifact["model"] = model
    artifact["scaler_X"] = scaler_X
    artifact["scaler_y"] = scaler_y
    return artifact


def build_repeated_scenario_sequence(
    hot_inlet_temp: float,
    hot_mass_flow: float,
    cold_mass_flow: float,
    heat_load_estimate: float,
    sequence_length: int,
) -> np.ndarray:
    lmtd_estimate = (hot_inlet_temp - COLD_INLET_TEMPERATURE_K) * 0.6
    row = np.array(
        [
            hot_inlet_temp,
            cold_mass_flow,
            heat_load_estimate,
            HOT_OUTLET_PRESSURE_PA,
            COLD_OUTLET_PRESSURE_PA,
            hot_mass_flow,
            cold_mass_flow,
            lmtd_estimate,
        ],
        dtype=float,
    )
    sequence = np.repeat(row.reshape(1, -1), sequence_length, axis=0)
    return sequence.reshape(1, sequence_length, -1)


def predict_restored_deep_model(
    artifact: dict,
    hot_inlet_temp: float,
    hot_mass_flow: float,
    cold_mass_flow: float,
    heat_load_estimate: float,
) -> float | None:
    try:
        sequence = build_repeated_scenario_sequence(
            hot_inlet_temp=hot_inlet_temp,
            hot_mass_flow=hot_mass_flow,
            cold_mass_flow=cold_mass_flow,
            heat_load_estimate=heat_load_estimate,
            sequence_length=int(artifact["sequence_length"]),
        )
        scaler_X: StandardScaler = artifact["scaler_X"]
        scaler_y: StandardScaler = artifact["scaler_y"]
        sequence_scaled = scaler_X.transform(sequence.reshape(-1, sequence.shape[-1])).reshape(sequence.shape)

        if artifact["model_type"] == "mlp_sequence":
            raw_pred = artifact["model"].predict(sequence_scaled.reshape(1, -1), verbose=0)
        elif artifact["model_type"] == "vanilla_lstm":
            raw_pred = artifact["model"].predict(sequence_scaled, verbose=0)
        else:
            return None

        pred = scaler_y.inverse_transform(raw_pred).reshape(-1)[0]
        return float(pred)
    except Exception:
        return None
