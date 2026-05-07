#!/usr/bin/env python3
"""Debug helper: load PI-LSTM artifact and weights and run a prediction outside Streamlit."""
import joblib
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from version_1.physics_informed_lstm import PhysicsInformedLSTM
except Exception as e:
    print("Failed importing PhysicsInformedLSTM:", e)
    raise

ARTIFACT = Path("version_2/results/low_data_125_pilstm_artifact.joblib")
WEIGHTS = Path("version_2/results/low_data_125_pilstm.weights.h5")

if not ARTIFACT.exists():
    print("Artifact not found:", ARTIFACT)
    raise SystemExit(1)
if not WEIGHTS.exists():
    print("Weights not found:", WEIGHTS)
    raise SystemExit(1)

artifact = joblib.load(ARTIFACT)
print("Loaded artifact keys:", list(artifact.keys()))

p = artifact
pilstm = PhysicsInformedLSTM(
    sequence_length=p["sequence_length"],
    lstm_units=p["lstm_units"],
    learning_rate=p["learning_rate"],
    use_hard_energy_balance=True,
)
print("Built PI-LSTM instance")

pilstm.build_model(input_shape=(p["sequence_length"], 8))
print("Built model layers. Loading weights (by_name=True)...")
loaded = False
try:
    pilstm.model.load_weights(str(WEIGHTS), by_name=True, skip_mismatch=True)
    print("Weights loaded with by_name=True + skip_mismatch=True")
    loaded = True
except TypeError:
    # older TF/Keras may not accept skip_mismatch with by_name
    try:
        pilstm.model.load_weights(str(WEIGHTS), by_name=True)
        print("Weights loaded with by_name=True")
        loaded = True
    except Exception:
        loaded = False
except Exception as e:
    print("load_weights by_name failed:", repr(e))

if not loaded:
    try:
        pilstm.model.load_weights(str(WEIGHTS), skip_mismatch=True)
        print("Weights loaded with skip_mismatch=True")
        loaded = True
    except Exception as e:
        print("Final load attempt failed:", repr(e))
        raise

# restore scalers if available
if "scaler_X_mean" in p:
    pilstm.scaler_X.mean_ = p["scaler_X_mean"]
    pilstm.scaler_X.scale_ = p["scaler_X_scale"]
if "scaler_y_mean" in p:
    pilstm.scaler_y.mean_ = p["scaler_y_mean"]
    pilstm.scaler_y.scale_ = p["scaler_y_scale"]

# prepare a dummy input sequence (use plausible values from artifact or defaults)
hot_inlet = 450.0
cold_flow = 3.0
heat_load = 0.5
hot_mass_flow = 1.0
lmtd = (hot_inlet - 293.15) * 0.6
row = [hot_inlet, cold_flow, heat_load, 500000.0, 100000.0, hot_mass_flow, cold_flow, lmtd]
seq = np.array([row] * pilstm.sequence_length, dtype=float).reshape(1, pilstm.sequence_length, -1)

print("Running pilstm.predict on test sequence...")
try:
    out = pilstm.predict(seq)
    print("Predict output shape:", getattr(out, 'shape', None))
    print("Predict output:", out)
    hot_pred = float(out[0, 0])
    hot_q = hot_mass_flow * float(pilstm.cp_hot) * (hot_inlet - hot_pred)
    q_loss = 0.0
    cold_pred = (hot_q - q_loss) / (cold_flow * float(pilstm.cp_cold) + 1e-12) + float(pilstm.cold_inlet_temperature_k)
    cold_q = cold_flow * float(pilstm.cp_cold) * (cold_pred - float(pilstm.cold_inlet_temperature_k))
    print("Hot-side heat transfer rate (kW):", hot_q)
    print("Cold-side heat transfer rate (kW):", cold_q)
    print("Heat loss to surroundings (kW):", q_loss)
    print("Energy gap (kW):", abs(hot_q - cold_q))
except Exception as e:
    print("Prediction failed:", repr(e))
    raise

print("Done.")
