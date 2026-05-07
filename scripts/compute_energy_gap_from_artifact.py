import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from version_1.physics_informed_lstm import PhysicsInformedLSTM

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'version_2' / 'results'
DATA_DIR = ROOT / 'version_2' / 'data'

SUBSETS = ['low_data_100', 'low_data_125']

cp_hot = 4.18
cp_cold = 4.18
cold_inlet_temperature_k = 293.15

reports = []
for subset in SUBSETS:
    art_path = RESULTS / f"{subset}_pilstm_artifact.joblib"
    weights_path = RESULTS / f"{subset}_pilstm.weights.h5"
    subset_file = DATA_DIR / f"{subset}.csv"
    if not art_path.exists() or not weights_path.exists() or not subset_file.exists():
        print('Missing files for', subset)
        continue

    art = joblib.load(art_path)
    seq_len = int(art.get('sequence_length', 10))
    lstm_units = int(art.get('lstm_units', 64))
    input_features = art.get('input_features', [])

    pilstm = PhysicsInformedLSTM(sequence_length=seq_len, lstm_units=lstm_units)

    # restore scalers
    scaler_X = StandardScaler()
    scaler_X.mean_ = np.array(art['scaler_X_mean'])
    scaler_X.scale_ = np.array(art['scaler_X_scale'])
    scaler_X.var_ = scaler_X.scale_ ** 2
    scaler_X.n_features_in_ = len(scaler_X.mean_)
    pilstm.scaler_X = scaler_X

    scaler_y = StandardScaler()
    scaler_y.mean_ = np.array(art['scaler_y_mean'])
    scaler_y.scale_ = np.array(art['scaler_y_scale'])
    scaler_y.var_ = scaler_y.scale_ ** 2
    scaler_y.n_features_in_ = len(scaler_y.mean_)
    pilstm.scaler_y = scaler_y

    # build and load weights. try default (2-output) first; if loading fails, retry with hard energy-balance (1-output)
    input_shape = (seq_len, len(input_features))
    pilstm.build_model(input_shape=input_shape)
    try:
        pilstm.model.load_weights(str(weights_path))
    except ValueError:
        # likely mismatch because weights were saved for a 1-output model
        pilstm = PhysicsInformedLSTM(sequence_length=seq_len, lstm_units=lstm_units, use_hard_energy_balance=True)
        pilstm.scaler_X = scaler_X
        pilstm.scaler_y = scaler_y
        pilstm.build_model(input_shape=input_shape)
        pilstm.model.load_weights(str(weights_path))

    # prepare sequences and splits
    df = pd.read_csv(subset_file)
    X, y = pilstm.prepare_data(df)
    X_seq, y_seq = pilstm.create_sequences(X, y)

    n_total = len(X_seq)
    n_test = max(1, int(round(n_total * 0.15)))
    test_start = n_total - n_test
    X_test = X_seq[test_start:]
    y_test = y_seq[test_start:]

    # predict
    y_pred = pilstm.predict(X_test)

    hot_inlet_idx = 0
    cold_flow_idx = 1
    hot_flow_idx = 5

    q_hot = []
    q_cold = []
    energy_gap = []
    errors_hot = []
    errors_cold = []

    for i in range(len(X_test)):
        X_last = X_test[i, -1, :]
        hot_inlet = float(X_last[hot_inlet_idx])
        cold_flow = float(X_last[cold_flow_idx])
        hot_flow = float(X_last[hot_flow_idx])
        hot_out_pred = float(y_pred[i, 0])
        cold_out_pred = float(y_pred[i, 1])
        hot_true = float(y_test[i, 0])
        cold_true = float(y_test[i, 1])

        Q_hot = hot_flow * cp_hot * (hot_inlet - hot_out_pred)
        Q_cold = cold_flow * cp_cold * (cold_out_pred - cold_inlet_temperature_k)
        q_hot.append(Q_hot)
        q_cold.append(Q_cold)
        energy_gap.append(abs(Q_hot - Q_cold))
        errors_hot.append(hot_true - hot_out_pred)
        errors_cold.append(cold_true - cold_out_pred)

    q_hot = np.array(q_hot)
    q_cold = np.array(q_cold)
    energy_gap = np.array(energy_gap)
    errors_hot = np.array(errors_hot)
    errors_cold = np.array(errors_cold)

    report = {
        'subset': subset,
        'n_points': int(len(energy_gap)),
        'mean_energy_gap_kw': float(np.mean(energy_gap)),
        'median_energy_gap_kw': float(np.median(energy_gap)),
        'max_energy_gap_kw': float(np.max(energy_gap)),
        'pct_gt_1kw': float((energy_gap > 1.0).mean() * 100.0),
        'rmse_hot_k': float(np.sqrt(np.mean(errors_hot ** 2))) if len(errors_hot) else None,
        'rmse_cold_k': float(np.sqrt(np.mean(errors_cold ** 2))) if len(errors_cold) else None,
    }
    reports.append(report)

out_path = RESULTS / 'pilstm_energy_gap_diagnostics_from_artifact.csv'
pd.DataFrame(reports).to_csv(out_path, index=False)
print('Saved diagnostics to', out_path)
for r in reports:
    print(r)
