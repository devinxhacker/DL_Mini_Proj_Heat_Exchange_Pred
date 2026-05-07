import math
import numpy as np
import pandas as pd
from pathlib import Path

from version_1.physics_informed_lstm import PhysicsInformedLSTM

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'version_2' / 'data'
RESULTS = ROOT / 'version_2' / 'results'
PRED_PATH = RESULTS / 'low_data_predictions.csv'

PILSTM_SEQ_LEN = 10
PILSTM_TEST_FRACTION = 0.15
PILSTM_VAL_FRACTION_OF_REMAINING = 0.176

pred_df = pd.read_csv(PRED_PATH)
subsets = pred_df['subset_name'].unique()

hot_idx = 0
cold_flow_idx = 1
hot_flow_idx = 5
cp_hot = 4.18
cp_cold = 4.18
cold_inlet_temperature_k = 293.15

reports = []
for subset in subsets:
    if subset not in ['low_data_100', 'low_data_125']:
        continue
    subset_file = DATA_DIR / f"{subset}.csv"
    if not subset_file.exists():
        print('Missing subset file', subset_file)
        continue
    df = pd.read_csv(subset_file)
    pilstm = PhysicsInformedLSTM(sequence_length=PILSTM_SEQ_LEN)
    X, y = pilstm.prepare_data(df)
    X_seq, y_seq = pilstm.create_sequences(X, y)
    n_total = len(X_seq)
    n_test = max(1, int(round(n_total * PILSTM_TEST_FRACTION)))
    test_start = n_total - n_test
    X_test = X_seq[test_start:]
    y_test = y_seq[test_start:]

    rows_hot = pred_df.loc[
        (pred_df['subset_name'] == subset) &
        (pred_df['model'] == 'PI-LSTM') &
        (pred_df['target'] == 'hot_outlet_temperature_k')
    ]
    rows_cold = pred_df.loc[
        (pred_df['subset_name'] == subset) &
        (pred_df['model'] == 'PI-LSTM') &
        (pred_df['target'] == 'cold_outlet_temperature_k')
    ]
    rows_hot = rows_hot.sort_values('point_index').reset_index(drop=True)
    rows_cold = rows_cold.sort_values('point_index').reset_index(drop=True)

    if len(rows_hot) != len(X_test) or len(rows_cold) != len(X_test):
        print(f"Warning: counts differ for {subset}: X_test={len(X_test)} hot_preds={len(rows_hot)} cold_preds={len(rows_cold)}")

    energy_gaps = []
    q_hot_list = []
    q_cold_list = []
    errors_hot = []
    errors_cold = []
    for i in range(min(len(X_test), len(rows_hot), len(rows_cold))):
        X_last = X_test[i, -1, :]
        hot_inlet = float(X_last[hot_idx])
        cold_flow = float(X_last[cold_flow_idx])
        hot_flow = float(X_last[hot_flow_idx])
        hot_out_pred = float(rows_hot.loc[i, 'predicted_value'])
        cold_out_pred = float(rows_cold.loc[i, 'predicted_value'])
        hot_true = float(rows_hot.loc[i, 'actual_value'])
        cold_true = float(rows_cold.loc[i, 'actual_value'])

        Q_hot = hot_flow * cp_hot * (hot_inlet - hot_out_pred)
        Q_cold = cold_flow * cp_cold * (cold_out_pred - cold_inlet_temperature_k)
        energy_gaps.append(abs(Q_hot - Q_cold))
        q_hot_list.append(Q_hot)
        q_cold_list.append(Q_cold)
        errors_hot.append(hot_true - hot_out_pred)
        errors_cold.append(cold_true - cold_out_pred)

    energy_gaps = np.array(energy_gaps)
    q_hot_list = np.array(q_hot_list)
    q_cold_list = np.array(q_cold_list)
    errors_hot = np.array(errors_hot)
    errors_cold = np.array(errors_cold)

    report = {
        'subset': subset,
        'n_points': len(energy_gaps),
        'mean_energy_gap_kw': float(np.mean(energy_gaps)),
        'median_energy_gap_kw': float(np.median(energy_gaps)),
        'max_energy_gap_kw': float(np.max(energy_gaps)),
        'pct_gt_1kw': float((energy_gaps > 1.0).mean() * 100.0),
        'rmse_hot_k': float(np.sqrt(np.mean(errors_hot ** 2))) if len(errors_hot) else None,
        'rmse_cold_k': float(np.sqrt(np.mean(errors_cold ** 2))) if len(errors_cold) else None,
    }
    reports.append(report)

print('\nEnergy-gap diagnostics for PI-LSTM predictions:')
for r in reports:
    print(r)

# Save detailed diagnostics
out_path = RESULTS / 'pilstm_energy_gap_diagnostics.csv'
df_out = pd.DataFrame(reports)
df_out.to_csv(out_path, index=False)
print('\nSaved diagnostics to', out_path)
