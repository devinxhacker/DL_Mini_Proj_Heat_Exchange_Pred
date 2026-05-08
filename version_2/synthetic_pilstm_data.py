from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
VERSION1_DATASET = ROOT_DIR / "version_1" / "heat_exchanger_dataset.csv"
DEFAULT_SYNTHETIC_PATH = Path(__file__).resolve().parent / "data" / "synthetic_training_data.csv"
DEFAULT_SYNTHETIC_TRAJECTORIES = 1500
DEFAULT_SYNTHETIC_STEPS_PER_TRAJECTORY = 28


def _safe_lmtd(hot_in: float, hot_out: float, cold_in: float, cold_out: float) -> float:
    delta_t1 = max(hot_in - cold_out, 1e-3)
    delta_t2 = max(hot_out - cold_in, 1e-3)
    if abs(delta_t1 - delta_t2) < 1e-6:
        return float(delta_t1)
    ratio = max(delta_t1 / delta_t2, 1e-6)
    return float((delta_t1 - delta_t2) / np.log(ratio))


def _clip(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


def _load_source_dataset() -> pd.DataFrame:
    source = pd.read_csv(VERSION1_DATASET).drop_duplicates().reset_index(drop=True)
    numeric = source.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return numeric


def generate_synthetic_pilstm_dataset(
    n_trajectories: int = DEFAULT_SYNTHETIC_TRAJECTORIES,
    steps_per_trajectory: int = DEFAULT_SYNTHETIC_STEPS_PER_TRAJECTORY,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate physics-consistent trajectories for PI-LSTM pretraining."""

    source = _load_source_dataset()
    source_records = source.to_dict(orient="records")
    rng = np.random.default_rng(random_state)
    cp_hot = 4.18
    cp_cold = 4.18

    rows: list[dict[str, float]] = []
    for trajectory_id in range(n_trajectories):
        base = source_records[int(rng.integers(0, len(source_records)))]
        hot_base = float(base["hot_inlet_temperature_k"])
        hot_flow_base = float(base["hot_outlet_mass_flow_kg_s"])
        cold_flow_base = float(base["cold_inlet_mass_flow_kg_s"])
        hot_pressure_base = float(base["hot_outlet_pressure_pa"])
        cold_pressure_base = float(base["cold_outlet_pressure_pa"])
        q_base = float(base["hx_1_heat_load_kw"])
        regime = int(rng.integers(0, 4))

        if regime == 0:
            hot_shift = rng.normal(8.0, 2.0)
            cold_drive_center = rng.uniform(18.0, 45.0)
            flow_jitter = 0.06
            eta_center = 0.985
        elif regime == 1:
            hot_shift = rng.normal(0.0, 3.0)
            cold_drive_center = rng.uniform(35.0, 70.0)
            flow_jitter = 0.10
            eta_center = 0.975
        elif regime == 2:
            hot_shift = rng.normal(-5.0, 2.5)
            cold_drive_center = rng.uniform(8.0, 25.0)
            flow_jitter = 0.08
            eta_center = 0.965
        else:
            hot_shift = rng.normal(12.0, 4.0)
            cold_drive_center = rng.uniform(50.0, 90.0)
            flow_jitter = 0.12
            eta_center = 0.955

        for step_index in range(steps_per_trajectory):
            phase = step_index / max(steps_per_trajectory - 1, 1)

            hot_in = _clip(
                hot_base + hot_shift + rng.normal(0.0, 1.8) + (phase - 0.5) * rng.normal(0.0, 6.0),
                275.0,
                405.0,
            )
            cold_drive = _clip(cold_drive_center + rng.normal(0.0, 6.0), 3.0, 105.0)
            cold_base = hot_base - cold_drive
            cold_in = _clip(
                cold_base + rng.normal(0.0, 1.8) + (phase - 0.5) * rng.normal(0.0, 8.0),
                245.0,
                360.0,
            )
            hot_flow = _clip(hot_flow_base + rng.normal(0.0, flow_jitter), 0.30, 3.50)
            cold_flow = _clip(cold_flow_base + rng.normal(0.0, flow_jitter), 0.30, 3.50)
            eta = _clip(eta_center + rng.normal(0.0, 0.012), 0.90, 1.0)

            capacity_hot = hot_flow * cp_hot
            capacity_cold = cold_flow * cp_cold
            temperature_drive = max(hot_in - cold_in, 5.0)
            q_limit = 0.92 * min(capacity_hot, capacity_cold) * temperature_drive
            q_seed = q_base * (0.80 + 0.45 * rng.random())
            q_hot = _clip(q_seed, 0.10 * q_limit, q_limit)

            hot_out = hot_in - q_hot / max(capacity_hot, 1e-8)
            if hot_out <= cold_in + 1.0:
                q_hot = 0.82 * min(q_hot, capacity_hot * max(hot_in - cold_in - 1.0, 1.0))
                hot_out = hot_in - q_hot / max(capacity_hot, 1e-8)

            q_cold = eta * q_hot
            cold_out = cold_in + q_cold / max(capacity_cold, 1e-8)
            lmtd = _safe_lmtd(hot_in, hot_out, cold_in, cold_out)

            hot_pressure = _clip(
                hot_pressure_base - 1200.0 * (hot_flow - hot_flow_base) + rng.normal(0.0, 750.0),
                90_000.0,
                220_000.0,
            )
            cold_pressure = _clip(
                cold_pressure_base - 1200.0 * (cold_flow - cold_flow_base) + rng.normal(0.0, 750.0),
                90_000.0,
                220_000.0,
            )

            rows.append(
                {
                    "trajectory_id": float(trajectory_id),
                    "step_index": float(step_index),
                    "hot_inlet_temperature_k": hot_in,
                    "cold_inlet_temperature_k": cold_in,
                    "cold_inlet_mass_flow_kg_s": cold_flow,
                    "hot_outlet_temperature_k": hot_out,
                    "cold_outlet_temperature_k": cold_out,
                    "q_hot_kw": q_hot,
                    "q_cold_kw": q_cold,
                    "hot_outlet_pressure_pa": hot_pressure,
                    "cold_outlet_pressure_pa": cold_pressure,
                    "hot_outlet_mass_flow_kg_s": hot_flow,
                    "cold_outlet_mass_flow_kg_s": cold_flow,
                    "hx_1_heat_load_kw": q_hot,
                    "hx_1_logarithmic_mean_temperature_difference_lmtd_k": lmtd,
                    "energy_proxy_noisy_kw": q_hot + rng.normal(0.0, max(1.5, 0.012 * q_hot)),
                }
            )

    synthetic = pd.DataFrame(rows)
    synthetic = synthetic.sort_values(["trajectory_id", "step_index"]).reset_index(drop=True)
    return synthetic


def ensure_synthetic_pilstm_dataset(
    output_path: Path = DEFAULT_SYNTHETIC_PATH,
    n_trajectories: int = DEFAULT_SYNTHETIC_TRAJECTORIES,
    steps_per_trajectory: int = DEFAULT_SYNTHETIC_STEPS_PER_TRAJECTORY,
    random_state: int = 42,
    force: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path

    synthetic = generate_synthetic_pilstm_dataset(
        n_trajectories=n_trajectories,
        steps_per_trajectory=steps_per_trajectory,
        random_state=random_state,
    )
    synthetic.to_csv(output_path, index=False)
    return output_path


def create_grouped_sequence_dataset(
    pilstm,
    frame: pd.DataFrame,
    group_column: str = "trajectory_id",
    order_column: str = "step_index",
) -> tuple[np.ndarray, np.ndarray]:
    if group_column not in frame.columns:
        X, y = pilstm.prepare_data(frame)
        return pilstm.create_sequences(X, y)

    grouped = frame.sort_values([group_column, order_column]).groupby(group_column, sort=False)
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for _, group in grouped:
        X, y = pilstm.prepare_data(group)
        X_seq, y_seq = pilstm.create_sequences(X, y)
        if len(X_seq) == 0:
            continue
        X_parts.append(X_seq)
        y_parts.append(y_seq)

    if not X_parts:
        return np.empty((0, pilstm.sequence_length, 0)), np.empty((0, 2))

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)