from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
VERSION1_DIR = ROOT_DIR / "version_1"
if str(VERSION1_DIR) not in sys.path:
    sys.path.insert(0, str(VERSION1_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from heat_exchanger_best_model import regression_metrics


VERSION2_DIR = Path(__file__).resolve().parent
DATA_DIR = VERSION2_DIR / "data"
RESULTS_DIR = VERSION2_DIR / "results"

PRIMARY_TARGET = "hx_1_heat_load_kw"
SECONDARY_TARGET = "hot_outlet_temperature_k"
COLD_OUTLET_TARGET = "cold_outlet_temperature_k"
RESEARCH_TARGETS = [PRIMARY_TARGET, SECONDARY_TARGET, COLD_OUTLET_TARGET]
HOT_OUTLET_TARGET = "hot_outlet_temperature_k"

TEMPERATURE_COLUMN = "hot_inlet_temperature_k"
FLOW_COLUMN = "cold_inlet_mass_flow_kg_s"

TARGET_LABELS = {
    PRIMARY_TARGET: "Heat load",
    SECONDARY_TARGET: "Hot outlet temperature",
    COLD_OUTLET_TARGET: "Cold outlet temperature",
}

TARGET_UNITS = {
    PRIMARY_TARGET: "kW",
    SECONDARY_TARGET: "K",
    COLD_OUTLET_TARGET: "K",
}


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return regression_metrics(y_true, y_pred)


def metric_row(
    subset_name: str,
    subset_size: int,
    target: str,
    model_name: str,
    model_family: str,
    split_strategy: str,
    n_train_rows: int,
    n_val_rows: int,
    n_test_rows: int,
    metrics: dict[str, float],
    notes: str,
    best_params: str | None = None,
) -> dict[str, object]:
    return {
        "subset_name": subset_name,
        "subset_size": subset_size,
        "target": target,
        "target_label": TARGET_LABELS.get(target, target),
        "target_unit": TARGET_UNITS.get(target, ""),
        "model": model_name,
        "family": model_family,
        "split_strategy": split_strategy,
        "n_train_rows": n_train_rows,
        "n_val_rows": n_val_rows,
        "n_test_rows": n_test_rows,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "MAPE": metrics["MAPE"],
        "accuracy_proxy": 100.0 - metrics["MAPE"],
        "best_params": best_params if best_params is not None else "",
        "notes": notes,
    }


def prediction_rows(
    subset_name: str,
    subset_size: int,
    target: str,
    model_name: str,
    frame: pd.DataFrame,
    actual_values: np.ndarray,
    predictions: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    actual_arr = np.asarray(actual_values, dtype=float)
    pred_arr = np.asarray(predictions, dtype=float)
    for idx, (actual, predicted) in enumerate(zip(actual_arr, pred_arr)):
        rows.append(
            {
                "subset_name": subset_name,
                "subset_size": subset_size,
                "target": target,
                "target_label": TARGET_LABELS.get(target, target),
                "target_unit": TARGET_UNITS.get(target, ""),
                "model": model_name,
                "point_index": idx,
                TEMPERATURE_COLUMN: float(frame[TEMPERATURE_COLUMN].iloc[idx]),
                FLOW_COLUMN: float(frame[FLOW_COLUMN].iloc[idx]),
                "actual_value": float(actual),
                "predicted_value": float(predicted),
                "error": float(actual - predicted),
            }
        )
    return rows


def add_rmse_ranks(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics_df.copy()
    ranked["rank_by_rmse"] = (
        ranked.groupby(["subset_name", "target"])["RMSE"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    return ranked.sort_values(
        ["subset_size", "target", "rank_by_rmse", "MAE"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def summarize_best_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for (subset_name, target), group in metrics_df.groupby(["subset_name", "target"], sort=True):
        ordered = group.sort_values(["RMSE", "MAE"], ascending=[True, True]).reset_index(drop=True)
        best = ordered.iloc[0]
        pi_rows = ordered.loc[ordered["model"] == "PI-LSTM"]
        pi_rank = None
        if not pi_rows.empty:
            pi_rank = int(pi_rows.index[0] + 1)
        summary_rows.append(
            {
                "subset_name": subset_name,
                "subset_size": int(best["subset_size"]),
                "target": target,
                "target_label": best["target_label"],
                "best_model": best["model"],
                "best_family": best["family"],
                "best_rmse": float(best["RMSE"]),
                "best_mape": float(best["MAPE"]),
                "pilstm_rank_by_rmse": pi_rank,
            }
        )
    return pd.DataFrame(summary_rows).sort_values(
        ["subset_size", "target"],
        ascending=[True, True],
    ).reset_index(drop=True)


def load_sampling_manifest() -> pd.DataFrame | None:
    manifest_path = DATA_DIR / "sampling_manifest.csv"
    if not manifest_path.exists():
        return None
    return pd.read_csv(manifest_path)
