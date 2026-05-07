from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


DEFAULT_CONFIG = {
    "random_state": 42,
    "test_size": 0.20,
    "validation_fraction_of_train": 0.20,
    "cold_inlet_temperature_k": 293.15,
    "assumed_hot_mass_flow_kg_s": 1.0,
    "cp_hot_kj_kgk": 4.18,
    "cp_cold_kj_kgk": 4.18,
    "rho_hot_kg_m3": 997.0,
    "rho_cold_kg_m3": 997.0,
    "mu_hot_pa_s": 0.0010,
    "mu_cold_pa_s": 0.0010,
    "k_hot_w_mk": 0.60,
    "k_cold_w_mk": 0.60,
}


FLUID_PRESETS = {
    "Water": {
        "cp_kj_kgk": 4.18,
        "rho_kg_m3": 997.0,
        "mu_pa_s": 0.0010,
        "k_w_mk": 0.60,
    },
    "Oil": {
        "cp_kj_kgk": 2.10,
        "rho_kg_m3": 870.0,
        "mu_pa_s": 0.0800,
        "k_w_mk": 0.145,
    },
    "Ethylene Glycol": {
        "cp_kj_kgk": 2.42,
        "rho_kg_m3": 1110.0,
        "mu_pa_s": 0.0160,
        "k_w_mk": 0.258,
    },
    "Refrigerant": {
        "cp_kj_kgk": 1.42,
        "rho_kg_m3": 1200.0,
        "mu_pa_s": 0.0003,
        "k_w_mk": 0.080,
    },
}


class QuantileClipper:
    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "QuantileClipper":
        X_arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.quantile(X_arr, self.lower_q, axis=0)
        self.upper_bounds_ = np.quantile(X_arr, self.upper_q, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class StandardScalerScratch:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScalerScratch":
        X_arr = np.asarray(X, dtype=float)
        self.mean_ = X_arr.mean(axis=0)
        self.std_ = X_arr.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class LinearRegressionGD:
    def __init__(self, lr: float = 0.03, epochs: int = 1200, l2: float = 0.0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w: np.ndarray | None = None
        self.b = 0.0
        self.history_: list[tuple[int, float]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n_samples, n_features = X_arr.shape
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.history_ = []

        for epoch in range(self.epochs):
            y_pred = X_arr @ self.w + self.b
            error = y_pred - y_arr

            grad_w = (2.0 / n_samples) * (X_arr.T @ error) + 2.0 * self.l2 * self.w
            grad_b = (2.0 / n_samples) * np.sum(error)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            if epoch % 25 == 0 or epoch == self.epochs - 1:
                loss = np.mean(error**2) + self.l2 * np.sum(self.w**2)
                self.history_.append((epoch, float(loss)))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return X_arr @ self.w + self.b


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def r2_score_scratch(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    return float(1.0 - (ss_res / (ss_tot + 1e-12)))


def mape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(
        np.mean(np.abs((y_true_arr - y_pred_arr) / (np.abs(y_true_arr) + 1e-12))) * 100.0
    )


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score_scratch(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def train_test_split_scratch(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    test_count = int(len(X) * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
    )


def apply_fluid_to_config(
    base_config: dict[str, float],
    hot_props: dict[str, float],
    cold_props: dict[str, float],
) -> dict[str, float]:
    config = dict(base_config)
    config["cp_hot_kj_kgk"] = hot_props["cp_kj_kgk"]
    config["rho_hot_kg_m3"] = hot_props["rho_kg_m3"]
    config["mu_hot_pa_s"] = hot_props["mu_pa_s"]
    config["k_hot_w_mk"] = hot_props["k_w_mk"]
    config["cp_cold_kj_kgk"] = cold_props["cp_kj_kgk"]
    config["rho_cold_kg_m3"] = cold_props["rho_kg_m3"]
    config["mu_cold_pa_s"] = cold_props["mu_pa_s"]
    config["k_cold_w_mk"] = cold_props["k_w_mk"]
    return config


def build_feature_frame(df_in: pd.DataFrame, config: dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame(index=df_in.index)
    cold_inlet_temperature_k = config["cold_inlet_temperature_k"]
    hot_mass_flow_kg_s = config["assumed_hot_mass_flow_kg_s"]

    cp_hot_kj_kgk = config["cp_hot_kj_kgk"]
    cp_cold_kj_kgk = config["cp_cold_kj_kgk"]
    cp_hot_j_kgk = cp_hot_kj_kgk * 1000.0
    cp_cold_j_kgk = cp_cold_kj_kgk * 1000.0

    rho_hot = config["rho_hot_kg_m3"]
    rho_cold = config["rho_cold_kg_m3"]
    mu_hot = config["mu_hot_pa_s"]
    mu_cold = config["mu_cold_pa_s"]
    k_hot = config["k_hot_w_mk"]
    k_cold = config["k_cold_w_mk"]

    out["hot_inlet_temperature_k"] = df_in["hot_inlet_temperature_k"]
    out["hot_inlet_temperature_k_noisy"] = df_in["hot_inlet_temperature_k_noisy"]
    out["cold_inlet_mass_flow_kg_s"] = df_in["cold_inlet_mass_flow_kg_s"]
    out["cold_inlet_mass_flow_kg_s_noisy"] = df_in["cold_inlet_mass_flow_kg_s_noisy"]
    out["cold_inlet_temperature_k_assumed"] = cold_inlet_temperature_k
    out["hot_mass_flow_kg_s_assumed"] = hot_mass_flow_kg_s

    out["hot_inlet_sensor_error_k"] = (
        out["hot_inlet_temperature_k_noisy"] - out["hot_inlet_temperature_k"]
    )
    out["cold_flow_sensor_error_kg_s"] = (
        out["cold_inlet_mass_flow_kg_s_noisy"] - out["cold_inlet_mass_flow_kg_s"]
    )

    out["delta_t_in_clean_k"] = out["hot_inlet_temperature_k"] - cold_inlet_temperature_k
    out["delta_t_in_noisy_k"] = (
        out["hot_inlet_temperature_k_noisy"] - cold_inlet_temperature_k
    )

    out["cp_hot_kj_kgk"] = cp_hot_kj_kgk
    out["cp_cold_kj_kgk"] = cp_cold_kj_kgk
    out["rho_hot_kg_m3"] = rho_hot
    out["rho_cold_kg_m3"] = rho_cold
    out["mu_hot_pa_s"] = mu_hot
    out["mu_cold_pa_s"] = mu_cold
    out["k_hot_w_mk"] = k_hot
    out["k_cold_w_mk"] = k_cold

    out["capacity_rate_hot_kw_per_k"] = hot_mass_flow_kg_s * cp_hot_kj_kgk
    out["capacity_rate_cold_clean_kw_per_k"] = (
        out["cold_inlet_mass_flow_kg_s"] * cp_cold_kj_kgk
    )
    out["capacity_rate_cold_noisy_kw_per_k"] = (
        out["cold_inlet_mass_flow_kg_s_noisy"] * cp_cold_kj_kgk
    )

    out["capacity_rate_ratio_clean"] = (
        out["capacity_rate_cold_clean_kw_per_k"] / out["capacity_rate_hot_kw_per_k"]
    )
    out["capacity_rate_ratio_noisy"] = (
        out["capacity_rate_cold_noisy_kw_per_k"] / out["capacity_rate_hot_kw_per_k"]
    )

    out["capacity_rate_min_clean"] = np.minimum(
        out["capacity_rate_hot_kw_per_k"], out["capacity_rate_cold_clean_kw_per_k"]
    )
    out["capacity_rate_max_clean"] = np.maximum(
        out["capacity_rate_hot_kw_per_k"], out["capacity_rate_cold_clean_kw_per_k"]
    )
    out["capacity_rate_min_noisy"] = np.minimum(
        out["capacity_rate_hot_kw_per_k"], out["capacity_rate_cold_noisy_kw_per_k"]
    )
    out["capacity_rate_max_noisy"] = np.maximum(
        out["capacity_rate_hot_kw_per_k"], out["capacity_rate_cold_noisy_kw_per_k"]
    )

    out["energy_proxy_clean_kw"] = (
        out["capacity_rate_cold_clean_kw_per_k"] * out["delta_t_in_clean_k"]
    )
    out["energy_proxy_noisy_kw"] = (
        out["capacity_rate_cold_noisy_kw_per_k"] * out["delta_t_in_noisy_k"]
    )
    out["normalized_flow_ratio_clean"] = (
        out["cold_inlet_mass_flow_kg_s"] / hot_mass_flow_kg_s
    )
    out["normalized_flow_ratio_noisy"] = (
        out["cold_inlet_mass_flow_kg_s_noisy"] / hot_mass_flow_kg_s
    )

    out["thermal_diffusivity_hot_m2_s"] = k_hot / (rho_hot * cp_hot_j_kgk)
    out["thermal_diffusivity_cold_m2_s"] = k_cold / (rho_cold * cp_cold_j_kgk)
    out["prandtl_hot"] = mu_hot * cp_hot_j_kgk / k_hot
    out["prandtl_cold"] = mu_cold * cp_cold_j_kgk / k_cold

    out["delta_t_in_clean_sq"] = out["delta_t_in_clean_k"] ** 2
    out["delta_t_in_noisy_sq"] = out["delta_t_in_noisy_k"] ** 2
    out["flow_x_delta_clean"] = out["cold_inlet_mass_flow_kg_s"] * out["delta_t_in_clean_k"]
    out["flow_x_delta_noisy"] = (
        out["cold_inlet_mass_flow_kg_s_noisy"] * out["delta_t_in_noisy_k"]
    )
    out["temp_flow_interaction_noisy"] = (
        out["hot_inlet_temperature_k_noisy"] * out["cold_inlet_mass_flow_kg_s_noisy"]
    )
    return out


@dataclass
class TrainedTargetModel:
    target_name: str
    model: LinearRegressionGD
    clipper: QuantileClipper
    scaler: StandardScalerScratch
    feature_columns: list[str]
    metrics: dict[str, float]

    def predict_from_frame(self, feature_frame: pd.DataFrame) -> np.ndarray:
        X = feature_frame[self.feature_columns].values
        X_processed = self.scaler.transform(self.clipper.transform(X))
        return self.model.predict(X_processed)


def train_best_linear_model(
    dataset_path: str,
    target_name: str,
    config: dict[str, float] | None = None,
) -> tuple[TrainedTargetModel, pd.DataFrame]:
    run_config = dict(DEFAULT_CONFIG if config is None else config)
    df = pd.read_csv(dataset_path).drop_duplicates().reset_index(drop=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    feature_frame = build_feature_frame(df, run_config)
    X_train, X_test, y_train, y_test = train_test_split_scratch(
        feature_frame,
        df[target_name],
        test_size=run_config["test_size"],
        random_state=run_config["random_state"],
    )

    clipper = QuantileClipper(lower_q=0.01, upper_q=0.99)
    scaler = StandardScalerScratch()

    X_train_processed = scaler.fit_transform(clipper.fit_transform(X_train.values))
    X_test_processed = scaler.transform(clipper.transform(X_test.values))

    model = LinearRegressionGD(lr=0.03, epochs=1200, l2=0.0)
    model.fit(X_train_processed, y_train.values)
    test_predictions = model.predict(X_test_processed)

    metrics = regression_metrics(y_test.values, test_predictions)
    trained_model = TrainedTargetModel(
        target_name=target_name,
        model=model,
        clipper=clipper,
        scaler=scaler,
        feature_columns=feature_frame.columns.tolist(),
        metrics=metrics,
    )

    prediction_frame = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": test_predictions,
            "Residual": y_test.values - test_predictions,
        }
    )
    return trained_model, prediction_frame


def build_single_scenario_frame(
    hot_inlet_temperature_k: float,
    hot_inlet_temperature_k_noisy: float,
    hot_mass_flow_kg_s: float,
    cold_inlet_mass_flow_kg_s: float,
    cold_inlet_mass_flow_kg_s_noisy: float,
    config: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hot_inlet_temperature_k": [hot_inlet_temperature_k],
            "hot_inlet_temperature_k_noisy": [hot_inlet_temperature_k_noisy],
            "hot_outlet_mass_flow_kg_s": [hot_mass_flow_kg_s],
            "cold_inlet_mass_flow_kg_s": [cold_inlet_mass_flow_kg_s],
            "cold_inlet_mass_flow_kg_s_noisy": [cold_inlet_mass_flow_kg_s_noisy],
        }
    )


def simulate_physics_summary(
    hot_inlet_temperature_k: float,
    hot_inlet_temperature_k_noisy: float,
    hot_mass_flow_kg_s: float,
    cold_inlet_mass_flow_kg_s: float,
    cold_inlet_mass_flow_kg_s_noisy: float,
    hot_props: dict[str, float],
    cold_props: dict[str, float],
    config: dict[str, float] | None = None,
) -> dict[str, float]:
    run_config = apply_fluid_to_config(dict(DEFAULT_CONFIG if config is None else config), hot_props, cold_props)
    run_config["assumed_hot_mass_flow_kg_s"] = hot_mass_flow_kg_s
    df = build_single_scenario_frame(
        hot_inlet_temperature_k=hot_inlet_temperature_k,
        hot_inlet_temperature_k_noisy=hot_inlet_temperature_k_noisy,
        hot_mass_flow_kg_s=hot_mass_flow_kg_s,
        cold_inlet_mass_flow_kg_s=cold_inlet_mass_flow_kg_s,
        cold_inlet_mass_flow_kg_s_noisy=cold_inlet_mass_flow_kg_s_noisy,
        config=run_config,
    )
    features = build_feature_frame(df, run_config).iloc[0]

    thermal_response_factor = (
        (hot_props["k_w_mk"] / cold_props["k_w_mk"])
        * (cold_props["cp_kj_kgk"] / hot_props["cp_kj_kgk"])
        * (cold_props["rho_kg_m3"] / hot_props["rho_kg_m3"]) ** 0.15
        * (hot_props["mu_pa_s"] / cold_props["mu_pa_s"]) ** 0.08
    )

    return {
        "delta_t_in_clean_k": float(features["delta_t_in_clean_k"]),
        "delta_t_in_noisy_k": float(features["delta_t_in_noisy_k"]),
        "capacity_rate_hot_kw_per_k": float(features["capacity_rate_hot_kw_per_k"]),
        "capacity_rate_cold_noisy_kw_per_k": float(features["capacity_rate_cold_noisy_kw_per_k"]),
        "capacity_rate_ratio_noisy": float(features["capacity_rate_ratio_noisy"]),
        "energy_proxy_noisy_kw": float(features["energy_proxy_noisy_kw"]),
        "prandtl_hot": float(features["prandtl_hot"]),
        "prandtl_cold": float(features["prandtl_cold"]),
        "thermal_response_factor": float(thermal_response_factor),
        "effectiveness_proxy": float(
            min(0.98, max(0.05, 0.18 + 0.09 * np.log1p(features["capacity_rate_ratio_noisy"] + 1.0)
            + 0.0025 * features["delta_t_in_noisy_k"] + 0.015 * thermal_response_factor))
        ),
    }


def predict_scenario(
    artifact: dict,
    hot_inlet_temperature_k: float,
    hot_inlet_temperature_k_noisy: float,
    hot_mass_flow_kg_s: float,
    cold_inlet_temperature_k: float,
    cold_inlet_mass_flow_kg_s: float,
    cold_inlet_mass_flow_kg_s_noisy: float,
    hot_props: dict[str, float],
    cold_props: dict[str, float],
) -> dict[str, float]:
    config = apply_fluid_to_config(dict(artifact["config"]), hot_props, cold_props)
    config["assumed_hot_mass_flow_kg_s"] = hot_mass_flow_kg_s
    scenario_df = build_single_scenario_frame(
        hot_inlet_temperature_k=hot_inlet_temperature_k,
        hot_inlet_temperature_k_noisy=hot_inlet_temperature_k_noisy,
        hot_mass_flow_kg_s=hot_mass_flow_kg_s,
        cold_inlet_mass_flow_kg_s=cold_inlet_mass_flow_kg_s,
        cold_inlet_mass_flow_kg_s_noisy=cold_inlet_mass_flow_kg_s_noisy,
        config=config,
    )
    scenario_features = build_feature_frame(scenario_df, config)

    heat_load_ml = float(
        artifact["models"]["hx_1_heat_load_kw"].predict_from_frame(scenario_features)[0]
    )
    hot_outlet_ml = float(
        artifact["models"]["hot_outlet_temperature_k"].predict_from_frame(scenario_features)[0]
    )
    cold_outlet_ml = float(
        artifact["models"]["cold_outlet_temperature_k"].predict_from_frame(scenario_features)[0]
    )

    physics = simulate_physics_summary(
        hot_inlet_temperature_k=hot_inlet_temperature_k,
        hot_inlet_temperature_k_noisy=hot_inlet_temperature_k_noisy,
        hot_mass_flow_kg_s=hot_mass_flow_kg_s,
        cold_inlet_mass_flow_kg_s=cold_inlet_mass_flow_kg_s,
        cold_inlet_mass_flow_kg_s_noisy=cold_inlet_mass_flow_kg_s_noisy,
        hot_props=hot_props,
        cold_props=cold_props,
        config=config,
    )

    baseline_hot_cp = DEFAULT_CONFIG["cp_hot_kj_kgk"]
    baseline_cold_cp = DEFAULT_CONFIG["cp_cold_kj_kgk"]
    cp_adjustment = 0.55 * (
        (cold_props["cp_kj_kgk"] / baseline_cold_cp)
        / max(0.35, hot_props["cp_kj_kgk"] / baseline_hot_cp)
    )
    conductivity_adjustment = 0.25 * (
        cold_props["k_w_mk"] / max(0.05, DEFAULT_CONFIG["k_cold_w_mk"])
    )
    viscosity_penalty = 0.18 * (
        DEFAULT_CONFIG["mu_cold_pa_s"] / max(1e-5, cold_props["mu_pa_s"])
    )

    hybrid_factor = max(
        0.45,
        min(
            1.85,
            0.55
            + cp_adjustment
            + conductivity_adjustment
            + viscosity_penalty
            + 0.15 * physics["thermal_response_factor"],
        ),
    )

    heat_load_hybrid = heat_load_ml * hybrid_factor
    hot_capacity_rate = physics["capacity_rate_hot_kw_per_k"]
    hot_temp_drop_hybrid = heat_load_hybrid / max(hot_capacity_rate, 1e-8)
    hot_outlet_hybrid = hot_inlet_temperature_k - hot_temp_drop_hybrid
    
    # Hybrid prediction for cold outlet using energy balance
    cold_capacity_rate = physics["capacity_rate_cold_noisy_kw_per_k"]
    cold_temp_rise_hybrid = heat_load_hybrid / max(cold_capacity_rate, 1e-8)
    cold_outlet_hybrid = cold_inlet_temperature_k + cold_temp_rise_hybrid

    return {
        "predicted_heat_load_kw_ml": heat_load_ml,
        "predicted_hot_outlet_k_ml": hot_outlet_ml,
        "predicted_cold_outlet_k_ml": cold_outlet_ml,
        "predicted_heat_load_kw_hybrid": float(heat_load_hybrid),
        "predicted_hot_outlet_k_hybrid": float(hot_outlet_hybrid),
        "predicted_cold_outlet_k_hybrid": float(cold_outlet_hybrid),
        "cold_inlet_temperature_k": cold_inlet_temperature_k,
        **physics,
        "hybrid_factor": float(hybrid_factor),
    }
