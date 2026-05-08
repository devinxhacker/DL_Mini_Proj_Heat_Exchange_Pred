from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

ROOT_DIR = Path(__file__).resolve().parent.parent
VERSION1_DIR = ROOT_DIR / "version_1"
if str(VERSION1_DIR) not in sys.path:
    sys.path.insert(0, str(VERSION1_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from heat_exchanger_best_model import (
    DEFAULT_CONFIG,
    FLUID_PRESETS,
    TrainedTargetModel,
)
from physics_informed_lstm import PhysicsInformedLSTM
from version_2.build_low_data_subsets import DEFAULT_DATASET, build_low_data_subset
from version_2.plain_deep_models import (
    build_mlp_sequence_model,
    build_vanilla_lstm_model,
    create_sequence_dataset,
    save_deep_artifact,
    scale_sequence_dataset,
)
from version_2.synthetic_pilstm_data import (
    create_grouped_sequence_dataset,
    ensure_synthetic_pilstm_dataset,
)
from version_2.scratch_experiment import build_version1_feature_frame, run_experiment
from version_2.study_utils import (
    DATA_DIR,
    COLD_OUTLET_TARGET,
    HOT_OUTLET_TARGET,
    PRIMARY_TARGET,
    RESEARCH_TARGETS,
    SECONDARY_TARGET,
    TARGET_LABELS,
    add_rmse_ranks,
    compute_metrics,
    ensure_results_dir,
    metric_row,
    prediction_rows,
    summarize_best_models,
)


RANDOM_STATE = 42
DEFAULT_SUBSET_SIZES = [100, 125]
PILSTM_SEQUENCE_LENGTH = 10
PILSTM_LSTM_UNITS = 128
PILSTM_LEARNING_RATE = 0.001
PILSTM_EPOCHS = 1000
PILSTM_BATCH_SIZE = 32
PILSTM_TEST_FRACTION = 0.15
PILSTM_VAL_FRACTION_OF_REMAINING = 0.176
PILSTM_PHYSICS_WEIGHT = 10.0
PILSTM_SYNTHETIC_PRETRAIN_EPOCHS = 150
PILSTM_SYNTHETIC_PRETRAIN_BATCH_SIZE = 64
PILSTM_SYNTHETIC_PRETRAIN_PATH = DATA_DIR / "synthetic_training_data.csv"

MODEL_FAMILY = {
    "MeanBaseline": "Reference baseline",
    "LinearRegressionGD": "Traditional ML",
    "LinearSVR": "Traditional ML",
    "KNN": "Traditional ML",
    "DecisionTree": "Traditional ML",
    "RandomForest": "Traditional ML",
    "GradientBoosting": "Traditional ML",
    "MLP": "Plain deep learning",
    "VanillaLSTM": "Plain sequence deep learning",
    "PI-LSTM": "Sequence-aware physics-informed deep learning",
}


def clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.drop_duplicates().reset_index(drop=True).copy()
    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned.dropna().reset_index(drop=True)


def ensure_subset_paths(subset_files: list[str]) -> list[Path]:
    if subset_files:
        return [Path(path) for path in subset_files]
    return [DATA_DIR / f"low_data_{size}.csv" for size in DEFAULT_SUBSET_SIZES]


def create_missing_subsets(subset_paths: list[Path]) -> None:
    missing_paths = [path for path in subset_paths if not path.exists()]
    if not missing_paths:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    full_df = pd.read_csv(DEFAULT_DATASET)
    for missing_path in missing_paths:
        subset_size = int(missing_path.stem.split("_")[-1])
        subset_df, summary_df = build_low_data_subset(
            df=full_df,
            target_size=subset_size,
            temp_column="hot_inlet_temperature_k",
            flow_column="cold_inlet_mass_flow_kg_s",
            temp_bins=10,
            random_state=RANDOM_STATE,
        )
        subset_df.to_csv(missing_path, index=False)
        summary_df.to_csv(DATA_DIR / f"low_data_{subset_size}_summary.csv", index=False)


def best_metric_record(target: str, metrics_df: pd.DataFrame) -> dict[str, object]:
    best_row = metrics_df.sort_values(["RMSE", "MAE"], ascending=[True, True]).iloc[0]
    return {
        "Target": target,
        "Model": best_row["Model"],
        "MAE": float(best_row["MAE"]),
        "RMSE": float(best_row["RMSE"]),
        "R2": float(best_row["R2"]),
        "MAPE": float(best_row["MAPE"]),
    }


def build_traditional_artifact(
    subset_name: str,
    subset_size: int,
    best_models: dict[str, TrainedTargetModel],
    metric_records: list[dict[str, object]],
) -> dict[str, object]:
    best_family = {row["Target"]: row["Model"] for row in metric_records}
    return {
        "artifact_name": f"Version 2 Low-Data Best Models ({subset_name})",
        "best_model_family": best_family,
        "selection_basis": (
            "Chosen from the same Version 1 from-scratch model search, but retrained on the "
            f"{subset_size}-row low-data subset while preserving temperature-range coverage."
        ),
        "config": dict(DEFAULT_CONFIG),
        "fluid_presets": FLUID_PRESETS,
        "subset_name": subset_name,
        "subset_size": subset_size,
        "models": best_models,
        "metrics": metric_records,
    }


def evaluate_traditional_models(
    subset_df: pd.DataFrame,
    subset_name: str,
    results_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    subset_size = len(subset_df)
    feature_frame = build_version1_feature_frame(subset_df, DEFAULT_CONFIG)

    metric_rows: list[dict[str, object]] = []
    prediction_output_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    best_models: dict[str, TrainedTargetModel] = {}
    artifact_metric_records: list[dict[str, object]] = []

    for target in RESEARCH_TARGETS:
        experiment = run_experiment(feature_frame, subset_df, target, DEFAULT_CONFIG)
        metrics_df = experiment["metrics_df"].copy()
        predictions_df = experiment["predictions_df"].copy()
        split_counts = experiment["split_counts"]
        x_test_original = experiment["X_test_original"].reset_index(drop=True)
        y_test = np.asarray(experiment["y_test"], dtype=float)

        for tune_row in experiment["tuning_df"].to_dict(orient="records"):
            tuning_rows.append(
                {
                    "subset_name": subset_name,
                    "subset_size": subset_size,
                    **tune_row,
                }
            )

        for _, row in metrics_df.iterrows():
            model_name = str(row["Model"])
            metric_rows.append(
                metric_row(
                    subset_name=subset_name,
                    subset_size=subset_size,
                    target=target,
                    model_name=model_name,
                    model_family=MODEL_FAMILY.get(model_name, "Traditional ML"),
                    split_strategy="Version 1 random holdout split with validation tuning",
                    n_train_rows=split_counts["train"],
                    n_val_rows=split_counts["val"],
                    n_test_rows=split_counts["test"],
                    metrics={
                        "MAE": float(row["MAE"]),
                        "RMSE": float(row["RMSE"]),
                        "R2": float(row["R2"]),
                        "MAPE": float(row["MAPE"]),
                    },
                    best_params=str(row["Best Params"]),
                    notes=(
                        "Same from-scratch feature engineering and model search space as Version 1; "
                        "only the training data has been reduced."
                    ),
                )
            )

            prediction_output_rows.extend(
                prediction_rows(
                    subset_name=subset_name,
                    subset_size=subset_size,
                    target=target,
                    model_name=model_name,
                    frame=x_test_original,
                    actual_values=y_test,
                    predictions=predictions_df[f"{model_name}_Predicted"].to_numpy(),
                )
            )

        artifact_payload = experiment["artifact"]
        best_row = metrics_df.iloc[0]
        best_models[target] = TrainedTargetModel(
            target_name=target,
            model=artifact_payload["best_model_object"],
            clipper=artifact_payload["clipper"],
            scaler=artifact_payload["scaler"],
            feature_columns=artifact_payload["feature_columns"],
            metrics={
                "MAE": float(best_row["MAE"]),
                "RMSE": float(best_row["RMSE"]),
                "R2": float(best_row["R2"]),
                "MAPE": float(best_row["MAPE"]),
            },
        )
        artifact_metric_records.append(best_metric_record(target, metrics_df))

    artifact = build_traditional_artifact(
        subset_name=subset_name,
        subset_size=subset_size,
        best_models=best_models,
        metric_records=artifact_metric_records,
    )
    joblib.dump(artifact, results_dir / f"{subset_name}_best_models.joblib")
    return metric_rows, prediction_output_rows, tuning_rows


def split_sequence_data(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_total = len(X_seq)
    n_test = max(1, int(round(n_total * PILSTM_TEST_FRACTION)))
    test_start = n_total - n_test

    X_temp = X_seq[:test_start]
    y_temp = y_seq[:test_start]
    X_test = X_seq[test_start:]
    y_test = y_seq[test_start:]

    n_val = max(1, int(round(len(X_temp) * PILSTM_VAL_FRACTION_OF_REMAINING)))
    val_start = len(X_temp) - n_val

    X_train = X_temp[:val_start]
    y_train = y_temp[:val_start]
    X_val = X_temp[val_start:]
    y_val = y_temp[val_start:]
    return X_train, y_train, X_val, y_val, X_test, y_test


_PRETRAINED_PILSTM_WEIGHTS_PATH: Path | None = None


def ensure_pretrained_pilstm_weights(results_dir: Path) -> Path | None:
    global _PRETRAINED_PILSTM_WEIGHTS_PATH
    if _PRETRAINED_PILSTM_WEIGHTS_PATH is not None and _PRETRAINED_PILSTM_WEIGHTS_PATH.exists():
        return _PRETRAINED_PILSTM_WEIGHTS_PATH
    if not TENSORFLOW_AVAILABLE:
        return None

    synthetic_path = ensure_synthetic_pilstm_dataset(PILSTM_SYNTHETIC_PRETRAIN_PATH, force=True)
    synthetic_df = clean_numeric_frame(pd.read_csv(synthetic_path))

    pretrain_model = PhysicsInformedLSTM(
        sequence_length=PILSTM_SEQUENCE_LENGTH,
        lstm_units=PILSTM_LSTM_UNITS,
        learning_rate=PILSTM_LEARNING_RATE,
        use_hard_energy_balance=True,
    )
    pretrain_model.physics_weight = PILSTM_PHYSICS_WEIGHT

    X_seq, y_seq = create_grouped_sequence_dataset(pretrain_model, synthetic_df)
    if len(X_seq) < 20:
        return None

    X_train, y_train, X_val, y_val, _, _ = split_sequence_data(X_seq, y_seq)
    if min(len(X_train), len(X_val)) <= 0:
        return None

    pretrain_model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    pretrain_model.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=PILSTM_SYNTHETIC_PRETRAIN_EPOCHS,
        batch_size=PILSTM_SYNTHETIC_PRETRAIN_BATCH_SIZE,
        verbose=0,
    )

    weights_path = results_dir / "pilstm_synthetic_pretrained.weights.h5"
    pretrain_model.model.save_weights(weights_path)
    _PRETRAINED_PILSTM_WEIGHTS_PATH = weights_path
    return weights_path


def evaluate_pilstm(
    subset_df: pd.DataFrame,
    subset_name: str,
    results_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not TENSORFLOW_AVAILABLE:
        return [], []

    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    tf.keras.backend.clear_session()

    subset_size = len(subset_df)
    pilstm = PhysicsInformedLSTM(
        sequence_length=PILSTM_SEQUENCE_LENGTH,
        lstm_units=PILSTM_LSTM_UNITS,
        learning_rate=PILSTM_LEARNING_RATE,
        use_hard_energy_balance=True,
    )

    # increase physics penalty to more strongly enforce energy conservation
    pilstm.physics_weight = PILSTM_PHYSICS_WEIGHT

    X, y = pilstm.prepare_data(subset_df)
    X_seq, y_seq = pilstm.create_sequences(X, y)
    if len(X_seq) < 10:
        return [], []

    X_train, y_train, X_val, y_val, X_test, y_test = split_sequence_data(X_seq, y_seq)
    if min(len(X_train), len(X_val), len(X_test)) <= 0:
        return [], []

    pilstm.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    pretrained_weights = ensure_pretrained_pilstm_weights(results_dir)
    if pretrained_weights is not None and pretrained_weights.exists():
        try:
            pilstm.model.load_weights(str(pretrained_weights))
        except Exception:
            pass
    pilstm.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=PILSTM_EPOCHS,
        batch_size=PILSTM_BATCH_SIZE,
        verbose=1,
    )

    val_pred = pilstm.predict(X_val)
    val_hot_pred = val_pred[:, 0]
    val_hot_true = y_val[:, 0]
    if len(val_hot_pred) >= 2:
        calibration_slope, calibration_intercept = np.polyfit(val_hot_pred, val_hot_true, 1)
    else:
        calibration_slope, calibration_intercept = 1.0, 0.0

    y_pred = pilstm.predict(X_test)
    hot_pred = y_pred[:, 0] * calibration_slope + calibration_intercept
    test_last_step = X_test[:, -1, :]
    hot_inlet = test_last_step[:, 0]
    cold_inlet = test_last_step[:, 1]
    cold_flow = test_last_step[:, 2]
    hot_flow = test_last_step[:, 6]
    q_hot = hot_flow * pilstm.cp_hot * (hot_inlet - hot_pred)
    cold_pred = q_hot / (cold_flow * pilstm.cp_cold + 1e-12) + cold_inlet
    hot_metrics = compute_metrics(y_test[:, 0], hot_pred)
    cold_metrics = compute_metrics(y_test[:, 1], cold_pred)

    weights_path = results_dir / f"{subset_name}_pilstm.weights.h5"
    pilstm.model.save_weights(weights_path)

    # Use the appropriate scaler based on learning mode
    if pilstm.residual_learning:
        scaler_y_mean = pilstm.scaler_residual.mean_
        scaler_y_scale = pilstm.scaler_residual.scale_
    else:
        scaler_y_mean = pilstm.scaler_y.mean_
        scaler_y_scale = pilstm.scaler_y.scale_

    artifact = {
        "subset_name": subset_name,
        "subset_size": subset_size,
        "sequence_length": pilstm.sequence_length,
        "lstm_units": pilstm.lstm_units,
        "learning_rate": pilstm.learning_rate,
        "residual_learning": pilstm.residual_learning,
        "use_bidirectional": pilstm.use_bidirectional,
        "scaler_X_mean": pilstm.scaler_X.mean_,
        "scaler_X_scale": pilstm.scaler_X.scale_,
        "scaler_y_mean": scaler_y_mean,
        "scaler_y_scale": scaler_y_scale,
        "hot_calibration_slope": float(calibration_slope),
        "hot_calibration_intercept": float(calibration_intercept),
        "input_features": [
            "hot_inlet_temperature_k",
            "cold_inlet_temperature_k",
            "cold_inlet_mass_flow_kg_s",
            "hx_1_heat_load_kw",
            "hot_outlet_pressure_pa",
            "cold_outlet_pressure_pa",
            "hot_outlet_mass_flow_kg_s",
            "cold_outlet_mass_flow_kg_s",
            "hx_1_logarithmic_mean_temperature_difference_lmtd_k",
        ],
        "weights_file": weights_path.name,
        "metrics": {
            "hot_outlet": hot_metrics,
            "cold_outlet": cold_metrics,
        },
        "notes": (
            "PI-LSTM mirrors the Version 1 sequence model setup, retrained on the low-data subset. "
            "Cold outlet remains diagnostic only because the source dataset keeps that column unrealistic."
        ),
    }
    joblib.dump(artifact, results_dir / f"{subset_name}_pilstm_artifact.joblib")

    sequence_target_frame = subset_df.iloc[PILSTM_SEQUENCE_LENGTH:].reset_index(drop=True)
    test_frame = sequence_target_frame.iloc[-len(X_test):].reset_index(drop=True)

    metric_rows = [
        metric_row(
            subset_name=subset_name,
            subset_size=subset_size,
            target=HOT_OUTLET_TARGET,
            model_name="PI-LSTM",
            model_family=MODEL_FAMILY["PI-LSTM"],
            split_strategy="Version 1 PI-LSTM sequential split after sequence creation",
            n_train_rows=len(X_train),
            n_val_rows=len(X_val),
            n_test_rows=len(X_test),
            metrics=hot_metrics,
            notes=artifact["notes"],
            best_params=(
                f"sequence_length={PILSTM_SEQUENCE_LENGTH}, lstm_units={PILSTM_LSTM_UNITS}, "
                f"epochs<={PILSTM_EPOCHS}, batch_size={PILSTM_BATCH_SIZE}"
            ),
        )
    ]

    prediction_output_rows = prediction_rows(
        subset_name=subset_name,
        subset_size=subset_size,
        target=HOT_OUTLET_TARGET,
        model_name="PI-LSTM",
        frame=test_frame,
        actual_values=y_test[:, 0],
        predictions=hot_pred,
    )
    return metric_rows, prediction_output_rows


def evaluate_plain_deep_models(
    subset_df: pd.DataFrame,
    subset_name: str,
    results_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not TENSORFLOW_AVAILABLE:
        return [], []

    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    subset_size = len(subset_df)
    X_seq, y_seq, eval_frame = create_sequence_dataset(
        subset_df,
        sequence_length=PILSTM_SEQUENCE_LENGTH,
    )
    if len(X_seq) < 10:
        return [], []

    X_train, y_train, X_val, y_val, X_test, y_test = split_sequence_data(X_seq, y_seq)
    if min(len(X_train), len(X_val), len(X_test)) <= 0:
        return [], []

    scaled = scale_sequence_dataset(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
    )
    test_frame = eval_frame.iloc[-len(X_test) :].reset_index(drop=True)

    model_specs = [
        {
            "model_name": "MLP",
            "builder": lambda: build_mlp_sequence_model(PILSTM_SEQUENCE_LENGTH, X_train.shape[2]),
            "train_inputs": scaled["X_train_scaled"].reshape(len(X_train), -1),
            "val_inputs": scaled["X_val_scaled"].reshape(len(X_val), -1),
            "test_inputs": scaled["X_test_scaled"].reshape(len(X_test), -1),
            "best_params": (
                f"sequence_length={PILSTM_SEQUENCE_LENGTH}, hidden_layers=[64,32], batch_size=8"
            ),
            "notes": (
                "Plain dense network on flattened low-data sequence windows. It sees the same recent operating window "
                "as the sequence models, but without recurrent memory or explicit physics guidance."
            ),
            "artifact_name": "mlp_sequence",
            "architecture": {"hidden_layers": [64, 32], "dropout": [0.25, 0.15]},
        },
        {
            "model_name": "VanillaLSTM",
            "builder": lambda: build_vanilla_lstm_model(PILSTM_SEQUENCE_LENGTH, X_train.shape[2]),
            "train_inputs": scaled["X_train_scaled"],
            "val_inputs": scaled["X_val_scaled"],
            "test_inputs": scaled["X_test_scaled"],
            "best_params": (
                f"sequence_length={PILSTM_SEQUENCE_LENGTH}, lstm_units=[32,16], batch_size=8"
            ),
            "notes": (
                "Plain LSTM baseline on the same low-data operating windows. This model adds sequence memory, "
                "but it does not include any physics-aware regularization."
            ),
            "artifact_name": "vanilla_lstm",
            "architecture": {"lstm_units": [32, 16], "dropout": [0.2, 0.2]},
        },
    ]

    metric_rows: list[dict[str, object]] = []
    prediction_output_rows: list[dict[str, object]] = []

    for spec in model_specs:
        tf.keras.backend.clear_session()
        model = spec["builder"]()
        callback_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
            )
        ]
        model.fit(
            spec["train_inputs"],
            scaled["y_train_scaled"],
            validation_data=(spec["val_inputs"], scaled["y_val_scaled"]),
            epochs=PILSTM_EPOCHS,
            batch_size=8,
            verbose=0,
            callbacks=callback_list,
        )

        # Avoid retracing warnings by using direct forward pass in eager mode
        pred_scaled = model(spec["test_inputs"], training=False).numpy()
        predictions = scaled["scaler_y"].inverse_transform(pred_scaled).reshape(-1)
        metrics = compute_metrics(y_test, predictions)

        artifact_path = results_dir / f"{subset_name}_{spec['artifact_name']}_artifact.joblib"
        weights_path = results_dir / f"{subset_name}_{spec['artifact_name']}.weights.h5"
        save_deep_artifact(
            artifact_path=artifact_path,
            weights_path=weights_path,
            model_type=spec["artifact_name"],
            model=model,
            scaler_X=scaled["scaler_X"],
            scaler_y=scaled["scaler_y"],
            sequence_length=PILSTM_SEQUENCE_LENGTH,
            metrics=metrics,
            notes=spec["notes"],
            architecture=spec["architecture"],
        )

        metric_rows.append(
            metric_row(
                subset_name=subset_name,
                subset_size=subset_size,
                target=HOT_OUTLET_TARGET,
                model_name=spec["model_name"],
                model_family=MODEL_FAMILY[spec["model_name"]],
                split_strategy="Shared low-data sequential split for deep hot-outlet comparison",
                n_train_rows=len(X_train),
                n_val_rows=len(X_val),
                n_test_rows=len(X_test),
                metrics=metrics,
                notes=spec["notes"],
                best_params=spec["best_params"],
            )
        )
        prediction_output_rows.extend(
            prediction_rows(
                subset_name=subset_name,
                subset_size=subset_size,
                target=HOT_OUTLET_TARGET,
                model_name=spec["model_name"],
                frame=test_frame,
                actual_values=y_test,
                predictions=predictions,
            )
        )

    return metric_rows, prediction_output_rows


def evaluate_subset(
    subset_path: Path,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset_name = subset_path.stem
    subset_df = clean_numeric_frame(pd.read_csv(subset_path))

    traditional_metric_rows, traditional_predictions, tuning_rows = evaluate_traditional_models(
        subset_df=subset_df,
        subset_name=subset_name,
        results_dir=results_dir,
    )
    deep_metric_rows, deep_predictions = evaluate_plain_deep_models(
        subset_df=subset_df,
        subset_name=subset_name,
        results_dir=results_dir,
    )
    pilstm_metric_rows, pilstm_predictions = evaluate_pilstm(
        subset_df=subset_df,
        subset_name=subset_name,
        results_dir=results_dir,
    )

    all_metric_rows = traditional_metric_rows + deep_metric_rows + pilstm_metric_rows
    all_prediction_rows = traditional_predictions + deep_predictions + pilstm_predictions
    return (
        pd.DataFrame(all_metric_rows),
        pd.DataFrame(all_prediction_rows),
        pd.DataFrame(tuning_rows),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Version 2 low-data study using the Version 1 model flow."
    )
    parser.add_argument(
        "--subset-files",
        nargs="*",
        default=[],
        help="Optional explicit subset CSV files. Defaults to the 100-row and 125-row low-data subsets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = ensure_results_dir()
    subset_paths = ensure_subset_paths(args.subset_files)
    create_missing_subsets(subset_paths)

    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_tuning: list[pd.DataFrame] = []

    for subset_path in subset_paths:
        metrics_df, predictions_df, tuning_df = evaluate_subset(subset_path, results_dir)
        all_metrics.append(metrics_df)
        all_predictions.append(predictions_df)
        all_tuning.append(tuning_df)

    metrics_output = add_rmse_ranks(pd.concat(all_metrics, ignore_index=True))
    predictions_output = pd.concat(all_predictions, ignore_index=True)
    tuning_output = pd.concat(all_tuning, ignore_index=True)
    best_models_output = summarize_best_models(metrics_output)

    metrics_output.to_csv(results_dir / "low_data_metrics.csv", index=False)
    predictions_output.to_csv(results_dir / "low_data_predictions.csv", index=False)
    tuning_output.to_csv(results_dir / "low_data_tuning.csv", index=False)
    best_models_output.to_csv(results_dir / "low_data_best_models.csv", index=False)

    print("Version 2 low-data study completed.")
    print(metrics_output.to_string(index=False))
    print()
    print("Best model summary:")
    print(best_models_output.to_string(index=False))


if __name__ == "__main__":
    main()
