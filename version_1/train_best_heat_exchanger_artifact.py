from pathlib import Path

import joblib
import pandas as pd

from heat_exchanger_best_model import (
    DEFAULT_CONFIG,
    FLUID_PRESETS,
    train_best_linear_model,
)


APP_DIR = Path(__file__).resolve().parent
DATASET_PATH = APP_DIR / "heat_exchanger_dataset.csv"
ARTIFACT_PATH = APP_DIR / "best_heat_exchanger_models.joblib"
METRICS_PATH = APP_DIR / "best_model_metrics.csv"
HEAT_LOAD_PREDICTIONS_PATH = APP_DIR / "best_model_heat_load_predictions.csv"
HOT_OUTLET_PREDICTIONS_PATH = APP_DIR / "best_model_hot_outlet_predictions.csv"


def main() -> None:
    heat_load_model, heat_load_predictions = train_best_linear_model(
        dataset_path=str(DATASET_PATH),
        target_name="hx_1_heat_load_kw",
        config=DEFAULT_CONFIG,
    )
    hot_outlet_model, hot_outlet_predictions = train_best_linear_model(
        dataset_path=str(DATASET_PATH),
        target_name="hot_outlet_temperature_k",
        config=DEFAULT_CONFIG,
    )
    cold_outlet_model, cold_outlet_predictions = train_best_linear_model(
        dataset_path=str(DATASET_PATH),
        target_name="cold_outlet_temperature_k",
        config=DEFAULT_CONFIG,
    )

    metrics_df = pd.DataFrame(
        [
            {"Target": heat_load_model.target_name, **heat_load_model.metrics},
            {"Target": hot_outlet_model.target_name, **hot_outlet_model.metrics},
            {"Target": cold_outlet_model.target_name, **cold_outlet_model.metrics},
        ]
    )

    artifact = {
        "artifact_name": "Heat Exchanger Best Models",
        "best_model_family": "LinearRegressionGD",
        "selection_basis": (
            "Chosen from notebook results shared by the user where LinearRegressionGD "
            "ranked first by RMSE for both research targets."
        ),
        "config": dict(DEFAULT_CONFIG),
        "fluid_presets": FLUID_PRESETS,
        "models": {
            heat_load_model.target_name: heat_load_model,
            hot_outlet_model.target_name: hot_outlet_model,
            cold_outlet_model.target_name: cold_outlet_model,
        },
        "metrics": metrics_df.to_dict(orient="records"),
    }

    joblib.dump(artifact, ARTIFACT_PATH)
    metrics_df.to_csv(METRICS_PATH, index=False)
    heat_load_predictions.to_csv(HEAT_LOAD_PREDICTIONS_PATH, index=False)
    hot_outlet_predictions.to_csv(HOT_OUTLET_PREDICTIONS_PATH, index=False)
    cold_outlet_predictions.to_csv(APP_DIR / "cold_outlet_predictions.csv", index=False)

    print(f"Saved artifact: {ARTIFACT_PATH.resolve()}")
    print(f"Saved metrics: {METRICS_PATH.resolve()}")
    print(f"Saved heat-load predictions: {HEAT_LOAD_PREDICTIONS_PATH.resolve()}")
    print(f"Saved hot-outlet predictions: {HOT_OUTLET_PREDICTIONS_PATH.resolve()}")
    print(f"Saved cold-outlet predictions: {(APP_DIR / 'cold_outlet_predictions.csv').resolve()}")
    print("\nMetrics summary:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
