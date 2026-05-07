from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
VERSION1_DIR = ROOT_DIR / "version_1"
if str(VERSION1_DIR) not in sys.path:
    sys.path.insert(0, str(VERSION1_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from heat_exchanger_best_model import (
    DEFAULT_CONFIG,
    QuantileClipper,
    StandardScalerScratch,
    LinearRegressionGD,
    build_feature_frame,
    regression_metrics,
    rmse,
)


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_fraction_of_train: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    test_count = int(len(X) * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    val_count = int(len(train_idx) * val_fraction_of_train)
    val_idx = train_idx[:val_count]
    train_only_idx = train_idx[val_count:]
    return (
        X.iloc[train_only_idx].reset_index(drop=True),
        X.iloc[val_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_only_idx].reset_index(drop=True),
        y.iloc[val_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
    )


class MeanRegressor:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanRegressor":
        self.mean_ = float(np.mean(y))
        self.history_ = None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(shape=(len(X),), fill_value=self.mean_, dtype=float)


class LinearSVRSubgradient:
    def __init__(self, lr: float = 0.01, epochs: int = 1200, epsilon: float = 0.5, C: float = 1.0, l2: float = 0.0):
        self.lr = lr
        self.epochs = epochs
        self.epsilon = epsilon
        self.C = C
        self.l2 = l2
        self.w = None
        self.b = 0.0
        self.history_: list[tuple[int, float]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVRSubgradient":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n_samples, n_features = X_arr.shape
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.history_ = []
        for epoch in range(self.epochs):
            y_pred = X_arr @ self.w + self.b
            residual = y_pred - y_arr
            abs_res = np.abs(residual)
            active = abs_res > self.epsilon
            sign = np.sign(residual)
            if np.any(active):
                grad_w_loss = self.C * np.mean((sign[active, None] * X_arr[active]), axis=0)
                grad_b_loss = self.C * np.mean(sign[active])
            else:
                grad_w_loss = np.zeros(n_features, dtype=float)
                grad_b_loss = 0.0
            grad_w = grad_w_loss + 2.0 * self.l2 * self.w
            grad_b = grad_b_loss
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            if epoch % 25 == 0 or epoch == self.epochs - 1:
                epsilon_loss = np.maximum(0.0, abs_res - self.epsilon).mean()
                objective = epsilon_loss + self.l2 * np.sum(self.w ** 2)
                self.history_.append((epoch, float(objective)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return X_arr @ self.w + self.b


class KNNRegressorScratch:
    def __init__(self, k: int = 7, weighted: bool = True):
        self.k = k
        self.weighted = weighted
        self.X_train = None
        self.y_train = None
        self.history_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressorScratch":
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        preds = []
        for row in X_arr:
            distances = np.sqrt(np.sum((self.X_train - row) ** 2, axis=1))
            neighbor_idx = np.argsort(distances)[: self.k]
            neighbor_targets = self.y_train[neighbor_idx]
            if self.weighted:
                weights = 1.0 / (distances[neighbor_idx] + 1e-8)
                pred = np.sum(weights * neighbor_targets) / np.sum(weights)
            else:
                pred = np.mean(neighbor_targets)
            preds.append(pred)
        return np.array(preds, dtype=float)


class TreeNode:
    def __init__(self, prediction: float | None = None, feature_index: int | None = None, threshold: float | None = None, left: "TreeNode" | None = None, right: "TreeNode" | None = None):
        self.prediction = prediction
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right


class DecisionTreeRegressorScratch:
    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 20,
        min_samples_leaf: int = 8,
        max_features: int | None = None,
        n_thresholds: int = 18,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state
        self.root: TreeNode | None = None
        self.feature_importance_splits_: np.ndarray | None = None
        self._rng = np.random.default_rng(random_state)
        self.history_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressorScratch":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        self.n_features_ = X_arr.shape[1]
        self.feature_importance_splits_ = np.zeros(self.n_features_, dtype=float)
        self.root = self._grow_tree(X_arr, y_arr, depth=0)
        return self

    def _variance(self, y: np.ndarray) -> float:
        return 0.0 if len(y) == 0 else float(np.var(y))

    def _candidate_features(self) -> np.ndarray:
        all_features = np.arange(self.n_features_)
        if self.max_features is None or self.max_features >= self.n_features_:
            return all_features
        return self._rng.choice(all_features, size=self.max_features, replace=False)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None]:
        best_feature, best_threshold, best_score = None, None, np.inf
        if self._variance(y) <= 1e-12:
            return None, None
        for feature_idx in self._candidate_features():
            column = X[:, feature_idx]
            unique_vals = np.unique(column)
            if len(unique_vals) <= 1:
                continue
            if len(unique_vals) > self.n_thresholds:
                thresholds = np.unique(np.quantile(column, np.linspace(0.05, 0.95, self.n_thresholds)))
            else:
                thresholds = unique_vals[:-1]
            for threshold in thresholds:
                left_mask = column <= threshold
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                y_left, y_right = y[left_mask], y[right_mask]
                weighted_var = (len(y_left) * self._variance(y_left) + len(y_right) * self._variance(y_right)) / len(y)
                if weighted_var < best_score:
                    best_score, best_feature, best_threshold = weighted_var, feature_idx, float(threshold)
        return best_feature, best_threshold

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        node = TreeNode(prediction=float(np.mean(y)))
        if depth >= self.max_depth or len(y) < self.min_samples_split or self._variance(y) <= 1e-12:
            return node
        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            return node
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return node
        self.feature_importance_splits_[feature_idx] += 1.0
        node.feature_index = feature_idx
        node.threshold = threshold
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _predict_row(self, row: np.ndarray, node: TreeNode) -> float:
        if node.feature_index is None:
            return node.prediction
        if row[node.feature_index] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.array([self._predict_row(row, self.root) for row in X_arr], dtype=float)


class RandomForestRegressorScratch:
    def __init__(
        self,
        n_estimators: int = 15,
        max_depth: int = 6,
        min_samples_split: int = 20,
        min_samples_leaf: int = 8,
        max_features: str | int = "sqrt",
        n_thresholds: int = 18,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state
        self.trees: list[DecisionTreeRegressorScratch] = []
        self.feature_importances_ = None
        self.history_ = None

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(self.max_features, int):
            return min(n_features, self.max_features)
        return n_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressorScratch":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n_samples, n_features = X_arr.shape
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        importances = np.zeros(n_features, dtype=float)
        max_features = self._resolve_max_features(n_features)
        for est_idx in range(self.n_estimators):
            bootstrap_idx = rng.choice(np.arange(n_samples), size=n_samples, replace=True)
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                n_thresholds=self.n_thresholds,
                random_state=self.random_state + est_idx,
            )
            tree.fit(X_arr[bootstrap_idx], y_arr[bootstrap_idx])
            self.trees.append(tree)
            importances += tree.feature_importance_splits_
        self.feature_importances_ = importances / importances.sum() if importances.sum() != 0 else importances
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.mean(np.array([tree.predict(X_arr) for tree in self.trees]), axis=0)


class GradientBoostingRegressorScratch:
    def __init__(
        self,
        n_estimators: int = 35,
        learning_rate: float = 0.08,
        max_depth: int = 2,
        min_samples_split: int = 20,
        min_samples_leaf: int = 8,
        n_thresholds: int = 16,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds
        self.random_state = random_state
        self.base_value_ = None
        self.trees_: list[DecisionTreeRegressorScratch] = []
        self.history_: list[tuple[int, float]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressorScratch":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        self.base_value_ = float(np.mean(y_arr))
        current_pred = np.full_like(y_arr, self.base_value_, dtype=float)
        self.trees_, self.history_ = [], []
        for i in range(self.n_estimators):
            residual = y_arr - current_pred
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=None,
                n_thresholds=self.n_thresholds,
                random_state=self.random_state + i,
            )
            tree.fit(X_arr, residual)
            update = tree.predict(X_arr)
            current_pred = current_pred + self.learning_rate * update
            self.trees_.append(tree)
            if i % 2 == 0 or i == self.n_estimators - 1:
                self.history_.append((i, float(np.mean((y_arr - current_pred) ** 2))))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        pred = np.full(shape=(len(X_arr),), fill_value=self.base_value_, dtype=float)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X_arr)
        return pred


MODEL_SEARCH_SPACE = {
    "MeanBaseline": [{}],
    "LinearRegressionGD": [
        {"lr": 0.03, "epochs": 1200, "l2": 0.0},
        {"lr": 0.02, "epochs": 1600, "l2": 1e-4},
    ],
    "LinearSVR": [
        {"lr": 0.01, "epochs": 1000, "epsilon": 0.35, "C": 0.8, "l2": 1e-4},
        {"lr": 0.008, "epochs": 1300, "epsilon": 0.50, "C": 1.0, "l2": 1e-4},
    ],
    "KNN": [
        {"k": 5, "weighted": True},
        {"k": 9, "weighted": True},
        {"k": 11, "weighted": False},
    ],
    "DecisionTree": [
        {"max_depth": 4, "min_samples_split": 25, "min_samples_leaf": 10, "max_features": None, "n_thresholds": 14, "random_state": 42},
        {"max_depth": 6, "min_samples_split": 25, "min_samples_leaf": 10, "max_features": None, "n_thresholds": 18, "random_state": 42},
    ],
    "RandomForest": [
        {"n_estimators": 10, "max_depth": 5, "min_samples_split": 25, "min_samples_leaf": 10, "max_features": "sqrt", "n_thresholds": 14, "random_state": 42},
        {"n_estimators": 16, "max_depth": 6, "min_samples_split": 25, "min_samples_leaf": 10, "max_features": "sqrt", "n_thresholds": 18, "random_state": 42},
    ],
    "GradientBoosting": [
        {"n_estimators": 25, "learning_rate": 0.08, "max_depth": 2, "min_samples_split": 25, "min_samples_leaf": 10, "n_thresholds": 14, "random_state": 42},
        {"n_estimators": 40, "learning_rate": 0.06, "max_depth": 2, "min_samples_split": 25, "min_samples_leaf": 10, "n_thresholds": 16, "random_state": 42},
    ],
}


def build_model(model_name: str, params: dict[str, object]) -> object:
    if model_name == "MeanBaseline":
        return MeanRegressor()
    if model_name == "LinearRegressionGD":
        return LinearRegressionGD(**params)
    if model_name == "LinearSVR":
        return LinearSVRSubgradient(**params)
    if model_name == "KNN":
        return KNNRegressorScratch(**params)
    if model_name == "DecisionTree":
        return DecisionTreeRegressorScratch(**params)
    if model_name == "RandomForest":
        return RandomForestRegressorScratch(**params)
    if model_name == "GradientBoosting":
        return GradientBoostingRegressorScratch(**params)
    raise ValueError(f"Unknown model name: {model_name}")


def prepare_data_for_target(feature_frame: pd.DataFrame, full_df: pd.DataFrame, target_column: str, config: dict[str, float]) -> dict[str, object]:
    X = feature_frame.copy()
    y = full_df[target_column].copy()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X,
        y,
        test_size=config["test_size"],
        val_fraction_of_train=config["validation_fraction_of_train"],
        random_state=config["random_state"],
    )
    clipper = QuantileClipper(lower_q=0.01, upper_q=0.99)
    scaler = StandardScalerScratch()
    X_train_proc = scaler.fit_transform(clipper.fit_transform(X_train.values))
    X_val_proc = scaler.transform(clipper.transform(X_val.values))
    X_test_proc = scaler.transform(clipper.transform(X_test.values))
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_proc": X_train_proc,
        "X_val_proc": X_val_proc,
        "X_test_proc": X_test_proc,
        "clipper": clipper,
        "scaler": scaler,
        "feature_columns": X.columns.tolist(),
    }


def refit_on_train_and_val(model_name: str, best_params: dict[str, object], prep_bundle: dict[str, object]) -> object:
    X_combined = np.vstack([prep_bundle["X_train_proc"], prep_bundle["X_val_proc"]])
    y_combined = np.concatenate([prep_bundle["y_train"].values, prep_bundle["y_val"].values])
    model = build_model(model_name, best_params)
    model.fit(X_combined, y_combined)
    return model


def run_experiment(feature_frame: pd.DataFrame, full_df: pd.DataFrame, target_column: str, config: dict[str, float] | None = None) -> dict[str, object]:
    run_config = dict(DEFAULT_CONFIG if config is None else config)
    prep = prepare_data_for_target(feature_frame, full_df, target_column, run_config)
    all_metrics = []
    tuning_rows = []
    predictions = pd.DataFrame({"Actual": prep["y_test"].values})
    model_objects = {}
    for model_name, candidate_list in MODEL_SEARCH_SPACE.items():
        best_rmse = np.inf
        best_params = None
        for params in candidate_list:
            model = build_model(model_name, params)
            model.fit(prep["X_train_proc"], prep["y_train"].values)
            # call prediction appropriately for sklearn-like vs Keras models
            if hasattr(model, "predict") and not callable(model):
                val_pred = model.predict(prep["X_val_proc"])
            else:
                val_pred = model(prep["X_val_proc"], training=False).numpy()
            val_rmse = rmse(prep["y_val"].values, val_pred)
            tuning_rows.append(
                {
                    "Target": target_column,
                    "Model": model_name,
                    "Params": str(params),
                    "Validation_RMSE": val_rmse,
                }
            )
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_params = params
        final_model = refit_on_train_and_val(model_name, best_params, prep)
        if hasattr(final_model, "predict") and not callable(final_model):
            test_pred = final_model.predict(prep["X_test_proc"])
        else:
            test_pred = final_model(prep["X_test_proc"], training=False).numpy()
        metrics = regression_metrics(prep["y_test"].values, test_pred)
        all_metrics.append({"Target": target_column, "Model": model_name, "Best Params": str(best_params), **metrics})
        predictions[f"{model_name}_Predicted"] = test_pred
        model_objects[model_name] = final_model
    metrics_df = pd.DataFrame(all_metrics).sort_values("RMSE").reset_index(drop=True)
    tuning_df = pd.DataFrame(tuning_rows).sort_values(["Model", "Validation_RMSE"]).reset_index(drop=True)
    best_model_name = metrics_df.iloc[0]["Model"]
    artifact = {
        "target": target_column,
        "best_model_name": best_model_name,
        "best_model_object": model_objects[best_model_name],
        "all_models": model_objects,
        "clipper": prep["clipper"],
        "scaler": prep["scaler"],
        "feature_columns": prep["feature_columns"],
        "config": run_config,
    }
    return {
        "metrics_df": metrics_df,
        "predictions_df": predictions,
        "tuning_df": tuning_df,
        "artifact": artifact,
        "y_test": prep["y_test"].values,
        "X_test_original": prep["X_test"],
        "split_counts": {
            "train": int(len(prep["y_train"])),
            "val": int(len(prep["y_val"])),
            "test": int(len(prep["y_test"])),
        },
    }


def build_version1_feature_frame(df: pd.DataFrame, config: dict[str, float] | None = None) -> pd.DataFrame:
    run_config = dict(DEFAULT_CONFIG if config is None else config)
    return build_feature_frame(df, run_config)
