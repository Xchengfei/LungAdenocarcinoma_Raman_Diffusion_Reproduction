from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from augmented_classification.metrics import compute_metrics


@dataclass(frozen=True)
class ModelSpec:
    name: str
    param_grid: list[dict[str, object]]
    builder: object
    refit_on_train_val: bool = False


def _build_svm(params: dict[str, object], seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel=str(params["kernel"]),
                    C=float(params["C"]),
                    gamma=params["gamma"],
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def _build_random_forest(params: dict[str, object], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def _build_mlp(params: dict[str, object], seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=params["hidden_layers"],
                    activation="relu",
                    solver="adam",
                    alpha=float(params["alpha"]),
                    learning_rate_init=float(params["learning_rate_init"]),
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=seed,
                ),
            ),
        ]
    )


def _build_pca_svm(params: dict[str, object], seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=int(params["pca_components"]), random_state=seed)),
            (
                "svm",
                SVC(
                    kernel=str(params["kernel"]),
                    C=float(params["C"]),
                    gamma=params["gamma"],
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def model_specs(max_pca_components: int) -> list[ModelSpec]:
    svm_params = [
        {"kernel": "linear", "C": 1.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 0.1, "gamma": "scale"},
        {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 1.0, "gamma": 0.01},
    ]
    rf_params = [
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 4, "max_features": "log2"},
    ]
    mlp_params = [
        {"hidden_layers": (32,), "alpha": 0.001, "learning_rate_init": 0.001},
        {"hidden_layers": (64,), "alpha": 0.001, "learning_rate_init": 0.001},
        {"hidden_layers": (64, 32), "alpha": 0.01, "learning_rate_init": 0.001},
        {"hidden_layers": (128, 64), "alpha": 0.01, "learning_rate_init": 0.0001},
    ]
    component_candidates = [n for n in [5, 10, 20, 30, 50, 80, 100] if n < max_pca_components]
    component_candidates = component_candidates[:3]
    pca_svm_params = [
        {"pca_components": n_components, "kernel": "linear", "C": 1.0, "gamma": "scale"}
        for n_components in component_candidates
    ] + [
        {"pca_components": n_components, "kernel": "rbf", "C": 1.0, "gamma": "scale"}
        for n_components in component_candidates
    ]
    return [
        ModelSpec("svm", svm_params, _build_svm, refit_on_train_val=True),
        ModelSpec("random_forest", rf_params, _build_random_forest, refit_on_train_val=False),
        ModelSpec("mlp", mlp_params, _build_mlp, refit_on_train_val=False),
        ModelSpec("pca_svm", pca_svm_params, _build_pca_svm, refit_on_train_val=True),
    ]


def predict_probability(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    decision = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-decision))


def select_best_model(
    spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[object, dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    best_model = None
    best_row: dict[str, object] | None = None

    for params in spec.param_grid:
        model = spec.builder(params, seed)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        y_prob = predict_probability(model, x_val)
        metrics = compute_metrics(y_val, y_pred, y_prob)
        row = {**params, **{f"val_{key}": value for key, value in metrics.items()}}
        rows.append(row)

        if best_row is None:
            best_model = model
            best_row = row
            continue
        current_score = (float(row["val_auc"]), float(row["val_accuracy"]), float(row["val_f1"]))
        best_score = (float(best_row["val_auc"]), float(best_row["val_accuracy"]), float(best_row["val_f1"]))
        if current_score > best_score:
            best_model = model
            best_row = row

    if best_model is None or best_row is None:
        raise RuntimeError(f"{spec.name} grid search did not train any model")

    best_params = {key: best_row[key] for key in spec.param_grid[0] if key in best_row}
    return best_model, best_params, pd.DataFrame(rows)


def refit_if_needed(
    spec: ModelSpec,
    best_params: dict[str, object],
    selected_model: object,
    x_train_val: np.ndarray,
    y_train_val: np.ndarray,
    seed: int,
) -> object:
    if not spec.refit_on_train_val:
        return selected_model
    final_model = spec.builder(best_params, seed)
    final_model.fit(x_train_val, y_train_val)
    return final_model
