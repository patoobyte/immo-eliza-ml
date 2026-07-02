import joblib
import numpy as np
import pandas as pd
from src import config
from src.features_engineer import prepare_training_data

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import FunctionTransformer

MODEL = "XGBoost"

def build_preprocessor():

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    binary_zero_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    binary_unknown_pipeline = Pipeline(steps=[
        ("to_object", FunctionTransformer(lambda X: X.astype("object").replace({pd.NA: np.nan}))),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("to_string", FunctionTransformer(lambda X: X.astype(str))),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("to_object", FunctionTransformer(lambda X: X.astype("object").replace({pd.NA: np.nan}))),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("to_string", FunctionTransformer(lambda X: X.astype(str))),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.NUMERIC_FEATURES),
            ("binary_zero", binary_zero_pipeline, config.BINARY_MISSING_AS_ZERO),
            ("binary_unknown", binary_unknown_pipeline, config.BINARY_MISSING_AS_UNKNOWN),
            ("categorical", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ]
    )

    print("Pipeline preprocessing complete")
    return preprocessor

def build_model_pipeline():
    print(f"[STARTING] Building {MODEL} pipeline...")
    preprocessor = build_preprocessor()
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=10,
            subsample=0.9,
            colsample_bytree=1.0,
            objective="reg:squarederror",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            min_child_weight=10,
            reg_alpha=1,
            reg_lambda=10,
            gamma=0.01
        ))
    ])
    print(f"[COMPLETED] {MODEL} pipeline")
    return model_pipeline

def train_model(model_pipeline, X_train, y_train):
    print(f"[STARTING] Training {MODEL} model...")
    y_train_log = np.log1p(y_train)
    model_pipeline.fit(X_train, y_train_log)
    print(f"[COMPLETED] Training {MODEL} model")
    return model_pipeline

def compute_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }

def print_metrics(label, metrics):
    print(f"\n{MODEL} {label} metrics:")
    print("MAE:", config.format_euros(round(metrics["mae"], 2)))
    print("RMSE:", config.format_euros(round(metrics["rmse"], 2)))
    print("R2:", round(metrics["r2"], 4))

def print_prediction_diagnostics(label, y_true, y_pred):
    errors = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
    })

    errors["error"] = errors["predicted"] - errors["actual"]
    errors["absolute_error"] = errors["error"].abs()

    print(f"\n {MODEL} {label} prediction diagnostics:")
    print("Min predicted price:", config.format_euros(errors["predicted"].min()))
    print("Max predicted price:", config.format_euros(errors["predicted"].max()))
    print("Negative predictions:", int((errors["predicted"] < 0).sum()))

    pd.set_option("display.float_format", "{:,.0f}".format)

    print(f"\n{MODEL} {label} worst prediction errors:")
    worst_errors = (
        errors
        .sort_values("absolute_error", ascending=False)
        .head(10)
        .copy()
    )
    for col in ["actual", "predicted", "error", "absolute_error"]:
        worst_errors[col] = worst_errors[col].map(config.format_euros)

    print(worst_errors.to_string())

def evaluate_model(model_pipeline, X_train, y_train, X_test, y_test):
    y_train_pred_log = model_pipeline.predict(X_train)
    y_train_pred = np.expm1(y_train_pred_log)

    y_test_pred_log = model_pipeline.predict(X_test)
    y_test_pred = np.expm1(y_test_pred_log)

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    print_metrics("Train", train_metrics)
    print_metrics("Test", test_metrics)
    print_overfitting(train_metrics, test_metrics)

    print_prediction_diagnostics("Test", y_test, y_test_pred)

    return y_test_pred, {
        "train": train_metrics,
        "test": test_metrics,
    }

def print_overfitting(train_metrics, test_metrics):
    mae_gap = test_metrics["mae"] - train_metrics["mae"]
    rmse_gap = test_metrics["rmse"] - train_metrics["rmse"]
    r2_gap = train_metrics["r2"] - test_metrics["r2"]

    print("\nOverfitting check:")
    print("MAE gap:", f"{mae_gap:,.0f} €")
    print("RMSE gap:", f"{rmse_gap:,.0f} €")
    print("R2 gap:", round(r2_gap, 4))

def main():
    print(f"\n{'=' * 60}")
    print(" train_xgboost.py ")
    print(f"{'=' * 60}")

    print("[STARTING] Loading and preparing data...")
    X_train, X_test, y_train, y_test = prepare_training_data()
    model_pipeline = build_model_pipeline()
    model_pipeline = train_model(model_pipeline, X_train, y_train)
    y_pred, metrics = evaluate_model(model_pipeline, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()