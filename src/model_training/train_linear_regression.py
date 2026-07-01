import joblib
import numpy as np
import pandas as pd
from src import config
from src.features_engineer import (
    load_data,
    remove_exact_duplicates,
    engineer_features,
    split_target_features,
    split_train_test,
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import FunctionTransformer

def build_preprocessor():

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
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
    print("[STARTING] Building linear regression pipeline...")
    preprocessor = build_preprocessor()
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])
    print("[COMPLETED] Linear Regression pipeline")
    return model_pipeline

def train_model(model_pipeline, X_train, y_train):
    print("[STARTING] Training Linear Regression model...")
    model_pipeline.fit(X_train, y_train)
    print("[COMPLETED] Trained Linear Regression model")
    return model_pipeline

def evaluate_model(model_pipeline, X_test, y_test):
    y_pred = model_pipeline.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        "r2": r2_score(y_test, y_pred),
    }

    errors = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred,
    })

    errors["error"] = errors["predicted"] - errors["actual"]
    errors["absolute_error"] = errors["error"].abs()

    print("\nLinear Regression Prediction diagnostics:")
    print("Min predicted price:", round(errors["predicted"].min(), 2))
    print("Max predicted price:", round(errors["predicted"].max(), 2))
    print("Negative predictions:", int((errors["predicted"] < 0).sum()))

    print("\nLinear Regression Worst prediction errors:")
    print(
        errors
        .sort_values("absolute_error", ascending=False)
        .head(10)
        .to_string()
    )

    print("\nLinear Regression Evaluation metrics:")
    print("MAE:", round(metrics["mae"], 2))
    print("RMSE:", round(metrics["rmse"], 2))
    print("R2:", round(metrics["r2"], 4))

    return y_pred, metrics

def main():
    print(f"\n{'=' * 60}")
    print(" train_linear_regression.py ")
    print(f"{'=' * 60}")

    print("[STARTING] Loading and preparing data...")
    df = load_data()
    df = remove_exact_duplicates(df)
    print("[COMPLETED] Initial setup")
    df = engineer_features(df)
    X, y = split_target_features(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model_pipeline = build_model_pipeline()
    model_pipeline = train_model(model_pipeline, X_train, y_train)
    y_pred, metrics = evaluate_model(model_pipeline, X_test, y_test)

if __name__ == "__main__":
    main()