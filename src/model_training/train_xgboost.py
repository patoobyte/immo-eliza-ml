"""
-----------------------------------------
XGBoost Model Trainer
-----------------------------------------

This script trains an XGBoost regression model to predict Belgian property prices by:
    1. Loading and preparing the cleaned dataset via the features pipeline
    2. Building a preprocessing pipeline (numeric, binary, categorical,
       target-encoded and ordinal transformers) followed by the XGBoost model
    3. Training the model on log1p-transformed prices
    4. Evaluating predictions on both train and test sets (expm1 back-transformed)
    5. Printing metrics (MAE, RMSE, R2), overfitting gaps and worst errors

Functions:
    - build_preprocessor()           : builds the ColumnTransformer with all feature pipelines
    - build_model_pipeline()         : assembles preprocessor + XGBRegressor into a single sklearn Pipeline
    - train_model()                  : fits the pipeline on log1p(y_train)
    - compute_metrics()              : returns MAE, RMSE and R2 as a dictionary
    - print_metrics()                : prints formatted train or test metrics
    - print_overfitting()            : prints MAE, RMSE and R2 gaps between train and test
    - print_prediction_diagnostics() : prints price range and the 10 worst prediction errors
    - evaluate_model()               : runs prediction on train and test, calls metrics and diagnostics printers
    - save_model()                   : saves the trained model pipeline to disk with joblib
    - main()                         : orchestrates the full training workflow
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, OrdinalEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import FunctionTransformer

from src import config
from src.features_engineer import prepare_training_data

MODEL = "XGBoost"

# Helpers to avoid using lambda functions
def nan_to_object(X):
    return X.astype("object").replace({pd.NA: np.nan})

def to_str(X):
    return X.astype(str)

# Build preprocessing pipelines for all feature groups
def build_preprocessor():

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    binary_zero_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    binary_unknown_pipeline = Pipeline(steps=[
        ("to_object", FunctionTransformer(nan_to_object)),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("to_string", FunctionTransformer(to_str)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("to_object", FunctionTransformer(nan_to_object)),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("to_string", FunctionTransformer(to_str)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    target_encoding_pipeline = Pipeline(steps=[
        ("to_object", FunctionTransformer(nan_to_object)),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("to_string", FunctionTransformer(to_str)),
        ("target_encoder", TargetEncoder(target_type="continuous", smooth="auto")),
    ])

    ordinal_pipeline = Pipeline(steps=[
        ("to_object", FunctionTransformer(nan_to_object)),
        ("ordinal", OrdinalEncoder(
            categories=list(config.ORDINAL_FEATURES.values()),
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.NUMERIC_FEATURES),
            ("binary_zero", binary_zero_pipeline, config.BINARY_MISSING_AS_ZERO),
            ("binary_unknown", binary_unknown_pipeline, config.BINARY_MISSING_AS_UNKNOWN),
            ("categorical", categorical_pipeline, config.CATEGORICAL_FEATURES),
            ("target_encoded", target_encoding_pipeline, config.TARGET_ENCODED_FEATURES),
            ("ordinal", ordinal_pipeline, list(config.ORDINAL_FEATURES.keys())),
        ]
    )

    print("Pipeline preprocessing complete")
    return preprocessor

# Assemble preprocessing and XGBoost into one sklearn Pipeline
def build_model_pipeline():
    print(f"[STARTING] Building {MODEL} pipeline...")
    preprocessor = build_preprocessor()
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(
            n_estimators=1000, 
            learning_rate=0.02, 
            max_depth=7,  
            subsample=0.8,
            colsample_bytree=0.7, 
            colsample_bynode=0.8, 
            objective="reg:squarederror",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            min_child_weight=20, 
            reg_alpha=1,
            reg_lambda=12, 
            gamma=0.01
        ))
    ])
    print(f"[COMPLETED] {MODEL} pipeline")
    return model_pipeline

# Train XGBoost on log-transformed prices
def train_model(model_pipeline, X_train, y_train):
    print(f"[STARTING] Training {MODEL} model...")
    y_train_log = np.log1p(y_train)
    model_pipeline.fit(X_train, y_train_log)
    print(f"[COMPLETED] Training {MODEL} model")
    return model_pipeline

# Compute regression metrics on real price scale
def compute_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }

# Print formatted metrics
def print_metrics(label, metrics):
    print(f"\n{MODEL} {label} metrics:")
    print("MAE:", config.format_euros(round(metrics["mae"], 2)))
    print("RMSE:", config.format_euros(round(metrics["rmse"], 2)))
    print("R2:", round(metrics["r2"], 4))

# Print prediction range and largest test errors
def print_prediction_diagnostics(label, y_true, y_pred):
    errors = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
    })

    errors["error"] = errors["predicted"] - errors["actual"]
    errors["absolute_error"] = errors["error"].abs()

    print(f"\n{MODEL} {label} prediction diagnostics:")
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

# Evaluate train and test predictions
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

# Compare train and test metrics to check overfitting
def print_overfitting(train_metrics, test_metrics):
    mae_gap = test_metrics["mae"] - train_metrics["mae"]
    rmse_gap = test_metrics["rmse"] - train_metrics["rmse"]
    r2_gap = train_metrics["r2"] - test_metrics["r2"]

    print("\nOverfitting check:")
    print("MAE gap:", f"{mae_gap:,.0f}")
    print("RMSE gap:", f"{rmse_gap:,.0f}")
    print("R2 gap:", round(r2_gap, 4))

# Save the fitted pipeline with joblib
def save_model(model_pipeline):
    model_path = config.MODEL_PATH / "xgboost_pipeline.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_pipeline, model_path)

    print(f"Saved {MODEL} model to: {model_path}")

# Orchestrator : run the full XGBoost training workflow
def main():
    print(f"\n{'=' * 60}")
    print(" train_xgboost.py ")
    print(f"{'=' * 60}")

    print("[STARTING] Loading and preparing data...")
    X_train, X_test, y_train, y_test = prepare_training_data()
    model_pipeline = build_model_pipeline()
    model_pipeline = train_model(model_pipeline, X_train, y_train)
    y_pred, metrics = evaluate_model(model_pipeline, X_train, y_train, X_test, y_test)
    save_model(model_pipeline)

if __name__ == "__main__":
    main()