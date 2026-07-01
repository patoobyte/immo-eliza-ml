"""
--------------------------------------
Trainer using Linear Regression Model
--------------------------------------

1. Cleans data
2. Create features
3. Split dataset into train/test
4. Build pipeline
5. Train model
6. Evaluate predictions

"""

## SETUP ## 
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

MODEL = "Linear Regression"

# STEP 1
def build_preprocessor() -> ColumnTransformer:
    """
    This function prepares data using transformers.
    """

    # Handles numeric features
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), # NaN = median of column
        ("scaler", StandardScaler()), # Standardizes
    ])

    # Handles binary features where NaN becomes 0
    binary_zero_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)), # NaN = 0
    ])

    # Handles binary features where NaN is kept
    binary_unknown_pipeline = Pipeline(steps=[
        ("to_object", FunctionTransformer(lambda X: X.astype("object").replace({pd.NA: np.nan}))), # Converts to object type
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), # NaN = "Unknown"
        ("to_string", FunctionTransformer(lambda X: X.astype(str))), # Converts to string
        ("onehot", OneHotEncoder(handle_unknown="ignore")), # Apply one-hot encoding
    ])

    # Handles categporical features (same as above)
    categorical_pipeline = Pipeline(steps=[
        ("to_object", FunctionTransformer(lambda X: X.astype("object").replace({pd.NA: np.nan}))),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("to_string", FunctionTransformer(lambda X: X.astype(str))),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Handles connecting features to the right pipeline
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

# Training step
def train_model(model_pipeline, X_train, y_train):
    print("[STARTING] Training Linear Regression model...")
    model_pipeline.fit(X_train, y_train)
    print("[COMPLETED] Trained Linear Regression model")
    return model_pipeline

def compute_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }

def print_metrics(label, metrics):
    print(f"\n{MODEL} {label} metrics:")
    print("MAE:", round(metrics["mae"], 2))
    print("RMSE:", round(metrics["rmse"], 2))
    print("R2:", round(metrics["r2"], 4))

def print_prediction_diagnostics(label, y_true, y_pred):
    errors = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
    })

    errors["error"] = errors["predicted"] - errors["actual"]
    errors["absolute_error"] = errors["error"].abs()

    print(f"\n {MODEL} {label} prediction diagnostics:")
    print("Min predicted price:", round(errors["predicted"].min(), 2))
    print("Max predicted price:", round(errors["predicted"].max(), 2))
    print("Negative predictions:", int((errors["predicted"] < 0).sum()))

    print(f"\n{MODEL} {label} worst prediction errors:")
    print(
        errors
        .sort_values("absolute_error", ascending=False)
        .head(10)
        .to_string()
    )

def evaluate_model(model_pipeline, X_train, y_train, X_test, y_test):
    y_train_pred = model_pipeline.predict(X_train)
    y_test_pred = model_pipeline.predict(X_test)

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    print_metrics("Train", train_metrics)
    print_metrics("Test", test_metrics)

    print_prediction_diagnostics("Test", y_test, y_test_pred)

    return y_test_pred, {
        "train": train_metrics,
        "test": test_metrics,
    }

def main():
    """
     Main training workflow:
    - Load and clean the dataset
    - Engineer features and separate inputs from target
    - Split data into training and testing sets
    - Build the pipeline 
    - Train the model on training data
    - Evaluate model performance on train/test data
    """

    # Prints header
    print(f"\n{'=' * 60}")
    print(" train_linear_regression.py ")
    print(f"{'=' * 60}")
    print("[STARTING] Loading and preparing data...")

    df = load_data() # Loads the dataset into a dataframe
    df = remove_exact_duplicates(df) # Cleans the dataset
    print("[COMPLETED] Initial setup")

    df = engineer_features(df) # Feature engineering
    X, y = split_target_features(df) # Split target from imput features
    X_train, X_test, y_train, y_test = split_train_test(X, y) # Split dataset to train/test
    model_pipeline = build_model_pipeline() # Creates the pipeline
    model_pipeline = train_model(model_pipeline, X_train, y_train) # Training
    y_pred, metrics = evaluate_model(model_pipeline, X_train, y_train, X_test, y_test) # Tests the trained model and show metrics

if __name__ == "__main__":
    main()