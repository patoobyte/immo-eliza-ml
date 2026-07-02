import numpy as np

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_validate

from src.features_engineer import (
    load_data,
    remove_exact_duplicates,
    engineer_features,
    split_target_features,
)
from src.model_training.train_xgboost import build_model_pipeline


MODEL = "XGBoost"


def build_log_target_model():
    base_pipeline = build_model_pipeline()

    return TransformedTargetRegressor(
        regressor=base_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def print_cv_results(results):
    mae_scores = -results["test_mae"]
    rmse_scores = -results["test_rmse"]
    r2_scores = results["test_r2"]

    print(f"\n{MODEL} cross-validation results")
    print("=" * 60)

    print("MAE scores:", np.round(mae_scores, 2))
    print("RMSE scores:", np.round(rmse_scores, 2))
    print("R2 scores:", np.round(r2_scores, 4))

    print(
        "\nMAE: {:.2f} (+/- {:.2f})".format(
            mae_scores.mean(),
            mae_scores.std(),
        )
    )
    print(
        "RMSE: {:.2f} (+/- {:.2f})".format(
            rmse_scores.mean(),
            rmse_scores.std(),
        )
    )
    print(
        "R2: {:.4f} (+/- {:.4f})".format(
            r2_scores.mean(),
            r2_scores.std(),
        )
    )


def main():
    print(f"\n{'=' * 60}")
    print(" cross_validation.py ")
    print(f"{'=' * 60}")

    print("[STARTING] Loading and preparing data...")
    df = load_data()
    df = remove_exact_duplicates(df)
    df = engineer_features(df)
    X, y = split_target_features(df)
    print("[COMPLETED] Data preparation")

    model = build_log_target_model()

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    print(f"[STARTING] {MODEL} 5-fold cross-validation...")
    results = cross_validate(
        model,
        X,
        y,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
    )
    print("[COMPLETED] Cross-validation")

    print_cv_results(results)


if __name__ == "__main__":
    main()