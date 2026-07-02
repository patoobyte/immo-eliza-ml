import numpy as np
from src import config
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_validate
from src.features_engineer import prepare_cv_data
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
    train_mae_scores = -results["train_mae"]
    test_mae_scores = -results["test_mae"]

    train_rmse_scores = -results["train_rmse"]
    test_rmse_scores = -results["test_rmse"]

    train_r2_scores = results["train_r2"]
    test_r2_scores = results["test_r2"]

    mae_gap = test_mae_scores.mean() - train_mae_scores.mean()
    rmse_gap = test_rmse_scores.mean() - train_rmse_scores.mean()
    r2_gap = train_r2_scores.mean() - test_r2_scores.mean()

    print("=" * 60)
    print(f"{MODEL} cross-validation results")
    print("=" * 60)

    print("Train MAE scores:", np.round(train_mae_scores, 2))
    print("Test MAE scores:", np.round(test_mae_scores, 2))

    print("Train RMSE scores:", np.round(train_rmse_scores, 2))
    print("Test RMSE scores:", np.round(test_rmse_scores, 2))

    print("Train R2 scores:", np.round(train_r2_scores, 4))
    print("Test R2 scores:", np.round(test_r2_scores, 4))

    print(
        "\nTrain MAE: {} (+/- {})".format(
            config.format_euros(train_mae_scores.mean()),
            config.format_euros(train_mae_scores.std()),
        )
    )
    print(
        "Test MAE: {} (+/- {})".format(
            config.format_euros(test_mae_scores.mean()),
            config.format_euros(test_mae_scores.std()),
        )
    )
    print("MAE gap:", config.format_euros(mae_gap))

    print(
        "\nTrain RMSE: {} (+/- {})".format(
            config.format_euros(train_rmse_scores.mean()),
            config.format_euros(train_rmse_scores.std()),
        )
    )
    print(
        "Test RMSE: {} (+/- {})".format(
            config.format_euros(test_rmse_scores.mean()),
            config.format_euros(test_rmse_scores.std()),
        )
    )
    print("RMSE gap:", config.format_euros(rmse_gap))

    print(
        "\nTrain R2: {:.4f} (+/- {:.4f})".format(
            train_r2_scores.mean(),
            train_r2_scores.std(),
        )
    )
    print(
        "Test R2: {:.4f} (+/- {:.4f})".format(
            test_r2_scores.mean(),
            test_r2_scores.std(),
        )
    )
    print("R2 gap:", round(r2_gap, 4))

def main():
    print(f"\n{'=' * 60}")
    print(" cross_validation.py ")
    print(f"{'=' * 60}")

    print("[STARTING] Loading and preparing data...")
    X, y = prepare_cv_data()
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
        return_train_score=True,
        n_jobs=-1,
    )
    print("[COMPLETED] Cross-validation")

    print_cv_results(results)


if __name__ == "__main__":
    main()