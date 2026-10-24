import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import joblib
import json
import dagshub

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# 1. Initialize remote MLflow tracking through DagsHub
dagshub.init(
    repo_owner="tanmoyghosh704-lang",
    repo_name="Synthetic_data_mlops_practice",
    mlflow=True,
    dvc=False
)

print("MLflow Tracking URI:", mlflow.get_tracking_uri())


def train_model():

    # 2. Load configuration
    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    params = config["train"]

    # 3. Load training and testing datasets
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df.drop(columns=["units"])
    y_train = train_df["units"]

    X_test = test_df.drop(columns=["units"])
    y_test = test_df["units"]

    # 4. Select MLflow experiment
    mlflow.set_experiment("sales-demand-prediction")

    # 5. Start MLflow run
    with mlflow.start_run(run_name="random-forest-baseline"):

        # Log hyperparameters
        mlflow.log_params(params)

        # 6. Initialize model
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=params["random_state"],
            n_jobs=-1
        )

        # 7. Train model
        model.fit(X_train, y_train)

        # 8. Generate predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # 9. Calculate metrics
        metrics = {
            "train_rmse": mean_squared_error(
                y_train, train_pred
            ) ** 0.5,

            "test_rmse": mean_squared_error(
                y_test, test_pred
            ) ** 0.5,

            "train_mae": mean_absolute_error(
                y_train, train_pred
            ),

            "test_mae": mean_absolute_error(
                y_test, test_pred
            ),

            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred)
        }

        # 10. Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # 11. Save trained model locally
        model_path = Path("models/random_forest.joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_path)

        # 12. Log model to remote MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        # 13. Save metrics locally for DVC
        metrics_path = Path("metrics/train_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metrics_path, "w") as file:
            json.dump(metrics, file, indent=4)

        print("MODEL TRAINING COMPLETED")
        print(metrics)


if __name__ == "__main__":
    train_model()