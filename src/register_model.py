import mlflow
from mlflow import MlflowClient


TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "sales-demand-prediction"
MODEL_NAME = "SalesDemandRF"


def register_best_model():

    # 1. Configure MLflow tracking and registry
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    client = MlflowClient()

    # 2. Find our experiment
    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise ValueError("Experiment not found!")

    # 3. Retrieve completed runs, ranked by test RMSE
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.test_rmse ASC"],
        max_results=100
    )

    if not runs:
        raise ValueError("No completed runs found!")

    best_run = runs[0]

    print("Best run ID:", best_run.info.run_id)
    print("Best test RMSE:", best_run.data.metrics["test_rmse"])
    print("Selected parameters:", best_run.data.params)

    # 4. Construct model URI
    model_uri = (
        f"runs:/{best_run.info.run_id}/model"
    )

    # 5. Register model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    # 6. Assign candidate alias
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="candidate",
        version=model_version.version
    )

    # 7. Load registered model to verify
    registered_model_uri = (
        f"models:/{MODEL_NAME}@candidate"
    )

    model = mlflow.sklearn.load_model(
        registered_model_uri
    )

    print("\nMODEL REGISTRATION SUCCESSFUL")
    print("Model name:", MODEL_NAME)
    print("Model version:", model_version.version)
    print("Model alias: candidate")
    print("Loaded model:", type(model))


if __name__ == "__main__":
    register_best_model()