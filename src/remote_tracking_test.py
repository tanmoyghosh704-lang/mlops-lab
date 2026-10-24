import dagshub
import mlflow

# 1. Initialize DagsHub remote MLflow tracking
dagshub.init(
    repo_owner="tanmoyghosh704-lang",
    repo_name="Synthetic_data_mlops_practice",
    mlflow=True,
    dvc=False
)

# 2. Verify tracking URI
print("Tracking URI:", mlflow.get_tracking_uri())

# 3. Create a test experiment
mlflow.set_experiment("dagshub-connection-test")

# 4. Log a test run
with mlflow.start_run(run_name="remote-test"):

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 500)

    mlflow.log_metric("test_rmse", 4.2266)
    mlflow.log_metric("test_r2", 0.7637)

    # Test artifact logging
    with open("remote_test.txt", "w") as file:
        file.write("DagsHub remote MLflow tracking successful!")

    mlflow.log_artifact("remote_test.txt")

print("REMOTE EXPERIMENT LOGGED SUCCESSFULLY!")