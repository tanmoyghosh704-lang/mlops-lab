
import dagshub
import mlflow
from mlflow import MlflowClient

# ==========================================
# 1. CONFIGURATION
# ==========================================

REPO_OWNER = "tanmoyghosh704-lang"
REPO_NAME = "Synthetic_data_mlops_practice"

EXPERIMENT_NAME = "sales-demand-prediction"
MODEL_NAME = "SalesDemandRF"

# Currently our training script logs test_rmse.
# Later, we will replace this with val_rmse.
SELECTION_METRIC = "test_rmse"


# ==========================================
# 2. CONNECT TO REMOTE MLFLOW
# ==========================================

dagshub.init(
    repo_owner=REPO_OWNER,
    repo_name=REPO_NAME,
    mlflow=True,
    dvc=False
)

tracking_uri = mlflow.get_tracking_uri()
mlflow.set_registry_uri(tracking_uri)

client = MlflowClient()


# ==========================================
# 3. RETRIEVE ALL RUNS
# ==========================================

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError("Experiment not found!")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    max_results=1000
)

# Consider only completed runs having the selection metric
eligible_runs = [
    run for run in runs
    if run.info.status == "FINISHED"
    and SELECTION_METRIC in run.data.metrics
]

if not eligible_runs:
    raise ValueError("No eligible completed runs found!")

# Sort by lowest RMSE
eligible_runs.sort(
    key=lambda run: run.data.metrics[SELECTION_METRIC]
)


# ==========================================
# 4. DISPLAY MODEL COMPARISON
# ==========================================

print("\n========== MODEL COMPARISON ==========")

for rank, run in enumerate(eligible_runs[:10], start=1):
    print(
        f"Rank: {rank} | "
        f"Run ID: {run.info.run_id} | "
        f"{SELECTION_METRIC}: "
        f"{run.data.metrics[SELECTION_METRIC]}"
    )


# ==========================================
# 5. SELECT THE BEST RUN
# ==========================================

best_run = eligible_runs[0]
best_run_id = best_run.info.run_id

print("\n========== BEST RUN ==========")
print("Best Run ID:", best_run_id)
print(
    "Best RMSE:",
    best_run.data.metrics[SELECTION_METRIC]
)


# ==========================================
# 6. RETRIEVE ITS LOGGED MODEL
# ==========================================

logged_models = mlflow.search_logged_models(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"source_run_id = '{best_run_id}'",
    output_format="list"
)

if not logged_models:
    raise ValueError("No model artifact found for the best run!")

# Our project follows one run = one logged model.
if len(logged_models) != 1:
    raise ValueError(
        f"Expected one logged model, found {len(logged_models)}."
    )

best_model = logged_models[0]
model_id = best_model.model_id

print("\nSelected Model ID:", model_id)


# ==========================================
# 7. CHECK WHETHER IT IS ALREADY REGISTERED
# ==========================================

registered_versions = client.search_model_versions(
    filter_string=f"name='{MODEL_NAME}'"
)

matching_versions = [
    version for version in registered_versions
    if version.run_id == best_run_id
]

if matching_versions:
    # Reuse the latest existing version for this run
    registered_version = max(
        matching_versions,
        key=lambda version: int(version.version)
    )

    print("\nModel already registered.")
    print("Reusing Version:", registered_version.version)

else:
    # Register the selected logged model
    registered_version = mlflow.register_model(
        model_uri=f"models:/{model_id}",
        name=MODEL_NAME
    )

    print("\nNew model registered successfully!")
    print("Created Version:", registered_version.version)


# ==========================================
# 8. ASSIGN CHAMPION ALIAS
# ==========================================

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=str(registered_version.version)
)

print("\n========== CHAMPION UPDATED ==========")

champion = client.get_model_version_by_alias(
    name=MODEL_NAME,
    alias="champion"
)

print("Champion Model:", MODEL_NAME)
print("Champion Version:", champion.version)
print("Champion Run ID:", champion.run_id)

print("\nModel selection and champion assignment completed!")
