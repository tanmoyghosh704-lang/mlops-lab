
from contextlib import asynccontextmanager
from typing import Literal

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


# ==========================================
# 1. CONFIGURATION
# ==========================================

REPO_OWNER = "tanmoyghosh704-lang"
REPO_NAME = "Synthetic_data_mlops_practice"

MODEL_NAME = "SalesDemandRF"
MODEL_ALIAS = "champion"

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
app=FastAPI()


# ==========================================
# 2. INPUT DATA SCHEMA
# ==========================================

class PredictionInput(BaseModel):

    day_of_week: int = Field(ge=0, le=6)
    price: float = Field(ge=0, le=100)
    promo: Literal[0, 1]
    temperature: float = Field(ge=-50, le=60)


# ==========================================
# 3. LOAD MODEL WHEN API STARTS
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Connecting to DagsHub...")

    dagshub.init(
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
        mlflow=True,
        dvc=False
    )

    tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_registry_uri(tracking_uri)

    print("Loading champion model...")

    model = mlflow.sklearn.load_model(MODEL_URI)

    app.state.model = model

    print("Champion model loaded successfully!")

    yield

    app.state.model = None


# ==========================================
# 4. INITIALIZE FASTAPI
# ==========================================

app = FastAPI(
    title="Sales Demand Prediction API",
    description="Predict product demand using the registered Random Forest model.",
    version="1.0",
    lifespan=lifespan
)


# ==========================================
# 5. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Sales Demand Prediction API is running!",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS
    }


# ==========================================
# 6. HEALTH CHECK ENDPOINT
# ==========================================

@app.get("/health")
def health(request: Request):

    model = getattr(request.app.state, "model", None)

    return {
        "status": "healthy" if model is not None else "unavailable",
        "model_loaded": model is not None
    }


# ==========================================
# 7. PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(
    input_data: PredictionInput,
    request: Request
):

    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available"
        )

    # Convert input JSON into a DataFrame
    input_df = pd.DataFrame(
        [input_data.model_dump()]
    )

    # Generate prediction
    prediction = model.predict(input_df)

    return {
        "predicted_units": float(prediction[0]),
        "model": MODEL_NAME,
        "model_alias": MODEL_ALIAS
    }
