import os
import joblib
from google.cloud import storage
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Google Cloud Storage Client
storage_client = storage.Client.from_service_account_json("spaceship-key.json")
BUCKET_NAME = 'spaceship_titanic_bucket'
MODELS_FOLDER = 'models'

# FastAPI app
app = FastAPI()

# Function to download a model from GCS
def download_model_from_gcs(model_name):
    """Download a model from GCS and load it."""
    # Ensure the models directory exists
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)  # Create the directory if it does not exist

    model_path = f"./models/{model_name}_model.pkl"
    blob_name = f"{MODELS_FOLDER}/{model_name}_model.pkl"

    # Access the model from GCS
    bucket = storage_client.bucket(BUCKET_NAME)
    model_blob = bucket.blob(blob_name)

    # Download the model to local storage
    model_blob.download_to_filename(model_path)
    print(f"Downloaded {model_name} model from GCS at {model_path}")

    # Load and return the model with joblib
    model = joblib.load(model_path)
    return model

# Use lifespan context manager to load models
@asynccontextmanager
async def lifespan(app):
    # Load models during the lifespan of the app
    app.state.lr_model = download_model_from_gcs("lr")
    app.state.rf_model = download_model_from_gcs("rf")
    app.state.xgb_model = download_model_from_gcs("xgb")
    yield

# Create FastAPI app with lifespan context manager
app = FastAPI(lifespan=lifespan)

# Define input model for prediction
class PredictionInput(BaseModel):
    Age: float
    CryoSleep: str  # 'True' or 'False'
    VIP: str        # 'True' or 'False'
    Cabin: str
    HomePlanet: str
    Destination: str
    PassengerId: int = None
    Name: str = None

# Prediction endpoint
@app.post("/predict/")
async def predict(input_data: PredictionInput):
    # Prepare input data
    input_dict = input_data.dict()

    # Convert 'CryoSleep' and 'VIP' to booleans
    input_dict['CryoSleep'] = True if input_dict['CryoSleep'] == 'True' else False
    input_dict['VIP'] = True if input_dict['VIP'] == 'True' else False

    # Select the model (for example, Logistic Regression)
    model = app.state.lr_model  # You can change this to use different models like rf_model or xgb_model

    # Prepare data (you may need preprocessing here depending on your model's requirements)
    # For simplicity, we assume the model expects raw input in the same format.

    # Predict
    prediction = model.predict([[
        input_dict['Age'],
        input_dict['CryoSleep'],
        input_dict['VIP'],
        input_dict['Cabin'],
        input_dict['HomePlanet'],
        input_dict['Destination']
    ]])

    return {"prediction": int(prediction[0])}
