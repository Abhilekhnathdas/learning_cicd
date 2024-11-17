import json
import numpy as np
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    input_data: list
    params: dict = {}
model = None

def init():
    global model
    model_path = "./models/learning_model.pkl"  # Path to the registered model in the container
    model = joblib.load(model_path)      # Load the model using joblib
init()

#uvicorn main:app --host 0.0.0.0 --port 8000
#curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @input.json
@app.post("/")
async def home():
    return "This app is up and running"

@app.post("/hello")
async def hellouser():
    return "Hello User"

@app.post("helloagin")
async def helloagin():
    return "hello again"
#curl -X POST iri-webapp-bhczgvgudbf3g6h6.eastus2-01.azurewebsites.net -H "Content-Type: application/json" -d @input.json
@app.post("/predict")
async def predict(data: InputData):
    try:
        # Extract input data and convert it to a numpy array
        input_array = np.array(data.input_data)
        
        # Make predictions using the loaded model
        predictions = model.predict(input_array)
        
        # Return predictions as a JSON response
        return {"predictions": predictions.tolist()}
    except Exception as e:
        # Return any errors encountered during processing
        return {"error": str(e)}
