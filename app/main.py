from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

# Initialize FastAPI
app = FastAPI()

# Load the trained model and vectorizer
model_path = '/Users/ebricks/Desktop/hallucination-detection/model/model.pkl'
vectorizer_path = '/Users/ebricks/Desktop/hallucination-detection/model/vectorizer.pkl'

# Check if model and vectorizer files exist
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Model or Vectorizer file not found. Ensure the paths are correct.")

try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

try:
    with open(vectorizer_path, 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
except Exception as e:
    raise RuntimeError(f"Error loading the vectorizer: {e}")

# Pydantic model for request validation
class PredictRequest(BaseModel):
    context: str
    prompt: str
    response: str

@app.post("/predict/")
def predict(request: PredictRequest):
    try:
        # Combine context and prompt for prediction (this is just an example)
        input_text = f"{request.context} {request.prompt}"

        # Transform the input text using the vectorizer
        input_vector = vectorizer.transform([input_text])

        # Make a prediction using the trained model
        prediction = model.predict(input_vector)

        # Return the prediction as a response
        return {"prediction": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
