from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
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
        # Combine context, prompt, and response for prediction
        input_text = f"{request.context} {request.prompt} {request.response}"

        # Transform the input text using the vectorizer
        input_vector = vectorizer.transform([input_text])

        # Make a prediction using the trained model
        prediction = model.predict(input_vector)
        prediction_proba = model.predict_proba(input_vector)[:, 1]  # Get the probability for the positive class

        # Map the prediction to a label
        prediction_label = 'yes' if prediction[0] == 1 else 'no'

        # Return the prediction as a response with probability
        return {
            "prediction": prediction_label,
            "probability": float(prediction_proba[0])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
