from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize FastAPI
app = FastAPI()

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('/Users/ebricks/Desktop/hallucination-detection/model')
tokenizer = BertTokenizer.from_pretrained('/Users/ebricks/Desktop/hallucination-detection/model')

# Pydantic model for request validation
class PredictRequest(BaseModel):
    context: str
    prompt: str
    response: str

@app.post("/predict/")
def predict(request: PredictRequest):
    # Combine context, prompt, and response
    input_text = f"{request.context} {request.prompt} {request.response}"

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()

    # Map prediction to 'yes' or 'no'
    prediction_label = 'yes' if prediction == 1 else 'no'
    return {"prediction": prediction_label}