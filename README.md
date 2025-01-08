# hallucination-detection
Train a machine learning (ML) model to detect whether a response to a question is hallucinated based on a given context.

## Setup and Installation

### Clone the repository & Navigate to the project directory:
```bash
    cd hallucination-detection
```
###  Install the required dependencies:
```bash
    pip install -r requirements.txt
```
###  Run the data preprocessing script:
```bash
    python scripts/preprocess.py
```
###  Run the training script:
```bash
    python scripts/train_model.py
```
###  Start the API server:
```bash
    uvicorn main:app --reload
```
###  Use curl to test the prediction endpoint:
```bash
    curl -X 'POST' \
    'http://127.0.0.1:8000/predict/' \
    -H 'Content-Type: application/json' \
    -d '{
    "context": "Paris is the capital of France.",
    "prompt": "What is the capital of France?",
    "response": "London"
    }'
```