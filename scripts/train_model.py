import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset

def compute_metrics(eval_pred):
    """Compute metrics like accuracy, precision, recall, and F1 score."""
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    labels = labels if isinstance(labels, np.ndarray) else labels.numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_model():
    # Load the dataset
    df = pd.read_csv("/Users/ebricks/Desktop/hallucination-detection/data/preprocessed_dataset.csv")
    df['context'] = df['context'].fillna('')
    df['prompt'] = df['prompt'].fillna('')
    df['response'] = df['response'].fillna('')
    
    # Combine the context, prompt, and response
    X = df["context"] + " " + df["prompt"] + " " + df["response"]
    y = df["hallucination"].map({'yes': 1, 'no': 0})

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use a smaller subset of data for faster experimentation
    X_train = X_train[:2000]  # Use only the first 1000 samples
    y_train = y_train[:2000]
    X_test = X_test[:400]    # Use only the first 200 samples
    y_test = y_test[:400]

    # Tokenize text using BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512)

    # Convert labels to tensors
    train_labels = torch.tensor(list(y_train))
    test_labels = torch.tensor(list(y_test))

    # Prepare datasets for training
    class HallucinationDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = HallucinationDataset(train_encodings, train_labels)
    test_dataset = HallucinationDataset(test_encodings, test_labels)

    # Leverage GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained BERT model for sequence classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    # Define optimized training arguments
    training_args = TrainingArguments(
        output_dir='./model_output',
        num_train_epochs=1,  # Reduced epochs for faster experimentation
        per_device_train_batch_size=32,  # Increased batch size
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",
        save_strategy="no",  # Disable saving during experimentation
        logging_dir='./logs',
        logging_steps=10,
        disable_tqdm=False,  # Show training progress
        report_to="none",
    )

    # Trainer setup with custom metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)

    # Save the model
    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')

    print("Training complete and model saved!")

if __name__ == "__main__":
    train_model()