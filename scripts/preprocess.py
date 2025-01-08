import pandas as pd
import re

def preprocess_data(input_csv, output_csv):
    """
    Cleans and preprocesses dataset for ML model.
    """
    # Load dataset
    df = pd.read_csv(input_csv)
    print("Columns in the dataset:", df.columns)

    # Check for required columns
    required_columns = ["context", "prompt", "response", "hallucination"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the dataset: {missing_columns}")

    # Drop rows with missing values in required columns
    df.dropna(subset=required_columns, inplace=True)

    # Clean text (lowercase, remove special characters)
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text

    # Apply cleaning to text columns
    for col in ["context", "prompt", "response"]:
        df[col] = df[col].apply(clean_text)

    # Save preprocessed data
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")

if __name__ == "__main__":
    preprocess_data(
        "/Users/ebricks/Desktop/hallucination-detection/data/raw_dataset.csv",
        "/Users/ebricks/Desktop/hallucination-detection/data/preprocessed_dataset.csv"
    )
