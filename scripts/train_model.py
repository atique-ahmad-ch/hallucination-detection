import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def train_model():
    """
    Trains a logistic regression model on the dataset.
    """
    # Load preprocessed data
    df = pd.read_csv("/Users/ebricks/Desktop/hallucination-detection/data/preprocessed_dataset.csv")
    
    # Print column names to verify
    print(df.columns)

    # Handle NaN values by replacing them with empty strings
    df['context'] = df['context'].fillna('')
    df['prompt'] = df['prompt'].fillna('')

    # Extract features (context and prompt) and target (hallucination)
    X = df["context"] + " " + df["prompt"]  # Concatenate context and prompt for better feature representation
    y = df["hallucination"]  # The target variable

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert text data to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = clf.predict(X_test_tfidf)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Create model directory if it doesn't exist
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model and vectorizer to disk
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as model_file:
        pickle.dump(clf, model_file)

    with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print("Model and vectorizer saved.")

if __name__ == "__main__":
    train_model()
