import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss, f1_score, precision_score, recall_score
from sklearn.ensemble import StackingClassifier
import pickle

def train_model():
    """
    Trains an ensemble stacking model (Logistic Regression, Random Forest, and SVM) on the dataset
    and calculates the loss, accuracy, F1 score, precision, and recall.
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
    # Convert 'hallucination' column to numerical values (1 for 'yes', 0 for 'no')
    y = df["hallucination"].map({'yes': 1, 'no': 0})  # The target variable


    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert text data to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Define base models for stacking
    base_learners = [
        ('logreg', LogisticRegression(random_state=42, max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]

    # Create Stacking Classifier with Logistic Regression as the meta-model
    clf = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression(random_state=42))

    # Train the stacked model
    clf.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = clf.predict(X_test_tfidf)
    y_pred_proba = clf.predict_proba(X_test_tfidf)[:, 1]  # Probabilities for log loss

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)  
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Print evaluation results
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Log Loss: {loss:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Create model directory if it doesn't exist
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model and vectorizer to disk
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as model_file:
        pickle.dump(clf, model_file)

    with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print("Stacked Model and vectorizer saved.")

if __name__ == "__main__":
    train_model()
