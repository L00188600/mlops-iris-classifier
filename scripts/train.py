# scripts/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
from scripts.preprocess import load_and_preprocess_data # Import preprocess function

def train_model():
    """
    Loads preprocessed data, trains a Logistic Regression model,
    evaluates it, and saves the trained model.
    """
    print("Loading preprocessed data...")
    X, y = load_and_preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained. Accuracy on test set: {accuracy:.4f}")

    # Save the trained model
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'iris_logistic_regression_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # For CT/CM, you might want to log metrics or save them to a file
    with open('metrics.txt', 'w') as f:
        f.write(f"accuracy={accuracy}\n")

    return accuracy

if __name__ == "__main_file__": # Changed to avoid direct execution when imported
    print("Starting model training...")
    train_model()
    print("Model training complete.")
