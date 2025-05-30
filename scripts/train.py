# scripts/train.py
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


def train_model():
    """Trains a Logistic Regression model on the Iris dataset."""
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Logistic Regression model...")
    # Increased max_iter for convergence warning prevention
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    # Ensure the 'models' directory exists
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'iris_logistic_regression_model.joblib')

    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print("Model saved successfully.")


if __name__ == '__main__':
    train_model()
