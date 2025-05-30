# scripts/preprocess.py
import pandas as pd
from sklearn.datasets import load_iris
import os

def load_and_preprocess_data():
    """
    Loads the Iris dataset and returns features (X) and target (y).
    In a real scenario, this would involve more complex preprocessing.
    """
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')

    # Example of a simple preprocessing step: renaming columns
    X.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in X.columns]

    # Save processed data temporarily (optional, but good for pipeline visibility)
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    X.to_csv(os.path.join(output_dir, 'processed_features.csv'), index=False)
    y.to_csv(os.path.join(output_dir, 'processed_target.csv'), index=False)

    print(f"Data preprocessed. Features saved to {os.path.join(output_dir, 'processed_features.csv')}")
    print(f"Target saved to {os.path.join(output_dir, 'processed_target.csv')}")

    return X, y

if __name__ == "__main__":
    print("Starting data preprocessing...")
    load_and_preprocess_data()
    print("Data preprocessing complete.")
