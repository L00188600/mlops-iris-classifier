# tests/test_preprocessing.py
import pandas as pd
import os
import sys

# Add the 'scripts' directory to the Python path so we can import preprocess.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.preprocess import load_and_preprocess_data

def test_load_and_preprocess_data_output_shape():
    """Test if the output of preprocessing has the correct shape."""
    X, y = load_and_preprocess_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0] # Number of samples should match
    assert X.shape[1] == 4 # Iris has 4 features

def test_load_and_preprocess_data_column_names():
    """Test if column names are correctly transformed."""
    X, _ = load_and_preprocess_data()
    expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    assert list(X.columns) == expected_columns

def test_load_and_preprocess_data_no_nulls():
    """Test if there are no null values after preprocessing."""
    X, y = load_and_preprocess_data()
    assert not X.isnull().any().any()
    assert not y.isnull().any()
