# tests/test_model_api.py
import pytest
from app import app # Import the Flask app instance
import json
import os

# Ensure the models directory exists and a dummy model is present for testing
@pytest.fixture(scope='session', autouse=True)
def setup_model_for_tests():
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'iris_logistic_regression_model.joblib')

    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        # Create a dummy joblib file if it doesn't exist for testing purposes
        # In a real scenario, you'd ensure your CI pipeline builds/provides the model
        print(f"Creating dummy model at {model_path} for testing...")
        try:
            import joblib
            from sklearn.linear_model import LogisticRegression
            from sklearn.datasets import load_iris
            iris = load_iris()
            dummy_model = LogisticRegression(max_iter=200)
            dummy_model.fit(iris.data, iris.target)
            joblib.dump(dummy_model, model_path)
        except ImportError:
            # Fallback if sklearn/joblib not available during setup fixture itself
            with open(model_path, 'w') as f:
                f.write("dummy content") # Just ensure the file exists
        print("Dummy model setup complete.")


@pytest.fixture
def client():
    """Creates a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_endpoint_valid_input(client):
    """Test the predict endpoint with valid input."""
    # The model is now loaded when the app module is imported, no need to trigger manually.
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post('/predict', data=json.dumps(test_data), content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'prediction_proba' in data
    assert isinstance(data['prediction'], str)
    assert isinstance(data['prediction_proba'], list)
    assert len(data['prediction_proba']) == 3 # For 3 iris classes

def test_predict_endpoint_missing_feature(client):
    """Test the predict endpoint with missing input features."""
    # The model is now loaded when the app module is imported, no need to trigger manually.
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_width": 0.2 # Missing petal_length
    }
    response = client.post('/predict', data=json.dumps(test_data), content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert "Missing expected feature" in data['error']

def test_predict_endpoint_invalid_data_type(client):
    """Test the predict endpoint with invalid data types."""
    # The model is now loaded when the app module is imported, no need to trigger manually.
    test_data = {
        "sepal_length": "invalid", # Invalid type
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post('/predict', data=json.dumps(test_data), content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert "could not convert string to float" in data['error'] or "Unsupported type" in data['error']

def test_home_endpoint(client):
    """Test the home endpoint."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"MLOps Iris Prediction API. Use /predict endpoint." in response.data
