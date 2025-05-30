# tests/test_model_api.py
import pytest
import sys
import os
import json

# Add the 'app' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))
from app import app # Import the Flask app instance

@pytest.fixture
def client():
    # Set the testing configuration
    app.config['TESTING'] = True
    # For testing, ensure the model exists or mock joblib.load
    # In a real CI, the model might be built into the Docker image
    # For now, let's assume it exists from a previous train.py run
    # Or for robust testing, mock it
    if not os.path.exists('../models/iris_logistic_regression_model.joblib'):
        # Create a dummy model for testing if it doesn't exist
        # This is a workaround for tests to pass without full training
        # In a real scenario, the model would be a build artifact
        from sklearn.linear_model import LogisticRegression
        import joblib
        dummy_model = LogisticRegression()
        dummy_model.classes_ = [0, 1, 2] # Mock classes for predict_proba
        os.makedirs('../models', exist_ok=True)
        joblib.dump(dummy_model, '../models/iris_logistic_regression_model.joblib')


    with app.test_client() as client:
        yield client

def test_home_endpoint(client):
    """Test the home endpoint."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"MLOps Iris Prediction API" in response.data

def test_predict_endpoint_valid_input(client):
    """Test the predict endpoint with valid input."""
    # Ensure the model is loaded before testing
    app.before_first_request_funcs[None][0]()

    valid_input = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post('/predict', data=json.dumps(valid_input), content_type='application/json')
    data = json.loads(response.data)

    assert response.status_code == 200
    assert 'prediction' in data
    assert 'prediction_proba' in data
    assert isinstance(data['prediction'], str)
    assert isinstance(data['prediction_proba'], list)
    assert len(data['prediction_proba']) == 3 # For 3 classes (Iris)

def test_predict_endpoint_missing_feature(client):
    """Test the predict endpoint with missing input features."""
    app.before_first_request_funcs[None][0]()

    invalid_input = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
        # petal_width is missing
    }
    response = client.post('/predict', data=json.dumps(invalid_input), content_type='application/json')
    data = json.loads(response.data)

    assert response.status_code == 400
    assert 'error' in data
    assert "Missing expected feature" in data['error']

def test_predict_endpoint_invalid_data_type(client):
    """Test the predict endpoint with invalid data types."""
    app.before_first_request_funcs[None][0]()

    invalid_input = {
        "sepal_length": "invalid", # Invalid data type
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post('/predict', data=json.dumps(invalid_input), content_type='application/json')
    data = json.loads(response.data)

    assert response.status_code == 400
    assert 'error' in data
