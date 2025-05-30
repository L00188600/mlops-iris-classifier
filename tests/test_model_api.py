import pytest
from app.app import app  # Import Flask app instance correctly
import json
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


@pytest.fixture(scope='session', autouse=True)
def setup_model_for_tests():
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'iris_logistic_regression_model.joblib')

    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Creating dummy model at {model_path} for testing...")
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=[
            'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
        ])
        dummy_model = LogisticRegression(max_iter=200)
        dummy_model.fit(X, iris.target)
        joblib.dump(dummy_model, model_path)
        print("Dummy model setup complete.")


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict_endpoint_valid_input(client):
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post('/predict', data=json.dumps(test_data),
                           content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'prediction_proba' in data
    assert isinstance(data['prediction'], str)
    assert isinstance(data['prediction_proba'], list)
    assert len(data['prediction_proba']) == 3


def test_predict_endpoint_missing_feature(client):
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_width": 0.2  # Missing petal_length
    }
    response = client.post('/predict', data=json.dumps(test_data),
                           content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert "Missing expected feature" in data['error']


def test_predict_endpoint_invalid_data_type(client):
    test_data = {
        "sepal_length": "invalid",  # Invalid type
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post('/predict', data=json.dumps(test_data),
                           content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert "could not convert string to float" in data['error'] or \
           "Unsupported type" in data['error']


def test_home_endpoint(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"MLOps Iris Prediction API." in response.data
