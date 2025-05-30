# app/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = 'models/iris_logistic_regression_model.joblib'
model = None

def load_model():
    """Loads the trained model from the specified path."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None # Ensure model is None if loading fails

@app.before_first_request
def initialize_app():
    """Ensures the model is loaded when the app starts."""
    load_model()

@app.route('/')
def home():
    return "MLOps Iris Prediction API. Use /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    try:
        json_data = request.get_json(force=True)
        # Expected input format:
        # {
        #   "sepal_length": 5.1,
        #   "sepal_width": 3.5,
        #   "petal_length": 1.4,
        #   "petal_width": 0.2
        # }

        # Convert input to DataFrame, ensuring column order matches training data
        # Based on preprocessed column names in scripts/preprocess.py
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        input_df = pd.DataFrame([json_data], columns=feature_names)

        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df).tolist()[0]

        # Map integer prediction to species name for better readability
        iris_species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        predicted_species = iris_species_map[prediction[0]]

        return jsonify({
            'prediction': predicted_species,
            'prediction_proba': prediction_proba
        })
    except KeyError as e:
        return jsonify({"error": f"Missing expected feature: {e}. Please provide all features: {feature_names}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Ensure the 'models' directory exists when running locally for development
    os.makedirs('models', exist_ok=True)
    # If running locally, you might want to train the model first or copy a pre-trained one
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please run 'python scripts/train.py' first.")
    app.run(debug=True, host='0.0.0.0', port=5000)
