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

