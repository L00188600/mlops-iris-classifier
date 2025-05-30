from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)  # Flask app instance

# Load the trained model once when the app starts
model = joblib.load('models/iris_logistic_regression_model.joblib')


@app.route('/')
def home():
    return "MLOps Iris Prediction API. Use /predict endpoint."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        expected_features = [
            'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
        ]

        # Check for missing features
        for feature in expected_features:
            if feature not in input_data:
                return jsonify({
                    "error": f"Missing expected feature: {feature}"
                }), 400

        # Create a DataFrame with the expected features
        input_df = pd.DataFrame([input_data])

        # Make prediction and probability
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0].tolist()

        return jsonify({
            "prediction": str(prediction),
            "prediction_proba": prediction_proba
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
