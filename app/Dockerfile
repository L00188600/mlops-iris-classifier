# app/Dockerfile
FROM python:3.9-slim-buster

WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the 'models' and 'scripts' directories exist and copy contents
# Note: In a real pipeline, the trained model would be downloaded from an artifact store
# For this assessment, we assume the model is built into the image or copied during deploy
COPY models/ models/
COPY scripts/ scripts/

# Expose the port the Flask app runs on
EXPOSE 5000

# Run the Flask app with Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.app:app"]
