# app/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Create a directory for models and ensure it exists
RUN mkdir -p models

# Expose port 5000 for the Flask application
EXPOSE 5000

# Run the Gunicorn server when the container launches
# Assuming your Flask app instance is named 'app' in 'app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.app:app"]
