#!/bin/bash
# Aviator Predictor Startup Script

echo "🚀 Starting Aviator Predictor System..."

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export DEBUG=true

# Start the application using virtual environment python
/home/oliver/Desktop/aviator_predictor/venv/bin/python app.py