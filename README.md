# Titanic Survival Prediction System

This project is an end-to-end Machine Learning system that predicts whether a passenger survived the Titanic disaster. It includes data preprocessing, model training, a REST API, and containerization.

## Project Structure
- `api/`: Contains the FastAPI application (`main.py`).
- `model/`: Contains the serialized scikit-learn pipeline (`titanic_pipeline.pkl`).
- `data/`: Contains the raw Titanic dataset.
- `src/`: Directory for additional source code/scripts.
- `Dockerfile`: Instructions for containerizing the API.
- `requirements.txt`: Python dependencies.

## Setup Instructions (Docker)

1. **Build the Docker Image:**
   ```bash
   docker build -t titanic-ml-api .