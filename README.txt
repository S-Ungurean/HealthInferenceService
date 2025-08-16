Python Inference Service

Python service exposes 1 API to perform predictions based on content passed. Works by pulling trained model weights and metadata from S3 buckets and then
performing inference.

HOW TO RUN

spin up venv in project root (os dependend)

# Install Requirements
pip install -r requirements.txt

# Run Server
uvicorn app:app --host 0.0.0.0 --port 8000