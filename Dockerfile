# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

# Copy app source code
COPY . .

# Expose port (default FastAPI Uvicorn port)
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]