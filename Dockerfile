FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . /app

# Command to start the Flask application
CMD ["python", "app.py"]
