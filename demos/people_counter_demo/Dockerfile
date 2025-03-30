FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy your script and utils directory
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    opencv-python-headless \
    numpy \
    torch \
    ultralytics \
    openvino-dev \
    supervision

# Default command
ENTRYPOINT ["python", "main.py"]
