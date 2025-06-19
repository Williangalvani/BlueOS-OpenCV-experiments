# Use Python slim image compatible with ARM architecture (Raspberry Pi)
FROM python:3.9-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libpng16-16 \
    libjpeg62-turbo \
    libtiff5-dev \
    libwebp-dev \
    libopenjp2-7 \
    libhdf5-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY webcam_optical_flow.py .
COPY dense_headless.py .
COPY dense_profiling.py .
COPY optimize_dense_params.py .
COPY motion_data.csv .

# Create directory for videos
RUN mkdir -p /app/videos

