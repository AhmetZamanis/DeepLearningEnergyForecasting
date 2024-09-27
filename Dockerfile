# Get CUDA image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Container working directory
WORKDIR /app

# Copy files, matching project root structure
COPY pyproject.toml setup.py .env ./
COPY src src
COPY scripts/deployment scripts/deployment

# Create empty folders for data
# Keep the /deployment/ subfolder so we can do 1 bind mount instead of 4
RUN mkdir -p /app/data/deployment/raw
RUN mkdir -p /app/data/deployment/processed
RUN mkdir -p /app/data/deployment/tuning-logs
RUN mkdir -p /app/data/deployment/predictions

# Create empty folders for models
# If other models (or loadables) are added, split into more subfolders like data 
RUN mkdir -p /app/models/deployment

# Install Torch, src & its dependencies
# Torch is not listed in pyproject.toml dependencies due to unique install
RUN pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir . 