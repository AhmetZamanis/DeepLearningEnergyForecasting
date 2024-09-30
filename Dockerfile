# Get CUDA image
# Older images may be scheduled for deletion, check nvidia/cuda
FROM nvidia/cuda:12.6.1-base-ubuntu24.04

# Install / update Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Create container working directory
WORKDIR /app

# Copy deployment code & files, matching project root structure
COPY pyproject.toml setup.py .env ./
COPY src src
COPY scripts/deployment scripts/deployment

# Create empty folders for data & models
# Create the /deployment/ subfolders so we can do 1 bind mount each for data/deployment & models/deployment
RUN mkdir -p /app/data/deployment/raw \
    && mkdir -p /app/data/deployment/processed \
    && mkdir -p /app/data/deployment/tuning-logs \
    && mkdir -p /app/data/deployment/predictions \
    && mkdir -p /app/models/deployment/transformer \
    && mkdir -p /app/models/deployment/scaler

# Install Torch, src & its dependencies
# Torch is not listed in pyproject.toml dependencies due to --index-url not being usable
RUN pip3 install --no-cache-dir --break-system-packages torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir --break-system-packages . 