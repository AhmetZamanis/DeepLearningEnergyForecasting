# Python image
FROM python:3.12-slim

# Install CUDA, or get an image?

# Container working directory
WORKDIR /app

# Copy files, matching project root structure
COPY pyproject.toml setup.py .env .
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

# Install src and its dependencies
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir .