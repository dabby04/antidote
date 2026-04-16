# Backend-only Dockerfile for the Antidote demo
# Builds a Python image that runs the FastAPI/uvicorn server.

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (for pandas/pyarrow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only backend runtime assets.
COPY utils.py ./utils.py
COPY Demo/server.py ./Demo/server.py

# Data used by /examples endpoint
COPY pi-detection-data/t1_llmail.parquet ./pi-detection-data/t1_llmail.parquet
COPY pi-detection-data/t2_hackaprompt.parquet ./pi-detection-data/t2_hackaprompt.parquet
COPY pi-detection-data/t3_bipia.parquet ./pi-detection-data/t3_bipia.parquet

# Stage checkpoints used by /simulate and /simulate_all_stages
COPY results/results-baselines/checkpoints/naive_sequential_after_T1_LLMail.pt ./results/results-baselines/checkpoints/naive_sequential_after_T1_LLMail.pt
COPY results/results-baselines/checkpoints/naive_sequential_after_T2_HackAPrompt.pt ./results/results-baselines/checkpoints/naive_sequential_after_T2_HackAPrompt.pt
COPY results/results-baselines/checkpoints/naive_sequential_after_T3_BIPIA.pt ./results/results-baselines/checkpoints/naive_sequential_after_T3_BIPIA.pt
COPY results/results-ewc+replay/checkpoints/ewc_plus_replay_after_T1_LLMail.pt ./results/results-ewc+replay/checkpoints/ewc_plus_replay_after_T1_LLMail.pt
COPY results/results-ewc+replay/checkpoints/ewc_plus_replay_after_T2_HackAPrompt.pt ./results/results-ewc+replay/checkpoints/ewc_plus_replay_after_T2_HackAPrompt.pt
COPY results/results-ewc+replay/checkpoints/ewc_plus_replay_after_T3_BIPIA.pt ./results/results-ewc+replay/checkpoints/ewc_plus_replay_after_T3_BIPIA.pt

# Work from the Demo folder where server.py lives
WORKDIR /app/Demo

# Expose the FastAPI port
EXPOSE 8000

# Start the uvicorn server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
