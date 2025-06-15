# AV Model Training

This repository contains utilities for training an autonomous vehicle model.

## Building the Docker image

```bash
docker build -t av-model .
```

## Running the container

```bash
docker run --gpus all -v $(pwd)/data:/app/data av-model bash scripts/run_training.sh --dataset-path /app/data
```

Adjust the mounted data directory and command line options as needed.
