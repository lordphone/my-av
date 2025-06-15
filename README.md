# AV Model Training

This repository contains utilities for training an autonomous vehicle model.

## Building the Docker image

```bash
docker build -t av-model .
```

## Running the container

```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -e DATASET_PATH=/app/data \
  -e NUM_WORKERS=4 \
  -e BATCH_SIZE=32 \
  av-model
```

You can override any training parameter using environment variables or by 
passing the corresponding command-line flags. Mount the dataset directory as 
needed for your environment.
