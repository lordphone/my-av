# AV Model Training

This project provides utilities for training an autonomous vehicle (AV) model. The recommended approach is to use the provided Docker environment so you do not need to install the deep learning stack locally.

## Prerequisites

- **Docker** with GPU support (e.g. NVIDIA Container Toolkit)
- A copy of the `comma2k19` dataset or your own dataset organised in the same structure

## Building the Docker image

Clone this repository and build the image:

```bash
# from the repository root
docker build -t av-model .
```

The build step creates a Conda environment defined in `environment.yml` and copies the project sources into `/app` inside the image. The container automatically runs all commands inside this environment.

## Running the training container

1. Place your dataset on the host machine. In the examples below we assume it lives in `~/data/comma2k19`.
2. Start the container and mount the dataset directory. The dataset path inside the container is passed through the `DATASET_PATH` environment variable. The startup script checks that this directory exists before training starts.

```bash
docker run --gpus all \
  -v ~/data/comma2k19:/app/data \
  -e DATASET_PATH=/app/data \
  -e NUM_WORKERS=4 \
  -e BATCH_SIZE=32 \
  av-model
```

The `run_training.sh` script invoked by the container reads these variables and forwards them to `src/training/train.py`. You can supply additional arguments by setting environment variables or editing the command line in the script.

### Customising parameters

- `DATASET_PATH` – location of the dataset inside the container. Must match the mounted volume.
- `NUM_WORKERS` – number of data loading workers.
- `BATCH_SIZE` – batch size for training.
- `CHECKPOINT_DIR`, `MODEL_DIR`, `LOG_DIR` – output locations for training artefacts.

If you want to run training manually you can execute the script directly:

```bash
bash scripts/run_training.sh --dataset-path /path/to/data --num-workers 4 --batch-size 32
```

All parameters accepted by `train.py` can be passed on the command line.

## Repository structure

- `src/` – source code for data loading, models and training.
- `scripts/` – helper scripts including `run_training.sh` used inside Docker.
- `tests/` – unit tests.
- `environment.yml` – Conda environment specification used in the Docker image.

