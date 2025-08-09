## Project Overview

An autonomous vehicle prediction system that forecasts near‑future steering angle and speed (T+100ms to T+500ms) from video and vehicle state. The model pairs a ResNet18 visual backbone with a GRU for temporal reasoning. It is trained on 20‑frame windows (1s at 20fps) built from 100ms‑separated frame pairs, and at inference it runs in a streaming manner with seq_len=1 by feeding successive frame‑pairs (current + T‑100ms) while carrying the GRU hidden state. The pipeline targets the Comma2k19 dataset with efficient windowing (non‑overlapping 20‑frame windows, 240×320 images) and worker‑friendly caching for high throughput. Optional visualization renders speed (mph), steering (degrees), and turn indicators, with HEVC video support.

## Training

One Python entrypoint for all environments: `python -m src.training.train`.

### Configs
- `configs/default.yaml` (shared defaults)
- `configs/local.yaml` (your machine)
- `configs/cloud.yaml` (Vertex AI)

Precedence: `default.yaml` < profile (`local`/`cloud` or custom YAML) < environment variables < CLI flags.

### How to run
- Local (host Python):
  - `python -m src.training.train --config local --dataset-path /path/to/data`

- Local (Docker):
  - `docker build -t my-av .`
  - `docker run --rm -v /path/to/data:/data my-av --config local --dataset-path /data`

- Cloud (Vertex AI UI with your Artifact Registry image):
  - Set the container image built from this repo.
  - In Job Args, pass training flags, e.g.: `--config cloud --dataset-path gs://your-bucket/data --epochs 50 --batch-size 32 --num-workers 4`.

### Common flags
- `--config local|cloud|/path/to/config.yaml`
- `--dataset-path PATH` (supports `gs://`)
- `--mode train|test` (default: `train`)
- `--epochs N` (default: 50 for `train`, 10 for `test` if not specified)
- `--batch-size N` (default: 20)
- `--num-workers N` (default: 2)
- `--window-size N` (default: 20)
- `--checkpoint-dir DIR` (default: `checkpoints`)
- `--model-dir DIR` (default: `models`)
- `--log-dir DIR` (default: `logs`)

### Environment variable overrides (optional)
- `AV_DATASET_PATH` → `data.dataset_path`
- `AV_NUM_WORKERS` → `data.num_workers`
- `AV_BATCH_SIZE` → `train.batch_size`
- `AV_EPOCHS` → `train.epochs`
- `AV_DEVICE` → `runtime.device`
- `AV_LOG_INTERVAL` → `runtime.log_interval`

Outputs are written to `checkpoints/`, `models/`, and `logs/` by default.

Note: The container entrypoint calls the Python module directly; the prior shell wrapper has been removed for a single, consistent workflow.
