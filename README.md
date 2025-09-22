## Project Overview

An autonomous vehicle prediction system that forecasts steering angles and vehicle speeds 100-500ms into the future from camera footage and current vehicle state.
<div align="center">
  <table>
      <td align="center">
        <h4>Steering on Highway</h4>
        <a href="https://youtu.be/Gu2GAtuMQag">
          <img src="https://img.youtube.com/vi/Gu2GAtuMQag/maxresdefault.jpg" width="400" alt="Steering on Highway">
        </a>
        <br>
      </td>
      <td align="center">
        <h4>Works at Night, Can Stop Too!</h4>
        <a href="https://youtu.be/pPa6Q-niG0c">
          <img src="https://img.youtube.com/vi/pPa6Q-niG0c/maxresdefault.jpg" width="400" alt="Works at Night, Can Stop Too!">
        </a>
        <br>
      </td>
  </table>
</div>

Finished training on 09/09/2025 (My Bday!), next steps are evaluation + simulation + intergration with openpilot.

Issues I ran into and learnt from are in [issues.txt](issues.txt).

### Architecture & Implementation Stuff

- **CV Pipeline:** Modified ResNet18 with 6-channel input for frame pairs (current + past)
- **Frame Pairing:** Stacked current+past (T-100ms) frame pairs as input for the GRU
- **Temporal Modeling:** 2-layer GRU (256 hidden units) with persistent state for streaming inference  
- **Multi-task Output:** Joint prediction of steering angles and vehicle speeds
- **Data Pipeline:** Custom windowing system for efficient processing of 107.5GB Comma2k19 dataset (33 hrs of highway driving footage), smart shuffling too
- **Dynamic Loss Weighting:** Adaptive balancing between steering and speed objectives during training
- **Streaming Design:** GRU hidden states persist between predictions for real-time deployment
- **Production Ready:** Complete checkpointing system with resumable training and model exports. Dockerized for training on the cloud too (GCP)

### Training Results

**Architecture:**
- **CNN Backbone:** ResNet18 (pre-trained) with modified 6-channel input layer
- **Temporal Model:** 2-layer GRU (256 hidden units) for sequential processing
- **Input:** 20-frame sequences (1 second) with frame pairs T and T-100ms, with current steering and speed values
- **Output:** 5 future predictions for steering angle and speed (T+100ms to T+500ms)

**Training Metrics:**
- **Total Epochs:** 50 (completed successfully)
- **Best Model:** Epoch 48 with validation loss of **0.0014**
- **Final Performance:**
  - Training Loss: **0.0002** (Steering: 0.0001, Speed: 0.0001)
  - Validation Loss: **0.0016** (Steering: 0.0011, Speed: 0.0000)
- **Training Split:** ~20 epochs on GCP, ~30 epochs locally with my gaming PC (like 2 weeks of loud fans spinning at night)

## Train it Yourself!

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
  - Enable Cloud Storage FUSE for your bucket so it is mounted inside the container at `/gcs/<bucket>/...`.
  - In Job Args, pass training flags, e.g.: `--config cloud --dataset-path gs://your-bucket/data --epochs 50 --batch-size 32 --num-workers 4`.
  - Note: The code uses standard file operations, so either enable Cloud Storage FUSE or modify the code to use `gcsfs` for direct `gs://` access.

### Common flags
- `--config local|cloud|/path/to/config.yaml`
- `--dataset-path PATH` (supports `gs://` with Cloud Storage FUSE enabled)
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
