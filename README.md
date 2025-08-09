## AV Training (Essentials)

Single codebase for local and cloud training. Use YAML configs to switch environments.

### Configs
- `configs/default.yaml` (shared defaults)
- `configs/local.yaml` (your machine)
- `configs/cloud.yaml` (Vertex AI)

Precedence: default.yaml < local/cloud < env vars < CLI flags

### Quick start
- Local (Python):
  - python -m src.training.train --config local --dataset-path /path/to/data

- Local (script):
  - bash scripts/run_training.sh --config local --dataset-path /path/to/data

- Cloud (Vertex AI):
  - python submit_training.py
  - or use `job_config.yaml` with gcloud/Vertex

### Common flags
- --config local|cloud (or path to YAML)
- --dataset-path PATH (supports gs:// on cloud)
- --num-workers N, --batch-size B, --epochs E

Outputs are written to `checkpoints/`, `models/`, and `logs/` (overridable).

### Options and defaults
- Precedence: default.yaml < local/cloud < environment variables < CLI flags

- CLI flags:
  - `--config` (optional): profile name (`local`|`cloud`) or path to YAML
  - `--dataset-path` (required if not set in config): local path or `gs://...`
  - `--mode` (optional, default: `train`): `train` or `test`
  - `--epochs` (optional): if omitted, defaults to 50 in `train` mode, 10 in `test` mode
  - `--batch-size` (optional, default: 20)
  - `--num-workers` (optional, default: 2)
  - `--window-size` (optional, default: 20)
  - `--checkpoint-dir` (optional, default: `checkpoints`)
  - `--model-dir` (optional, default: `models`)
  - `--log-dir` (optional, default: `logs`)

- Environment variables (all optional):
  - `AV_DATASET_PATH` → `data.dataset_path`
  - `AV_NUM_WORKERS` → `data.num_workers`
  - `AV_BATCH_SIZE` → `train.batch_size`
  - `AV_EPOCHS` → `train.epochs`
  - `AV_DEVICE` → `runtime.device`
  - `AV_LOG_INTERVAL` → `runtime.log_interval`

- Config keys (from YAML):
  - `data.dataset_path` (required if not provided via CLI/env)
  - `data.num_workers` (default: 2)
  - `train.batch_size` (default: 16)
  - `train.epochs` (default: 30)
  - `train.window_size` (default: 20)
  - `runtime.device` (default: `auto`)
  - `runtime.log_interval` (default: 1000)
