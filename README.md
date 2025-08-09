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
