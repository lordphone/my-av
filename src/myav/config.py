"""
Training config: defaults here, optionally overridden by a YAML file, then by command line flags.
Every field is also a flag, e.g. `--batch-size 16` or `--no-pretrained`.
"""

import argparse
from dataclasses import asdict, dataclass, fields

import yaml


@dataclass
class Config:
    dataset: str = ""             # comma2k19 root, the folder containing Chunk_1 ... Chunk_10
    runs_dir: str = "runs"        # each run writes its config, log, metrics and checkpoints to runs_dir/name
    name: str = ""                # run name, defaults to a timestamp
    resume: bool = False          # continue runs_dir/name from its last checkpoint
    seed: int = 0

    # data
    val_fraction: float = 0.1     # fraction of routes held out for validation
    window: int = 40              # frames per training window (2 s at 20 fps)
    burn_in: int = 10             # frames at the start of each window that only warm up the GRU, not scored
    windows_per_segment: int = 20
    segments_in_flight: int = 4   # segments each worker draws windows from at once; more = better mixing, more RAM
    workers: int = 8

    # model
    hidden_size: int = 256
    num_layers: int = 2
    pretrained: bool = True       # start the CNN from ImageNet weights

    # optimisation
    epochs: int = 10
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True              # bf16 autocast on CUDA
    log_every: int = 50           # batches between progress lines


def load_config(argv=None) -> Config:
    parser = argparse.ArgumentParser(description="Train the lateral model")
    parser.add_argument("--config", help="YAML file with any Config fields")
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(flag, type=f.type, default=None)
    args = parser.parse_args(argv)

    values = {}
    if args.config:
        with open(args.config) as fh:
            values.update(yaml.safe_load(fh) or {})
    values.update({k: v for k, v in vars(args).items() if v is not None and k != "config"})

    known = {f.name for f in fields(Config)}
    unknown = set(values) - known
    if unknown:
        parser.error(f"unknown config keys: {', '.join(sorted(unknown))}")
    cfg = Config(**values)
    if not cfg.dataset:
        parser.error("--dataset is required (or set dataset in the config file)")
    return cfg


def save_config(cfg: Config, path) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(asdict(cfg), fh, sort_keys=False)
