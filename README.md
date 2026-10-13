# my-av

A CNN-GRU steering model trained on [comma2k19](https://github.com/commaai/comma2k19) that runs inside
[openpilot](https://github.com/commaai/openpilot) on a comma 3X. It takes over steering only. Openpilot's stock
model still handles gas, braking, lead cars and everything else.

The first version of this project, which predicted future steering angle and speed, is in [legacy/](legacy/),
including its demo videos and training results.

## How it works

**Input**: exactly what openpilot's own driving model sees, so the same frames work in training and on the car:

- the road camera warped into openpilot's model frame, as 6 packed YUV channels at 128x256 (`myav/data/frames.py`)
- current speed (`v_ego`)
- the steering delay to plan for (`action_t`)

**Output**: one number, the lateral acceleration (m/s², left positive) the car should have `action_t` seconds from
now. Openpilot turns this into `desiredCurvature`. Training targets come from comma2k19's GPS/IMU pose
(`myav/data/targets.py`).

**Model** (`myav/models/lateral.py`), run once per frame:

1. A ResNet18 encodes the previous and current frame stacked together (12 channels). The pair lets it see motion.
2. A GRU updates its memory from those features plus speed and delay.
3. A small head predicts lateral acceleration from the memory.

Training runs the exact same per-frame step over 40-frame windows and scores a prediction at every frame. The
first 10 frames are a burn-in: the GRU starts from an empty memory there, so those frames are not scored. On the
car, modeld calls the same step every frame and feeds the memory back in. A test checks that the two paths give
identical outputs.

What the legacy model did differently: it scored only the last frame of each window, and reset the GRU's memory
every window in training while carrying it across the whole video at inference. It also split train/val by
segment, so neighbouring minutes of the same drive ended up on both sides. This pipeline splits by drive.

## Setup (Linux + NVIDIA)

Needs Python 3.10+ and `ffmpeg` (`sudo apt install ffmpeg`). The RTX 5090 needs a PyTorch build for CUDA 12.8 or
newer, so install torch from PyTorch's index first:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
```

## Train

```bash
python -m myav.train --dataset /path/to/comma2k19 --name first-run
```

`--dataset` is the folder containing `Chunk_1` … `Chunk_10`. Every setting in `src/myav/config.py` is also a
flag (`--batch-size 16`, `--window 60`, `--no-pretrained`, …), or put them in a YAML file and pass `--config`.

Each run writes to `runs/<name>/`:

- `config.yaml`: the exact settings used
- `train.log`, `metrics.csv`: per-epoch losses. `val_mae` is the average error in m/s². Compare it with
  `val_zero_mae`, the error of always driving straight; the model is only useful if it clearly beats that.
- `best.pt`: weights with the lowest validation loss
- `last.pt`: everything needed to resume after a crash:

```bash
python -m myav.train --config runs/first-run/config.yaml --resume
```

## Run it on the car

```bash
python -m myav.export_onnx runs/first-run/best.pt myav.onnx
```

Then follow [openpilot/README.md](openpilot/README.md) to patch modeld in your openpilot fork.

## Tests

```bash
pytest
```

The tests run on CPU and need no dataset. They use a fake comma2k19 tree with random frames.

## Layout

```
src/myav/
  config.py          settings, YAML and command line flags
  train.py           training loop, metrics, checkpoints
  export_onnx.py     ONNX export with the inputs/outputs modeld expects
  data/comma2k19.py  finds segments, splits train/val by drive
  data/frames.py     decodes video and warps it into openpilot's model frame
  data/targets.py    lateral acceleration from pose
  data/dataset.py    streams training windows
  models/lateral.py  the CNN-GRU
openpilot/           files and patch for running the model in openpilot's modeld
docs/                lessons learned and the original roadmap
legacy/              the first steering + speed model (kept for reference, not maintained)
```
