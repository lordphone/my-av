# Running my-av on a comma 3X

The stock driving model keeps running. When `MYAV=1` is set, `desiredCurvature` comes from the my-av model
instead, while everything else (longitudinal, lane lines, leads, alerts) stays stock.
Written against openpilot `master` at `7ee974a`.

1. Train and export:
   ```bash
   python -m src.training.train_op --dataset /path/to/comma2k19
   python -m scripts.export_onnx checkpoints/op_model.pth models/myav.onnx
   ```
2. In your openpilot fork:
   ```bash
   cp myav_lateral.py <openpilot>/openpilot/selfdrive/modeld/
   cp models/myav.onnx <openpilot>/openpilot/selfdrive/modeld/models/
   cd <openpilot> && git apply <my-av>/openpilot/modeld.patch
   ```
3. Compile on the device (same flags as modeld's SConscript):
   ```bash
   cd /data/openpilot
   DEV=QCOM IMAGE=1 FLOAT16=1 NOLOCALS=1 JIT_BATCH_SIZE=0 OPENPILOT_HACKS=1 PYTHONPATH=tinygrad_repo \
     python3 tinygrad_repo/examples/openpilot/compile_onnx.py \
     openpilot/selfdrive/modeld/models/myav.onnx openpilot/selfdrive/modeld/models/myav_tinygrad.pkl --out-of-band
   ```
4. Run with `MYAV=1` set in modeld's environment (e.g. in `system/manager/process_config.py`).

Not handled: the chestnut (big model) path, and the UI path still draws the stock plan.
