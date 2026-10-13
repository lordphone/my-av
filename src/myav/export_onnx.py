"""
Exports a trained LateralModel to ONNX with the input/output layout openpilot's modeld expects.
State inputs X have a matching output next_X so modeld can feed them back each frame.

    python -m myav.export_onnx runs/first-run/best.pt myav.onnx
"""

import argparse

import torch
import torch.nn as nn

from myav.models.lateral import LateralModel


class OpenpilotWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, new_img, prev_img, hidden, v_ego, action_t):
        # new_img is the warp output from modeld: [road, wide] frames, only the road camera is used
        cur = new_img[0:1]
        lat_accel, next_hidden = self.model.step(prev_img, cur, v_ego, action_t, hidden)
        return lat_accel, cur.float(), next_hidden


def export(checkpoint, out_path):
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        cfg = ckpt["config"]
        model = LateralModel(cfg["hidden_size"], cfg["num_layers"], pretrained=False)
        model.load_state_dict(ckpt["model"])
    else:
        model = LateralModel(pretrained=False)  # untrained, for checking the export itself
    wrapper = OpenpilotWrapper(model).eval()

    inputs = (
        torch.zeros(2, 6, 128, 256, dtype=torch.uint8),               # new_img
        torch.zeros(1, 6, 128, 256),                                  # prev_img
        torch.zeros(model.num_layers, 1, model.hidden_size),          # hidden
        torch.zeros(1, 1),                                            # v_ego
        torch.zeros(1, 1),                                            # action_t
    )
    torch.onnx.export(
        wrapper, inputs, out_path,
        input_names=["new_img", "prev_img", "hidden", "v_ego", "action_t"],
        output_names=["lat_accel", "next_prev_img", "next_hidden"],
        opset_version=17,
        dynamo=False,
    )
    print(f"exported {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=None)
    parser.add_argument("out", nargs="?", default="myav.onnx")
    args = parser.parse_args()
    export(args.checkpoint, args.out)
