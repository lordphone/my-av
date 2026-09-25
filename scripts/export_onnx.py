# export_onnx.py
# Exports a trained OpModel to ONNX with the input/output layout openpilot's modeld expects.
# State inputs X have a matching output next_X so modeld can feed them back each frame.
#
# usage: python -m scripts.export_onnx checkpoints/op_model.pth models/myav.onnx

import argparse
import torch
import torch.nn as nn

from src.models.op_model import OpModel


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
    model = OpModel(pretrained=False)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
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
    parser.add_argument("out", nargs="?", default="models/myav.onnx")
    args = parser.parse_args()
    export(args.checkpoint, args.out)
