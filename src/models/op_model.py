# op_model.py
# Lateral model that consumes openpilot's warped YUV frames and predicts lateral acceleration.
# Runs one frame per step at inference so it can be dropped into openpilot's modeld.

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

IN_CHANNELS = 12  # previous + current frame, 6 packed YUV channels each
V_EGO_SCALE = 30.0
LAT_ACCEL_SCALE = 3.0


class GRUCell(nn.Module):
    """Plain GRU cell, written out so the ONNX export only uses basic ops tinygrad can compile."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.x2h = nn.Linear(input_size, 3 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, x, h):
        xr, xz, xn = self.x2h(x).chunk(3, dim=-1)
        hr, hz, hn = self.h2h(h).chunk(3, dim=-1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        return (1 - z) * n + z * h


class OpModel(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, pretrained=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Identity()

        # 12 channel conv1, initialised from the RGB weights averaged across channels and
        # scaled so activations keep the same magnitude as the pretrained network
        orig = self.backbone.conv1.weight.data
        self.backbone.conv1 = nn.Conv2d(IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.backbone.conv1.weight.copy_(orig.mean(dim=1, keepdim=True).repeat(1, IN_CHANNELS, 1, 1) * 3 / IN_CHANNELS)

        self.cells = nn.ModuleList([
            GRUCell(512 + 2 if i == 0 else hidden_size, hidden_size) for i in range(num_layers)
        ])
        self.head = nn.Sequential(nn.Linear(hidden_size + 1, 64), nn.ReLU(), nn.Linear(64, 1))

    def init_hidden(self, batch_size, device=None):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def encode(self, prev_img, cur_img):
        # uint8 YUV -> [-1, 1]
        x = torch.cat([prev_img, cur_img], dim=1).float() / 127.5 - 1.0
        return self.backbone(x)

    def recurrent(self, feats, v_ego, action_t, hidden):
        x = torch.cat([feats, v_ego / V_EGO_SCALE, action_t], dim=-1)
        new_hidden = []
        for i, cell in enumerate(self.cells):
            x = cell(x, hidden[i])
            new_hidden.append(x)
        out = self.head(torch.cat([x, action_t], dim=-1)) * LAT_ACCEL_SCALE
        return out, torch.stack(new_hidden)

    def step(self, prev_img, cur_img, v_ego, action_t, hidden):
        """
        Single inference step.
        prev_img, cur_img: [B, 6, 128, 256]
        v_ego, action_t:   [B, 1]
        hidden:            [num_layers, B, hidden_size]
        """
        return self.recurrent(self.encode(prev_img, cur_img), v_ego, action_t, hidden)

    def forward(self, imgs, v_ego, action_t, hidden=None):
        """
        Training forward over a window.
        imgs:            [B, T+1, 6, 128, 256] uint8
        v_ego, action_t: [B, T]
        returns lat_accel [B, T], hidden
        """
        B, T1, C, H, W = imgs.shape
        T = T1 - 1
        prev = imgs[:, :-1].reshape(B * T, C, H, W)
        cur = imgs[:, 1:].reshape(B * T, C, H, W)
        feats = self.encode(prev, cur).reshape(B, T, -1)

        if hidden is None:
            hidden = self.init_hidden(B, imgs.device)
        preds = []
        for i in range(T):
            out, hidden = self.recurrent(feats[:, i], v_ego[:, i:i + 1], action_t[:, i:i + 1], hidden)
            preds.append(out)
        return torch.cat(preds, dim=1), hidden
