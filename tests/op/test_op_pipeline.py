import os
import tempfile
import unittest
import numpy as np
import torch

from src.data.op_frames import get_warp_matrix, warp_yuv_frame, i420_frame_size, EON_W, EON_H
from src.models.op_model import OpModel


class TestOpPipeline(unittest.TestCase):

    def test_warp_shape_and_crop(self):
        yuv = np.random.randint(0, 256, i420_frame_size(EON_W, EON_H), dtype=np.uint8)
        out = warp_yuv_frame(yuv, get_warp_matrix())
        y = yuv[:EON_W * EON_H].reshape(EON_H, EON_W)
        self.assertEqual(out.shape, (6, 128, 256))
        self.assertEqual(out.dtype, np.uint8)
        # with zero calibration and equal focal lengths the warp is a pure crop
        x0, y0 = round(EON_W / 2 - 256), round(EON_H / 2 - 47.6)
        np.testing.assert_array_equal(out[0], y[y0:y0 + 256:2, x0:x0 + 512:2])

    def test_lateral_accel_left_turn_positive(self):
        from src.data.op_dataset import lateral_accel_from_pose
        # counter-clockwise circle (left turn) on a plane tangent to the earth at the north pole
        t = np.arange(0, 10, 0.05)
        r, w, R = 50.0, 0.2, 6.4e6
        pos = np.stack([r * np.cos(w * t), r * np.sin(w * t), np.full_like(t, R)], axis=1)
        vel = np.stack([-r * w * np.sin(w * t), r * w * np.cos(w * t), np.zeros_like(t)], axis=1)
        with tempfile.TemporaryDirectory() as d:
            for name, arr in [("frame_times", t), ("frame_positions", pos), ("frame_velocities", vel)]:
                with open(os.path.join(d, name), "wb") as f:
                    np.save(f, arr)
            _, speed, lat = lateral_accel_from_pose(d)
        np.testing.assert_allclose(speed, r * w, rtol=1e-6)
        np.testing.assert_allclose(lat[10:-10], r * w * w, rtol=1e-2)

    def test_step_matches_forward(self):
        model = OpModel(pretrained=False).eval()
        imgs = torch.randint(0, 256, (1, 4, 6, 128, 256), dtype=torch.uint8)
        v_ego = torch.rand(1, 3) * 30
        action_t = torch.rand(1, 3)
        with torch.no_grad():
            preds, _ = model(imgs, v_ego, action_t)
            hidden = model.init_hidden(1)
            for i in range(3):
                out, hidden = model.step(imgs[:, i], imgs[:, i + 1], v_ego[:, i:i + 1], action_t[:, i:i + 1], hidden)
                self.assertAlmostEqual(out.item(), preds[0, i].item(), places=4)

    def test_onnx_export(self):
        try:
            import onnx
        except ImportError:
            self.skipTest("onnx not installed")
        from scripts.export_onnx import export
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "myav.onnx")
            export(None, path)
            graph = onnx.load(path).graph
            self.assertEqual([i.name for i in graph.input], ["new_img", "prev_img", "hidden", "v_ego", "action_t"])
            self.assertEqual([o.name for o in graph.output], ["lat_accel", "next_prev_img", "next_hidden"])


if __name__ == "__main__":
    unittest.main()
