import pytest
import torch

from myav.models.lateral import LateralModel


def test_step_matches_forward():
    """Streaming one frame at a time (as on the car) must give the same result as the training forward."""
    model = LateralModel(pretrained=False).eval()
    imgs = torch.randint(0, 256, (1, 4, 6, 128, 256), dtype=torch.uint8)
    v_ego = torch.rand(1, 3) * 30
    action_t = torch.rand(1, 3)
    with torch.no_grad():
        preds, _ = model(imgs, v_ego, action_t)
        hidden = model.init_hidden(1)
        for i in range(3):
            out, hidden = model.step(imgs[:, i], imgs[:, i + 1], v_ego[:, i:i + 1], action_t[:, i:i + 1], hidden)
            assert out.item() == pytest.approx(preds[0, i].item(), abs=1e-4)


def test_onnx_export(tmp_path):
    onnx = pytest.importorskip("onnx")
    from myav.export_onnx import export

    path = tmp_path / "myav.onnx"
    export(None, str(path))
    graph = onnx.load(path).graph
    assert [i.name for i in graph.input] == ["new_img", "prev_img", "hidden", "v_ego", "action_t"]
    assert [o.name for o in graph.output] == ["lat_accel", "next_prev_img", "next_hidden"]
