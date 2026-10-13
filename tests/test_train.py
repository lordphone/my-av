import torch

from myav.export_onnx import export
from myav.train import main


def test_train_resume_and_export(fake_dataset, tmp_path):
    args = ["--dataset", str(fake_dataset), "--runs-dir", str(tmp_path / "runs"), "--name", "smoke",
            "--epochs", "1", "--batch-size", "2", "--window", "8", "--burn-in", "2",
            "--windows-per-segment", "2", "--workers", "0", "--no-pretrained", "--val-fraction", "0.25"]
    main(args)
    run = tmp_path / "runs" / "smoke"
    assert (run / "best.pt").exists() and (run / "config.yaml").exists()
    assert len((run / "metrics.csv").read_text().splitlines()) == 2

    main(["--config", str(run / "config.yaml"), "--resume", "--epochs", "2"])
    assert torch.load(run / "last.pt")["epoch"] == 1
    assert len((run / "metrics.csv").read_text().splitlines()) == 3

    try:
        import onnx  # noqa: F401
    except ImportError:
        return
    export(str(run / "best.pt"), str(tmp_path / "myav.onnx"))
