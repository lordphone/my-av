"""
Trains the lateral model on comma2k19.

    python -m myav.train --dataset /path/to/comma2k19 --name first-run
    python -m myav.train --config runs/first-run/config.yaml --resume
"""

import csv
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from myav.config import Config, load_config, save_config
from myav.data.comma2k19 import find_segments, split_by_route
from myav.data.dataset import WindowDataset
from myav.models.lateral import LateralModel

log = logging.getLogger("myav")


def setup_run(cfg: Config) -> Path:
    cfg.name = cfg.name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.runs_dir) / cfg.name
    if run_dir.exists() and not cfg.resume:
        raise SystemExit(f"{run_dir} already exists, pick another --name or pass --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log")])
    return run_dir


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, cfg, device, optimizer=None):
    """One pass over the loader. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    criterion = nn.SmoothL1Loss()
    totals = {"loss": 0.0, "mae": 0.0, "zero_mae": 0.0}
    batches = 0

    for batch in loader:
        imgs = batch["imgs"].to(device, non_blocking=True)
        v_ego = batch["v_ego"].to(device)
        action_t = batch["action_t"].to(device)
        target = batch["target"].to(device)[:, cfg.burn_in:]

        with torch.set_grad_enabled(training), \
                torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"):
            preds, _ = model(imgs, v_ego, action_t)
        preds = preds[:, cfg.burn_in:].float()
        loss = criterion(preds, target)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        totals["loss"] += loss.item()
        totals["mae"] += (preds - target).abs().mean().item()
        totals["zero_mae"] += target.abs().mean().item()  # baseline: always drive straight
        batches += 1
        if training and batches % cfg.log_every == 0:
            log.info(f"  batch {batches} loss {totals['loss'] / batches:.4f}")

    return {k: v / max(batches, 1) for k, v in totals.items()}


def main(argv=None):
    cfg = load_config(argv)
    run_dir = setup_run(cfg)
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"run {run_dir} on {device}")

    segments = find_segments(cfg.dataset)
    if not segments:
        raise SystemExit(f"no segments found under {cfg.dataset}")
    train_segs, val_segs = split_by_route(segments, cfg.val_fraction, cfg.seed)
    log.info(f"{len(train_segs)} train segments, {len(val_segs)} val segments")

    def loader(segs, train):
        ds = WindowDataset(segs, cfg.window, cfg.windows_per_segment, cfg.segments_in_flight, train, cfg.seed)
        return DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.workers, pin_memory=device.type == "cuda")

    train_loader, val_loader = loader(train_segs, True), loader(val_segs, False)

    model = LateralModel(cfg.hidden_size, cfg.num_layers, cfg.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    start_epoch, best = 0, float("inf")
    if cfg.resume and (run_dir / "last.pt").exists():
        ckpt = torch.load(run_dir / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best = ckpt["epoch"] + 1, ckpt["best_val_loss"]
        log.info(f"resumed from epoch {start_epoch}")

    metrics_path = run_dir / "metrics.csv"
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, cfg, device, optimizer)
        with torch.no_grad():
            val_m = run_epoch(model, val_loader, cfg, device)
        scheduler.step()

        log.info(f"epoch {epoch + 1}/{cfg.epochs} ({time.time() - t0:.0f}s) "
                 f"train loss {train_m['loss']:.4f} | val loss {val_m['loss']:.4f} "
                 f"mae {val_m['mae']:.3f} m/s^2 (drive-straight baseline {val_m['zero_mae']:.3f})")

        row = {"epoch": epoch + 1, "lr": scheduler.get_last_lr()[0],
               **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}}
        new_file = not metrics_path.exists()
        with open(metrics_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

        if val_m["loss"] < best:
            best = val_m["loss"]
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch, "val": val_m},
                       run_dir / "best.pt")
            log.info(f"  new best, saved {run_dir / 'best.pt'}")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "config": asdict(cfg), "epoch": epoch,
                    "best_val_loss": best}, run_dir / "last.pt")


if __name__ == "__main__":
    main()
