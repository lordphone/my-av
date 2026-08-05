# train_op.py
# Trains OpModel (openpilot input format, lateral accel output) on comma2k19

import os
import argparse
import logging
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from src.data.comma2k19dataset import Comma2k19Dataset
from src.data.op_dataset import OpWindowDataset
from src.models.op_model import OpModel

BURN_IN = 10  # frames at the start of each window used only to warm up the GRU


def split_by_route(dataset, val_fraction=0.1, seed=0):
    routes = sorted({s["route_id"] for s in dataset.samples})
    random.Random(seed).shuffle(routes)
    val_routes = set(routes[:max(1, int(len(routes) * val_fraction))])
    train_idx = [i for i, s in enumerate(dataset.samples) if s["route_id"] not in val_routes]
    val_idx = [i for i, s in enumerate(dataset.samples) if s["route_id"] in val_routes]
    return train_idx, val_idx


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for batch in loader:
        imgs = batch["imgs"].to(device, non_blocking=True)
        v_ego = batch["v_ego"].to(device)
        action_t = batch["action_t"].to(device)
        target = batch["target"].to(device)

        with torch.set_grad_enabled(training), autocast(device.type, enabled=device.type == "cuda"):
            preds, _ = model(imgs, v_ego, action_t)
            loss = criterion(preds[:, BURN_IN:].float(), target[:, BURN_IN:])

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        total += loss.item()
        count += 1
        if training and count % 50 == 0:
            logging.info(f"  batch {count} loss {total / count:.4f}")
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="checkpoints/op_model.pth")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = Comma2k19Dataset(args.dataset)
    train_idx, val_idx = split_by_route(base)
    logging.info(f"{len(train_idx)} train segments, {len(val_idx)} val segments")

    train_loader = DataLoader(OpWindowDataset(base, train_idx, args.window), batch_size=args.batch_size,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(OpWindowDataset(base, val_idx, args.window, shuffle=False), batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=True)

    model = OpModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    criterion = nn.SmoothL1Loss()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best = float("inf")
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        logging.info(f"epoch {epoch + 1}/{args.epochs} train {train_loss:.4f} val {val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, args.out)
            logging.info(f"saved {args.out}")


if __name__ == "__main__":
    main()
