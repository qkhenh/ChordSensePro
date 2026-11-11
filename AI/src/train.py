from __future__ import annotations
import csv
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.augment import Augmenter
from src.config import Config, load_config
from src.dataset import ChordDataset
from src.evaluate import compute_metrics
from src.losses import ChordLoss, compute_smoothed_class_weights
from src.model import build_model

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    total_steps = max(1, total_steps)
    warmup_steps = int(total_steps * warmup_ratio)
    if 0 < warmup_steps < total_steps:
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: ChordLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    cfg: Config,
    scheduler=None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = cfg.train.fp16 and device.type == "cuda"

    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        root = batch["root"].to(device, non_blocking=True)
        quality = batch["quality"].to(device, non_blocking=True)
        ext = batch["ext"].to(device, non_blocking=True)
        bass = batch["bass"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            r, q, e, b = model(wav)
            loss, details = criterion(r, q, e, b, root, quality, ext, bass)

        if use_amp:
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scaler.get_scale() >= scale_before:
                scheduler.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += details["loss_total"]
        n_batches += 1

    return {"train_loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: ChordLoss,
    device: torch.device,
    cfg: Config,
) -> dict[str, float]:
    model.eval()
    all_root_pred, all_root_true = [], []
    all_qual_pred, all_qual_true = [], []
    all_ext_pred, all_ext_true = [], []
    all_bass_pred, all_bass_true = [], []
    total_loss = 0.0
    n_batches = 0
    use_amp = cfg.train.fp16 and device.type == "cuda"

    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        root = batch["root"].to(device, non_blocking=True)
        quality = batch["quality"].to(device, non_blocking=True)
        ext = batch["ext"].to(device, non_blocking=True)
        bass = batch["bass"].to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            r, q, e, b = model(wav)
            loss, _ = criterion(r, q, e, b, root, quality, ext, bass)

        if r.dim() == 3:
            T = min(r.shape[1], root.shape[1])
            r, q, e, b = r[:, :T], q[:, :T], e[:, :T], b[:, :T]
            root, quality, ext, bass = root[:, :T], quality[:, :T], ext[:, :T], bass[:, :T]

        total_loss += loss.item()
        n_batches += 1

        all_root_pred.append(r.argmax(-1).cpu())
        all_root_true.append(root.cpu())
        all_qual_pred.append(q.argmax(-1).cpu())
        all_qual_true.append(quality.cpu())
        all_ext_pred.append(e.argmax(-1).cpu())
        all_ext_true.append(ext.cpu())
        all_bass_pred.append(b.argmax(-1).cpu())
        all_bass_true.append(bass.cpu())

    root_pred = torch.cat(all_root_pred).reshape(-1).numpy()
    root_true = torch.cat(all_root_true).reshape(-1).numpy()
    qual_pred = torch.cat(all_qual_pred).reshape(-1).numpy()
    qual_true = torch.cat(all_qual_true).reshape(-1).numpy()
    ext_pred = torch.cat(all_ext_pred).reshape(-1).numpy()
    ext_true = torch.cat(all_ext_true).reshape(-1).numpy()
    bass_pred = torch.cat(all_bass_pred).reshape(-1).numpy()
    bass_true = torch.cat(all_bass_true).reshape(-1).numpy()

    metrics = compute_metrics(
        root_pred, root_true,
        qual_pred, qual_true,
        ext_pred, ext_true,
        bass_pred, bass_true,
    )
    metrics["val_loss"] = total_loss / max(n_batches, 1)
    return metrics


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None, strict: bool = True):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as exc:
            log.warning(f"Could not restore scheduler state ({exc}); continuing with fresh scheduler")
    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


def append_results(csv_path: str, row: dict) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run(cfg: Config, resume_path: str | None = None, phase1_ckpt: str | None = None) -> None:
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.train.fp16 and device.type == "cuda"
    log.info(f"Device: {device} | AMP fp16: {use_amp}")

    augmenter = Augmenter(cfg.augment) if cfg.augment.enabled else None
    train_ds = ChordDataset(cfg, split="train", augment_fn=augmenter)
    val_ds = ChordDataset(cfg, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg.model)

    if phase1_ckpt:
        load_checkpoint(Path(phase1_ckpt), model, strict=False)
        log.info(f"Loaded Phase 1 weights from {phase1_ckpt}")

    if cfg.lora.enabled:
        model.apply_lora(cfg.lora)
        log.info(f"LoRA applied: r={cfg.lora.r}, alpha={cfg.lora.alpha}, modules={cfg.lora.target_modules}")

    model = model.to(device)

    train_pq = Path(cfg.data_dir) / "train" / "chords.parquet"
    qual_cw, ext_cw = None, None
    if train_pq.exists():
        qual_cw = compute_smoothed_class_weights(train_pq, "quality_id", cfg.model.n_quality)
        ext_cw = compute_smoothed_class_weights(train_pq, "ext_id", cfg.model.n_ext)
        log.info(f"Computed smoothed class weights for quality ({qual_cw.tolist()}) and ext")

    criterion = ChordLoss(
        weights=cfg.train.loss_weights,
        focal_gamma=cfg.train.focal_gamma,
        quality_class_weight=qual_cw,
        ext_class_weight=ext_cw,
    ).to(device)

    param_groups = model.get_trainable_params()
    optimizer = AdamW([
        {"params": g["params"], "lr": getattr(cfg.train, g["lr_key"])}
        for g in param_groups
    ], weight_decay=cfg.train.weight_decay)

    total_steps = len(train_loader) * cfg.train.epochs
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_ratio=cfg.train.warmup_ratio)
    scaler = GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if resume_path:
        start_epoch, prev_metrics = load_checkpoint(
            Path(resume_path), model, optimizer, scheduler, scaler
        )
        best_val_loss = prev_metrics.get("val_loss", float("inf"))
        log.info(f"Resumed from epoch {start_epoch}")

    log.info(
        f"Train: {len(train_ds._valid_songs)} songs ({len(train_ds)} crops/epoch) | "
        f"Val: {len(val_ds._valid_songs)} songs ({len(val_ds)} tiles)"
    )
    log.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg, scheduler=scheduler
        )
        val_metrics = validate(model, val_loader, criterion, device, cfg)

        elapsed = time.time() - t0
        val_loss = val_metrics["val_loss"]
        cur_lr = optimizer.param_groups[0]["lr"]

        log.info(
            f"Epoch {epoch+1}/{cfg.train.epochs} ({elapsed:.0f}s, lr={cur_lr:.2e}) | "
            f"train_loss={train_metrics['train_loss']:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"root={val_metrics.get('root_acc', 0):.3f} | "
            f"qual={val_metrics.get('quality_acc', 0):.3f} | "
            f"ext={val_metrics.get('ext_acc', 0):.3f} | "
            f"bass={val_metrics.get('bass_acc', 0):.3f} | "
            f"slash={val_metrics.get('bass_slash_acc', 0):.3f} | "
            f"chord={val_metrics.get('chord_acc', 0):.3f}"
        )

        all_metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1, "elapsed_s": elapsed, "lr": cur_lr}
        append_results(cfg.results_csv, all_metrics)

        ckpt_dir = Path(cfg.checkpoint_dir)
        save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, val_metrics, ckpt_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, val_metrics, ckpt_dir / "best.pt")
            log.info(f"  ↑ New best val_loss={val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.early_stop_patience:
                log.info(f"Early stopping at epoch {epoch + 1}")
                break


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="resume Phase 2 checkpoint")
    parser.add_argument("--phase1_ckpt", default=None, help="Phase 1 best.pt to init Phase 2")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, resume_path=args.resume, phase1_ckpt=args.phase1_ckpt)
