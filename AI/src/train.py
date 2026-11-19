from __future__ import annotations
import csv
import logging
import math
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
from src.dataset import CachedSequenceDataset, ChordDataset, normalize_chunks_clamped
from src.evaluate import compute_metrics, decode_v5_batch
from src.losses import (
    ChordLoss,
    ChordLossV5,
    compute_smoothed_class_weights,
    compute_type_class_weights,
)
from src.model import build_model, compute_chunked_frame_rms_db

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


@torch.no_grad()
def ensure_phase3_embedding_cache(
    model: torch.nn.Module,
    ds: ChordDataset,
    cache_split_dir: Path,
    device: torch.device,
    use_amp: bool,
) -> None:
    """Extract and cache `(T_song, 256)` float16 MERT+LoRA embeddings (+ `(T_song, 60)` CQT) + `frame_rms_db` per song."""
    cache_split_dir.mkdir(parents=True, exist_ok=True)
    need_cqt = bool(getattr(model, "bass_first", False))

    def _cache_valid(sid_chk: str) -> bool:
        p = cache_split_dir / f"{sid_chk}.npz"
        if not p.exists():
            return False
        if not need_cqt:
            return True
        try:
            with np.load(p) as zf:
                return "cqt" in zf
        except Exception:
            return False

    missing = [sid for sid in ds._valid_songs if not _cache_valid(sid)]
    if not missing:
        log.info(f"[Phase 3 Cache] '{ds.split}' already cached ({len(ds._valid_songs)} songs) at {cache_split_dir}")
        return

    log.info(f"[Phase 3 Cache] Extracting {len(missing)} '{ds.split}' songs -> {cache_split_dir} ...")
    model.eval()
    chunk_samples = ds.chunk_samples
    batch_chunks = 16

    for idx_s, sid in enumerate(missing, 1):
        audio = np.load(ds.audio_dir / f"{sid}.npy").astype(np.float32) / 32768.0
        n_chunks = max(1, int(math.ceil(len(audio) / float(chunk_samples))))
        padded_len = n_chunks * chunk_samples
        if len(audio) < padded_len:
            pad_buf = np.zeros(padded_len, dtype=np.float32)
            pad_buf[:len(audio)] = audio
        else:
            pad_buf = audio[:padded_len]

        rms_db = compute_chunked_frame_rms_db(pad_buf, n_chunks, chunk_samples)
        norm_wav = normalize_chunks_clamped(pad_buf, chunk_samples)
        chunks_2d = norm_wav.reshape(n_chunks, chunk_samples)

        emb_parts: list[np.ndarray] = []
        cqt_parts: list[np.ndarray] = []
        for c0 in range(0, n_chunks, batch_chunks):
            c_slice = torch.from_numpy(chunks_2d[c0: c0 + batch_chunks]).to(device)
            with autocast("cuda", enabled=use_amp):
                emb_b = model.encode_chunks(c_slice)  # (B_c, 374, 256)
                cqt_b = model.encode_cqt(c_slice) if need_cqt else None
            emb_parts.append(emb_b.reshape(-1, emb_b.shape[-1]).half().cpu().numpy())
            if cqt_b is not None:
                cqt_parts.append(cqt_b.reshape(-1, cqt_b.shape[-1]).half().cpu().numpy())

        full_emb = np.concatenate(emb_parts, axis=0)
        save_kw: dict[str, np.ndarray] = {
            "emb": full_emb,
            "rms_db": rms_db[:len(full_emb)].astype(np.float16),
        }
        if cqt_parts:
            save_kw["cqt"] = np.concatenate(cqt_parts, axis=0)[:len(full_emb)]
        np.savez(cache_split_dir / f"{sid}.npz", **save_kw)
        if idx_s % 250 == 0 or idx_s == len(missing):
            log.info(f"  [{ds.split}] cached {idx_s}/{len(missing)} songs")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    cfg: Config,
    scheduler=None,
) -> dict[str, float]:
    model.train()
    is_v5 = getattr(cfg.model, "arch_version", 4) >= 5
    is_phase3 = is_v5 and cfg.train.phase == 3
    if is_phase3:
        # Keep frozen MERT in eval mode during Phase 3 sequence polish
        model.mert.eval()

    total_loss = 0.0
    n_batches = 0
    use_amp = cfg.train.fp16 and device.type == "cuda"
    accum_steps = max(1, int(getattr(cfg.train, "grad_accum_steps", 1)))
    num_loader_batches = len(loader)

    optimizer.zero_grad(set_to_none=True)

    for step_idx, batch in enumerate(loader):
        is_last_batch = (step_idx + 1) == num_loader_batches
        should_step = ((step_idx + 1) % accum_steps == 0) or is_last_batch
        # Exact divisor for trailing partial accumulation window
        eff_div = accum_steps if ((step_idx + 1) % accum_steps == 0 or not is_last_batch) else ((step_idx % accum_steps) + 1)

        if is_v5:
            gpu_batch = {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            with autocast("cuda", enabled=use_amp):
                if "emb" in gpu_batch:
                    outputs = model.decode_context(
                        gpu_batch["emb"],
                        frame_rms_db=gpu_batch["frame_rms_db"],
                        cqt=gpu_batch.get("cqt"),
                    )
                else:
                    outputs = model(gpu_batch["wav"], frame_rms_db=gpu_batch["frame_rms_db"])
                loss, details = criterion(outputs, gpu_batch, model.crf)
        else:
            wav = batch["wav"].to(device, non_blocking=True)
            root = batch["root"].to(device, non_blocking=True)
            quality = batch["quality"].to(device, non_blocking=True)
            ext = batch["ext"].to(device, non_blocking=True)
            bass = batch["bass"].to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                r, q, e, b = model(wav)
                loss, details = criterion(r, q, e, b, root, quality, ext, bass)

        scaled_loss = loss / float(eff_div)

        if use_amp:
            scaler.scale(scaled_loss).backward()
            if should_step:
                scale_before = scaler.get_scale()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and scaler.get_scale() >= scale_before:
                    scheduler.step()
        else:
            scaled_loss.backward()
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

        total_loss += details["loss_total"]
        n_batches += 1

    return {"train_loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    cfg: Config,
) -> dict[str, float]:
    model.eval()
    is_v5 = getattr(cfg.model, "arch_version", 4) >= 5
    use_amp = cfg.train.fp16 and device.type == "cuda"

    if is_v5:
        all_rp, all_rt = [], []
        all_qp, all_qt = [], []
        all_ep, all_et = [], []
        all_bp, all_bt = [], []
        all_sp, all_st = [], []
        all_kp, all_kt = [], []
        all_tbp, all_tbt = [], []
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            gpu_batch = {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            with autocast("cuda", enabled=use_amp):
                if "emb" in gpu_batch:
                    outputs = model.decode_context(
                        gpu_batch["emb"],
                        frame_rms_db=gpu_batch["frame_rms_db"],
                        cqt=gpu_batch.get("cqt"),
                    )
                else:
                    outputs = model(gpu_batch["wav"], frame_rms_db=gpu_batch["frame_rms_db"])
                loss, _ = criterion(outputs, gpu_batch, model.crf)

            preds = decode_v5_batch(outputs, model.crf, crf_stride=cfg.model.crf_stride)
            T = min(preds["root"].shape[1], gpu_batch["root"].shape[1])

            total_loss += float(loss.item())
            n_batches += 1
            all_rp.append(preds["root"][:, :T].cpu())
            all_rt.append(gpu_batch["root"][:, :T].cpu())
            all_qp.append(preds["quality"][:, :T].cpu())
            all_qt.append(gpu_batch["quality"][:, :T].cpu())
            all_ep.append(preds["ext"][:, :T].cpu())
            all_et.append(gpu_batch["ext"][:, :T].cpu())
            all_bp.append(preds["bass"][:, :T].cpu())
            all_bt.append(gpu_batch["bass"][:, :T].cpu())
            all_sp.append(preds["skel"][:, :T].cpu())
            all_st.append(gpu_batch["skel_id"][:, :T].cpu())
            all_kp.append(preds["key"][:, :T].cpu())
            all_kt.append(gpu_batch["local_key"][:, :T].cpu())
            if "true_bass" in gpu_batch:
                all_tbp.append(preds["true_bass"][:, :T].cpu())
                all_tbt.append(gpu_batch["true_bass"][:, :T].cpu())

        metrics = compute_metrics(
            torch.cat(all_rp).reshape(-1).numpy(),
            torch.cat(all_rt).reshape(-1).numpy(),
            torch.cat(all_qp).reshape(-1).numpy(),
            torch.cat(all_qt).reshape(-1).numpy(),
            torch.cat(all_ep).reshape(-1).numpy(),
            torch.cat(all_et).reshape(-1).numpy(),
            torch.cat(all_bp).reshape(-1).numpy(),
            torch.cat(all_bt).reshape(-1).numpy(),
            skel_pred=torch.cat(all_sp).reshape(-1).numpy(),
            skel_true=torch.cat(all_st).reshape(-1).numpy(),
            key_pred=torch.cat(all_kp).reshape(-1).numpy(),
            key_true=torch.cat(all_kt).reshape(-1).numpy(),
            true_bass_pred=torch.cat(all_tbp).reshape(-1).numpy() if all_tbp else None,
            true_bass_true=torch.cat(all_tbt).reshape(-1).numpy() if all_tbt else None,
        )
        metrics["val_loss"] = total_loss / max(n_batches, 1)
        return metrics

    all_root_pred, all_root_true = [], []
    all_qual_pred, all_qual_true = [], []
    all_ext_pred, all_ext_true = [], []
    all_bass_pred, all_bass_true = [], []
    total_loss = 0.0
    n_batches = 0

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


def load_warm_compatible_weights(path: Path, model: torch.nn.Module, lora_cfg=None) -> int:
    """Warm-start matching layers (including merging prior LoRA MERT weights if present) across architecture upgrades."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    src_state = ckpt.get("model_state_dict", ckpt)
    has_src_lora = any("lora_A" in k for k in src_state.keys())
    if has_src_lora and hasattr(model, "apply_lora") and hasattr(model, "merge_lora") and lora_cfg is not None:
        from src.config import LoRACfg
        # Infer r from checkpoint lora_A shape so any prior LoRA checkpoint merges cleanly
        sample_a = next(v for k, v in src_state.items() if "lora_A" in k)
        r_src = int(sample_a.shape[0])
        tmp_lora = LoRACfg(
            enabled=True,
            r=r_src,
            alpha=r_src * 2,
            dropout=0.0,
            target_modules=list(lora_cfg.target_modules) if lora_cfg.target_modules else ["q_proj", "v_proj", "k_proj", "out_proj"],
        )
        model.apply_lora(tmp_lora)
        cur_sd = model.state_dict()
        mert_match = {k: v for k, v in src_state.items() if k.startswith("mert.") and k in cur_sd and cur_sd[k].shape == v.shape}
        model.load_state_dict(mert_match, strict=False)
        model.merge_lora()

    cur_sd = model.state_dict()
    compatible = {
        k: v for k, v in src_state.items()
        if not k.startswith("mert.") and k in cur_sd and cur_sd[k].shape == v.shape
    }
    if not has_src_lora:
        mert_plain = {
            k: v for k, v in src_state.items()
            if k.startswith("mert.") and k in cur_sd and cur_sd[k].shape == v.shape
        }
        compatible.update(mert_plain)
    model.load_state_dict(compatible, strict=False)
    return len(compatible)


def run(
    cfg: Config,
    resume_path: str | None = None,
    phase1_ckpt: str | None = None,
    phase2_ckpt: str | None = None,
    warm_ckpt: str | None = None,
) -> None:
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.train.fp16 and device.type == "cuda"
    is_v5 = getattr(cfg.model, "arch_version", 4) >= 5
    log.info(f"Device: {device} | AMP fp16: {use_amp} | Arch: v{cfg.model.arch_version} | Phase: {cfg.train.phase}")

    augmenter = Augmenter(cfg.augment) if cfg.augment.enabled else None
    base_train_ds = ChordDataset(cfg, split="train", augment_fn=augmenter)
    base_val_ds = ChordDataset(cfg, split="val")

    model = build_model(cfg.model, sample_rate=cfg.sample_rate)

    if warm_ckpt and Path(warm_ckpt).exists():
        n_warm = load_warm_compatible_weights(Path(warm_ckpt), model, lora_cfg=cfg.lora)
        log.info(f"Warm-started {n_warm} compatible tensors (+ merged MERT acoustic weights) from {warm_ckpt}")

    if phase1_ckpt:
        load_checkpoint(Path(phase1_ckpt), model, strict=False)
        log.info(f"Loaded Phase 1 weights from {phase1_ckpt}")

    if cfg.lora.enabled:
        model.apply_lora(cfg.lora)
        log.info(f"LoRA applied: r={cfg.lora.r}, alpha={cfg.lora.alpha}, modules={cfg.lora.target_modules}")

    if phase2_ckpt:
        ckpt_p2 = torch.load(phase2_ckpt, map_location="cpu", weights_only=False)
        st_p2 = ckpt_p2.get("model_state_dict", ckpt_p2)
        if any("lora_A" in k for k in st_p2.keys()) and not cfg.lora.enabled:
            model.apply_lora(cfg.lora)
        load_checkpoint(Path(phase2_ckpt), model, strict=True)
        log.info(f"Loaded Phase 2 weights from {phase2_ckpt}")

    model = model.to(device)

    # Phase 3: Full-song sequence polish on cached 256-dim MERT+LoRA embeddings
    if is_v5 and cfg.train.phase == 3:
        cache_root = Path(cfg.train.cache_dir or (Path(cfg.checkpoint_dir).parent / "v5_emb_cache"))
        ensure_phase3_embedding_cache(model, base_train_ds, cache_root / "train", device, use_amp)
        ensure_phase3_embedding_cache(model, base_val_ds, cache_root / "val", device, use_amp)

        # Freeze acoustic front-end & offload MERT to CPU RAM so GPU runs batch_size=16 60s sequences easily
        for p in model.mert.parameters():
            p.requires_grad = False
        model.layer_weights.requires_grad = False
        for p in model.input_proj.parameters():
            p.requires_grad = False
        model.mert.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        train_ds = CachedSequenceDataset(base_train_ds, cache_root / "train", max_seq_s=cfg.crop_s)
        val_ds = CachedSequenceDataset(base_val_ds, cache_root / "val", max_seq_s=cfg.crop_s)
    else:
        train_ds = base_train_ds
        val_ds = base_val_ds

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(train_ds) >= cfg.train.batch_size * 2),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    train_pq = Path(cfg.data_dir) / "train" / "chords.parquet"
    if is_v5:
        type_cw = compute_type_class_weights(train_pq) if train_pq.exists() else None
        criterion = ChordLossV5(
            weights=cfg.train.loss_weights,
            focal_gamma=cfg.train.focal_gamma,
            type_class_weight=type_cw,
            crf_stride=cfg.model.crf_stride,
        ).to(device)
    else:
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
        for g in param_groups if g["params"]
    ], weight_decay=cfg.train.weight_decay)

    accum_steps = max(1, int(getattr(cfg.train, "grad_accum_steps", 1)))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / float(accum_steps)))
    total_steps = steps_per_epoch * cfg.train.epochs
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
        f"Train: {len(base_train_ds._valid_songs)} songs ({len(train_ds)} items/epoch, accum={accum_steps}) | "
        f"Val: {len(base_val_ds._valid_songs)} songs ({len(val_ds)} items)"
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

        skel_str = f" | skel={val_metrics['skeleton_acc']:.3f}" if "skeleton_acc" in val_metrics else ""
        tbass_str = f" | tbass={val_metrics['true_bass_acc']:.3f}" if "true_bass_acc" in val_metrics else ""
        key_str = f" | key={val_metrics['key_rel_acc']:.3f}" if "key_rel_acc" in val_metrics else ""
        log.info(
            f"Epoch {epoch+1}/{cfg.train.epochs} ({elapsed:.0f}s, lr={cur_lr:.2e}) | "
            f"train_loss={train_metrics['train_loss']:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"root={val_metrics.get('root_acc', 0):.3f}{tbass_str}{skel_str}{key_str} | "
            f"qual={val_metrics.get('quality_acc', 0):.3f} | "
            f"ext={val_metrics.get('ext_acc', 0):.3f} | "
            f"slash_f1={val_metrics.get('bass_slash_f1', 0):.3f} | "
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
    parser.add_argument("--resume", default=None, help="Resume checkpoint")
    parser.add_argument("--phase1_ckpt", default=None, help="Phase 1 best.pt to init Phase 2")
    parser.add_argument("--phase2_ckpt", default=None, help="Phase 2 best.pt to init Phase 3 full-song polish")
    parser.add_argument("--warm_ckpt", default=None, help="Optional prior checkpoint to warm-start MERT & matching layers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, resume_path=args.resume, phase1_ckpt=args.phase1_ckpt, phase2_ckpt=args.phase2_ckpt, warm_ckpt=args.warm_ckpt)
