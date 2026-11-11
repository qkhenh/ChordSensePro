from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from src.config import Config, load_config
from src.model import MERT_HOP_SAMPLES, build_model, mert_output_length
from src.serve import _smooth_jitter, _triangle_weights, chord_name, weighted_chord_conf
from src.train import load_checkpoint

log = logging.getLogger(__name__)


def load_combo_mask(classes_json: str | None) -> torch.Tensor | None:
    candidates = [classes_json] if classes_json else []
    candidates.extend([
        "../DE/data/export/v3/classes.json",
        "../DE/data/export/v2/classes.json",
        "DE/data/export/v3/classes.json",
    ])
    for cand in candidates:
        if cand and Path(cand).exists():
            data = json.loads(Path(cand).read_text(encoding="utf-8"))
            if "valid_combo_mask" in data:
                return torch.tensor(data["valid_combo_mask"], dtype=torch.bool)
    return None


def sliding_window(
    audio: np.ndarray,
    sr: int,
    crop_samples: int,
    step_samples: int,
) -> list[tuple[int, np.ndarray]]:
    windows = []
    audio_len = len(audio)
    if audio_len < crop_samples:
        pad = np.zeros(crop_samples, dtype=audio.dtype)
        pad[:audio_len] = audio
        windows.append((0, pad))
        return windows

    pos = 0
    while pos + crop_samples <= audio_len:
        windows.append((pos, audio[pos: pos + crop_samples]))
        pos += step_samples

    if pos < audio_len and pos + crop_samples > audio_len:
        last_start = audio_len - crop_samples
        if last_start != pos - step_samples:
            windows.append((last_start, audio[last_start: last_start + crop_samples]))

    return windows


def rms_gate(crop: np.ndarray, threshold_db: float = -40.0) -> bool:
    rms = np.sqrt(np.mean(crop.astype(np.float32) ** 2) + 1e-10)
    rms_db = 20 * np.log10(rms / 32768.0 + 1e-10)
    return rms_db >= threshold_db


@torch.no_grad()
def infer_song(
    model: torch.nn.Module,
    audio: np.ndarray,
    cfg: Config,
    device: torch.device,
    step_s: float = 2.5,
    rms_threshold_db: float = -40.0,
    confidence_threshold: float = 0.30,
    combo_mask: torch.Tensor | None = None,
) -> list[dict]:
    sr = cfg.sample_rate
    crop_samples = cfg.crop_samples
    use_amp = device.type == "cuda"

    if cfg.model.frame_level:
        step_samples = max(MERT_HOP_SAMPLES, int(round(step_s * sr / MERT_HOP_SAMPLES)) * MERT_HOP_SAMPLES)
        windows = sliding_window(audio, sr, crop_samples, step_samples)
        audio_len = len(audio)
        duration_ms = round(audio_len / sr * 1000.0, 1)
        total_frames = max(1, mert_output_length(max(audio_len, crop_samples)))
        if audio_len < crop_samples:
            total_frames = max(1, int(np.ceil(audio_len / MERT_HOP_SAMPLES)))

        acc_r = np.zeros((total_frames, cfg.model.n_root), dtype=np.float32)
        acc_q = np.zeros((total_frames, cfg.model.n_quality), dtype=np.float32)
        acc_e = np.zeros((total_frames, cfg.model.n_ext), dtype=np.float32)
        acc_b = np.zeros((total_frames, cfg.model.n_bass), dtype=np.float32)
        acc_w = np.zeros(total_frames, dtype=np.float32)

        for pos, crop in windows:
            if not rms_gate(crop, rms_threshold_db):
                continue
            wav = crop.astype(np.float32) / 32768.0
            std = wav.std()
            if std > 1e-6:
                wav = (wav - wav.mean()) / std
            wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)
            with autocast("cuda", enabled=use_amp):
                r, q, e, b = model(wav_t)
            rp = F.softmax(r[0].float(), dim=-1).cpu().numpy()
            qp = F.softmax(q[0].float(), dim=-1).cpu().numpy()
            ep = F.softmax(e[0].float(), dim=-1).cpu().numpy()
            bp = F.softmax(b[0].float(), dim=-1).cpu().numpy()

            n_f = len(rp)
            tw = _triangle_weights(n_f)
            start_f = int(round(pos / float(MERT_HOP_SAMPLES)))
            end_f = min(total_frames, start_f + n_f)
            span = end_f - start_f
            if span <= 0:
                continue
            w_col = tw[:span, None]
            acc_r[start_f:end_f] += rp[:span] * w_col
            acc_q[start_f:end_f] += qp[:span] * w_col
            acc_e[start_f:end_f] += ep[:span] * w_col
            acc_b[start_f:end_f] += bp[:span] * w_col
            acc_w[start_f:end_f] += tw[:span]

        combo_np = combo_mask.cpu().numpy() if combo_mask is not None else None
        frame_ms = MERT_HOP_SAMPLES / float(sr) * 1000.0
        frames = []
        for g in range(total_frames):
            if acc_w[g] <= 0:
                r_id, q_id, e_id, b_id, conf, silent = -1, -1, -1, 12, 0.0, True
            else:
                rp_g = acc_r[g] / acc_w[g]
                qp_g = acc_q[g] / acc_w[g]
                ep_g = acc_e[g] / acc_w[g]
                bp_g = acc_b[g] / acc_w[g]
                r_id = int(rp_g.argmax())
                q_id = int(qp_g.argmax())
                if combo_np is not None:
                    em = ep_g.copy()
                    em[~combo_np[q_id]] = 0.0
                    s = em.sum()
                    if s > 0:
                        em /= s
                    e_id = int(em.argmax())
                    e_conf = float(em[e_id])
                else:
                    e_id = int(ep_g.argmax())
                    e_conf = float(ep_g[e_id])
                b_id = int(bp_g.argmax())
                conf = float(weighted_chord_conf(float(rp_g[r_id]), float(qp_g[q_id]), e_conf))
                if conf < confidence_threshold:
                    r_id, q_id, e_id, b_id = -1, -1, -1, 12
                silent = False

            s_ms = round(g * frame_ms, 1)
            e_ms = duration_ms if g == total_frames - 1 else round((g + 1) * frame_ms, 1)
            frames.append({
                "start_ms": s_ms,
                "end_ms": e_ms,
                "root_id": r_id,
                "quality_id": q_id,
                "ext_id": e_id,
                "bass_id": b_id,
                "chord": chord_name(r_id, q_id, e_id, b_id),
                "confidence": round(conf, 4),
                "silent": silent,
            })

        min_frames = max(1, int(round(160.0 / frame_ms)))
        smoothed = _smooth_jitter(frames, max_jitter_frames=min_frames)
        # Convert keys to legacy infer_song format + chord name
        compact = [
            {
                "start_ms": f["start_ms"],
                "end_ms": f["end_ms"],
                "root": f["root_id"],
                "quality": f["quality_id"],
                "ext": f["ext_id"],
                "bass": f["bass_id"],
                "chord": f["chord"],
                "confidence": f["confidence"],
                "is_silence": f["silent"],
            }
            for f in smoothed
        ]
        return merge_consecutive(compact)

    step_samples = int(step_s * sr)
    windows = sliding_window(audio, sr, crop_samples, step_samples)
    results = []

    for pos, crop in windows:
        if not rms_gate(crop, rms_threshold_db):
            results.append({
                "start_ms": pos / sr * 1000,
                "end_ms": (pos + crop_samples) / sr * 1000,
                "root": -1,
                "quality": -1,
                "ext": -1,
                "bass": 12,
                "chord": "N",
                "confidence": 0.0,
                "is_silence": True,
            })
            continue

        wav = crop.astype(np.float32) / 32768.0
        std = wav.std()
        if std > 1e-6:
            wav = (wav - wav.mean()) / std

        wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)

        with autocast("cuda", enabled=use_amp):
            r, q, e, b = model(wav_t)

        root_prob = F.softmax(r.float(), dim=-1)
        qual_prob = F.softmax(q.float(), dim=-1)
        ext_prob = F.softmax(e.float(), dim=-1)
        bass_prob = F.softmax(b.float(), dim=-1)

        root_id = root_prob.argmax(-1).item()
        root_conf = root_prob[0, root_id].item()

        qual_id = qual_prob.argmax(-1).item()
        qual_conf = qual_prob[0, qual_id].item()
        bass_id = bass_prob.argmax(-1).item()

        if combo_mask is not None:
            masked_ext_prob = ext_prob.clone()
            valid = combo_mask[qual_id]
            masked_ext_prob[0, ~valid] = 0.0
            s = masked_ext_prob.sum()
            if s > 0:
                masked_ext_prob /= s
            ext_id = masked_ext_prob.argmax(-1).item()
            ext_conf = masked_ext_prob[0, ext_id].item()
        else:
            ext_id = ext_prob.argmax(-1).item()
            ext_conf = ext_prob[0, ext_id].item()

        avg_conf = float(weighted_chord_conf(root_conf, qual_conf, ext_conf))

        if avg_conf < confidence_threshold:
            root_id = -1
            qual_id = -1
            ext_id = -1
            bass_id = 12

        results.append({
            "start_ms": pos / sr * 1000,
            "end_ms": (pos + crop_samples) / sr * 1000,
            "root": root_id,
            "quality": qual_id,
            "ext": ext_id,
            "bass": bass_id,
            "chord": chord_name(root_id, qual_id, ext_id, bass_id),
            "confidence": round(avg_conf, 4),
            "is_silence": False,
        })

    return merge_consecutive(results)


def merge_consecutive(results: list[dict]) -> list[dict]:
    if not results:
        return results

    merged = [results[0].copy()]
    for r in results[1:]:
        prev = merged[-1]
        if (r["root"] == prev["root"]
                and r["quality"] == prev["quality"]
                and r["ext"] == prev["ext"]
                and r.get("bass", 12) == prev.get("bass", 12)):
            prev["end_ms"] = r["end_ms"]
            prev["confidence"] = max(prev["confidence"], r["confidence"])
        else:
            merged.append(r.copy())
    return merged


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v4_5s_p2.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True, help="path to .npy int16 mono audio")
    parser.add_argument("--step_s", type=float, default=2.5)
    parser.add_argument("--classes_json", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if any("lora_A" in k for k in state.keys()):
        model.apply_lora(cfg.lora)
    load_checkpoint(Path(args.checkpoint), model, strict=True)
    model = model.to(device).eval()

    combo_mask = load_combo_mask(args.classes_json or str(Path(cfg.data_dir) / "classes.json"))
    if combo_mask is not None:
        combo_mask = combo_mask.to(device)

    audio = np.load(args.audio)
    log.info(f"Audio length: {len(audio)/cfg.sample_rate:.1f}s")

    results = infer_song(model, audio, cfg, device, step_s=args.step_s, combo_mask=combo_mask)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        log.info(f"Saved {len(results)} segments to {args.output}")
    else:
        for r in results:
            log.info(f"{r['start_ms']/1000:6.2f}s - {r['end_ms']/1000:6.2f}s | {r['chord']:<12} | conf={r['confidence']:.3f}")


if __name__ == "__main__":
    main()
