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
from src.model import build_model
from src.train import load_checkpoint

log = logging.getLogger(__name__)


def load_combo_mask(classes_json: str) -> torch.Tensor | None:
    path = Path(classes_json)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
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
    step_s: float = 1.0,
    rms_threshold_db: float = -40.0,
    confidence_threshold: float = 0.3,
    combo_mask: torch.Tensor | None = None,
) -> list[dict]:
    sr = cfg.sample_rate
    crop_samples = cfg.crop_samples
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
                "confidence": 0.0,
                "is_silence": True,
            })
            continue

        wav = crop.astype(np.float32) / 32768.0
        std = wav.std()
        if std > 1e-6:
            wav = (wav - wav.mean()) / std

        wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)

        with autocast("cuda"):
            r, q, e = model(wav_t)

        root_prob = F.softmax(r, dim=-1)
        qual_prob = F.softmax(q, dim=-1)
        ext_prob = F.softmax(e, dim=-1)

        root_id = root_prob.argmax(-1).item()
        root_conf = root_prob[0, root_id].item()

        qual_id = qual_prob.argmax(-1).item()

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

        avg_conf = (root_conf + qual_prob[0, qual_id].item() + ext_conf) / 3

        if root_conf < confidence_threshold:
            root_id = -1
            qual_id = -1
            ext_id = -1

        results.append({
            "start_ms": pos / sr * 1000,
            "end_ms": (pos + crop_samples) / sr * 1000,
            "root": root_id,
            "quality": qual_id,
            "ext": ext_id,
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
                and r["ext"] == prev["ext"]):
            prev["end_ms"] = r["end_ms"]
            prev["confidence"] = max(prev["confidence"], r["confidence"])
        else:
            merged.append(r.copy())
    return merged


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True, help="path to .npy int16 mono audio")
    parser.add_argument("--step_s", type=float, default=1.0)
    parser.add_argument("--classes_json", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg.model).to(device)
    load_checkpoint(Path(args.checkpoint), model)
    model.eval()

    combo_mask = None
    if args.classes_json:
        combo_mask = load_combo_mask(args.classes_json)
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
            log.info(r)


if __name__ == "__main__":
    main()

