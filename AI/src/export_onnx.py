from __future__ import annotations

import argparse
import inspect
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.model import build_model, mert_output_length
from src.train import load_checkpoint

log = logging.getLogger(__name__)


class ExportWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, wav: torch.Tensor):
        r, q, e, b = self.model(wav)
        return r, q, e, b


class V5EncoderExportWrapper(torch.nn.Module):
    """Exports MERT (with merged LoRA) + 25-layer weighted sum + 256d projection (+ 5-Octave CQT if `bass_first`)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, wav: torch.Tensor):
        emb = self.model._encode_single_chunk_batch(wav)
        if getattr(self.model, "bass_first", False):
            cqt = self.model.cqt_bank(wav)
            return emb, cqt
        return emb


class V5DecoderExportWrapper(torch.nn.Module):
    """Exports Full-Song Bi-Conformer + OctavePitchBass + DualChroma + Sustain + LocalKey + Joint Head + CRF params."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, emb: torch.Tensor, frame_rms_db: torch.Tensor, cqt: torch.Tensor | None = None):
        out = self.model.decode_context(emb, frame_rms_db=frame_rms_db, cqt=cqt)
        if "true_bass_logits" in out:
            return (
                out["crf_emissions"],
                self.model.crf.transitions,
                self.model.crf.start_transitions,
                self.model.crf.end_transitions,
                out["type_logits_by_root"],
                out["bass_logits"],
                out["key_logits"],
                out["sustain_env"],
                out["true_bass_logits"],
            )
        return (
            out["crf_emissions"],
            self.model.crf.transitions,
            self.model.crf.start_transitions,
            self.model.crf.end_transitions,
            out["type_logits_by_root"],
            out["bass_logits"],
            out["key_logits"],
            out["sustain_env"],
        )


class V6EncoderExportWrapper(torch.nn.Module):
    """Exports v6 MERT (with merged LoRA) + 25-layer weighted sum + 256d proj + 96-bin Harmonic CQT."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, wav: torch.Tensor):
        emb = self.model._encode_single_chunk_batch(wav)
        cqt = self.model.cqt_bank(wav)
        return emb, cqt


class V6DecoderExportWrapper(torch.nn.Module):
    """Exports v6 Full-Song Parallel-Kernel Conformer + Cross-Section Attention + Factorized Heads + Dynamic CRF params."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, emb: torch.Tensor, frame_rms_db: torch.Tensor, cqt: torch.Tensor):
        out = self.model.decode_context(emb, frame_rms_db=frame_rms_db, cqt=cqt)
        stay_s, boost = self.model.crf._effective_params()
        crf_dyn_params = torch.stack([stay_s, boost], dim=0)  # (2,)
        return (
            out["crf_emissions"],
            self.model.crf.transitions,
            self.model.crf.start_transitions,
            self.model.crf.end_transitions,
            crf_dyn_params,
            out["tension_logits_by_root"],
            out["inv_logits_by_root"],
            out["bass_logits"],
            out["true_bass_logits"],
            out["boundary_logits"],
            out["key_logits"],
            out["sustain_env"],
            out["fast_chord_env"],
        )


def companion_decoder_path(encoder_path: str | Path) -> Path:
    p = Path(encoder_path)
    return p.with_name(f"{p.stem}_decoder{p.suffix}")


def companion_meta_path(encoder_path: str | Path) -> Path:
    p = Path(encoder_path)
    return p.with_suffix(".meta.json")


def _base_onnx_kwargs(opset: int) -> dict:
    kw = dict(opset_version=opset, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False
    return kw


def export_onnx(
    cfg_path: str,
    checkpoint_path: str,
    output_path: str,
    opset: int = 17,
) -> None:
    cfg = load_config(cfg_path)
    arch = getattr(cfg.model, "arch_version", 4)
    is_v6 = arch >= 6
    is_v5 = arch >= 5
    device = torch.device("cpu")

    model = build_model(cfg.model, sample_rate=cfg.sample_rate).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    has_lora = any("lora_A" in k for k in state.keys())
    if has_lora:
        log.info("Checkpoint has LoRA weights — applying LoRA before loading")
        model.apply_lora(cfg.lora)

    load_checkpoint(Path(checkpoint_path), model, strict=True)
    if is_v5 and has_lora and hasattr(model, "merge_lora"):
        log.info("Merging LoRA weights into MERT linear layers for zero-overhead ONNX graph")
        model.merge_lora()
    model.eval()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    common_kw = _base_onnx_kwargs(opset)

    if is_v6:
        chunk_samples = cfg.chunk_samples
        t_chunk = mert_output_length(chunk_samples)
        cqt_bins = int(getattr(model, "cqt_bins", 96))

        enc_wrapper = V6EncoderExportWrapper(model).eval()
        dummy_wav = torch.randn(1, chunk_samples, device=device)
        torch.onnx.export(
            enc_wrapper,
            (dummy_wav,),
            str(out_p),
            input_names=["wav"],
            output_names=["emb", "cqt"],
            dynamic_axes={
                "wav": {0: "batch"},
                "emb": {0: "batch", 1: "time"},
                "cqt": {0: "batch", 1: "time"},
            },
            **common_kw,
        )
        log.info(f"Exported v6 Acoustic Encoder ONNX (cqt_bins={cqt_bins}) -> {out_p}")

        dec_path = companion_decoder_path(out_p)
        dec_wrapper = V6DecoderExportWrapper(model).eval()
        dummy_emb = torch.randn(1, t_chunk * 4, cfg.model.context_dim, device=device)
        dummy_rms = torch.full((1, t_chunk * 4), -22.0, device=device)
        dummy_cqt = torch.rand(1, t_chunk * 4, cqt_bins, device=device)
        dec_out_names = [
            "crf_emissions",
            "crf_transitions",
            "crf_start",
            "crf_end",
            "crf_dyn_params",
            "tension_logits_by_root",
            "inv_logits_by_root",
            "bass_logits",
            "true_bass_logits",
            "boundary_logits",
            "key_logits",
            "sustain_env",
            "fast_chord_env",
        ]
        dec_dyn_axes = {
            "emb": {0: "batch", 1: "time"},
            "frame_rms_db": {0: "batch", 1: "time"},
            "cqt": {0: "batch", 1: "time"},
            "crf_emissions": {0: "batch", 1: "crf_time"},
            "tension_logits_by_root": {0: "batch", 1: "time"},
            "inv_logits_by_root": {0: "batch", 1: "time"},
            "bass_logits": {0: "batch", 1: "time"},
            "true_bass_logits": {0: "batch", 1: "time"},
            "boundary_logits": {0: "batch", 1: "time"},
            "key_logits": {0: "batch", 1: "time"},
            "sustain_env": {0: "batch", 1: "time"},
            "fast_chord_env": {0: "batch", 1: "time"},
        }
        torch.onnx.export(
            dec_wrapper,
            (dummy_emb, dummy_rms, dummy_cqt),
            str(dec_path),
            input_names=["emb", "frame_rms_db", "cqt"],
            output_names=dec_out_names,
            dynamic_axes=dec_dyn_axes,
            **common_kw,
        )
        log.info(f"Exported v6 Full-Song Context Decoder ONNX -> {dec_path}")

        meta = {
            "arch_version": 6,
            "bass_first": True,
            "cqt_bins": cqt_bins,
            "n_skeletons": 14,
            "n_tensions": 7,
            "n_inversions": 5,
            "decoder_onnx": dec_path.name,
            "sample_rate": cfg.sample_rate,
            "chunk_s": cfg.model.chunk_s,
            "context_dim": cfg.model.context_dim,
            "crf_stride": cfg.model.crf_stride,
            "silence_db_threshold": cfg.model.silence_db_threshold,
        }
        companion_meta_path(out_p).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return

    if is_v5:
        chunk_samples = cfg.chunk_samples
        t_chunk = mert_output_length(chunk_samples)
        bass_first = bool(getattr(model, "bass_first", False))
        cqt_bins = int(getattr(model, "cqt_bins", 60))

        enc_wrapper = V5EncoderExportWrapper(model).eval()
        dummy_wav = torch.randn(1, chunk_samples, device=device)
        if bass_first:
            enc_out_names = ["emb", "cqt"]
            enc_dyn_axes = {
                "wav": {0: "batch"},
                "emb": {0: "batch", 1: "time"},
                "cqt": {0: "batch", 1: "time"},
            }
        else:
            enc_out_names = ["emb"]
            enc_dyn_axes = {"wav": {0: "batch"}, "emb": {0: "batch", 1: "time"}}

        torch.onnx.export(
            enc_wrapper,
            (dummy_wav,),
            str(out_p),
            input_names=["wav"],
            output_names=enc_out_names,
            dynamic_axes=enc_dyn_axes,
            **common_kw,
        )
        log.info(f"Exported v5 Acoustic Encoder ONNX (bass_first={bass_first}) -> {out_p}")

        dec_path = companion_decoder_path(out_p)
        dec_wrapper = V5DecoderExportWrapper(model).eval()
        dummy_emb = torch.randn(1, t_chunk * 4, cfg.model.context_dim, device=device)
        dummy_rms = torch.full((1, t_chunk * 4), -22.0, device=device)
        dec_out_names = [
            "crf_emissions",
            "crf_transitions",
            "crf_start",
            "crf_end",
            "type_logits_by_root",
            "bass_logits",
            "key_logits",
            "sustain_env",
        ]
        dec_dyn_axes = {
            "emb": {0: "batch", 1: "time"},
            "frame_rms_db": {0: "batch", 1: "time"},
            "crf_emissions": {0: "batch", 1: "crf_time"},
            "type_logits_by_root": {0: "batch", 1: "time"},
            "bass_logits": {0: "batch", 1: "time"},
            "key_logits": {0: "batch", 1: "time"},
            "sustain_env": {0: "batch", 1: "time"},
        }
        if bass_first:
            dummy_cqt = torch.rand(1, t_chunk * 4, cqt_bins, device=device)
            dec_inputs = (dummy_emb, dummy_rms, dummy_cqt)
            dec_in_names = ["emb", "frame_rms_db", "cqt"]
            dec_out_names.append("true_bass_logits")
            dec_dyn_axes["cqt"] = {0: "batch", 1: "time"}
            dec_dyn_axes["true_bass_logits"] = {0: "batch", 1: "time"}
        else:
            dec_inputs = (dummy_emb, dummy_rms)
            dec_in_names = ["emb", "frame_rms_db"]

        torch.onnx.export(
            dec_wrapper,
            dec_inputs,
            str(dec_path),
            input_names=dec_in_names,
            output_names=dec_out_names,
            dynamic_axes=dec_dyn_axes,
            **common_kw,
        )
        log.info(f"Exported v5 Full-Song Context Decoder ONNX -> {dec_path}")

        meta = {
            "arch_version": 5,
            "bass_first": bass_first,
            "cqt_bins": cqt_bins,
            "decoder_onnx": dec_path.name,
            "sample_rate": cfg.sample_rate,
            "chunk_s": cfg.model.chunk_s,
            "context_dim": cfg.model.context_dim,
            "crf_stride": cfg.model.crf_stride,
            "silence_db_threshold": cfg.model.silence_db_threshold,
        }
        companion_meta_path(out_p).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return

    wrapper = ExportWrapper(model).eval()
    dummy = torch.randn(1, cfg.crop_samples, device=device)
    dynamic_axes = {
        "wav": {0: "batch"},
        "root_logits": {0: "batch"},
        "quality_logits": {0: "batch"},
        "ext_logits": {0: "batch"},
        "bass_logits": {0: "batch"},
    }
    if cfg.model.frame_level:
        for name in ["root_logits", "quality_logits", "ext_logits", "bass_logits"]:
            dynamic_axes[name][1] = "time"

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_p),
        input_names=["wav"],
        output_names=["root_logits", "quality_logits", "ext_logits", "bass_logits"],
        dynamic_axes=dynamic_axes,
        **common_kw,
    )
    log.info(f"Exported ONNX to {out_p}")


def benchmark_pytorch(cfg_path: str, checkpoint_path: str, n_runs: int = 50) -> None:
    cfg = load_config(cfg_path)
    is_v5 = getattr(cfg.model, "arch_version", 4) >= 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg.model, sample_rate=cfg.sample_rate).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if any("lora_A" in k for k in state.keys()):
        model.apply_lora(cfg.lora)
    load_checkpoint(Path(checkpoint_path), model, strict=True)
    if is_v5 and hasattr(model, "merge_lora"):
        model.merge_lora()
    model.eval()

    n_samp = cfg.chunk_samples if is_v5 else cfg.crop_samples
    dummy = torch.randn(1, n_samp, device=device)

    for _ in range(5):
        with torch.no_grad():
            model.encode_chunks(dummy) if is_v5 else model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.encode_chunks(dummy) if is_v5 else model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    log.info(
        f"PyTorch {device} (5s chunk) | mean={times_ms.mean():.1f}ms | "
        f"std={times_ms.std():.1f}ms | p95={np.percentile(times_ms, 95):.1f}ms"
    )


def benchmark_onnx(onnx_path: str, chunk_samples: int, n_runs: int = 50) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed, skipping ONNX benchmark")
        return

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    actual_provider = sess.get_providers()[0]

    dummy = np.random.randn(1, chunk_samples).astype(np.float32)
    for _ in range(5):
        sess.run(None, {"wav": dummy})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"wav": dummy})
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    log.info(
        f"ONNX Encoder ({actual_provider}) | mean={times_ms.mean():.1f}ms | "
        f"std={times_ms.std():.1f}ms | p95={np.percentile(times_ms, 95):.1f}ms"
    )

    dec_p = companion_decoder_path(onnx_path)
    if dec_p.exists():
        dec_sess = ort.InferenceSession(str(dec_p), providers=providers)
        t_60s = 4488
        in_meta = {m.name: m.shape for m in dec_sess.get_inputs()}
        c_dim = int(in_meta["emb"][2]) if isinstance(in_meta["emb"][2], int) else 256
        feed: dict[str, np.ndarray] = {
            "emb": np.random.randn(1, t_60s, c_dim).astype(np.float32),
            "frame_rms_db": np.full((1, t_60s), -20.0, dtype=np.float32),
        }
        if "cqt" in in_meta:
            cqt_bins = int(in_meta["cqt"][2]) if isinstance(in_meta["cqt"][2], int) else 60
            feed["cqt"] = np.random.rand(1, t_60s, cqt_bins).astype(np.float32)
        for _ in range(5):
            dec_sess.run(None, feed)
        d_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            dec_sess.run(None, feed)
            d_times.append(time.perf_counter() - t0)
        d_ms = np.array(d_times) * 1000
        log.info(
            f"ONNX Full-Song 60s Decoder ({actual_provider}) | mean={d_ms.mean():.2f}ms | "
            f"p95={np.percentile(d_ms, 95):.2f}ms"
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/model.onnx")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--n_runs", type=int, default=50)
    args = parser.parse_args()

    export_onnx(args.config, args.checkpoint, args.output)

    if args.benchmark:
        cfg = load_config(args.config)
        benchmark_pytorch(args.config, args.checkpoint, args.n_runs)
        n_samp = cfg.chunk_samples if getattr(cfg.model, "arch_version", 4) >= 5 else cfg.crop_samples
        benchmark_onnx(args.output, n_samp, args.n_runs)


if __name__ == "__main__":
    main()
