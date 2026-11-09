from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.model import build_model
from src.train import load_checkpoint

log = logging.getLogger(__name__)


class ExportWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, wav: torch.Tensor):
        r, q, e = self.model(wav)
        return r, q, e


def export_onnx(
    cfg_path: str,
    checkpoint_path: str,
    output_path: str,
    opset: int = 17,
) -> None:
    cfg = load_config(cfg_path)
    device = torch.device("cpu")

    model = build_model(cfg.model).to(device)
    load_checkpoint(Path(checkpoint_path), model)
    model.eval()

    wrapper = ExportWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, cfg.crop_samples, device=device)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy,),
        output_path,
        input_names=["wav"],
        output_names=["root_logits", "quality_logits", "ext_logits"],
        dynamic_axes={
            "wav": {0: "batch"},
            "root_logits": {0: "batch"},
            "quality_logits": {0: "batch"},
            "ext_logits": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    log.info(f"Exported ONNX to {output_path}")


def benchmark_pytorch(cfg_path: str, checkpoint_path: str, n_runs: int = 50) -> None:
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg.model).to(device)
    load_checkpoint(Path(checkpoint_path), model)
    model.eval()

    dummy = torch.randn(1, cfg.crop_samples, device=device)

    for _ in range(5):
        with torch.no_grad():
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000
    log.info(f"PyTorch {device} | mean={times.mean():.1f}ms | std={times.std():.1f}ms | p95={np.percentile(times, 95):.1f}ms")


def benchmark_onnx(onnx_path: str, crop_samples: int, n_runs: int = 50) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed, skipping ONNX benchmark")
        return

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    actual_provider = sess.get_providers()[0]

    dummy = np.random.randn(1, crop_samples).astype(np.float32)

    for _ in range(5):
        sess.run(None, {"wav": dummy})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"wav": dummy})
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000
    log.info(f"ONNX ({actual_provider}) | mean={times.mean():.1f}ms | std={times.std():.1f}ms | p95={np.percentile(times, 95):.1f}ms")


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
        benchmark_pytorch(args.config, args.checkpoint, args.n_runs)

        cfg = load_config(args.config)
        benchmark_onnx(args.output, cfg.crop_samples, args.n_runs)


if __name__ == "__main__":
    main()

