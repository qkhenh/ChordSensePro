from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelCfg:
    name: str = "m-a-p/MERT-v1-330M"
    n_layers: int = 25
    hidden_dim: int = 1024
    n_root: int = 12
    n_quality: int = 6
    n_ext: int = 15
    n_bass: int = 13
    head_hidden: int = 128
    ext_head_hidden: int = 256
    ext_head_mid: int = 64
    bass_head_hidden: int = 128
    dropout: float = 0.3
    detach_logits: bool = True
    pooling: str = "mean"
    frame_level: bool = False


@dataclass
class TrainCfg:
    phase: int = 1
    epochs: int = 10
    batch_size: int = 16
    lr_head: float = 2e-4
    lr_lora: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    early_stop_patience: int = 3
    loss_weights: list[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])
    focal_gamma: float = 2.0
    fp16: bool = True
    seed: int = 42
    num_workers: int = 2


@dataclass
class LoRACfg:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class AugmentCfg:
    enabled: bool = False
    pitch_shift_range: list[int] = field(default_factory=lambda: [-5, 6])
    gain_db: float = 6.0
    noise_snr_range: list[int] = field(default_factory=lambda: [10, 30])
    noise_prob: float = 0.5
    reverb_prob: float = 0.0
    time_mask_prob: float = 0.0
    time_mask_max_ms: int = 150
    n_time_masks: int = 2


@dataclass
class EvalCfg:
    bootstrap_n: int = 1000


@dataclass
class Config:
    data_dir: str = "../DE/data/export/v2"
    crop_s: int = 5
    sample_rate: int = 24000
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    lora: LoRACfg = field(default_factory=LoRACfg)
    augment: AugmentCfg = field(default_factory=AugmentCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    checkpoint_dir: str = "results/checkpoints"
    results_csv: str = "results/results.csv"

    @property
    def crop_samples(self) -> int:
        return self.crop_s * self.sample_rate


def load_config(path: str | Path) -> Config:
    _load_dotenv()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = Config(
        data_dir=raw.get("data_dir", Config.data_dir),
        crop_s=raw.get("crop_s", Config.crop_s),
        sample_rate=raw.get("sample_rate", Config.sample_rate),
        checkpoint_dir=raw.get("checkpoint_dir", Config.checkpoint_dir),
        results_csv=raw.get("results_csv", Config.results_csv),
    )
    if "model" in raw:
        cfg.model = ModelCfg(**raw["model"])
    if "train" in raw:
        cfg.train = TrainCfg(**raw["train"])
    if "lora" in raw:
        cfg.lora = LoRACfg(**raw["lora"])
    if "augment" in raw:
        cfg.augment = AugmentCfg(**raw["augment"])
    if "eval" in raw:
        cfg.eval = EvalCfg(**raw["eval"])
    _apply_env_overrides(cfg)
    return cfg


def _load_dotenv() -> None:
    for candidate in [".env", "../.env"]:
        p = Path(candidate)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if key and key not in os.environ:
                    os.environ[key] = val
            break


def _env(key: str, default=None):
    return os.environ.get(key, default)


def _env_float(key: str, default=None):
    v = _env(key)
    return float(v) if v is not None else default


def _env_int(key: str, default=None):
    v = _env(key)
    return int(v) if v is not None else default


def _env_bool(key: str, default=None):
    v = _env(key)
    if v is None:
        return default
    return v.lower() in ("true", "1", "yes")


def _env_list_float(key: str, default=None):
    v = _env(key)
    if v is None:
        return default
    return [float(x.strip()) for x in v.split(",")]


def _env_list_int(key: str, default=None):
    v = _env(key)
    if v is None:
        return default
    return [int(x.strip()) for x in v.split(",")]


def _apply_env_overrides(cfg: Config) -> None:
    if _env("DATA_DIR"):
        cfg.data_dir = _env("DATA_DIR")
    if _env("CHECKPOINT_DIR"):
        cfg.checkpoint_dir = _env("CHECKPOINT_DIR")
    if _env("RESULTS_CSV"):
        cfg.results_csv = _env("RESULTS_CSV")
    if _env_int("CROP_S") is not None:
        cfg.crop_s = _env_int("CROP_S")
    if _env_int("SAMPLE_RATE") is not None:
        cfg.sample_rate = _env_int("SAMPLE_RATE")

    if _env("MERT_MODEL"):
        cfg.model.name = _env("MERT_MODEL")
    if _env("POOLING"):
        cfg.model.pooling = _env("POOLING")
    if _env_bool("FRAME_LEVEL") is not None:
        cfg.model.frame_level = _env_bool("FRAME_LEVEL")
    if _env_bool("DETACH_LOGITS") is not None:
        cfg.model.detach_logits = _env_bool("DETACH_LOGITS")
    if _env_float("DROPOUT") is not None:
        cfg.model.dropout = _env_float("DROPOUT")

    if _env_int("EPOCHS") is not None:
        cfg.train.epochs = _env_int("EPOCHS")
    if _env_int("BATCH_SIZE") is not None:
        cfg.train.batch_size = _env_int("BATCH_SIZE")
    if _env_float("LR_HEAD") is not None:
        cfg.train.lr_head = _env_float("LR_HEAD")
    if _env_float("LR_LORA") is not None:
        cfg.train.lr_lora = _env_float("LR_LORA")
    if _env_float("WEIGHT_DECAY") is not None:
        cfg.train.weight_decay = _env_float("WEIGHT_DECAY")
    if _env_float("GRAD_CLIP") is not None:
        cfg.train.grad_clip = _env_float("GRAD_CLIP")
    if _env_int("EARLY_STOP_PATIENCE") is not None:
        cfg.train.early_stop_patience = _env_int("EARLY_STOP_PATIENCE")
    if _env_bool("FP16") is not None:
        cfg.train.fp16 = _env_bool("FP16")
    if _env_int("SEED") is not None:
        cfg.train.seed = _env_int("SEED")
    if _env_int("NUM_WORKERS") is not None:
        cfg.train.num_workers = _env_int("NUM_WORKERS")
    if _env_list_float("LOSS_WEIGHTS") is not None:
        cfg.train.loss_weights = _env_list_float("LOSS_WEIGHTS")
    if _env_float("FOCAL_GAMMA") is not None:
        cfg.train.focal_gamma = _env_float("FOCAL_GAMMA")

    if _env_bool("LORA_ENABLED") is not None:
        cfg.lora.enabled = _env_bool("LORA_ENABLED")
    if _env_int("LORA_R") is not None:
        cfg.lora.r = _env_int("LORA_R")
    if _env_int("LORA_ALPHA") is not None:
        cfg.lora.alpha = _env_int("LORA_ALPHA")
    if _env_float("LORA_DROPOUT") is not None:
        cfg.lora.dropout = _env_float("LORA_DROPOUT")

    if _env_bool("AUGMENT_ENABLED") is not None:
        cfg.augment.enabled = _env_bool("AUGMENT_ENABLED")
    if _env_list_int("PITCH_SHIFT_RANGE") is not None:
        cfg.augment.pitch_shift_range = _env_list_int("PITCH_SHIFT_RANGE")
    if _env_float("GAIN_DB") is not None:
        cfg.augment.gain_db = _env_float("GAIN_DB")
    if _env_float("NOISE_PROB") is not None:
        cfg.augment.noise_prob = _env_float("NOISE_PROB")

