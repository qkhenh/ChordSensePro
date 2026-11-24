from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augment import Augmenter
from src.config import AugmentCfg, Config, LoRACfg, ModelCfg
from src.conformer import BiConformerEncoder, MultiScaleSustainLayer
from src.crf import MarginalLinearChainCRF, viterbi_crf_numpy
from src.dataset import CachedSequenceDataset, ChordDataset, normalize_chunks_clamped
from src.evaluate import compute_metrics, decode_v5_batch
from src.export_onnx import companion_decoder_path, companion_meta_path, export_onnx
from src.losses import ChordLossV5
from src.model import BassPedalHoldLayer, ChordModelV5, HarmonicCQTFilterbank, mert_output_length
from src.music_theory import (
    N_CRF_STATES,
    N_SKELETONS,
    N_TYPES,
    QE_TO_TYPE,
    REL_PITCH_INDEX,
    ROOT_SILENCE_ID,
    SKEL_TYPE_MASK,
    chord_chroma_targets,
    decode_crf_skeleton_state,
    encode_crf_skeleton_state,
    estimate_local_keys,
)
from src.serve import ONNXPredictor, detect_solo_clean_audio, infer_audio
from src.train import ensure_phase3_embedding_cache, save_checkpoint


@dataclass
class _FakeMERTOut:
    hidden_states: tuple
    last_hidden_state: torch.Tensor = None


class _FakeMERT(torch.nn.Module):

    def __init__(self, n_layers: int = 25, hidden_dim: int = 64):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.proj = torch.nn.Linear(16, hidden_dim)

    def forward(self, wav):
        bsz, length = wav.shape
        t = mert_output_length(length)
        # Deterministic downsampled projection so ONNX tracing sees real tensor ops on `wav`
        frames = wav[:, : t * 320].reshape(bsz, t, 20, 16).mean(dim=2)
        base = self.proj(frames)
        hs = tuple(base for _ in range(self.n_layers))
        return _FakeMERTOut(hidden_states=hs, last_hidden_state=base)


def test_music_theory_bijections_and_key_estimation():
    assert N_TYPES == 41
    assert N_SKELETONS == 12
    assert N_CRF_STATES == 145
    assert SKEL_TYPE_MASK.sum() == 41
    assert np.all(SKEL_TYPE_MASK.sum(axis=0) == 1)  # Exact partition across 12 Skeletons

    for tid in range(N_TYPES):
        q, e = np.argwhere(QE_TO_TYPE == tid)[0]
        t_out, sk_out, crf_out = encode_crf_skeleton_state(5, int(q), int(e))
        assert t_out == tid
        r_dec, s_dec = decode_crf_skeleton_state(crf_out)
        assert r_dec == 5 and s_dec == sk_out

    # Silence state round-trip
    assert encode_crf_skeleton_state(ROOT_SILENCE_ID, -100, -100) == (-100, -100, 0)
    assert decode_crf_skeleton_state(0) == (ROOT_SILENCE_ID, -1)

    # Fadd9 (root=5, qual=0, ext=6): shell={F(5), A(9), C(0)}, full adds G(7)
    sh, fu = chord_chroma_targets(5, 0, 6)
    assert sh[5] == 1.0 and sh[9] == 1.0 and sh[0] == 1.0 and sh[7] == 0.0
    assert fu[7] == 1.0

    # Progression F -> G -> C -> Am must detect C Major (key_id = 0)
    starts = np.array([0.0, 2000.0, 4000.0, 6000.0])
    ends = np.array([2000.0, 4000.0, 6000.0, 8000.0])
    roots = np.array([5, 7, 0, 9])
    quals = np.array([0, 0, 0, 1])
    exts = np.array([0, 0, 0, 4])
    keys = estimate_local_keys(starts, ends, roots, quals, exts)
    assert np.all(keys == 0)

    # Progression Dm7 -> Gm7 -> A7 -> Dm must detect D Minor (key_id = 12 + 2 = 14)
    keys_dm = estimate_local_keys(
        starts, ends, np.array([2, 7, 9, 2]), np.array([1, 1, 2, 1]), np.array([4, 4, 5, 0])
    )
    assert np.all(keys_dm == 14)


def test_sustain_layer_separates_sustained_tension_from_fast_melody():
    layer = MultiScaleSustainLayer(short_frames=7, sustain_frames=39).eval()
    # 150 frames (~2.0s @ 75 Hz), 12 pitch classes initialized to -5.0 (off)
    logits = torch.full((1, 150, 12), -5.0)
    # Pitch 7 (G = 9th of F) sustained for 90 frames (~1.2s)
    logits[0, 20:110, 7] = 4.0
    # Pitch 2 (D = fast passing melody note) active for only 12 frames (~160ms)
    logits[0, 60:72, 2] = 4.0

    with torch.no_grad():
        sustain = layer(logits)

    sustained_peak = float(sustain[0, 65, 7].item())
    melody_blip = float(sustain[0, 65, 2].item())
    assert sustained_peak > 0.70
    assert melody_blip < 0.12


def test_marginal_crf_handles_gaps_and_matches_numpy_viterbi():
    crf = MarginalLinearChainCRF(num_tags=145)
    emissions = torch.randn(2, 40, 145, requires_grad=True)

    # Sequence with -100 unlabelled gap in the middle
    targets = torch.randint(0, 145, (2, 40))
    targets[0, 12:20] = -100
    loss = crf(emissions, targets)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)
    loss.backward()
    assert emissions.grad is not None

    # All -100 targets must return exact 0.0 without NaN
    all_masked = torch.full((2, 40), -100, dtype=torch.long)
    loss_zero = crf(emissions, all_masked)
    assert loss_zero.item() == 0.0

    # Exact parity between PyTorch batch Viterbi and NumPy Viterbi
    torch_path = crf.decode(emissions)[0].detach().cpu().numpy()
    np_path = viterbi_crf_numpy(
        emissions[0].detach().cpu().numpy(),
        crf.transitions.detach().cpu().numpy(),
        crf.start_transitions.detach().cpu().numpy(),
        crf.end_transitions.detach().cpu().numpy(),
    )
    np.testing.assert_array_equal(torch_path, np_path)


def test_chord_model_v5_forward_loss_and_pitch_rotation_equivariance():
    cfg = ModelCfg(
        name="fake",
        arch_version=5,
        n_layers=25,
        hidden_dim=64,
        context_dim=64,
        conformer_layers=2,
        conformer_heads=4,
        conformer_conv_kernel=7,
        head_hidden=64,
        type_head_hidden=64,
        bass_head_hidden=64,
        crf_stride=4,
        chunk_s=5,
        dropout=0.1,
    )
    with patch("src.model.AutoModel") as mock_auto:
        mock_auto.from_pretrained.return_value = _FakeMERT(25, 64)
        model = ChordModelV5(cfg, sample_rate=24000)

    # 20s window = 4 * 120,000 = 480,000 samples -> 4 * 374 = 1,496 frames
    wav = torch.randn(2, 480000)
    rms_db = torch.full((2, 1496), -22.0)
    rms_db[:, :80] = -65.0  # Silent intro

    outputs = model(wav, frame_rms_db=rms_db)
    assert outputs["root_logits"].shape == (2, 1496, 13)
    assert outputs["type_logits_by_root"].shape == (2, 1496, 12, 41)
    assert outputs["joint_chord_logp"].shape == (2, 1496, 12, 41)
    assert outputs["crf_emissions"].shape == (2, 374, 145)
    assert outputs["bass_logits"].shape == (2, 1496, 13)
    assert outputs["key_logits"].shape == (2, 1496, 24)

    # Verify REL_PITCH_INDEX maps F (5) + major 3rd (+4) -> A (9)
    assert REL_PITCH_INDEX[5, 4] == 9
    assert REL_PITCH_INDEX[0, 4] == 4

    batch = {
        "root": torch.randint(0, 12, (2, 1496)),
        "type_id": torch.randint(0, 41, (2, 1496)),
        "crf_state": torch.randint(1, 145, (2, 1496)),
        "bass": torch.full((2, 1496), 12, dtype=torch.long),
        "local_key": torch.zeros((2, 1496), dtype=torch.long),
        "shell_chroma": torch.zeros((2, 1496, 12)),
        "full_chroma": torch.zeros((2, 1496, 12)),
    }
    batch["root"][:, :80] = ROOT_SILENCE_ID
    batch["crf_state"][:, :80] = 0
    batch["type_id"][:, :80] = -100
    batch["bass"][:, :80] = -100

    criterion = ChordLossV5(crf_stride=4)
    loss, details = criterion(outputs, batch, model.crf)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
    loss.backward()
    assert model.head_type[0].weight.grad is not None

    decoded = decode_v5_batch(outputs, model.crf, crf_stride=4)
    assert decoded["root"].shape == (2, 1496)
    assert decoded["quality"].shape == (2, 1496)
    assert decoded["ext"].shape == (2, 1496)


def test_v5_dataset_short_song_melody_augment_and_phase3_cache(tmp_path):
    sr = 24000
    for split in ("train", "val"):
        audio_dir = tmp_path / split / "audio"
        audio_dir.mkdir(parents=True)
        # 12s song (< 20s crop_s!) with 1.2s silent intro (-70 dBFS) then loud F major chord
        wav = np.zeros(sr * 12, dtype=np.int16)
        t = np.arange(sr * 10, dtype=np.float32) / sr
        chord_f = (0.4 * np.sin(2 * np.pi * 174.61 * t) + 0.3 * np.sin(2 * np.pi * 220.0 * t)) * 32767
        wav[int(1.2 * sr): int(11.2 * sr)] = chord_f.astype(np.int16)
        np.save(audio_dir / "short_clip.npy", wav)

        df = pd.DataFrame([{
            "song_id": "short_clip",
            "start_ms": 1200,
            "end_ms": 11200,
            "label_raw": "F:maj",
            "label_status": "ok",
            "root_id": 5,
            "quality_id": 0,
            "ext_id": 0,
            "bass_id": 12,
        }])
        df.to_parquet(tmp_path / split / "chords.parquet")

    cfg = Config(data_dir=str(tmp_path), crop_s=20, sample_rate=sr)
    cfg.model = ModelCfg(
        name="fake", arch_version=5, hidden_dim=64, context_dim=64,
        conformer_layers=1, conformer_heads=4, chunk_s=5, frame_level=True,
    )
    cfg.augment = AugmentCfg(
        enabled=True, pitch_shift_range=[0, 0], gain_db=0.0, noise_prob=0.0,
        melody_overlay_prob=1.0, sustained_tension_prob=1.0, silence_intro_prob=0.0,
    )
    aug = Augmenter(cfg.augment)
    ds_train = ChordDataset(cfg, split="train", augment_fn=aug)
    assert "short_clip" in ds_train._valid_songs  # 12s clip is NOT dropped!

    item = ds_train[0]
    assert item["wav"].shape == (sr * 20,)
    assert item["root"].shape == (1496,)
    # Silent padded frames must be labelled N (12), and sustained 9th augment upgrades F -> Fadd9 (ext=6)
    assert (item["root"] == ROOT_SILENCE_ID).any()
    assert (item["ext"] == 6).any()

    # Verify soft-clamped normalization does not blow up -70 dBFS room hiss
    quiet = np.random.randn(sr * 5).astype(np.float32) * 3e-4
    norm_q = normalize_chunks_clamped(quiet, sr * 5)
    assert float(norm_q.std()) < 0.25

    # Verify Phase 3 embedding cache + CachedSequenceDataset
    with patch("src.model.AutoModel") as mock_auto:
        mock_auto.from_pretrained.return_value = _FakeMERT(25, 64)
        model = ChordModelV5(cfg.model, sample_rate=sr)

    cache_dir = tmp_path / "cache" / "train"
    ensure_phase3_embedding_cache(model, ds_train, cache_dir, torch.device("cpu"), use_amp=False)
    cached_ds = CachedSequenceDataset(ds_train, cache_dir, max_seq_s=20)
    c_item = cached_ds[0]
    assert c_item["emb"].shape == (1496, 64)


def test_v5_onnx_export_and_serve_hides_no_chord_output(tmp_path):
    import importlib.util
    from src.export_onnx import V5DecoderExportWrapper, V5EncoderExportWrapper

    sr = 24000
    cfg_yaml = tmp_path / "v5_test.yaml"
    cfg_yaml.write_text(
        "\n".join([
            f'data_dir: "{tmp_path.as_posix()}"',
            "crop_s: 20",
            "sample_rate: 24000",
            "model:",
            '  name: "fake"',
            "  arch_version: 5",
            "  n_layers: 25",
            "  hidden_dim: 64",
            "  context_dim: 64",
            "  conformer_layers: 1",
            "  conformer_heads: 4",
            "  conformer_conv_kernel: 7",
            "  head_hidden: 64",
            "  type_head_hidden: 64",
            "  bass_head_hidden: 64",
            "  crf_stride: 4",
            "  chunk_s: 5",
            "  frame_level: true",
        ]),
        encoding="utf-8",
    )

    with patch("src.model.AutoModel") as mock_auto:
        mock_auto.from_pretrained.return_value = _FakeMERT(25, 64)
        from src.config import load_config
        cfg = load_config(cfg_yaml)
        model = ChordModelV5(cfg.model, sample_rate=sr).eval()
        # Bias RootHead strongly toward C (0) when loud, and N (12) on quiet frames via silence gate
        with torch.no_grad():
            model.head_root[-1].bias.fill_(-3.0)
            model.head_root[-1].bias[0] = 4.0
            model.head_type[-1].bias.fill_(-3.0)
            model.head_type[-1].bias[0] = 5.0  # C major

        ckpt_path = tmp_path / "best.pt"
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        save_checkpoint(model, opt, None, scaler, 1, {"val_loss": 0.5}, ckpt_path)

        has_onnx_pkg = importlib.util.find_spec("onnx") is not None
        if has_onnx_pkg:
            onnx_out = tmp_path / "model_v5.onnx"
            export_onnx(str(cfg_yaml), str(ckpt_path), str(onnx_out))
            assert onnx_out.exists()
            assert companion_decoder_path(onnx_out).exists()
            assert companion_meta_path(onnx_out).exists()
            pred = ONNXPredictor(str(onnx_out))
            assert pred.is_v5 is True
        else:
            # Verify export wrappers directly and bridge into v5 serving pipeline
            enc_w = V5EncoderExportWrapper(model).eval()
            dec_w = V5DecoderExportWrapper(model).eval()

            class _LocalV5PredictorStub:
                is_v5 = True
                is_frame_level = True
                crf_stride = 4
                context_dim = 64
                v5_meta = {"chunk_s": 5, "crf_stride": 4, "context_dim": 64}

                @torch.no_grad()
                def encode_chunk_v5(self, wav_f32: np.ndarray) -> np.ndarray:
                    res = enc_w(torch.from_numpy(wav_f32).unsqueeze(0))
                    return (res[0] if isinstance(res, tuple) else res)[0].numpy()

                @torch.no_grad()
                def encode_chunk_v5_full(self, wav_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
                    res = enc_w(torch.from_numpy(wav_f32).unsqueeze(0))
                    if isinstance(res, tuple):
                        return res[0][0].numpy(), res[1][0].numpy()
                    return res[0].numpy(), None

                @torch.no_grad()
                def decode_song_v5(
                    self, song_emb: np.ndarray, frame_rms_db: np.ndarray, song_cqt: np.ndarray | None = None
                ) -> dict:
                    cqt_t = torch.from_numpy(song_cqt).unsqueeze(0) if song_cqt is not None else None
                    outs = dec_w(
                        torch.from_numpy(song_emb).unsqueeze(0),
                        torch.from_numpy(frame_rms_db).unsqueeze(0),
                        cqt_t,
                    )
                    out_d = {
                        "crf_emissions": outs[0][0].numpy(),
                        "crf_transitions": outs[1].numpy(),
                        "crf_start": outs[2].numpy(),
                        "crf_end": outs[3].numpy(),
                        "type_logits_by_root": outs[4][0].numpy(),
                        "bass_logits": outs[5][0].numpy(),
                        "key_logits": outs[6][0].numpy(),
                        "sustain_env": outs[7][0].numpy(),
                    }
                    if len(outs) >= 9:
                        out_d["true_bass_logits"] = outs[8][0].numpy()
                    return out_d

            pred = _LocalV5PredictorStub()

    # Construct 6s test audio: first 1.5s pure silence (0), remaining 4.5s clean harmonic piano wave
    audio = np.zeros(sr * 6, dtype=np.int16)
    t = np.arange(int(4.5 * sr), dtype=np.float32) / sr
    piano_c = (0.5 * np.sin(2 * np.pi * 261.63 * t) + 0.3 * np.sin(2 * np.pi * 329.63 * t)) * 32767
    audio[int(1.5 * sr):] = piano_c.astype(np.int16)

    # Pure harmonic wave (<4.5 kHz, zero percussion) should be detected as clean solo audio
    assert detect_solo_clean_audio(audio.astype(np.float32) / 32768.0, sr) is True

    # Default `include_silence=False` MUST omit the 0.00s - 1.50s `N` segment from output!
    segs_clean = infer_audio(pred, audio, sample_rate=sr, step_s=2.5, include_silence=False)
    assert len(segs_clean) >= 1
    assert all(s["chord"] != "N" and not s["silent"] for s in segs_clean)
    assert segs_clean[0]["start_ms"] >= 1350.0  # Starts right where the piano enters (~1.5s), NOT 0.0s!

    # `include_silence=True` retains the leading `N` segment for debugging
    segs_with_n = infer_audio(pred, audio, sample_rate=sr, step_s=2.5, include_silence=True)
    assert segs_with_n[0]["chord"] == "N"
    assert segs_with_n[0]["start_ms"] == 0.0

    # Also check v5 extended metrics helper
    m = compute_metrics(
        np.array([0, 12]), np.array([0, 12]),
        np.array([0, -1]), np.array([0, -100]),
        np.array([0, -1]), np.array([0, -100]),
        np.array([12, 12]), np.array([12, -100]),
        skel_pred=np.array([0, -1]), skel_true=np.array([0, -100]),
        key_pred=np.array([21, 0]), key_true=np.array([0, -100]),  # A minor (21) is relative minor of C major (0)
    )
    assert m["root_acc"] == 1.0
    assert m["silence_acc"] == 1.0
    assert m["skeleton_acc"] == 1.0
    assert m["key_rel_acc"] == 1.0


def test_v5_full_train_p1_p3_and_evaluate_split_smoke(tmp_path):
    """End-to-end smoke test of `train.run` (Phase 1 -> Phase 3 cache polish) + `evaluate_split`."""
    from src.evaluate import evaluate_split
    from src.train import run as train_run

    sr = 24000
    for split in ("train", "val", "test"):
        audio_dir = tmp_path / split / "audio"
        audio_dir.mkdir(parents=True)
        records = []
        for idx_s in range(2):
            sid = f"{split}_{idx_s}"
            t = np.arange(sr * 10, dtype=np.float32) / sr
            wav = (0.3 * np.sin(2 * np.pi * 261.63 * t) * 32767).astype(np.int16)
            np.save(audio_dir / f"{sid}.npy", wav)
            records.append({
                "song_id": sid,
                "start_ms": 0,
                "end_ms": 5000,
                "label_raw": "C:maj",
                "label_status": "ok",
                "root_id": 0,
                "quality_id": 0,
                "ext_id": 0,
                "bass_id": 12,
            })
            records.append({
                "song_id": sid,
                "start_ms": 5000,
                "end_ms": 10000,
                "label_raw": "G:7",
                "label_status": "ok",
                "root_id": 7,
                "quality_id": 2,
                "ext_id": 5,
                "bass_id": 11,
            })
        pd.DataFrame(records).to_parquet(tmp_path / split / "chords.parquet")

    cfg_p1 = Config(
        data_dir=str(tmp_path),
        crop_s=10,
        sample_rate=sr,
        checkpoint_dir=str(tmp_path / "ckpt_p1"),
        results_csv=str(tmp_path / "ckpt_p1" / "res.csv"),
    )
    cfg_p1.model = ModelCfg(
        name="fake", arch_version=5, n_layers=25, hidden_dim=64, context_dim=64,
        conformer_layers=1, conformer_heads=4, conformer_conv_kernel=5,
        head_hidden=64, type_head_hidden=64, bass_head_hidden=64, chunk_s=5, frame_level=True,
    )
    cfg_p1.train.phase = 1
    cfg_p1.train.epochs = 1
    cfg_p1.train.batch_size = 2
    cfg_p1.train.grad_accum_steps = 2
    cfg_p1.train.num_workers = 0
    cfg_p1.train.fp16 = False

    with patch("src.model.AutoModel") as mock_auto:
        mock_auto.from_pretrained.side_effect = lambda *a, **kw: _FakeMERT(25, 64)
        train_run(cfg_p1)
        p1_best = tmp_path / "ckpt_p1" / "best.pt"
        assert p1_best.exists()

        # Run Phase 3 using P1 checkpoint as base + cached sequence polish
        cfg_p3 = Config(
            data_dir=str(tmp_path),
            crop_s=15,
            sample_rate=sr,
            checkpoint_dir=str(tmp_path / "ckpt_p3"),
            results_csv=str(tmp_path / "ckpt_p3" / "res.csv"),
        )
        cfg_p3.model = cfg_p1.model
        cfg_p3.train.phase = 3
        cfg_p3.train.epochs = 1
        cfg_p3.train.batch_size = 2
        cfg_p3.train.num_workers = 0
        cfg_p3.train.fp16 = False
        cfg_p3.train.cache_dir = str(tmp_path / "emb_cache")
        cfg_p3.lora = LoRACfg(enabled=False)

        train_run(cfg_p3, phase2_ckpt=str(p1_best))
        p3_best = tmp_path / "ckpt_p3" / "best.pt"
        assert p3_best.exists()

        # Save YAML config and run `evaluate_split` on `test`
        eval_yaml = tmp_path / "eval_v5.yaml"
        eval_yaml.write_text(
            "\n".join([
                f'data_dir: "{tmp_path.as_posix()}"',
                "crop_s: 10",
                "sample_rate: 24000",
                "model:",
                '  name: "fake"',
                "  arch_version: 5",
                "  n_layers: 25",
                "  hidden_dim: 64",
                "  context_dim: 64",
                "  conformer_layers: 1",
                "  conformer_heads: 4",
                "  conformer_conv_kernel: 5",
                "  head_hidden: 64",
                "  type_head_hidden: 64",
                "  bass_head_hidden: 64",
                "  chunk_s: 5",
                "  frame_level: true",
                "train:",
                "  batch_size: 2",
                "  num_workers: 0",
                "  fp16: false",
            ]),
            encoding="utf-8",
        )
        metrics = evaluate_split(str(eval_yaml), str(p3_best), split="test")
        assert "root_acc" in metrics
        assert "skeleton_acc" in metrics
        assert "true_bass_acc" in metrics
        assert metrics["n_songs"] == 2


def test_v5_1_bass_first_cqt_pedal_hold_dropout_and_warm_start(tmp_path):
    """Verify v5.1 root-cause Bass-First components:
    1. 5-Octave `HarmonicCQTFilterbank` separates Octave-3 `E3` bass from Octave-4/5 `C4/C5` melody
    2. `BassPedalHoldLayer` sustains Beat-1 left-hand bass attacks across decaying bars
    3. `BassRegisterDropout` attenuates `<100Hz` sub-bass (`>20 dB`) while preserving `E3 (165Hz)`
    4. `load_warm_compatible_weights` merges prior v5.0 LoRA checkpoints cleanly into v5.1
    """
    from src.augment import _highpass_subbass_dropout
    from src.train import load_warm_compatible_weights

    sr = 24000
    t = np.arange(sr * 5, dtype=np.float32) / sr
    # Synthesize E3 (164.81 Hz) in bass + C4 (261.63 Hz) & C5 (523.25 Hz) in upper register
    sig = 0.45 * np.sin(2 * np.pi * 164.81 * t) + 0.50 * np.sin(2 * np.pi * 261.63 * t) + 0.45 * np.sin(2 * np.pi * 523.25 * t)
    cqt_bank = HarmonicCQTFilterbank(sample_rate=sr)
    cqt = cqt_bank(torch.from_numpy(sig).unsqueeze(0))
    assert cqt.shape == (1, 374, 60)
    cqt_oct = cqt[0].mean(dim=0).reshape(5, 12)  # Octaves 1..5 x 12 pitch classes
    # Octave 3 (index 2): E (4) is strong, C (0) is near zero!
    assert float(cqt_oct[2, 4]) > 0.50
    assert float(cqt_oct[2, 0]) < 0.08
    # Octave 4 & 5 (indices 3, 4): C (0) is strong!
    assert float(cqt_oct[3, 0]) > 0.50
    assert float(cqt_oct[4, 0]) > 0.50

    # BassPedalHoldLayer sustains a Beat-1 E bass strike (frames 10..35) well into frames 60..90
    hold_layer = BassPedalHoldLayer()
    logits = torch.full((1, 150, 12), -6.0)
    logits[0, 10:35, 4] = 5.0  # Beat-1 E strike
    held = hold_layer(logits)
    assert float(held[0, 12, 4]) > 0.40   # Immediate onset response (0ms lag)
    assert float(held[0, 75, 4]) > 0.22   # Held across Beat 2-3 after strike ended
    assert float(held[0, 75, 0]) < 0.02   # Non-struck notes stay near zero

    # BassRegisterDropout attenuates C2 (65.4 Hz) > 20 dB while preserving E3 (164.8 Hz)
    c2_wave = np.sin(2 * np.pi * 65.41 * t).astype(np.float32)
    e3_wave = np.sin(2 * np.pi * 164.81 * t).astype(np.float32)
    c2_hp = _highpass_subbass_dropout(c2_wave, sr, cutoff_hz=155.0)
    e3_hp = _highpass_subbass_dropout(e3_wave, sr, cutoff_hz=155.0)
    assert float(np.sqrt(np.mean(c2_hp ** 2))) < 0.10 * float(np.sqrt(np.mean(c2_wave ** 2)))
    assert float(np.sqrt(np.mean(e3_hp ** 2))) > 0.65 * float(np.sqrt(np.mean(e3_wave ** 2)))

    # Warm-start from a legacy v5.0 checkpoint (`bass_first=False`) into v5.1 (`bass_first=True`)
    cfg_old = ModelCfg(
        name="fake", arch_version=5, bass_first=False, n_layers=25, hidden_dim=64, context_dim=64,
        conformer_layers=1, conformer_heads=4, conformer_conv_kernel=5,
        head_hidden=64, type_head_hidden=64, bass_head_hidden=64, chunk_s=5, frame_level=True,
    )
    cfg_new = ModelCfg(
        name="fake", arch_version=5, bass_first=True, n_layers=25, hidden_dim=64, context_dim=64,
        conformer_layers=1, conformer_heads=4, conformer_conv_kernel=5,
        head_hidden=64, type_head_hidden=64, bass_head_hidden=64, chunk_s=5, frame_level=True,
    )
    with patch("src.model.AutoModel") as mock_auto:
        mock_auto.from_pretrained.side_effect = lambda *a, **kw: _FakeMERT(25, 64)
        m_old = ChordModelV5(cfg_old, sample_rate=sr)
        with torch.no_grad():
            m_old.layer_weights.fill_(1.25)
        old_ckpt = tmp_path / "old_v5.pt"
        torch.save({"model_state_dict": m_old.state_dict()}, old_ckpt)

        m_new = ChordModelV5(cfg_new, sample_rate=sr)
        n_loaded = load_warm_compatible_weights(old_ckpt, m_new)
        assert n_loaded > 5
        assert torch.allclose(m_new.layer_weights, torch.full_like(m_new.layer_weights, 1.25))


def test_jazz_chord_theory_equivalences_and_hallucination_pruner():
    """Verify Step-A & Step-B zero-retrain Jazz theory rules (`m6`, `9sus4`, `Abdim7`, `9(13)`, `7(#5b9)` & pruning)."""
    from src.music_theory import resolve_jazz_chord_theory
    from src.serve import apply_key_spelling, _prune_short_segments

    rel_s = np.zeros(12, dtype=np.float32)
    # 1. Cm7b5/Eb (root=0, qual=3, ext=4, bass=3) -> Ebm6 (root=3, sfx="m6")
    r, q, e, b, sfx = resolve_jazz_chord_theory(0, 3, 4, 3, rel_s)
    assert (r, b, sfx) == (3, 12, "m6")
    spelled = apply_key_spelling([
        {"start_ms": 0.0, "end_ms": 1500.0, "root_id": r, "quality_id": q, "ext_id": e, "bass_id": b, "chord": "", "confidence": 0.7, "silent": False, "_suffix_override": sfx}
    ], prefer_flats=True)
    assert spelled[0]["chord"] == "Ebm6"
    assert "_suffix_override" not in spelled[0]

    # 2. C11 (root=0, qual=2, ext=10) with sustained 9th & 4th and absent 3rd -> C9sus4 (qual=5, ext=8)
    rel_sus = np.zeros(12, dtype=np.float32)
    rel_sus[2] = 0.80  # 9th
    rel_sus[5] = 0.85  # 4th
    rel_sus[10] = 0.70  # b7
    assert resolve_jazz_chord_theory(0, 2, 10, 12, rel_sus)[:4] == (0, 5, 8, 12)

    # 3. Rootless E7b9 over Ab bass (root=4, qual=2, ext=7, bass=8) -> Abdim7 (root=8, qual=3, ext=5)
    assert resolve_jazz_chord_theory(4, 2, 7, 8, rel_s)[:4] == (8, 3, 5, 12)

    # 4. Ab7 + sustained 9th & 13th -> Ab9(13)
    rel_dom = np.zeros(12, dtype=np.float32)
    rel_dom[4] = 0.70  # 3rd
    rel_dom[2] = 0.65  # 9th
    rel_dom[9] = 0.75  # 13th
    assert resolve_jazz_chord_theory(8, 2, 5, 12, rel_dom) == (8, 2, 12, 12, "9(13)")

    # 5. Low-confidence <820ms transition hallucination is absorbed into strong neighbors
    segs = [
        {"start_ms": 0.0, "end_ms": 2000.0, "root_id": 5, "quality_id": 1, "ext_id": 4, "bass_id": 12, "chord": "Fm7", "confidence": 0.72, "silent": False},
        {"start_ms": 2000.0, "end_ms": 2550.0, "root_id": 9, "quality_id": 0, "ext_id": 2, "bass_id": 12, "chord": "Aaug", "confidence": 0.38, "silent": False},
        {"start_ms": 2550.0, "end_ms": 4500.0, "root_id": 10, "quality_id": 1, "ext_id": 4, "bass_id": 12, "chord": "Bbm7", "confidence": 0.70, "silent": False},
    ]
    cleaned = _prune_short_segments(segs, min_duration_ms=300.0, prune_low_conf=True)
    assert len(cleaned) == 2
    assert [c["chord"] for c in cleaned] == ["Fm7", "Bbm7"]


def test_v5_1_2_pa1_taxonomy_and_diatonic_key_fallback():
    """Verify PA1 removed combos, new data-backed colors (`add4, madd4, maj9#11, 6, m6, 9#11, 13#11, 13b9`)
    and Diatonic Key Fallback on `1 + 5` missing-3rd chords.
    """
    from src.music_theory import BANNED_COMBO_TUPLES, resolve_jazz_chord_theory
    from src.serve import canonicalize_chord_tuple, chord_name

    # 1. None of the 7 banned combos can ever survive `resolve_jazz_chord_theory` or `canonicalize_chord_tuple`
    banned_names = {"Cmaj11", "C11", "Cm7b9", "Cm7#11", "Cm7b13", "Cb5", "C7sus2"}
    empty_s = np.zeros(12, dtype=np.float32)
    for q_b, e_b in BANNED_COMBO_TUPLES:
        nr, nq, ne, nb, sfx = resolve_jazz_chord_theory(0, q_b, e_b, 12, empty_s)
        rc, qc, ec, bc = canonicalize_chord_tuple(nr, nq, ne, nb)
        assert chord_name(rc, qc, ec, bc, suffix_override=sfx) not in banned_names
        rc2, qc2, ec2, bc2 = canonicalize_chord_tuple(0, q_b, e_b, 12)
        assert chord_name(rc2, qc2, ec2, bc2) not in banned_names

    # 2. Diatonic Key Fallback for `1 + 5` missing-3rd power chords (`C + G`, absent `3` & `b3`)
    s_power = np.zeros(12, dtype=np.float32)
    s_power[0] = 0.85  # root
    s_power[7] = 0.80  # 5th
    s_power[3] = 0.05  # b3 absent
    s_power[4] = 0.06  # 3 absent
    # In Key of C Major (`key_id=0`): even if raw quality was `Cm (1)`, `C5` resolves to `C` (`qual=0`)!
    assert resolve_jazz_chord_theory(0, 1, 0, 12, s_power, key_id=0)[:4] == (0, 0, 0, 12)
    # In Key of C Minor (`key_id=12`): `C5` resolves to `Cm` (`qual=1`)!
    assert resolve_jazz_chord_theory(0, 0, 0, 12, s_power, key_id=12)[:4] == (0, 1, 0, 12)
    # In Key of C Major (`key_id=0`): `A5` (`root=9`) resolves to `Am` (`qual=1`)!
    assert resolve_jazz_chord_theory(9, 0, 0, 12, s_power, key_id=0)[:4] == (9, 1, 0, 12)
    # In Key of A Minor (`key_id=21`): `C5` (`root=0`, bIII) resolves to `C` (`qual=0`)!
    assert resolve_jazz_chord_theory(0, 1, 0, 12, s_power, key_id=21)[:4] == (0, 0, 0, 12)

    # 3. New PA1 colors: `add4`, `madd4`, `maj9#11`, `6`, `m6`, `9#11`, `13#11`, `13b9`
    s_add4 = np.zeros(12, dtype=np.float32)
    s_add4[[0, 4, 5, 7]] = [0.8, 0.65, 0.72, 0.7]
    assert resolve_jazz_chord_theory(0, 0, 0, 12, s_add4)[4] == "add4"

    s_madd4 = np.zeros(12, dtype=np.float32)
    s_madd4[[0, 3, 5, 7]] = [0.8, 0.65, 0.72, 0.7]
    assert resolve_jazz_chord_theory(0, 1, 0, 12, s_madd4)[4] == "madd4"

    s_maj9s11 = np.zeros(12, dtype=np.float32)
    s_maj9s11[[0, 2, 4, 6, 11]] = [0.8, 0.6, 0.6, 0.62, 0.55]
    assert resolve_jazz_chord_theory(0, 0, 8, 12, s_maj9s11)[4] == "maj9#11"

    s_maj6 = np.zeros(12, dtype=np.float32)
    s_maj6[[0, 4, 7, 9]] = [0.8, 0.65, 0.6, 0.75]
    assert resolve_jazz_chord_theory(0, 0, 0, 12, s_maj6)[4] == "6"

    s_min6 = np.zeros(12, dtype=np.float32)
    s_min6[[0, 3, 7, 9]] = [0.8, 0.65, 0.6, 0.72]
    assert resolve_jazz_chord_theory(0, 1, 0, 12, s_min6)[4] == "m6"

    s_dom_9s11 = np.zeros(12, dtype=np.float32)
    s_dom_9s11[[0, 2, 4, 6, 10]] = [0.8, 0.62, 0.65, 0.60, 0.65]
    assert resolve_jazz_chord_theory(7, 2, 5, 12, s_dom_9s11)[4] == "9#11"

    s_dom_13s11 = np.zeros(12, dtype=np.float32)
    s_dom_13s11[[0, 4, 6, 9, 10]] = [0.8, 0.65, 0.60, 0.62, 0.65]
    assert resolve_jazz_chord_theory(7, 2, 5, 12, s_dom_13s11)[4] == "13#11"

    s_dom_13b9 = np.zeros(12, dtype=np.float32)
    s_dom_13b9[[0, 1, 4, 9, 10]] = [0.8, 0.55, 0.65, 0.62, 0.65]
    assert resolve_jazz_chord_theory(7, 2, 7, 12, s_dom_13b9)[4] == "13b9"
