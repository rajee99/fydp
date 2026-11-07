"""Unit tests for the 5-class PyTorch model, dataset, and preprocessing pipeline."""

from pathlib import Path
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from class_config import CLASSES, CLASS_TO_ID, NUM_CLASSES
from dataset import (
    MelSpectrogramExtractor,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    N_MELS,
    run_preflight_dataset_check,
)
from model import CompactAudioCNN, count_parameters


def test_canonical_5_classes():
    """Verify exact 5-class canonical mapping."""
    expected = ["scream", "glass_breaking", "gunshot", "siren", "other"]
    assert CLASSES == expected
    assert NUM_CLASSES == 5
    for idx, name in enumerate(expected):
        assert CLASS_TO_ID[name] == idx


def test_compact_audio_cnn_forward():
    """Verify CompactAudioCNN accepts (batch, 1, 64, 1001) and outputs (batch, 5)."""
    model = CompactAudioCNN(num_classes=5)
    batch_size = 3
    x = torch.randn(batch_size, 1, 64, 1001)

    logits = model(x)
    assert logits.shape == (batch_size, 5)

    probs = model.predict_proba(x)
    assert probs.shape == (batch_size, 5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    total_params, trainable_params = count_parameters(model)
    assert total_params > 100_000
    assert total_params == trainable_params


def test_mel_spectrogram_extractor():
    """Verify log-mel extraction produces (1, 64, 1001) for exact, short, and long clips."""
    extractor = MelSpectrogramExtractor()

    # Exact 10 seconds (160,000 samples)
    exact_wave = torch.randn(1, TARGET_SAMPLES)
    out_exact = extractor(exact_wave)
    assert out_exact.shape == (1, N_MELS, 1001)

    # Short clip (5 seconds / 80,000 samples) -> padded
    short_wave = torch.randn(1, 80000)
    out_short = extractor(short_wave)
    assert out_short.shape == (1, N_MELS, 1001)

    # Long clip (12 seconds / 192,000 samples) -> truncated
    long_wave = torch.randn(1, 192000)
    out_long = extractor(long_wave)
    assert out_long.shape == (1, N_MELS, 1001)


def test_preflight_dataset_check():
    """Verify that run_preflight_dataset_check scans existing audio correctly."""
    train_dir = ROOT / "data" / "raw" / "train"
    is_ready, issues, stats = run_preflight_dataset_check(train_dir)

    # Check emergency classes
    for ec in ["scream", "glass_breaking", "gunshot", "siren"]:
        assert stats["classes_found"].get(ec, 0) > 0

    assert stats["sample_rate_violations"] == 0
    assert stats["channel_violations"] == 0
    assert stats["format_errors"] == 0
