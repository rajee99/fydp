"""Unit tests for PANNs CNN14 transfer learning architecture and integration."""

from pathlib import Path
import sys
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from class_config import CLASSES, CLASS_TO_ID, NUM_CLASSES
from dataset import (
    AudioSetDataset,
    TARGET_SAMPLES,
    load_split_manifest,
    extract_video_id_from_path,
)
from model import PannsCNN14, PannsCNN14Transfer, count_parameters

PRETRAINED_CKPT = ROOT / "models" / "pretrained" / "Cnn14_16k_mAP=0.438.pth"
TRAIN_AUDIO_DIR = ROOT / "data" / "raw" / "train"
TRAIN_SPLIT_MANIFEST = ROOT / "data" / "processed" / "train_split_manifest.csv"
VAL_SPLIT_MANIFEST = ROOT / "data" / "processed" / "val_split_manifest.csv"


def test_panns_cnn14_instantiation():
    """Verify PannsCNN14 initializes with 5-class transfer head and correct structure."""
    model = PannsCNN14(num_classes=NUM_CLASSES, freeze_backbone=True)
    assert model.num_classes == 5
    assert isinstance(model.fc_transfer, nn.Linear)
    assert model.fc_transfer.in_features == 2048
    assert model.fc_transfer.out_features == 5

    # Check alias
    assert PannsCNN14Transfer is PannsCNN14

    total_p, train_p = count_parameters(model)
    assert total_p == 79_963_653
    # Backbone frozen initially
    assert train_p == 10_245


def test_panns_cnn14_freezing_and_unfreezing():
    """Verify controlled freezing and unfreezing of backbone parameters."""
    model = PannsCNN14(num_classes=5, freeze_backbone=True)

    # Initial state: only fc_transfer is trainable
    for name, param in model.named_parameters():
        if "fc_transfer" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False

    # Full unfreezing
    model.unfreeze_backbone()
    total_p, train_p = count_parameters(model)
    assert total_p == train_p == 79_963_653

    # Controlled top-1 block unfreezing (conv_block6 + fc1 + fc_transfer)
    model.freeze_backbone()
    model.unfreeze_backbone(num_blocks=1)
    assert model.fc_transfer.weight.requires_grad is True
    assert model.fc1.weight.requires_grad is True
    assert model.conv_block6.conv1.weight.requires_grad is True
    assert model.conv_block5.conv1.weight.requires_grad is False
    assert model.conv_block1.conv1.weight.requires_grad is False


def test_panns_cnn14_pretrained_weights_loading():
    """Verify pretrained weights load correctly and report missing/unexpected keys."""
    if not PRETRAINED_CKPT.exists():
        pytest.skip(f"Pretrained checkpoint not found at {PRETRAINED_CKPT}")

    model = PannsCNN14(num_classes=5, freeze_backbone=True)
    load_res = model.load_from_pretrain(PRETRAINED_CKPT)

    assert load_res["matched_keys_count"] == 82
    assert set(load_res["missing_keys"]) == {"fc_transfer.weight", "fc_transfer.bias"}
    assert set(load_res["unexpected_keys"]) == {"fc_audioset.weight", "fc_audioset.bias"}
    assert load_res["loaded_param_count"] == 79_953_408


def test_panns_cnn14_forward_waveform():
    """Verify forward pass from raw 16 kHz waveform of shape (batch, 160000)."""
    model = PannsCNN14(num_classes=5, freeze_backbone=True)
    model.eval()

    batch_size = 2
    dummy_audio = torch.randn(batch_size, TARGET_SAMPLES)
    logits = model(dummy_audio)
    assert logits.shape == (batch_size, 5)

    probs = model.predict_proba(dummy_audio)
    assert probs.shape == (batch_size, 5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_panns_cnn14_forward_spectrogram():
    """Verify forward pass from precomputed log-mel spectrogram."""
    model = PannsCNN14(num_classes=5, freeze_backbone=True)
    model.eval()

    batch_size = 2
    # Standard PANNs format (batch, 1, 1001, 64)
    spec1 = torch.randn(batch_size, 1, 1001, 64)
    logits1 = model(spec1)
    assert logits1.shape == (batch_size, 5)

    # Transposed format (batch, 1, 64, 1001)
    spec2 = torch.randn(batch_size, 1, 64, 1001)
    logits2 = model(spec2)
    assert logits2.shape == (batch_size, 5)


def test_panns_cnn14_gradient_isolation():
    """Verify loss gradient updates ONLY fc_transfer when backbone is frozen."""
    model = PannsCNN14(num_classes=5, freeze_backbone=True)
    model.train()

    batch_size = 2
    x = torch.randn(batch_size, TARGET_SAMPLES)
    y = torch.tensor([0, 3], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    # Backbone gradients must be None
    assert model.conv_block1.conv1.weight.grad is None
    assert model.conv_block6.conv2.weight.grad is None
    assert model.fc1.weight.grad is None

    # Head gradients must exist
    assert model.fc_transfer.weight.grad is not None
    assert model.fc_transfer.bias.grad is not None


def test_audioset_dataset_raw_mode():
    """Verify AudioSetDataset with return_raw=True returns 1D waveform of length 160,000."""
    dataset = AudioSetDataset(TRAIN_AUDIO_DIR, return_raw=True)
    assert len(dataset) > 0

    wav, class_id, path_str = dataset[0]
    assert wav.ndim == 1
    assert wav.shape[0] == TARGET_SAMPLES
    assert 0 <= class_id < 5
    assert Path(path_str).exists()


def test_split_manifest_loading_isolation():
    """Verify load_split_manifest loads exact samples with zero train/val overlap."""
    if not (TRAIN_SPLIT_MANIFEST.exists() and VAL_SPLIT_MANIFEST.exists()):
        pytest.skip("Split manifests not generated yet.")

    train_samples = load_split_manifest(TRAIN_SPLIT_MANIFEST, root_dir=ROOT)
    val_samples = load_split_manifest(VAL_SPLIT_MANIFEST, root_dir=ROOT)

    assert len(train_samples) > 0
    assert len(val_samples) > 0

    train_vids = set(extract_video_id_from_path(p) for p, _, _ in train_samples)
    val_vids = set(extract_video_id_from_path(p) for p, _, _ in val_samples)

    overlap = train_vids.intersection(val_vids)
    assert len(overlap) == 0, f"Detected leakage between train and val: {overlap}"


def test_panns_cnn14_partial_finetune_setup():
    """Verify partial fine-tuning: unfreeze conv_block6 and fc_transfer only."""
    ckpt_path = ROOT / "models" / "experiments" / "panns_cnn14" / "best_model.pt"
    if not ckpt_path.exists():
        pytest.skip("Frozen checkpoint not found.")

    model = PannsCNN14(num_classes=5, freeze_backbone=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Unfreeze only conv_block6 and fc_transfer
    for name, param in model.named_parameters():
        if "conv_block6" in name or "fc_transfer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_p = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    assert trainable_p == 56_641_541
    assert frozen_p == 23_322_112

    # Forward + backward gradient check
    model.train()
    optimizer = torch.optim.AdamW([
        {"params": model.conv_block6.parameters(), "lr": 1e-5},
        {"params": model.fc_transfer.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)

    batch_x = torch.randn(2, TARGET_SAMPLES)
    batch_y = torch.tensor([1, 4], dtype=torch.long)
    optimizer.zero_grad()
    logits = model(batch_x)
    loss = nn.CrossEntropyLoss()(logits, batch_y)
    loss.backward()

    # Verify gradient flow
    assert model.conv_block6.conv1.weight.grad is not None
    assert model.conv_block6.conv2.weight.grad is not None
    assert model.fc_transfer.weight.grad is not None

    # Verify frozen layers have NO gradients
    assert model.conv_block5.conv1.weight.grad is None
    assert model.conv_block1.conv1.weight.grad is None
    assert model.fc1.weight.grad is None

