"""Unit tests for training entry point, evaluation isolation, and checkpoint reproducibility."""

import json
from pathlib import Path
import sys
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from class_config import CLASSES, CLASS_TO_ID
from dataset import (
    AudioSetDataset,
    create_stratified_split,
    save_split_manifest,
    extract_video_id_from_path,
)
from train import (
    set_seed,
    check_eval_isolation,
    save_checkpoint,
    compute_classification_metrics,
)


def test_set_seed_reproducibility():
    """Verify that set_seed ensures reproducible random tensor generation."""
    set_seed(1234)
    t1 = torch.randn(10)
    set_seed(1234)
    t2 = torch.randn(10)
    assert torch.equal(t1, t2)


def test_eval_isolation_guard(tmp_path):
    """Verify that check_eval_isolation flags any cross-split leakage."""
    eval_csv = tmp_path / "eval.csv"
    eval_csv.write_text("video_id,start_seconds,end_seconds,target_class\nLEAK_VID,0.0,10.0,siren\n")

    class MockDataset:
        def __init__(self, samples):
            self.samples = samples

    # Overlapping case should raise ValueError
    leaky_dataset = MockDataset([(Path("siren_LEAK_VID_0_10.wav"), 3, "siren")])
    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        check_eval_isolation(leaky_dataset, eval_csv)

    # Clean case should succeed
    clean_dataset = MockDataset([(Path("siren_CLEAN_VID_0_10.wav"), 3, "siren")])
    check_eval_isolation(clean_dataset, eval_csv)


def test_save_checkpoint_reproducibility(tmp_path):
    """Verify that checkpoint saving persists state dict, class mappings, and config JSON."""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    config = {"lr": 0.001, "batch_size": 16}
    metrics = {"val_acc": 0.85}

    save_checkpoint(
        checkpoint_dir=tmp_path,
        epoch=1,
        model=model,
        optimizer=optimizer,
        loss=0.45,
        metrics=metrics,
        config=config,
        is_best=True,
    )

    ckpt_file = tmp_path / "checkpoint_epoch_001.pt"
    meta_file = tmp_path / "checkpoint_epoch_001_meta.json"
    best_file = tmp_path / "best_model.pt"

    assert ckpt_file.exists()
    assert meta_file.exists()
    assert best_file.exists()

    # Load and verify state
    state = torch.load(ckpt_file, weights_only=False)
    assert state["epoch"] == 1
    assert state["classes"] == CLASSES
    assert state["class_to_id"] == CLASS_TO_ID
    assert state["config"]["lr"] == 0.001

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["metrics"]["val_acc"] == 0.85
    assert meta["classes"] == CLASSES


def test_create_stratified_split():
    """Verify that create_stratified_split partitions dataset cleanly with zero video ID overlap."""
    train_dir = ROOT / "data" / "raw" / "train"
    dataset = AudioSetDataset(train_dir, classes=CLASSES, class_to_id=CLASS_TO_ID)

    train_samples, val_samples = create_stratified_split(dataset, val_ratio=0.2, seed=42)

    assert len(train_samples) + len(val_samples) == len(dataset)
    assert len(train_samples) > 0
    assert len(val_samples) > 0

    train_vids = set(extract_video_id_from_path(p) for p, _, _ in train_samples)
    val_vids = set(extract_video_id_from_path(p) for p, _, _ in val_samples)

    # Strict zero leakage between train and val
    assert len(train_vids.intersection(val_vids)) == 0

    # Every class represented in both
    train_classes = set(c for _, _, c in train_samples)
    val_classes = set(c for _, _, c in val_samples)
    assert train_classes == set(CLASSES)
    assert val_classes == set(CLASSES)


def test_compute_classification_metrics():
    """Verify metric calculations for precision, recall, and macro F1."""
    classes = ["scream", "glass_breaking", "gunshot", "siren", "other"]
    # 2 samples per class, perfect predictions
    y_true = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    y_pred = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    metrics = compute_classification_metrics(y_true, y_pred, classes)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class"]["scream"]["f1_score"] == 1.0

    # Imperfect predictions
    y_imperfect = [0, 1, 1, 1, 2, 2, 3, 3, 4, 4]
    m2 = compute_classification_metrics(y_true, y_imperfect, classes)
    assert m2["accuracy"] == 0.9
    assert m2["per_class"]["scream"]["recall"] == 0.5

