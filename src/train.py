"""Baseline Training Experiment for 5-Class Emergency Sound Detection.

Canonical 5-Class Mapping:
  0: scream
  1: glass_breaking
  2: gunshot
  3: siren
  4: other

Features:
- Pre-flight dataset readiness check (16 kHz mono 16-bit PCM WAV validation)
- Source/video-ID-grouped stratified train/val split (guarantees 0 video ID overlap between splits)
- Strict evaluation split isolation (guarantees 0 overlap with held-out eval_manifest.csv)
- Class imbalance mitigation: inverse-frequency weighted CrossEntropyLoss
- CompactAudioCNN baseline model (4 ConvBlocks, Global AvgPool, Dense Head)
- Learning rate scheduling (ReduceLROnPlateau) and Early Stopping
- Complete per-epoch metric tracking (Accuracy, Loss, Macro F1, Per-class Precision/Recall/F1)
- Reproducible artifact persistence (best checkpoint, metadata JSON, training history JSON, config JSON)
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Configure UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

try:
    from class_config import (
        CLASSES,
        CLASS_TO_ID,
        ID_TO_CLASS,
        NUM_CLASSES,
    )
    from dataset import (
        AudioSetDataset,
        SAMPLE_RATE,
        TARGET_DURATION_SEC,
        TARGET_SAMPLES,
        N_MELS,
        run_preflight_dataset_check,
        create_stratified_split,
        save_split_manifest,
        load_split_manifest,
        extract_video_id_from_path,
    )
    from model import CompactAudioCNN, PannsCNN14, count_parameters
except ImportError:
    from src.class_config import (
        CLASSES,
        CLASS_TO_ID,
        ID_TO_CLASS,
        NUM_CLASSES,
    )
    from src.dataset import (
        AudioSetDataset,
        SAMPLE_RATE,
        TARGET_DURATION_SEC,
        TARGET_SAMPLES,
        N_MELS,
        run_preflight_dataset_check,
        create_stratified_split,
        save_split_manifest,
        load_split_manifest,
        extract_video_id_from_path,
    )
    from src.model import CompactAudioCNN, PannsCNN14, count_parameters

TRAIN_AUDIO_DIR = ROOT / "data" / "raw" / "train"
EVAL_MANIFEST = ROOT / "data" / "processed" / "eval_manifest.csv"
TRAIN_SPLIT_MANIFEST = ROOT / "data" / "processed" / "train_split_manifest.csv"
VAL_SPLIT_MANIFEST = ROOT / "data" / "processed" / "val_split_manifest.csv"
DEFAULT_CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
DEFAULT_PANNS_EXPERIMENT_DIR = ROOT / "models" / "experiments" / "panns_cnn14"
DEFAULT_PANNS_FINETUNED_DIR = ROOT / "models" / "experiments" / "panns_cnn14_finetuned"
DEFAULT_PANNS_COMBINED_DIR = ROOT / "models" / "experiments" / "panns_cnn14_combined"
DEFAULT_PRETRAINED_PATH = ROOT / "models" / "pretrained" / "Cnn14_16k_mAP=0.438.pth"
COMBINED_TRAIN_SPLIT_MANIFEST = ROOT / "data" / "processed" / "combined_train_split_manifest.csv"
COMBINED_VAL_SPLIT_MANIFEST = ROOT / "data" / "processed" / "combined_val_split_manifest.csv"


def set_seed(seed: int = 42):
    """Set random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_eval_isolation(dataset_or_samples, eval_manifest_path: Path):
    """Verify zero overlap between training audio files and evaluation manifest video IDs."""
    if not eval_manifest_path.exists():
        print(f"[WARNING] Evaluation manifest not found at {eval_manifest_path}. Skipping isolation check.")
        return

    import csv
    with open(eval_manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        eval_vids = set(r["video_id"].strip() for r in reader if r.get("video_id"))

    if hasattr(dataset_or_samples, "samples"):
        samples = dataset_or_samples.samples
    else:
        samples = dataset_or_samples

    overlapping = []
    for item in samples:
        fpath = item[0] if isinstance(item, (tuple, list)) else item
        vid = extract_video_id_from_path(fpath)
        if vid in eval_vids:
            overlapping.append((Path(fpath).name, vid))

    if overlapping:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! {len(overlapping)} training files overlap with evaluation manifest: {overlapping[:3]}"
        )
    print(f"  [OK] Zero overlap with evaluation manifest ({len(eval_vids)} eval video IDs verified).")


def get_manifest_metadata(manifest_path: Path) -> Dict:
    """Inspect split manifest and compute breakdown by source dataset, source_id, and class."""
    import csv
    stats = {
        "by_dataset": {},
        "by_class": {},
        "source_ids": set(),
        "file_hashes": set(),
        "video_ids": set(),
        "total": 0,
    }
    if not manifest_path.exists():
        return stats

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            stats["total"] += 1
            ds = r.get("source_dataset", "AudioSet")
            tc = r.get("target_class", "unknown")
            stats["by_dataset"].setdefault(ds, {})
            stats["by_dataset"][ds][tc] = stats["by_dataset"][ds].get(tc, 0) + 1
            stats["by_class"][tc] = stats["by_class"].get(tc, 0) + 1
            if r.get("source_id"):
                stats["source_ids"].add(r["source_id"].strip())
            if r.get("file_hash"):
                stats["file_hashes"].add(r["file_hash"].strip())
            if r.get("video_id"):
                stats["video_ids"].add(r["video_id"].strip())
    return stats


def compute_classification_metrics(y_true: List[int], y_pred: List[int], classes: List[str]) -> Dict:
    """Compute overall accuracy, macro F1, and per-class precision, recall, and F1."""
    num_classes = len(classes)
    total_samples = len(y_true)
    if total_samples == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "confusion_matrix": []}

    # Confusion matrix [true, pred]
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    correct = sum(cm[i][i] for i in range(num_classes))
    accuracy = correct / float(total_samples)

    per_class = {}
    f1_list = []
    precision_list = []
    recall_list = []

    for i, cname in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[k][i] for k in range(num_classes) if k != i)
        fn = sum(cm[i][k] for k in range(num_classes) if k != i)

        prec = (tp / float(tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / float(tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        support = sum(cm[i])
        per_class[cname] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support,
        }
        f1_list.append(f1)
        precision_list.append(prec)
        recall_list.append(rec)

    macro_f1 = sum(f1_list) / float(num_classes)
    macro_precision = sum(precision_list) / float(num_classes)
    macro_recall = sum(recall_list) / float(num_classes)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    classes: List[str],
) -> Tuple[float, Dict]:
    """Evaluate model on a dataloader and return (average_loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_data, batch_labels, _ in dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_data)
            loss = criterion(logits, batch_labels)

            total_loss += loss.item() * batch_data.size(0)
            total_samples += batch_data.size(0)

            preds = torch.argmax(logits, dim=-1)
            y_true.extend(batch_labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
    metrics = compute_classification_metrics(y_true, y_pred, classes)
    return avg_loss, metrics


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: float,
    metrics: Dict,
    config: Dict,
    is_best: bool = False,
):
    """Save training checkpoint and metadata JSON for full reproducibility."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics,
        "classes": list(CLASSES),
        "class_to_id": dict(CLASS_TO_ID),
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Epoch checkpoint
    epoch_ckpt = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(state, epoch_ckpt)

    epoch_meta = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}_meta.json"
    with open(epoch_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": epoch,
                "val_loss": loss,
                "metrics": metrics,
                "classes": list(CLASSES),
                "class_to_id": dict(CLASS_TO_ID),
                "config": config,
                "timestamp": state["timestamp"],
            },
            f,
            indent=2,
        )

    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(state, best_path)
        with open(checkpoint_dir / "best_model_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "val_loss": loss,
                    "metrics": metrics,
                    "classes": list(CLASSES),
                    "class_to_id": dict(CLASS_TO_ID),
                    "config": config,
                    "timestamp": state["timestamp"],
                },
                f,
                indent=2,
            )

    # Always persist latest checkpoint
    latest_path = checkpoint_dir / "latest_checkpoint.pt"
    torch.save(state, latest_path)

    # Prune earlier intermediate epoch checkpoints to conserve disk space,
    # preserving all epoch metadata JSON files, best_model.pt, and latest_checkpoint.pt
    for old_ckpt in checkpoint_dir.glob("checkpoint_epoch_*.pt"):
        if old_ckpt.name != epoch_ckpt.name:
            try:
                old_ckpt.unlink()
            except OSError:
                pass


def run_training_experiment(
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.2,
    patience: int = 5,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    seed: int = 42,
    device_name: Optional[str] = None,
    preflight_only: bool = False,
    dry_run: bool = False,
    model_name: str = "compact_cnn",
    pretrained_path: Optional[Union[str, Path]] = None,
    freeze_backbone: bool = True,
    early_stopping_metric: str = "val_loss",
    finetune_from: Optional[Union[str, Path]] = None,
    lr_head: Optional[float] = None,
    lr_backbone: Optional[float] = None,
    train_manifest: Optional[Union[str, Path]] = None,
    val_manifest: Optional[Union[str, Path]] = None,
    optimizer_name: str = "adam",
):
    # Determine checkpoint directory and display name
    is_combined = False
    if train_manifest is not None:
        train_manifest_path = Path(train_manifest)
        val_manifest_path = Path(val_manifest) if val_manifest else VAL_SPLIT_MANIFEST
        if "combined" in str(train_manifest_path).lower():
            is_combined = True
    else:
        train_manifest_path = None
        val_manifest_path = None

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        if finetune_from is not None:
            model_name = "panns_cnn14"
            model_display_name = "PANNs CNN14 (Partial Fine-Tuning: conv_block6 + head)"
        elif is_combined or (model_name == "panns_cnn14" and "combined" in str(checkpoint_dir)):
            model_name = "panns_cnn14"
            model_display_name = "PANNs CNN14 (Combined AudioSet + NIGENS Transfer Learning)"
        elif model_name == "panns_cnn14":
            model_display_name = "PANNs CNN14 (Transfer Learning)"
        else:
            model_display_name = "CompactAudioCNN (Baseline)"
    elif finetune_from is not None:
        model_name = "panns_cnn14"
        checkpoint_dir = DEFAULT_PANNS_FINETUNED_DIR
        model_display_name = "PANNs CNN14 (Partial Fine-Tuning: conv_block6 + head)"
    elif is_combined:
        model_name = "panns_cnn14"
        checkpoint_dir = DEFAULT_PANNS_COMBINED_DIR
        model_display_name = "PANNs CNN14 (Combined AudioSet + NIGENS Transfer Learning)"
    elif model_name == "panns_cnn14":
        checkpoint_dir = DEFAULT_PANNS_EXPERIMENT_DIR
        model_display_name = "PANNs CNN14 (Transfer Learning)"
    else:
        checkpoint_dir = DEFAULT_CHECKPOINT_DIR
        model_display_name = "CompactAudioCNN (Baseline)"

    if finetune_from:
        effective_head_lr = lr_head if lr_head is not None else 1e-4
        effective_backbone_lr = lr_backbone if lr_backbone is not None else 1e-5
    else:
        effective_head_lr = lr_head if lr_head is not None else lr
        effective_backbone_lr = None

    print("=" * 80)
    print(f"  TRAINING EXPERIMENT: {model_display_name}")
    print("=" * 80)
    print(f"Canonical Classes: {CLASSES}")
    print(f"Class-to-ID: {CLASS_TO_ID}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz | Target Samples: {TARGET_SAMPLES} ({TARGET_DURATION_SEC}s)")
    if finetune_from:
        print(f"Model: {model_display_name} | Epochs: {epochs} | Batch Size: {batch_size}")
        print(f"Fine-Tuning LRs: head={effective_head_lr:.1e}, conv_block6={effective_backbone_lr:.1e} | Weight Decay: {weight_decay} | Optimizer: {optimizer_name.upper()}")
    else:
        print(f"Model: {model_display_name} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {effective_head_lr} | Optimizer: {optimizer_name.upper()}")
    try:
        rel_ckpt = checkpoint_dir.relative_to(ROOT)
    except ValueError:
        rel_ckpt = checkpoint_dir
    print(f"Checkpoint Dir: {rel_ckpt}")
    print(f"Early Stopping Metric: {early_stopping_metric} (Patience: {patience} epochs) | Random Seed: {seed}")
    print("-" * 80)

    set_seed(seed)

    # 1. Preflight Dataset Checks & Split Loading
    train_meta = None
    val_meta = None

    if train_manifest_path is not None and val_manifest_path is not None:
        print("\n[Step 1/6] Running preflight dataset readiness check (Manifest-Guided)...")
        if not train_manifest_path.exists():
            raise FileNotFoundError(f"Train manifest not found: {train_manifest_path}")
        if not val_manifest_path.exists():
            raise FileNotFoundError(f"Val manifest not found: {val_manifest_path}")

        train_samples = load_split_manifest(train_manifest_path, root_dir=ROOT)
        val_samples = load_split_manifest(val_manifest_path, root_dir=ROOT)
        train_meta = get_manifest_metadata(train_manifest_path)
        val_meta = get_manifest_metadata(val_manifest_path)

        print(f"  Train Manifest: {train_manifest_path.name} ({len(train_samples)} samples)")
        print(f"  Val Manifest:   {val_manifest_path.name} ({len(val_samples)} samples)")

        # Verify all files exist
        missing_train = [p for p, _, _ in train_samples if not p.exists()]
        missing_val = [p for p, _, _ in val_samples if not p.exists()]
        if missing_train or missing_val:
            raise RuntimeError(f"Missing audio files: {len(missing_train)} in train, {len(missing_val)} in val!")
        print(f"  [OK] All {len(train_samples) + len(val_samples)} audio files verified to exist on disk.")

        # Verify all 5 classes are present
        train_classes = set(c for _, _, c in train_samples)
        val_classes = set(c for _, _, c in val_samples)
        for c in CLASSES:
            if c not in train_classes:
                raise RuntimeError(f"Class '{c}' missing from train split!")
            if c not in val_classes:
                raise RuntimeError(f"Class '{c}' missing from val split!")
        print(f"  [OK] All 5 classes present in both splits.")

        # Verify source group isolation
        if train_meta["source_ids"] and val_meta["source_ids"]:
            src_overlap = train_meta["source_ids"].intersection(val_meta["source_ids"])
            if src_overlap:
                raise RuntimeError(f"Source group leakage: {len(src_overlap)} overlapping source IDs: {list(src_overlap)[:3]}")
            print(f"  [OK] Source group overlap: 0 ({len(train_meta['source_ids'])} train / {len(val_meta['source_ids'])} val groups verified).")

        # Verify hash isolation
        if train_meta["file_hashes"] and val_meta["file_hashes"]:
            h_overlap = train_meta["file_hashes"].intersection(val_meta["file_hashes"])
            if h_overlap:
                raise RuntimeError(f"File hash leakage: {len(h_overlap)} overlapping hashes!")
            print(f"  [OK] File hash overlap: 0 ({len(train_meta['file_hashes'])} train / {len(val_meta['file_hashes'])} val hashes).")

        # Verify AudioSet video ID isolation
        if train_meta["video_ids"] and val_meta["video_ids"]:
            v_overlap = train_meta["video_ids"].intersection(val_meta["video_ids"])
            if v_overlap:
                raise RuntimeError(f"Video ID leakage: {len(v_overlap)} overlapping video IDs!")
            print(f"  [OK] AudioSet video ID overlap: 0 ({len(train_meta['video_ids'])} train / {len(val_meta['video_ids'])} val video IDs).")

        # Evaluation isolation
        print("\n[Step 2/6] Verifying evaluation isolation...")
        check_eval_isolation(train_samples + val_samples, EVAL_MANIFEST)
        print("  -> Preflight check: PASSED (all manifest samples verified).")
        print("\n[Step 3/6] Split manifests loaded and verified.")
    else:
        print("\n[Step 1/6] Running preflight dataset readiness check...")
        is_ready, issues, stats = run_preflight_dataset_check(TRAIN_AUDIO_DIR)

        print(f"  Training Directory: {TRAIN_AUDIO_DIR.relative_to(ROOT)}")
        for cname in CLASSES:
            count = stats["classes_found"].get(cname, 0)
            status_tag = "[OK]" if count > 0 else "[EMPTY]"
            print(f"  {status_tag} {cname:<16}: {count:4d} valid audio files")

        print(f"  Total Valid Files: {stats['total_valid_files']}")
        print(f"  Format Errors:     {stats['format_errors']}")
        print(f"  Rate Violations:   {stats['sample_rate_violations']}")

        if not is_ready:
            print("\n[PREFLIGHT FAILED] Dataset issues detected:", file=sys.stderr)
            for issue in issues:
                print(f"  - {issue}", file=sys.stderr)
            raise RuntimeError("Preflight check failed. Halting training pipeline.")
        print("  -> Preflight check: PASSED (all 5 classes present and format-verified).")

        # 2. Build Full Dataset & Verify Held-Out Eval Isolation
        print("\n[Step 2/6] Loading full dataset and verifying evaluation isolation...")
        full_dataset = AudioSetDataset(TRAIN_AUDIO_DIR, classes=CLASSES, class_to_id=CLASS_TO_ID)
        print(f"  Total verified training audio files loaded: {len(full_dataset)}")
        check_eval_isolation(full_dataset, EVAL_MANIFEST)

        # 3. Create or Load Reproducible Stratified Train / Val Split
        train_manifest_path = TRAIN_SPLIT_MANIFEST
        val_manifest_path = VAL_SPLIT_MANIFEST
        if TRAIN_SPLIT_MANIFEST.exists() and VAL_SPLIT_MANIFEST.exists():
            print(f"\n[Step 3/6] Loading existing split manifests ({TRAIN_SPLIT_MANIFEST.name}, {VAL_SPLIT_MANIFEST.name})...")
            train_samples = load_split_manifest(TRAIN_SPLIT_MANIFEST, root_dir=ROOT)
            val_samples = load_split_manifest(VAL_SPLIT_MANIFEST, root_dir=ROOT)
        else:
            print("\n[Step 3/6] Generating video-ID-grouped stratified train/val split (80/20)...")
            train_samples, val_samples = create_stratified_split(full_dataset, val_ratio=val_ratio, seed=seed)
            save_split_manifest(train_samples, TRAIN_SPLIT_MANIFEST, root_dir=ROOT)
            save_split_manifest(val_samples, VAL_SPLIT_MANIFEST, root_dir=ROOT)

    return_raw = (model_name == "panns_cnn14")
    train_dataset = AudioSetDataset(
        TRAIN_AUDIO_DIR,
        classes=CLASSES,
        class_to_id=CLASS_TO_ID,
        samples=train_samples,
        return_raw=return_raw,
    )
    val_dataset = AudioSetDataset(
        TRAIN_AUDIO_DIR,
        classes=CLASSES,
        class_to_id=CLASS_TO_ID,
        samples=val_samples,
        return_raw=return_raw,
    )

    train_counts = train_dataset.get_class_counts()
    val_counts = val_dataset.get_class_counts()

    print(f"  Train split: {len(train_dataset):3d} samples -> {train_manifest_path.name}")
    print(f"  Val split:   {len(val_dataset):3d} samples -> {val_manifest_path.name}")

    print("\n  Split Distribution per Class:")
    print(f"    {'Class Name':<16} | {'Train':<6} | {'Val':<6} | {'Total':<6} | {'Val Ratio':<8}")
    print("    " + "-" * 52)
    for c in CLASSES:
        t_cnt = train_counts[c]
        v_cnt = val_counts[c]
        tot = t_cnt + v_cnt
        ratio = (v_cnt / float(tot)) * 100.0 if tot > 0 else 0.0
        print(f"    {c:<16} | {t_cnt:6d} | {v_cnt:6d} | {tot:6d} | {ratio:6.1f}%")

    if train_meta and train_meta.get("by_dataset"):
        print("\n  Dataset Origin Breakdown (AudioSet vs NIGENS):")
        for ds_name in sorted(train_meta["by_dataset"].keys()):
            t_ds_tot = sum(train_meta["by_dataset"][ds_name].values())
            v_ds_tot = sum(val_meta["by_dataset"].get(ds_name, {}).values()) if val_meta else 0
            tot_ds = t_ds_tot + v_ds_tot
            pct = (v_ds_tot / float(tot_ds) * 100.0) if tot_ds > 0 else 0.0
            print(f"    {ds_name:<16}: {t_ds_tot:4d} train | {v_ds_tot:4d} val | {tot_ds:4d} total ({pct:5.1f}% val)")

    # 4. Class Imbalance Mitigation (Computed strictly on Train split)
    print("\n[Step 4/6] Computing class-weighted loss weights (Train split only)...")
    train_weights = train_dataset.get_class_weights()
    for idx, cname in enumerate(CLASSES):
        print(f"    Class {idx} ({cname:<14}): {train_counts[cname]:3d} train samples | Loss Weight: {train_weights[idx].item():.4f}")

    if preflight_only:
        print("\n" + "=" * 80)
        print("  PREFLIGHT-ONLY MODE COMPLETE: Dataset, splits, and weights verified.")
        print("=" * 80)
        return

    # Select Device
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTarget computation device: {device}")

    # Build DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss(weight=train_weights.to(device))

    # 5. Initialize Model Architecture & Optimizer
    if finetune_from is not None:
        print("\n[Step 5/6] Initializing PANNs CNN14 Architecture for Partial Fine-Tuning...")
        model = PannsCNN14(num_classes=NUM_CLASSES, freeze_backbone=True)
        finetune_path = Path(finetune_from)
        if not finetune_path.exists():
            raise FileNotFoundError(f"Checkpoint to fine-tune from not found: {finetune_path}")
        ckpt_state = torch.load(finetune_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_state["model_state_dict"])
        print(f"  Loaded weights from frozen checkpoint: {finetune_path.name} (epoch {ckpt_state.get('epoch')})")

        # Unfreeze ONLY conv_block6 and fc_transfer
        for name, param in model.named_parameters():
            if "conv_block6" in name or "fc_transfer" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        head_lr = lr_head if lr_head is not None else 1e-4
        backbone_lr = lr_backbone if lr_backbone is not None else 1e-5

        optim_cls = torch.optim.Adam if optimizer_name.lower() == "adam" else torch.optim.AdamW
        optimizer = optim_cls([
            {"params": model.conv_block6.parameters(), "lr": backbone_lr},
            {"params": model.fc_transfer.parameters(), "lr": head_lr},
        ], weight_decay=weight_decay)

        model_desc = f"PANNs CNN14 (Partial Fine-Tuning: conv_block6 @ lr={backbone_lr}, head @ lr={head_lr}, {optimizer_name.upper()})"
        p_path = finetune_path
    elif model_name == "panns_cnn14":
        print("\n[Step 5/6] Initializing PANNs CNN14 Architecture...")
        model = PannsCNN14(num_classes=NUM_CLASSES, freeze_backbone=freeze_backbone)
        p_path = (
            Path(pretrained_path)
            if pretrained_path is not None
            else DEFAULT_PRETRAINED_PATH
        )
        load_res = model.load_from_pretrain(p_path)
        print(f"  Pretrained Checkpoint: {p_path.name}")
        print(f"  Loaded Tensors:        {load_res['matched_keys_count']} ({load_res['loaded_param_count']:,} parameters)")
        print(f"  Missing Keys:          {load_res['missing_keys']}")
        print(f"  Unexpected Keys:       {load_res['unexpected_keys']}")
        model_desc = f"PANNs CNN14 (Pretrained backbone, frozen={freeze_backbone}, 5-class transfer head, {optimizer_name.upper()})"

        optim_cls = torch.optim.Adam if optimizer_name.lower() == "adam" else torch.optim.AdamW
        optimizer = optim_cls(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=effective_head_lr,
            weight_decay=weight_decay,
        )
    else:
        print("\n[Step 5/6] Initializing CompactAudioCNN Architecture...")
        model = CompactAudioCNN(num_classes=NUM_CLASSES)
        p_path = None
        model_desc = f"CompactAudioCNN (4 ConvBlocks, Global AvgPool, Dense Head, {optimizer_name.upper()})"

        optim_cls = torch.optim.Adam if optimizer_name.lower() == "adam" else torch.optim.AdamW
        optimizer = optim_cls(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=effective_head_lr,
            weight_decay=weight_decay,
        )

    model = model.to(device)
    total_p, train_p = count_parameters(model)
    print(f"  Model: {model_desc}")
    print(f"  Parameters: Total={total_p:,} | Trainable={train_p:,} | Frozen={total_p - train_p:,}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    config = {
        "model": "PannsCNN14_FineTuned" if finetune_from else ("PannsCNN14" if model_name == "panns_cnn14" else "CompactAudioCNN"),
        "finetuned_from": str(finetune_from) if finetune_from else None,
        "finetuned_layers": ["conv_block6", "fc_transfer"] if finetune_from else None,
        "lr_head": effective_head_lr,
        "lr_backbone": effective_backbone_lr,
        "pretrained_path": str(p_path) if p_path is not None else None,
        "freeze_backbone": freeze_backbone if model_name == "panns_cnn14" and not finetune_from else None,
        "early_stopping_metric": early_stopping_metric,
        "total_parameters": total_p,
        "trainable_parameters": train_p,
        "num_classes": NUM_CLASSES,
        "classes": list(CLASSES),
        "class_to_id": dict(CLASS_TO_ID),
        "sample_rate": SAMPLE_RATE,
        "target_duration_sec": TARGET_DURATION_SEC,
        "target_samples": TARGET_SAMPLES,
        "n_mels": N_MELS,
        "epochs": epochs,
        "batch_size": batch_size,
        "initial_lr": effective_head_lr,
        "weight_decay": weight_decay,
        "optimizer": optimizer_name.upper(),
        "train_manifest": str(train_manifest_path.relative_to(ROOT)) if (train_manifest_path and train_manifest_path.is_relative_to(ROOT)) else (str(train_manifest_path) if train_manifest_path else str(TRAIN_SPLIT_MANIFEST.name)),
        "val_manifest": str(val_manifest_path.relative_to(ROOT)) if (val_manifest_path and val_manifest_path.is_relative_to(ROOT)) else (str(val_manifest_path) if val_manifest_path else str(VAL_SPLIT_MANIFEST.name)),
        "imbalance_strategy": "class_weights",
        "train_class_weights": [round(w.item(), 4) for w in train_weights],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_counts": train_counts,
        "val_counts": val_counts,
        "dataset_distribution": {
            "train": train_meta.get("by_dataset") if train_meta else {"AudioSet": train_counts},
            "val": val_meta.get("by_dataset") if val_meta else {"AudioSet": val_counts},
        },
        "val_ratio": val_ratio,
        "seed": seed,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if dry_run:
        print("\n  [DRY RUN] Executing single forward/backward verification step...")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / (1024 ** 2)

        model.train()
        for batch_data, batch_labels, _ in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_data)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            print(f"  [DRY RUN OK] Input shape: {tuple(batch_data.shape)} | Logits shape: {tuple(logits.shape)} | Loss: {loss.item():.4f}")
            break

        if finetune_from is not None:
            assert model.conv_block6.conv1.weight.grad is not None, "conv_block6 must have gradients"
            assert model.fc_transfer.weight.grad is not None, "fc_transfer must have gradients"
            assert model.conv_block5.conv1.weight.grad is None, "conv_block5 must be frozen"
            assert model.fc1.weight.grad is None, "fc1 must be frozen"
            print("  [DRY RUN OK] Gradient flow verified: conv_block6 & head have gradients; conv_block1..5 & fc1 are frozen.")
        elif model_name == "panns_cnn14" and freeze_backbone:
            assert model.conv_block1.conv1.weight.grad is None, "Frozen backbone conv1 should have None grad"
            assert model.fc_transfer.weight.grad is not None, "Trainable head must have computed gradients"
            print("  [DRY RUN OK] Gradient flow verified: backbone gradients are None, head gradients computed.")

        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device, CLASSES)
        print(f"  [DRY RUN OK] Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

        if device.type == "cuda":
            mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            mem_cur = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"  [DRY RUN OK] CUDA Memory: current={mem_cur:.2f} MB | peak={mem_peak:.2f} MB")

        print("\n" + "=" * 80)
        print("  DRY-RUN VERIFICATION SUCCESSFUL: Pipeline initialized cleanly.")
        print("=" * 80)
        return

    # 6. Execute Training Loop
    print("\n[Step 6/6] Beginning Model Training...")
    print(f"  Total Batches per Epoch: {len(train_loader)} train | {len(val_loader)} val")
    print("-" * 80)

    best_val_loss = float("inf")
    best_val_macro_f1 = float("-inf")
    best_epoch = 0
    best_val_metrics = {}
    history = []
    epochs_without_improvement = 0

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training Phase
        model.train()
        train_loss_total = 0.0
        train_samples_count = 0
        train_y_true = []
        train_y_pred = []

        for batch_data, batch_labels, _ in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_data)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_size_cur = batch_data.size(0)
            train_loss_total += loss.item() * batch_size_cur
            train_samples_count += batch_size_cur

            preds = torch.argmax(logits, dim=-1)
            train_y_true.extend(batch_labels.cpu().tolist())
            train_y_pred.extend(preds.cpu().tolist())

        train_loss = train_loss_total / train_samples_count
        train_metrics = compute_classification_metrics(train_y_true, train_y_pred, CLASSES)

        # Validation Phase
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device, CLASSES)

        # Update Learning Rate Scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.time() - epoch_start

        # Record History Entry
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": round(val_loss, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_per_class": val_metrics["per_class"],
            "learning_rate": current_lr,
            "duration_sec": round(epoch_duration, 2),
        }
        history.append(epoch_record)

        # Check Best Model based on early_stopping_metric
        if early_stopping_metric == "macro_f1":
            cur_metric_val = val_metrics.get("macro_f1", 0.0)
            is_best = cur_metric_val > best_val_macro_f1
        else:
            cur_metric_val = val_loss
            is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_val_macro_f1 = val_metrics.get("macro_f1", 0.0)
            best_epoch = epoch
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss=val_loss,
                metrics=val_metrics,
                config=config,
                is_best=True,
            )
            marker = " [*BEST]"
        else:
            epochs_without_improvement += 1
            marker = ""

        # Print Epoch Summary
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train F1: {train_metrics['macro_f1']:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | "
            f"LR: {current_lr:.1e} ({epoch_duration:.1f}s){marker}"
        )

        # Early Stopping Check
        if epochs_without_improvement >= patience:
            print(f"\n[EARLY STOPPING] Validation {early_stopping_metric} did not improve for {patience} consecutive epochs. Stopping at epoch {epoch}.")
            break

    total_training_time = time.time() - start_train_time

    # Save Full Training History JSON
    history_path = checkpoint_dir / "training_history.json"
    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
        if device.type == "cuda"
        else None
    )
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "early_stopping_metric": early_stopping_metric,
                "best_val_loss": round(best_val_loss, 4),
                "best_val_accuracy": best_val_metrics.get("accuracy"),
                "best_val_macro_f1": best_val_metrics.get("macro_f1"),
                "total_training_time_sec": round(total_training_time, 2),
                "peak_gpu_memory_mb": peak_vram_mb,
                "epochs_completed": len(history),
                "history": history,
            },
            f,
            indent=2,
        )

    # Print Final Report
    print("\n" + "=" * 80)
    print("  TRAINING EXPERIMENT SUMMARY & AUDIT")
    print("=" * 80)
    print(f"Total Training Time:    {total_training_time:.1f}s ({len(history)} epochs completed)")
    if peak_vram_mb is not None:
        print(f"Peak GPU Memory:        {peak_vram_mb:.2f} MB")
    print(f"Selection Metric:       {early_stopping_metric}")
    print(f"Best Validation Epoch:  {best_epoch}")
    print(f"Best Validation Loss:   {best_val_loss:.4f}")
    print(f"Best Validation Acc:    {best_val_metrics.get('accuracy', 0.0):.4f}")
    print(f"Best Validation MacroF1:{best_val_metrics.get('macro_f1', 0.0):.4f}")
    print(f"Best Checkpoint File:   {checkpoint_dir / 'best_model.pt'}")
    print(f"Training History JSON:  {history_path}")
    print("-" * 80)
    print("Validation Split Performance (Best Epoch Per-Class Metrics):")
    print(f"  {'Class Name':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("  " + "-" * 60)
    for cname in CLASSES:
        cm = best_val_metrics.get("per_class", {}).get(cname, {})
        p = cm.get("precision", 0.0)
        r = cm.get("recall", 0.0)
        f = cm.get("f1_score", 0.0)
        s = cm.get("support", 0)
        print(f"  {cname:<16} | {p:10.4f} | {r:10.4f} | {f:10.4f} | {s:8d}")

    print("\nValidation Confusion Matrix (Rows: True, Columns: Predicted):")
    conf_mat = best_val_metrics.get("confusion_matrix", [])
    col_headers = "".join(f"{c[:6]:>8}" for c in CLASSES)
    print(f"  {'':<16} {col_headers}")
    for i, row in enumerate(conf_mat):
        row_str = "".join(f"{val:>8d}" for val in row)
        print(f"  {CLASSES[i]:<16} {row_str}")

    # Write Training Report Markdown
    report_path = checkpoint_dir / "training_report.md"
    try:
        with open(report_path, "w", encoding="utf-8") as rf:
            rf.write(f"# Training Experiment Report: {model_display_name}\n\n")
            rf.write(f"**Execution Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            rf.write(f"**Model Architecture:** `{config['model']}`  \n")
            rf.write(f"**Total / Trainable Parameters:** {total_p:,} / {train_p:,}  \n")
            rf.write(f"**Optimizer:** {config.get('optimizer', 'ADAM')}  \n")
            rf.write(f"**Head Learning Rate:** `{config.get('lr_head', effective_head_lr)}`  \n")
            rf.write(f"**Backbone Learning Rate:** `{config.get('lr_backbone', 'Frozen')}`  \n")
            rf.write(f"**Checkpoint Directory:** [`{rel_ckpt}`](file:///{checkpoint_dir})  \n")
            rf.write(f"**Best Checkpoint File:** [`best_model.pt`](file:///{checkpoint_dir / 'best_model.pt'})  \n")
            rf.write(f"**Training Manifest:** [`{train_manifest_path.name}`](file:///{train_manifest_path})  \n")
            rf.write(f"**Validation Manifest:** [`{val_manifest_path.name}`](file:///{val_manifest_path})  \n\n")
            rf.write("---\n\n## 1. Executive Summary\n\n")
            rf.write(f"- **Best Epoch:** {best_epoch} (out of {len(history)} epochs completed)\n")
            rf.write(f"- **Best Validation Accuracy:** **{best_val_metrics.get('accuracy', 0.0) * 100:.2f}%**\n")
            rf.write(f"- **Best Validation Macro F1:** **{best_val_metrics.get('macro_f1', 0.0):.4f}**\n")
            rf.write(f"- **Best Validation Loss:** {best_val_loss:.4f} (Selection Metric: {early_stopping_metric})\n")
            rf.write(f"- **Total Training Duration:** {total_training_time:.1f}s ({total_training_time / 60.0:.2f} min)\n")
            if peak_vram_mb is not None:
                rf.write(f"- **Peak GPU Memory:** {peak_vram_mb:.2f} MB\n")

            rf.write("\n---\n\n## 2. Dataset Distribution\n\n")
            rf.write(f"- **Training Partition:** {len(train_dataset)} samples\n- **Validation Partition:** {len(val_dataset)} samples\n\n")
            rf.write("| Class Name | Train Samples | Val Samples | Total Samples | Train Class Weight |\n")
            rf.write("|---|---:|---:|---:|---:|\n")
            for idx, cname in enumerate(CLASSES):
                t_cnt = train_counts[cname]
                v_cnt = val_counts[cname]
                w = train_weights[idx].item()
                rf.write(f"| `{cname}` | {t_cnt} | {v_cnt} | {t_cnt + v_cnt} | {w:.4f} |\n")
            rf.write(f"| **Total** | **{len(train_dataset)}** | **{len(val_dataset)}** | **{len(train_dataset) + len(val_dataset)}** | - |\n\n")

            baseline_meta_path = ROOT / "models" / "experiments" / "panns_cnn14" / "best_model_meta.json"
            if baseline_meta_path.exists():
                try:
                    with open(baseline_meta_path, "r", encoding="utf-8") as bf:
                        bdata = json.load(bf)
                    b_acc = bdata.get("metrics", {}).get("accuracy", 0.0)
                    b_f1 = bdata.get("metrics", {}).get("macro_f1", 0.0)
                    b_per_class = bdata.get("metrics", {}).get("per_class", {})

                    rf.write("---\n\n## 3. Comparison Against Frozen AudioSet Baseline\n\n")
                    rf.write(f"| Metric / Class | Baseline (AudioSet Val N=105) | Combined Model (AudioSet+NIGENS Val N={len(val_dataset)}) | Delta |\n")
                    rf.write("|---|---:|---:|---:|\n")
                    acc_cur = best_val_metrics.get("accuracy", 0.0)
                    f1_cur = best_val_metrics.get("macro_f1", 0.0)
                    rf.write(f"| **Overall Accuracy** | {b_acc * 100:.2f}% | {acc_cur * 100:.2f}% | {(acc_cur - b_acc) * 100:+.2f}% |\n")
                    rf.write(f"| **Macro F1 Score** | {b_f1:.4f} | {f1_cur:.4f} | {(f1_cur - b_f1):+.4f} |\n")
                    for cname in CLASSES:
                        b_c_f1 = b_per_class.get(cname, {}).get("f1_score", 0.0)
                        cur_c_f1 = best_val_metrics.get("per_class", {}).get(cname, {}).get("f1_score", 0.0)
                        rf.write(f"| `{cname}` F1 | {b_c_f1:.4f} | {cur_c_f1:.4f} | {(cur_c_f1 - b_c_f1):+.4f} |\n")
                    rf.write("\n")
                except Exception as e:
                    print(f"  [WARNING] Could not load baseline meta for comparison: {e}")

            rf.write("---\n\n## 4. Best Epoch Per-Class Performance\n\n")
            rf.write("| Class Name | Precision | Recall | F1-Score | Validation Support |\n")
            rf.write("|---|---:|---:|---:|---:|\n")
            for cname in CLASSES:
                cm = best_val_metrics.get("per_class", {}).get(cname, {})
                rf.write(f"| `{cname}` | {cm.get('precision', 0.0):.4f} | {cm.get('recall', 0.0):.4f} | {cm.get('f1_score', 0.0):.4f} | {cm.get('support', 0)} |\n")

            rf.write("\n---\n\n## 5. Validation Confusion Matrix\n\n")
            rf.write("Rows indicate Ground Truth, columns indicate Predictions:\n\n")
            rf.write("| True \\ Pred | " + " | ".join(f"`{c}`" for c in CLASSES) + " |\n")
            rf.write("|---|" + "---:|" * len(CLASSES) + "\n")
            conf_mat = best_val_metrics.get("confusion_matrix", [])
            for i, row in enumerate(conf_mat):
                row_str = " | ".join(str(val) for val in row)
                rf.write(f"| `{CLASSES[i]}` | {row_str} |\n")
            rf.write("\n")
        print(f"Training Report Markdown: {report_path}")
    except Exception as e:
        print(f"  [WARNING] Could not write training report markdown: {e}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Baseline & Transfer Learning Training Experiment for 5-Class Emergency Sound Detection.")
    parser.add_argument("--model", type=str, default="compact_cnn", choices=["compact_cnn", "panns_cnn14"], help="Model architecture ('compact_cnn' or 'panns_cnn14').")
    parser.add_argument("--pretrained-path", type=str, default=None, help="Path to pretrained PANNs checkpoint (default: models/pretrained/Cnn14_16k_mAP=0.438.pth).")
    parser.add_argument("--finetune-from", type=str, default=None, help="Path to best frozen checkpoint to fine-tune from (e.g. models/experiments/panns_cnn14/best_model.pt).")
    parser.add_argument("--lr-head", type=float, default=None, help="Learning rate for classification head during fine-tuning (default: 1e-4 for fine-tuning, or --lr for standard training).")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Learning rate for conv_block6 during fine-tuning (default: 1e-5).")
    parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze backbone weights during transfer learning (default: True).")
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false", help="Do not freeze backbone weights.")
    parser.add_argument("--early-stopping-metric", type=str, default=None, choices=["val_loss", "macro_f1"], help="Metric for best model selection and early stopping (default: macro_f1 for panns_cnn14, val_loss for compact_cnn).")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam/AdamW weight decay (default: 1e-4).")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio (default: 0.2).")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience in epochs (default: 5).")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save checkpoints (defaults: models/checkpoints for compact_cnn, models/experiments/panns_cnn14 for panns_cnn14, models/experiments/panns_cnn14_finetuned for fine-tuning).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--device", type=str, default=None, help="Target device ('cuda' or 'cpu').")
    parser.add_argument("--preflight-only", action="store_true", help="Perform preflight checks and print dataset inventory without training.")
    parser.add_argument("--dry-run", action="store_true", help="Run a single forward/backward batch step to verify pipeline.")
    parser.add_argument("--train-manifest", type=str, default=None, help="Path to training split manifest CSV (default: data/processed/train_split_manifest.csv).")
    parser.add_argument("--val-manifest", type=str, default=None, help="Path to validation split manifest CSV (default: data/processed/val_split_manifest.csv).")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"], help="Optimizer to use ('adam' or 'adamw'). Default: adam.")

    args = parser.parse_args()
    stopping_metric = args.early_stopping_metric
    if stopping_metric is None:
        stopping_metric = "macro_f1" if (args.model == "panns_cnn14" or args.finetune_from) else "val_loss"

    run_training_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        device_name=args.device,
        preflight_only=args.preflight_only,
        dry_run=args.dry_run,
        model_name=args.model,
        pretrained_path=args.pretrained_path,
        freeze_backbone=args.freeze_backbone,
        early_stopping_metric=stopping_metric,
        finetune_from=args.finetune_from,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        optimizer_name=args.optimizer,
    )


if __name__ == "__main__":
    main()
