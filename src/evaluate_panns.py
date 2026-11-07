"""Independent Evaluation of PANNs CNN14 on AudioSet Held-Out Evaluation Dataset.

Loads best checkpoint from `models/experiments/panns_cnn14/best_model.pt`
and evaluates against validated evaluation WAV files in `data/raw/eval/`.

Integrity guarantees:
- No training, fine-tuning, or threshold tuning.
- Exact checkpoint state dict loaded.
- Cross-split isolation check verified before evaluation.
- Per-class metrics, confusion matrix, coverage analysis, and failure audit reported.
"""

from datetime import datetime
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Configure UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

try:
    from class_config import CLASSES, CLASS_TO_ID, ID_TO_CLASS, NUM_CLASSES
    from dataset import (
        AudioSetDataset,
        SAMPLE_RATE,
        TARGET_DURATION_SEC,
        TARGET_SAMPLES,
        run_preflight_dataset_check,
        extract_video_id_from_path,
    )
    from model import PannsCNN14, count_parameters
    from train import compute_classification_metrics
except ImportError:
    from src.class_config import CLASSES, CLASS_TO_ID, ID_TO_CLASS, NUM_CLASSES
    from src.dataset import (
        AudioSetDataset,
        SAMPLE_RATE,
        TARGET_DURATION_SEC,
        TARGET_SAMPLES,
        run_preflight_dataset_check,
        extract_video_id_from_path,
    )
    from src.model import PannsCNN14, count_parameters
    from src.train import compute_classification_metrics

EVAL_AUDIO_DIR = ROOT / "data" / "raw" / "eval"
TRAIN_AUDIO_DIR = ROOT / "data" / "raw" / "train"
EVAL_MANIFEST = ROOT / "data" / "processed" / "eval_manifest.csv"
EVAL_RESULTS_CSV = ROOT / "data" / "processed" / "eval_download_results.csv"
CHECKPOINT_PATH = ROOT / "models" / "experiments" / "panns_cnn14" / "best_model.pt"
OUTPUT_REPORT_PATH = ROOT / "models" / "experiments" / "panns_cnn14" / "eval_report.json"


def evaluate_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[List[float]]]:
    """Run forward inference over the evaluation dataset."""
    model.eval()
    y_true = []
    y_pred = []
    all_probs = []

    with torch.no_grad():
        for batch_data, batch_labels, _ in dataloader:
            batch_data = batch_data.to(device)
            logits = model(batch_data)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            y_true.extend(batch_labels.tolist())
            y_pred.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return y_true, y_pred, all_probs


def main():
    print("=" * 80)
    print("  INDEPENDENT EVALUATION: PANNs CNN14 (FROZEN BACKBONE)")
    print("=" * 80)
    print(f"Checkpoint File:       {CHECKPOINT_PATH.relative_to(ROOT)}")
    print(f"Evaluation Audio Dir:  {EVAL_AUDIO_DIR.relative_to(ROOT)}")
    print(f"Evaluation Manifest:   {EVAL_MANIFEST.relative_to(ROOT)}")
    print(f"Canonical Classes:     {CLASSES}")
    print("-" * 80)

    # 1. Preflight Dataset Check
    print("\n[Step 1/5] Running preflight check on evaluation audio...")
    is_ready, issues, stats = run_preflight_dataset_check(EVAL_AUDIO_DIR)
    for cname in CLASSES:
        cnt = stats["classes_found"].get(cname, 0)
        print(f"  [OK] {cname:<16}: {cnt:4d} valid audio files")

    print(f"  Total Valid Files:   {stats['total_valid_files']}")
    print(f"  Format Errors:       {stats['format_errors']}")
    print(f"  Rate Violations:     {stats['sample_rate_violations']}")

    if stats["total_valid_files"] == 0:
        raise RuntimeError("No valid audio files found in evaluation directory!")

    # 2. Strict Cross-Split Isolation Check
    print("\n[Step 2/5] Verifying strict cross-split video ID isolation...")
    eval_files = list(EVAL_AUDIO_DIR.rglob("*.wav"))
    eval_vids = set(extract_video_id_from_path(p) for p in eval_files)

    train_files = list(TRAIN_AUDIO_DIR.rglob("*.wav"))
    train_vids = set(extract_video_id_from_path(p) for p in train_files)

    overlap = eval_vids.intersection(train_vids)
    if overlap:
        raise RuntimeError(f"FATAL: Cross-split leakage detected! Overlapping video IDs: {list(overlap)[:5]}")
    print(f"  Eval valid files: {len(eval_files)} ({len(eval_vids)} unique video IDs)")
    print(f"  Train valid files: {len(train_files)} ({len(train_vids)} unique video IDs)")
    print(f"  [OK] Cross-split video ID overlap: 0 (Strict isolation confirmed).")

    # 3. Load Checkpoint & Model
    print("\n[Step 3/5] Loading PANNs CNN14 model and best checkpoint...")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = PannsCNN14(num_classes=NUM_CLASSES, freeze_backbone=True)
    ckpt_state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_state["model_state_dict"])
    model = model.to(device)
    model.eval()

    total_p, train_p = count_parameters(model)
    best_epoch = ckpt_state.get("epoch", "unknown")
    best_val_f1 = ckpt_state.get("metrics", {}).get("macro_f1", "unknown")
    print(f"  Model loaded successfully from epoch {best_epoch} (val macro F1: {best_val_f1})")
    print(f"  Parameters: Total={total_p:,} | Head Trainable={train_p:,}")

    # 4. Build Evaluation DataLoader
    print("\n[Step 4/5] Preparing Evaluation DataLoader...")
    eval_dataset = AudioSetDataset(EVAL_AUDIO_DIR, classes=CLASSES, class_to_id=CLASS_TO_ID, return_raw=True)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"  Total evaluation batches: {len(eval_loader)} (batch size: 16)")

    # 5. Execute Evaluation
    print("\n[Step 5/5] Executing forward evaluation inference...")
    y_true, y_pred, all_probs = evaluate_dataset(model, eval_loader, device)

    metrics = compute_classification_metrics(y_true, y_pred, CLASSES)

    # Coverage calculations
    with open(EVAL_MANIFEST, "r", encoding="utf-8") as f:
        manifest_entries = list(csv.DictReader(f))
    total_manifest = len(manifest_entries)
    evaluated_count = len(y_true)
    coverage_pct = (evaluated_count / float(total_manifest)) * 100.0 if total_manifest > 0 else 0.0

    # Failure Audit from results CSV
    download_audit = {"total_manifest": total_manifest, "evaluated_valid": evaluated_count}
    if EVAL_RESULTS_CSV.exists():
        df_results = pd.read_csv(EVAL_RESULTS_CSV)
        status_counts = df_results["status"].value_counts().to_dict()
        download_audit["status_counts"] = status_counts
        unavailable_df = df_results[df_results["status"] != "SUCCESS"]
        download_audit["total_unavailable"] = len(unavailable_df)
        download_audit["failure_reasons"] = {}
        for msg, cnt in unavailable_df["error_message"].value_counts().head(10).items():
            download_audit["failure_reasons"][str(msg)] = int(cnt)

    # Print Final Audit Report
    print("\n" + "=" * 80)
    print("  INDEPENDENT EVALUATION RESULTS & AUDIT")
    print("=" * 80)
    print(f"Evaluation Coverage:    {evaluated_count}/{total_manifest} clips ({coverage_pct:.2f}%)")
    print(f"Overall Accuracy:       {metrics['accuracy'] * 100:.2f}% ({metrics['accuracy']:.4f})")
    print(f"Macro Precision:        {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:           {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:         {metrics['macro_f1']:.4f}")
    print("-" * 80)
    print("Per-Class Evaluation Performance:")
    print(f"  {'Class Name':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("  " + "-" * 60)
    for cname in CLASSES:
        cm = metrics["per_class"][cname]
        print(f"  {cname:<16} | {cm['precision']:10.4f} | {cm['recall']:10.4f} | {cm['f1_score']:10.4f} | {cm['support']:8d}")

    print("\nEvaluation Confusion Matrix (Rows: True, Columns: Predicted):")
    conf_mat = metrics["confusion_matrix"]
    col_headers = "".join(f"{c[:6]:>8}" for c in CLASSES)
    print(f"  {'':<16} {col_headers}")
    for i, row in enumerate(conf_mat):
        row_str = "".join(f"{val:>8d}" for val in row)
        print(f"  {CLASSES[i]:<16} {row_str}")

    print("\nDownload & Availability Breakdown:")
    print(f"  Total Manifest Rows:  {total_manifest}")
    print(f"  Successfully Evaluated:{evaluated_count} ({coverage_pct:.1f}%)")
    print(f"  Unavailable / Failed:  {total_manifest - evaluated_count} ({(100.0 - coverage_pct):.1f}%)")
    if "status_counts" in download_audit:
        for st, c in download_audit["status_counts"].items():
            print(f"    - {st:<20}: {c}")

    print("=" * 80)

    # Save Evaluation Report JSON
    report_dict = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(CHECKPOINT_PATH),
        "best_training_epoch": best_epoch,
        "best_training_val_macro_f1": best_val_f1,
        "evaluation_coverage": {
            "total_manifest_entries": total_manifest,
            "evaluated_clips": evaluated_count,
            "coverage_percentage": round(coverage_pct, 2),
            "unavailable_clips": total_manifest - evaluated_count,
        },
        "metrics": metrics,
        "download_audit": download_audit,
    }

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\nFull evaluation report saved to: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
