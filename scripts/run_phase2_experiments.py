"""Phase 2 E4 Category Ablations and Matched Random Controls Runner.

Executes the 15 approved Phase 2 conditions across 3 seeds (42, 123, 2026):
- Condition 0: E2 Full (646 clips: 422 baseline + 224 targeted hard negatives).
- Conditions 1-7: 7 Category-Removal Ablations (602 to 627 clips).
- Conditions 8-14: 7 Matched-Size Random Controls (602 to 627 clips).
- Architecture: PANNs CNN14 (frozen backbone, 5-class linear transfer head).
- Target: Isolate category-specific semantic importance of hard-negative acoustic confusers.
- Invariant Validation: Identical 105 samples (shared_val_split_manifest.csv) across all runs.
- Checkpoint Isolation: Saved to models/experiments/phase2_e4/<condition>_seed<seed>/
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.class_config import CLASSES, CLASS_TO_ID, NUM_CLASSES
from src.dataset import AudioSetDataset, load_split_manifest
from src.model import PannsCNN14
from src.train import run_training_experiment, compute_classification_metrics

PROCESSED_DIR = ROOT / "data" / "processed"
E4_DIR = PROCESSED_DIR / "e4_ablations"
SHARED_VAL_PATH = PROCESSED_DIR / "shared_val_split_manifest.csv"
OUTPUT_BASE_DIR = ROOT / "models" / "experiments" / "phase2_e4"
PRETRAINED_PATH = ROOT / "models" / "pretrained" / "Cnn14_16k_mAP=0.438.pth"

CONDITIONS = {
    "e2_full": {
        "display_name": "E2 Full Reference",
        "train_manifest": PROCESSED_DIR / "e2_train_split_manifest.csv",
        "train_samples_expected": 646,
        "type": "unablated_parent_reference",
        "target_subdomain": "all",
        "description": "Unablated parent ceiling with all 224 targeted hard negatives",
    },
    # 7 Category Ablations
    "e4_minus_human_shouting": {
        "display_name": "E4 Minus Human Shouting",
        "train_manifest": E4_DIR / "e4_minus_human_shouting_train.csv",
        "train_samples_expected": 602,
        "type": "category_ablation",
        "target_subdomain": "human_shouting",
        "description": "Category ablation removing all 44 human shouting hard negatives",
    },
    "e4_minus_metal_impacts": {
        "display_name": "E4 Minus Metal Impacts",
        "train_manifest": E4_DIR / "e4_minus_metal_impacts_train.csv",
        "train_samples_expected": 606,
        "type": "category_ablation",
        "target_subdomain": "metal_impacts",
        "description": "Category ablation removing all 40 metal impact hard negatives",
    },
    "e4_minus_door_slams": {
        "display_name": "E4 Minus Door Slams",
        "train_manifest": E4_DIR / "e4_minus_door_slams_train.csv",
        "train_samples_expected": 613,
        "type": "category_ablation",
        "target_subdomain": "door_slams",
        "description": "Category ablation removing all 33 door slam hard negatives",
    },
    "e4_minus_construction_tools": {
        "display_name": "E4 Minus Construction Tools",
        "train_manifest": E4_DIR / "e4_minus_construction_tools_train.csv",
        "train_samples_expected": 616,
        "type": "category_ablation",
        "target_subdomain": "construction_tools",
        "description": "Category ablation removing all 30 construction tool hard negatives",
    },
    "e4_minus_car_alarms_detectors": {
        "display_name": "E4 Minus Car Alarms & Detectors",
        "train_manifest": E4_DIR / "e4_minus_car_alarms_detectors_train.csv",
        "train_samples_expected": 617,
        "type": "category_ablation",
        "target_subdomain": "car_alarms_detectors",
        "description": "Category ablation removing all 29 car alarm hard negatives",
    },
    "e4_minus_fireworks_firecrackers": {
        "display_name": "E4 Minus Fireworks & Firecrackers",
        "train_manifest": E4_DIR / "e4_minus_fireworks_firecrackers_train.csv",
        "train_samples_expected": 618,
        "type": "category_ablation",
        "target_subdomain": "fireworks_firecrackers",
        "description": "Category ablation removing all 28 fireworks hard negatives",
    },
    "e4_minus_whistles_high_pitch": {
        "display_name": "E4 Minus Whistles & High Pitch",
        "train_manifest": E4_DIR / "e4_minus_whistles_high_pitch_train.csv",
        "train_samples_expected": 627,
        "type": "category_ablation",
        "target_subdomain": "whistles_high_pitch",
        "description": "Category ablation removing all 19 whistle hard negatives",
    },
    # 7 Matched-Size Random Controls
    "e4_matched_control_human_shouting": {
        "display_name": "E4 Matched Control Shouting",
        "train_manifest": E4_DIR / "e4_matched_control_human_shouting_train.csv",
        "train_samples_expected": 602,
        "type": "matched_size_control",
        "target_subdomain": "human_shouting",
        "description": "Matched control removing 44 random hard negatives across subdomains",
    },
    "e4_matched_control_metal_impacts": {
        "display_name": "E4 Matched Control Metal Impacts",
        "train_manifest": E4_DIR / "e4_matched_control_metal_impacts_train.csv",
        "train_samples_expected": 606,
        "type": "matched_size_control",
        "target_subdomain": "metal_impacts",
        "description": "Matched control removing 40 random hard negatives across subdomains",
    },
    "e4_matched_control_door_slams": {
        "display_name": "E4 Matched Control Door Slams",
        "train_manifest": E4_DIR / "e4_matched_control_door_slams_train.csv",
        "train_samples_expected": 613,
        "type": "matched_size_control",
        "target_subdomain": "door_slams",
        "description": "Matched control removing 33 random hard negatives across subdomains",
    },
    "e4_matched_control_construction_tools": {
        "display_name": "E4 Matched Control Construction Tools",
        "train_manifest": E4_DIR / "e4_matched_control_construction_tools_train.csv",
        "train_samples_expected": 616,
        "type": "matched_size_control",
        "target_subdomain": "construction_tools",
        "description": "Matched control removing 30 random hard negatives across subdomains",
    },
    "e4_matched_control_car_alarms_detectors": {
        "display_name": "E4 Matched Control Car Alarms",
        "train_manifest": E4_DIR / "e4_matched_control_car_alarms_detectors_train.csv",
        "train_samples_expected": 617,
        "type": "matched_size_control",
        "target_subdomain": "car_alarms_detectors",
        "description": "Matched control removing 29 random hard negatives across subdomains",
    },
    "e4_matched_control_fireworks_firecrackers": {
        "display_name": "E4 Matched Control Fireworks",
        "train_manifest": E4_DIR / "e4_matched_control_fireworks_firecrackers_train.csv",
        "train_samples_expected": 618,
        "type": "matched_size_control",
        "target_subdomain": "fireworks_firecrackers",
        "description": "Matched control removing 28 random hard negatives across subdomains",
    },
    "e4_matched_control_whistles_high_pitch": {
        "display_name": "E4 Matched Control Whistles",
        "train_manifest": E4_DIR / "e4_matched_control_whistles_high_pitch_train.csv",
        "train_samples_expected": 627,
        "type": "matched_size_control",
        "target_subdomain": "whistles_high_pitch",
        "description": "Matched control removing 19 random hard negatives across subdomains",
    },
}

ALL_SEEDS = [42, 123, 2026]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_checkpoint_predictions(ckpt_path: Path, val_manifest_path: Path, device: torch.device):
    """Load checkpoint and generate per-sample validation predictions."""
    model = PannsCNN14(num_classes=NUM_CLASSES, freeze_backbone=True)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()

    val_samples = load_split_manifest(val_manifest_path, root_dir=ROOT)
    val_dataset = AudioSetDataset(val_manifest_path.parent, samples=val_samples, return_raw=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    rows = []
    y_true = []
    y_pred = []
    sample_idx = 0

    with torch.no_grad():
        for batch_audio, batch_labels, _ in val_loader:
            batch_audio = batch_audio.to(device)
            logits = model(batch_audio)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            for b in range(len(batch_labels)):
                fpath, cid, cname = val_samples[sample_idx]
                sample_idx += 1
                t_id = int(batch_labels[b].item())
                p_id = int(preds[b])
                y_true.append(t_id)
                y_pred.append(p_id)

                row_dict = {
                    "sample_index": sample_idx,
                    "file_path": str(fpath.relative_to(ROOT)).replace("\\", "/"),
                    "video_id": fpath.stem.split("_")[1] if len(fpath.stem.split("_")) > 1 else fpath.stem,
                    "true_id": t_id,
                    "true_class": CLASSES[t_id],
                    "pred_id": p_id,
                    "pred_class": CLASSES[p_id],
                    "is_correct": bool(t_id == p_id),
                }
                for c_i, c_n in enumerate(CLASSES):
                    row_dict[f"prob_{c_n}"] = float(probs[b, c_i])

                rows.append(row_dict)

    pred_df = pd.DataFrame(rows)
    return pred_df, y_true, y_pred


def compute_extended_metrics(y_true: list, y_pred: list) -> dict:
    """Compute standard metrics plus Emergency Recall/Precision and False Emergency Rate."""
    base_metrics = compute_classification_metrics(y_true, y_pred, CLASSES)
    cm = np.array(base_metrics["confusion_matrix"])

    emerg_indices = [0, 1, 2, 3]
    other_idx = 4

    emerg_recalls = [base_metrics["per_class"][CLASSES[i]]["recall"] for i in emerg_indices]
    emerg_precisions = [base_metrics["per_class"][CLASSES[i]]["precision"] for i in emerg_indices]

    emerg_recall_macro = float(np.mean(emerg_recalls))
    emerg_precision_macro = float(np.mean(emerg_precisions))

    total_emerg_support = sum(base_metrics["per_class"][CLASSES[i]]["support"] for i in emerg_indices)
    exact_emerg_correct = sum(cm[i, i] for i in emerg_indices)
    emerg_recall_exact = float(exact_emerg_correct / total_emerg_support) if total_emerg_support > 0 else 0.0

    detected_emerg = sum(cm[i, j] for i in emerg_indices for j in emerg_indices)
    emerg_detection_rate = float(detected_emerg / total_emerg_support) if total_emerg_support > 0 else 0.0

    other_support = base_metrics["per_class"]["other"]["support"]
    other_recall = float(base_metrics["per_class"]["other"]["recall"])
    other_precision = float(base_metrics["per_class"]["other"]["precision"])

    other_predicted_as_emerg = sum(cm[other_idx, j] for j in emerg_indices)
    false_emergency_rate = float(other_predicted_as_emerg / other_support) if other_support > 0 else 0.0

    return {
        "accuracy": base_metrics["accuracy"],
        "macro_precision": base_metrics["macro_precision"],
        "macro_recall": base_metrics["macro_recall"],
        "macro_f1": base_metrics["macro_f1"],
        "emergency_recall_macro": round(emerg_recall_macro, 4),
        "emergency_recall_exact": round(emerg_recall_exact, 4),
        "emergency_detection_rate": round(emerg_detection_rate, 4),
        "emergency_precision_macro": round(emerg_precision_macro, 4),
        "other_recall": round(other_recall, 4),
        "other_precision": round(other_precision, 4),
        "false_emergency_rate": round(false_emergency_rate, 4),
        "other_support": other_support,
        "other_predicted_as_emerg": int(other_predicted_as_emerg),
        "per_class": base_metrics["per_class"],
        "confusion_matrix": base_metrics["confusion_matrix"],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2 E4 Category Ablation & Matched Random Control Training Suite"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS.keys()),
        choices=list(CONDITIONS.keys()),
        help="Subset of conditions to execute (defaults to all 15 conditions)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=ALL_SEEDS,
        help="Seeds to execute (defaults to [42, 123, 2026])",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify manifests, configurations, and print execution plan without training",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly permit overwriting existing checkpoints (refused by default)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (cuda or cpu)",
    )
    return parser.parse_args()


def run_phase2():
    args = parse_args()

    print("=" * 80)
    print("PHASE 2 E4: CATEGORY ABLATIONS & MATCHED CONTROLS EXECUTION RUNNER")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target Base Output Directory: {OUTPUT_BASE_DIR}")
    print(f"Selected Conditions ({len(args.conditions)}): {args.conditions}")
    print(f"Selected Seeds ({len(args.seeds)}): {args.seeds}")
    print(f"Total Planned Runs: {len(args.conditions) * len(args.seeds)}")
    print(f"Dry-Run Mode: {args.dry_run}")
    print(f"Overwrite Existing Checkpoints: {args.overwrite}")

    if not PRETRAINED_PATH.exists():
        raise FileNotFoundError(f"Missing pretrained PANNs checkpoint: {PRETRAINED_PATH}")

    if not SHARED_VAL_PATH.exists():
        raise FileNotFoundError(f"Missing shared validation manifest: {SHARED_VAL_PATH}")

    shared_val_hash = compute_sha256(SHARED_VAL_PATH)
    print(f"Shared Validation SHA-256: {shared_val_hash}")

    # Verify all manifest paths exist and sample counts match
    plan = []
    for cond_key in args.conditions:
        cinfo = CONDITIONS[cond_key]
        mpath = cinfo["train_manifest"]
        if not mpath.exists():
            raise FileNotFoundError(f"Missing train manifest for {cond_key}: {mpath}")

        mhash = compute_sha256(mpath)
        df = pd.read_csv(mpath)
        actual_rows = len(df)
        expected_rows = cinfo["train_samples_expected"]
        if actual_rows != expected_rows:
            raise ValueError(
                f"Row count mismatch for {cond_key}: expected {expected_rows}, found {actual_rows}"
            )

        for seed in args.seeds:
            run_name = f"{cond_key}_seed{seed}"
            run_dir = OUTPUT_BASE_DIR / run_name
            ckpt_exists = (run_dir / "best_model.pt").exists()
            plan.append({
                "condition_key": cond_key,
                "display_name": cinfo["display_name"],
                "type": cinfo["type"],
                "target_subdomain": cinfo["target_subdomain"],
                "seed": seed,
                "train_manifest": str(mpath),
                "train_manifest_hash": mhash,
                "train_samples": actual_rows,
                "val_manifest": str(SHARED_VAL_PATH),
                "val_manifest_hash": shared_val_hash,
                "run_dir": run_dir,
                "checkpoint_exists": ckpt_exists,
            })

    print(f"\nManifest pre-flight verification: 100% passed for all {len(plan)} runs.")

    # In dry-run mode, display execution matrix and exit cleanly
    if args.dry_run:
        print("\n" + "#" * 80)
        print("  DRY-RUN EXECUTION PLAN (NO TRAINING PERFORMED)")
        print("#" * 80)
        print(f"{'Index':<5} | {'Run Name':<42} | {'Train N':<7} | {'Type':<24} | {'Status'}")
        print("-" * 95)
        for idx, item in enumerate(plan, start=1):
            run_name = f"{item['condition_key']}_seed{item['seed']}"
            status = "EXISTS (SKIP)" if item["checkpoint_exists"] and not args.overwrite else "QUEUED"
            if item["checkpoint_exists"] and args.overwrite:
                status = "EXISTS (WILL OVERWRITE)"
            print(f"{idx:<5} | {run_name:<42} | {item['train_samples']:<7} | {item['type']:<24} | {status}")
        print("-" * 95)
        print("Dry-run complete. All manifests verified. Zero models trained.")
        return

    # Non-dry-run: determine execution device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nExecution Device: {device}")

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()
    completed_runs = 0

    for idx, item in enumerate(plan, start=1):
        run_name = f"{item['condition_key']}_seed{item['seed']}"
        run_dir = item["run_dir"]

        ckpt_file = run_dir / "best_model.pt"
        pred_file = run_dir / "val_predictions.csv"
        metrics_file = run_dir / "metrics.json"

        # Check if already completed
        if ckpt_file.exists() and pred_file.exists() and metrics_file.exists() and not args.overwrite:
            print(f"\n[{idx}/{len(plan)}] SKIPPING {run_name}: Checkpoint and evaluation already complete at {run_dir}")
            continue

        print("\n" + "=" * 80)
        print(f"  [{idx}/{len(plan)}] EXECUTING RUN: {run_name.upper()} (Seed: {item['seed']})")
        print(f"  Type: {item['type']} | Train Samples: {item['train_samples']}")
        print(f"  Output: {run_dir}")
        print("=" * 80)

        run_dir.mkdir(parents=True, exist_ok=True)
        run_start = time.time()

        # Record run metadata before execution
        run_meta = {
            "timestamp": datetime.now().isoformat(),
            "run_name": run_name,
            "condition_key": item["condition_key"],
            "display_name": item["display_name"],
            "type": item["type"],
            "target_subdomain": item["target_subdomain"],
            "seed": item["seed"],
            "train_manifest": item["train_manifest"],
            "train_manifest_sha256": item["train_manifest_hash"],
            "train_samples": item["train_samples"],
            "val_manifest": item["val_manifest"],
            "val_manifest_sha256": item["val_manifest_hash"],
            "hyperparameters": {
                "model_name": "panns_cnn14",
                "pretrained_path": str(PRETRAINED_PATH),
                "freeze_backbone": True,
                "epochs": 30,
                "batch_size": 16,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "optimizer": "adam",
                "patience": 7,
                "early_stopping_metric": "macro_f1",
            },
            "command_line": " ".join(sys.argv),
        }
        with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)

        try:
            # Execute training
            run_training_experiment(
                epochs=30,
                batch_size=16,
                lr=0.001,
                weight_decay=0.0001,
                patience=7,
                checkpoint_dir=run_dir,
                seed=item["seed"],
                device_name=str(device),
                model_name="panns_cnn14",
                pretrained_path=PRETRAINED_PATH,
                freeze_backbone=True,
                early_stopping_metric="macro_f1",
                train_manifest=Path(item["train_manifest"]),
                val_manifest=Path(item["val_manifest"]),
                optimizer_name="adam",
            )

            run_duration = time.time() - run_start

            # Evaluate best checkpoint and generate val_predictions.csv
            best_ckpt = run_dir / "best_model.pt"
            pred_df, y_true, y_pred = evaluate_checkpoint_predictions(best_ckpt, SHARED_VAL_PATH, device)
            pred_csv_path = run_dir / "val_predictions.csv"
            pred_df.to_csv(pred_csv_path, index=False)

            # Compute extended validation metrics
            ext_metrics = compute_extended_metrics(y_true, y_pred)
            ext_metrics["runtime_seconds"] = round(run_duration, 2)
            ext_metrics["seed"] = item["seed"]
            ext_metrics["condition"] = item["condition_key"]

            hist_path = run_dir / "training_history.json"
            if hist_path.exists():
                with open(hist_path, "r", encoding="utf-8") as hf:
                    hist_data = json.load(hf)
                    ext_metrics["best_epoch"] = hist_data.get("best_epoch")
                    ext_metrics["epochs_completed"] = hist_data.get("epochs_completed")
                    ext_metrics["peak_gpu_memory_mb"] = hist_data.get("peak_gpu_memory_mb")

            with open(run_dir / "metrics.json", "w", encoding="utf-8") as mf:
                json.dump(ext_metrics, mf, indent=2)

            # Prune intermediate epoch weights for this run to conserve disk space,
            # preserving best_model.pt, val_predictions.csv, metrics.json, and all JSON metadata
            for old_ckpt in run_dir.glob("checkpoint_epoch_*.pt"):
                try:
                    old_ckpt.unlink()
                except OSError:
                    pass
            latest_ckpt = run_dir / "latest_checkpoint.pt"
            if latest_ckpt.exists():
                try:
                    latest_ckpt.unlink()
                except OSError:
                    pass

            completed_runs += 1
            print(f"Run {run_name} complete in {run_duration:.1f}s | Val Acc: {ext_metrics['accuracy']:.4f} | Val F1: {ext_metrics['macro_f1']:.4f}")

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(f"\n[FATAL ERROR] Run {run_name} failed with exception:\n{err_msg}")
            fail_record = {
                "timestamp": datetime.now().isoformat(),
                "run_name": run_name,
                "condition_key": item["condition_key"],
                "seed": item["seed"],
                "error": str(e),
                "traceback": err_msg,
            }
            with open(run_dir / "failure_log.json", "w", encoding="utf-8") as ff:
                json.dump(fail_record, ff, indent=2)
            raise

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"PHASE 2 TRAINING SWEEP COMPLETE: {completed_runs} runs executed in {total_time:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_phase2()
