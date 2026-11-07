"""Phase 1 Main Hypothesis Controlled Experiment Runner.

Executes a 3x3 factorial experiment:
- Conditions: E1 Baseline (422), E2 Matched (634), E3 General (634).
- Seeds: 42, 123, 2026.
- Architecture: PANNs CNN14 (frozen backbone, 5-class transfer head).
- Target: Validate whether E2 Matched reduces false emergency alarms compared to E3 General.
- Invariant Validation: Identical 105 samples (shared_val_split_manifest.csv) across all runs.
- Checkpoint isolation: Saved to models/experiments/phase1_main/<condition>_seed<seed>/
"""

from datetime import datetime
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
SHARED_VAL_PATH = PROCESSED_DIR / "shared_val_split_manifest.csv"
OUTPUT_BASE_DIR = ROOT / "models" / "experiments" / "phase1_main"
PRETRAINED_PATH = ROOT / "models" / "pretrained" / "Cnn14_16k_mAP=0.438.pth"

CONDITIONS = {
    "e1_baseline": {
        "display_name": "E1 Baseline",
        "train_manifest": PROCESSED_DIR / "e1_train_split_manifest.csv",
        "train_samples_expected": 422,
        "description": "Original AudioSet baseline training dataset",
    },
    "e2_matched": {
        "display_name": "E2 Matched",
        "train_manifest": PROCESSED_DIR / "e2_matched_train_split_manifest.csv",
        "train_samples_expected": 634,
        "description": "422 baseline + 212 targeted hard-negative additions (Seed 42)",
    },
    "e3_general": {
        "display_name": "E3 General",
        "train_manifest": PROCESSED_DIR / "e3_train_split_manifest.csv",
        "train_samples_expected": 634,
        "description": "422 baseline + 212 general background additions",
    },
}

SEEDS = [42, 123, 2026]


def evaluate_checkpoint_predictions(ckpt_path: Path, val_manifest_path: Path, device: torch.device):
    """Load best checkpoint and compute per-sample validation predictions and probabilities."""
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
    """Compute standard metrics plus Emergency Recall, Emergency Precision, Other Recall, and False Emergency Rate."""
    base_metrics = compute_classification_metrics(y_true, y_pred, CLASSES)
    cm = np.array(base_metrics["confusion_matrix"])

    # Classes: 0: scream, 1: glass, 2: gunshot, 3: siren, 4: other
    emerg_indices = [0, 1, 2, 3]
    other_idx = 4

    # Emergency metrics
    emerg_recalls = [base_metrics["per_class"][CLASSES[i]]["recall"] for i in emerg_indices]
    emerg_precisions = [base_metrics["per_class"][CLASSES[i]]["precision"] for i in emerg_indices]

    emerg_recall_macro = float(np.mean(emerg_recalls))
    emerg_precision_macro = float(np.mean(emerg_precisions))

    # Exact emergency correct / total emergency support
    total_emerg_support = sum(base_metrics["per_class"][CLASSES[i]]["support"] for i in emerg_indices)
    exact_emerg_correct = sum(cm[i, i] for i in emerg_indices)
    emerg_recall_exact = float(exact_emerg_correct / total_emerg_support) if total_emerg_support > 0 else 0.0

    # Detection: any emergency predicted for emergency ground truth
    detected_emerg = sum(cm[i, j] for i in emerg_indices for j in emerg_indices)
    emerg_detection_rate = float(detected_emerg / total_emerg_support) if total_emerg_support > 0 else 0.0

    # Other metrics
    other_support = base_metrics["per_class"]["other"]["support"]
    other_tp = cm[other_idx, other_idx]
    other_recall = float(base_metrics["per_class"]["other"]["recall"])
    other_precision = float(base_metrics["per_class"]["other"]["precision"])

    # False emergency rate: actual other predicted as any emergency / actual other support
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


def run_all_phase1_experiments():
    print("=" * 80)
    print("PHASE 1: MAIN HYPOTHESIS CONTROLLED EXPERIMENTS (3 CONDITIONS x 3 SEEDS)")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target Output Directory: {OUTPUT_BASE_DIR}")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Seeds: {SEEDS}")
    print(f"Shared Validation Set: {SHARED_VAL_PATH.name} (105 samples)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    all_results = {}
    start_total_time = time.time()

    for cond_key, cond_info in CONDITIONS.items():
        all_results[cond_key] = {
            "display_name": cond_info["display_name"],
            "description": cond_info["description"],
            "runs": {},
        }

        for seed in SEEDS:
            run_name = f"{cond_key}_seed{seed}"
            run_dir = OUTPUT_BASE_DIR / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "#" * 80)
            print(f"  STARTING RUN: {run_name.upper()} (Seed: {seed})")
            print("#" * 80)

            run_start = time.time()

            # Execute training
            run_training_experiment(
                epochs=30,
                batch_size=16,
                lr=0.001,
                weight_decay=0.0001,
                patience=7,
                checkpoint_dir=run_dir,
                seed=seed,
                device_name=str(device),
                model_name="panns_cnn14",
                pretrained_path=PRETRAINED_PATH,
                freeze_backbone=True,
                early_stopping_metric="macro_f1",
                train_manifest=cond_info["train_manifest"],
                val_manifest=SHARED_VAL_PATH,
                optimizer_name="adam",
            )

            run_duration = time.time() - run_start

            # Evaluate best checkpoint and generate val_predictions.csv
            best_ckpt = run_dir / "best_model.pt"
            pred_df, y_true, y_pred = evaluate_checkpoint_predictions(best_ckpt, SHARED_VAL_PATH, device)
            pred_csv_path = run_dir / "val_predictions.csv"
            pred_df.to_csv(pred_csv_path, index=False)

            # Compute extended metrics
            ext_metrics = compute_extended_metrics(y_true, y_pred)
            ext_metrics["runtime_seconds"] = round(run_duration, 2)
            ext_metrics["seed"] = seed
            ext_metrics["condition"] = cond_key

            # Load history to extract best epoch and peak memory
            hist_path = run_dir / "training_history.json"
            if hist_path.exists():
                with open(hist_path, "r", encoding="utf-8") as hf:
                    hist_data = json.load(hf)
                    ext_metrics["best_epoch"] = hist_data.get("best_epoch")
                    ext_metrics["epochs_completed"] = hist_data.get("epochs_completed")
                    ext_metrics["peak_gpu_memory_mb"] = hist_data.get("peak_gpu_memory_mb")

            # Save metrics.json
            metrics_json_path = run_dir / "metrics.json"
            with open(metrics_json_path, "w", encoding="utf-8") as mf:
                json.dump(ext_metrics, mf, indent=2)

            all_results[cond_key]["runs"][str(seed)] = ext_metrics

            print(f"\n[COMPLETED RUN {run_name}] Duration: {run_duration:.1f}s")
            print(f"  Macro F1: {ext_metrics['macro_f1']:.4f} | Accuracy: {ext_metrics['accuracy']:.4f}")
            print(f"  Emergency Recall: {ext_metrics['emergency_recall_macro']:.4f} | Emergency Precision: {ext_metrics['emergency_precision_macro']:.4f}")
            print(f"  Other Recall: {ext_metrics['other_recall']:.4f} | False Emergency Rate: {ext_metrics['false_emergency_rate'] * 100:.2f}% ({ext_metrics['other_predicted_as_emerg']}/{ext_metrics['other_support']})")

    total_wall_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f"ALL 9 PHASE 1 EXPERIMENTS COMPLETED IN {total_wall_time / 60.0:.2f} MINUTES")
    print("=" * 80)

    # Compute Aggregate Stats (Mean +/- Std)
    summary_stats = {}
    metric_keys = [
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "emergency_recall_macro", "emergency_recall_exact", "emergency_detection_rate",
        "emergency_precision_macro", "other_recall", "false_emergency_rate"
    ]

    for cond_key, cond_data in all_results.items():
        summary_stats[cond_key] = {"display_name": cond_data["display_name"], "metrics": {}}
        runs = cond_data["runs"]

        for m in metric_keys:
            vals = [runs[str(s)][m] for s in SEEDS]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary_stats[cond_key]["metrics"][m] = {
                "values": vals,
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "formatted": f"{mean_val:.4f} +/- {std_val:.4f}",
            }

    aggregated_output = {
        "timestamp": datetime.now().isoformat(),
        "total_wall_time_seconds": round(total_wall_time, 2),
        "seeds": SEEDS,
        "summary_statistics": summary_stats,
        "conditions": all_results,
    }

    agg_path = OUTPUT_BASE_DIR / "phase1_aggregated_results.json"
    with open(agg_path, "w", encoding="utf-8") as af:
        json.dump(aggregated_output, af, indent=2)
    print(f"Saved aggregated results to: {agg_path}")

    return aggregated_output


if __name__ == "__main__":
    run_all_phase1_experiments()
