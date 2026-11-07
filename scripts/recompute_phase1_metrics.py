"""Recompute Phase 1 validation metrics and predictions using exact trained checkpoints.

Loads best_model.pt for each of the 9 trained Phase 1 runs with return_raw=True,
extracts sample-level predictions into val_predictions.csv,
computes extended metrics into metrics.json,
and generates phase1_aggregated_results.json with Mean +/- Std.
"""

from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_phase1_experiments import (
    CONDITIONS, SEEDS, OUTPUT_BASE_DIR, SHARED_VAL_PATH,
    evaluate_checkpoint_predictions, compute_extended_metrics
)


def main():
    print("=" * 80)
    print("RECOMPUTING PHASE 1 EXTENDED METRICS ACROSS ALL 9 TRAINED RUNS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = {}

    for cond_key, cond_info in CONDITIONS.items():
        all_results[cond_key] = {
            "display_name": cond_info["display_name"],
            "description": cond_info["description"],
            "runs": {},
        }

        for seed in SEEDS:
            run_name = f"{cond_key}_seed{seed}"
            run_dir = OUTPUT_BASE_DIR / run_name
            best_ckpt = run_dir / "best_model.pt"

            if not best_ckpt.exists():
                raise FileNotFoundError(f"Missing checkpoint: {best_ckpt}")

            print(f"\nProcessing {run_name}...")

            # Evaluate with return_raw=True
            pred_df, y_true, y_pred = evaluate_checkpoint_predictions(best_ckpt, SHARED_VAL_PATH, device)
            pred_csv_path = run_dir / "val_predictions.csv"
            pred_df.to_csv(pred_csv_path, index=False)

            # Compute extended metrics
            ext_metrics = compute_extended_metrics(y_true, y_pred)
            ext_metrics["seed"] = seed
            ext_metrics["condition"] = cond_key

            # Read training history for best epoch, duration, and peak memory
            hist_path = run_dir / "training_history.json"
            if hist_path.exists():
                with open(hist_path, "r", encoding="utf-8") as hf:
                    hist_data = json.load(hf)
                    ext_metrics["best_epoch"] = hist_data.get("best_epoch")
                    ext_metrics["epochs_completed"] = hist_data.get("epochs_completed")
                    ext_metrics["total_training_time_sec"] = hist_data.get("total_training_time_sec")
                    ext_metrics["peak_gpu_memory_mb"] = hist_data.get("peak_gpu_memory_mb")

            # Save metrics.json
            metrics_json_path = run_dir / "metrics.json"
            with open(metrics_json_path, "w", encoding="utf-8") as mf:
                json.dump(ext_metrics, mf, indent=2)

            all_results[cond_key]["runs"][str(seed)] = ext_metrics

            print(f"  Accuracy: {ext_metrics['accuracy']:.4f} | Macro F1: {ext_metrics['macro_f1']:.4f}")
            print(f"  Emergency Recall: {ext_metrics['emergency_recall_macro']:.4f} | Emergency Precision: {ext_metrics['emergency_precision_macro']:.4f}")
            print(f"  Other Recall: {ext_metrics['other_recall']:.4f} | False Emergency Rate: {ext_metrics['false_emergency_rate'] * 100:.2f}% ({ext_metrics['other_predicted_as_emerg']}/{ext_metrics['other_support']})")

    # Aggregate statistics
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
        "seeds": SEEDS,
        "summary_statistics": summary_stats,
        "conditions": all_results,
    }

    agg_path = OUTPUT_BASE_DIR / "phase1_aggregated_results.json"
    with open(agg_path, "w", encoding="utf-8") as af:
        json.dump(aggregated_output, af, indent=2)
    print(f"\nSaved updated aggregated results to: {agg_path}")


if __name__ == "__main__":
    main()
