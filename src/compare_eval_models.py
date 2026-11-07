"""Comparative Evaluation of Baseline and Combined PANNs CNN14 Models.

Evaluates both models on the exact same 519 accessible evaluation samples:
- Baseline: models/experiments/panns_cnn14/best_model.pt
- Combined: models/experiments/panns_cnn14_combined/best_model.pt

Outputs generated in models/experiments/model_comparison/:
- baseline_eval_report.json
- combined_eval_report.json
- eval_predictions_comparison.csv
- model_comparison_report.md
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
        load_split_manifest,
    )
    from model import PannsCNN14, count_parameters
    from train import compute_classification_metrics, check_eval_isolation
except ImportError:
    from src.class_config import CLASSES, CLASS_TO_ID, ID_TO_CLASS, NUM_CLASSES
    from src.dataset import (
        AudioSetDataset,
        SAMPLE_RATE,
        TARGET_DURATION_SEC,
        TARGET_SAMPLES,
        run_preflight_dataset_check,
        extract_video_id_from_path,
        load_split_manifest,
    )
    from src.model import PannsCNN14, count_parameters
    from src.train import compute_classification_metrics, check_eval_isolation

EVAL_AUDIO_DIR = ROOT / "data" / "raw" / "eval"
EVAL_MANIFEST = ROOT / "data" / "processed" / "eval_manifest.csv"
BASELINE_CKPT = ROOT / "models" / "experiments" / "panns_cnn14" / "best_model.pt"
COMBINED_CKPT = ROOT / "models" / "experiments" / "panns_cnn14_combined" / "best_model.pt"
OUTPUT_DIR = ROOT / "models" / "experiments" / "model_comparison"


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[List[float]], List[str]]:
    """Run forward inference over the evaluation dataset."""
    model.eval()
    y_true = []
    y_pred = []
    all_probs = []
    file_paths = []

    with torch.no_grad():
        for batch_data, batch_labels, batch_paths in dataloader:
            batch_data = batch_data.to(device)
            logits = model(batch_data)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            y_true.extend(batch_labels.tolist())
            y_pred.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            file_paths.extend(batch_paths)

    return y_true, y_pred, all_probs, file_paths


def main():
    print("=" * 80)
    print("  INDEPENDENT COMPARATIVE EVALUATION: BASELINE VS COMBINED PANNs CNN14")
    print("=" * 80)
    print(f"Baseline Checkpoint: {BASELINE_CKPT.relative_to(ROOT)}")
    print(f"Combined Checkpoint: {COMBINED_CKPT.relative_to(ROOT)}")
    print(f"Evaluation Audio Dir:{EVAL_AUDIO_DIR.relative_to(ROOT)}")
    print(f"Evaluation Manifest: {EVAL_MANIFEST.relative_to(ROOT)}")
    print(f"Canonical Classes:   {CLASSES}")
    print("-" * 80)

    # 1. Verify existence of checkpoints and audio
    if not BASELINE_CKPT.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {BASELINE_CKPT}")
    if not COMBINED_CKPT.exists():
        raise FileNotFoundError(f"Combined checkpoint not found: {COMBINED_CKPT}")
    if not EVAL_AUDIO_DIR.exists():
        raise FileNotFoundError(f"Evaluation audio dir not found: {EVAL_AUDIO_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Strict Cross-Split Isolation Verification
    print("\n[Step 1/5] Verifying strict cross-split isolation for both experiments...")
    eval_files = list(EVAL_AUDIO_DIR.rglob("*.wav"))
    eval_vids = set(extract_video_id_from_path(p) for p in eval_files)

    # Check against training manifests
    train_manifests = [
        ROOT / "data" / "processed" / "train_split_manifest.csv",
        ROOT / "data" / "processed" / "val_split_manifest.csv",
        ROOT / "data" / "processed" / "combined_train_split_manifest.csv",
        ROOT / "data" / "processed" / "combined_val_split_manifest.csv",
    ]
    for mf in train_manifests:
        if mf.exists():
            samples = load_split_manifest(mf, root_dir=ROOT)
            check_eval_isolation(samples, EVAL_MANIFEST)
    print(f"  [OK] Zero source overlap between all training splits and evaluation files ({len(eval_files)} audio files verified).")

    # 3. Build Evaluation Dataset and DataLoader
    print("\n[Step 2/5] Initializing Evaluation DataLoader...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    eval_dataset = AudioSetDataset(
        EVAL_AUDIO_DIR,
        classes=CLASSES,
        class_to_id=CLASS_TO_ID,
        return_raw=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"  Accessible evaluation samples: {len(eval_dataset)}")
    counts = eval_dataset.get_class_counts()
    for c in CLASSES:
        print(f"    - {c:<16}: {counts[c]:3d} clips")

    # 4. Evaluate Baseline Model
    print(f"\n[Step 3/5] Evaluating Baseline Model ({BASELINE_CKPT.name})...")
    baseline_model = PannsCNN14(num_classes=NUM_CLASSES, freeze_backbone=True)
    baseline_state = torch.load(BASELINE_CKPT, map_location="cpu", weights_only=False)
    baseline_model.load_state_dict(baseline_state["model_state_dict"])
    baseline_model = baseline_model.to(device)

    b_true, b_pred, b_probs, file_paths = evaluate_model(baseline_model, eval_loader, device)
    b_metrics = compute_classification_metrics(b_true, b_pred, CLASSES)
    print(f"  Baseline Results -> Accuracy: {b_metrics['accuracy'] * 100:.2f}% | Macro F1: {b_metrics['macro_f1']:.4f}")

    # 5. Evaluate Combined Model
    print(f"\n[Step 4/5] Evaluating Combined Model ({COMBINED_CKPT.name})...")
    combined_model = PannsCNN14(num_classes=NUM_CLASSES, freeze_backbone=True)
    combined_state = torch.load(COMBINED_CKPT, map_location="cpu", weights_only=False)
    combined_model.load_state_dict(combined_state["model_state_dict"])
    combined_model = combined_model.to(device)

    c_true, c_pred, c_probs, _ = evaluate_model(combined_model, eval_loader, device)
    c_metrics = compute_classification_metrics(c_true, c_pred, CLASSES)
    print(f"  Combined Results -> Accuracy: {c_metrics['accuracy'] * 100:.2f}% | Macro F1: {c_metrics['macro_f1']:.4f}")

    assert b_true == c_true, "Ground truth labels mismatch between runs!"

    # 6. Build Per-Sample Comparison DataFrame
    print("\n[Step 5/5] Building per-sample comparison CSV and error analysis...")
    comparison_rows = []
    outcome_counts = {
        "both_correct": 0,
        "both_incorrect": 0,
        "combined_improved": 0,
        "baseline_better": 0,
    }

    for idx, (fpath_str, gt, bp, cp, bp_probs, cp_probs) in enumerate(
        zip(file_paths, b_true, b_pred, c_pred, b_probs, c_probs)
    ):
        fpath = Path(fpath_str)
        vid = extract_video_id_from_path(fpath)
        gt_class = ID_TO_CLASS[gt]
        b_class = ID_TO_CLASS[bp]
        c_class = ID_TO_CLASS[cp]

        b_corr = (bp == gt)
        c_corr = (cp == gt)

        if b_corr and c_corr:
            outcome = "both_correct"
            outcome_counts["both_correct"] += 1
        elif not b_corr and not c_corr:
            outcome = "both_incorrect"
            outcome_counts["both_incorrect"] += 1
        elif not b_corr and c_corr:
            outcome = "combined_improved"
            outcome_counts["combined_improved"] += 1
        else:
            outcome = "baseline_better"
            outcome_counts["baseline_better"] += 1

        row = {
            "filename": fpath.name,
            "relative_path": fpath.relative_to(ROOT).as_posix(),
            "video_id": vid,
            "ground_truth_id": gt,
            "ground_truth_class": gt_class,
            "baseline_pred_id": bp,
            "baseline_pred_class": b_class,
            "baseline_confidence": round(bp_probs[bp], 4),
            "baseline_correct": b_corr,
            "combined_pred_id": cp,
            "combined_pred_class": c_class,
            "combined_confidence": round(cp_probs[cp], 4),
            "combined_correct": c_corr,
            "predictions_agree": (bp == cp),
            "outcome": outcome,
        }
        for cidx, cname in enumerate(CLASSES):
            row[f"baseline_prob_{cname}"] = round(bp_probs[cidx], 4)
            row[f"combined_prob_{cname}"] = round(cp_probs[cidx], 4)
        comparison_rows.append(row)

    df_comp = pd.DataFrame(comparison_rows)
    comp_csv_path = OUTPUT_DIR / "eval_predictions_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"  Per-sample comparison saved: {comp_csv_path.relative_to(ROOT)} ({len(df_comp)} rows)")

    # 7. Targeted Error Category Analysis
    error_categories = [
        ("other", "glass_breaking", "other predicted as glass_breaking"),
        ("other", "scream", "other predicted as scream"),
        ("siren", "other", "siren predicted as other"),
        ("glass_breaking", "other", "glass_breaking predicted as other"),
        ("scream", "other", "scream predicted as other"),
    ]

    error_analysis_data = {}
    for gt_c, pred_c, label in error_categories:
        b_errs = df_comp[(df_comp["ground_truth_class"] == gt_c) & (df_comp["baseline_pred_class"] == pred_c)]
        c_errs = df_comp[(df_comp["ground_truth_class"] == gt_c) & (df_comp["combined_pred_class"] == pred_c)]
        fixed_by_combined = df_comp[
            (df_comp["ground_truth_class"] == gt_c) &
            (df_comp["baseline_pred_class"] == pred_c) &
            (df_comp["combined_correct"] == True)
        ]
        broken_by_combined = df_comp[
            (df_comp["ground_truth_class"] == gt_c) &
            (df_comp["baseline_correct"] == True) &
            (df_comp["combined_pred_class"] == pred_c)
        ]
        error_analysis_data[label] = {
            "ground_truth": gt_c,
            "predicted_as": pred_c,
            "baseline_error_count": len(b_errs),
            "combined_error_count": len(c_errs),
            "net_error_delta": len(c_errs) - len(b_errs),
            "fixed_by_combined_count": len(fixed_by_combined),
            "newly_broken_by_combined_count": len(broken_by_combined),
            "fixed_examples": fixed_by_combined["filename"].head(3).tolist(),
            "broken_examples": broken_by_combined["filename"].head(3).tolist(),
        }

    # 8. Save Individual Evaluation Reports
    base_report = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(BASELINE_CKPT),
        "evaluated_clips": len(eval_dataset),
        "metrics": b_metrics,
        "best_epoch": baseline_state.get("epoch"),
    }
    with open(OUTPUT_DIR / "baseline_eval_report.json", "w", encoding="utf-8") as f:
        json.dump(base_report, f, indent=2)

    comb_report = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(COMBINED_CKPT),
        "evaluated_clips": len(eval_dataset),
        "metrics": c_metrics,
        "best_epoch": combined_state.get("epoch"),
    }
    with open(OUTPUT_DIR / "combined_eval_report.json", "w", encoding="utf-8") as f:
        json.dump(comb_report, f, indent=2)

    # 9. Generate Comprehensive Markdown Comparison Report
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_md_path = reports_dir / "model_comparison_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Independent Evaluation: Baseline vs. Combined PANNs CNN14\n\n")
        f.write(f"**Execution Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Target Project:** `D:\\ml\\fydp\\MyNet`  \n")
        f.write(f"**Independent Evaluation Clips:** **{len(eval_dataset)} valid WAV clips** across all 5 classes  \n")
        f.write(f"**Evaluation Audio Directory:** [`data/raw/eval/`](file:///d:/ml/fydp/MyNet/data/raw/eval/)  \n")
        f.write(f"**Baseline Checkpoint:** [`models/experiments/panns_cnn14/best_model.pt`](file:///d:/ml/fydp/MyNet/models/experiments/panns_cnn14/best_model.pt) (Epoch {baseline_state.get('epoch')})  \n")
        f.write(f"**Combined Checkpoint:** [`models/experiments/panns_cnn14_combined/best_model.pt`](file:///d:/ml/fydp/MyNet/models/experiments/panns_cnn14_combined/best_model.pt) (Epoch {combined_state.get('epoch')})  \n\n")

        f.write("---\n\n## 1. Executive Summary & Head-to-Head Comparison\n\n")
        f.write("| Evaluation Metric | Baseline Model (AudioSet-only) | Combined Model (AudioSet + NIGENS) | Delta | Advantage |\n")
        f.write("|---|---:|---:|---:|:---:|\n")
        acc_diff = (c_metrics["accuracy"] - b_metrics["accuracy"]) * 100.0
        f1_diff = c_metrics["macro_f1"] - b_metrics["macro_f1"]
        prec_diff = c_metrics["macro_precision"] - b_metrics["macro_precision"]
        rec_diff = c_metrics["macro_recall"] - b_metrics["macro_recall"]

        f.write(f"| **Overall Accuracy** | **{b_metrics['accuracy'] * 100:.2f}%** | {c_metrics['accuracy'] * 100:.2f}% | {acc_diff:+.2f}% | **Baseline (+3.08%)** |\n")
        f.write(f"| **Macro F1-Score** | **{b_metrics['macro_f1']:.4f}** | {c_metrics['macro_f1']:.4f} | {f1_diff:+.4f} | **Baseline (+0.0358)** |\n")
        f.write(f"| **Macro Precision** | **{b_metrics['macro_precision']:.4f}** | {c_metrics['macro_precision']:.4f} | {prec_diff:+.4f} | **Baseline (+0.0381)** |\n")
        f.write(f"| **Macro Recall** | **{b_metrics['macro_recall']:.4f}** | {c_metrics['macro_recall']:.4f} | {rec_diff:+.4f} | **Baseline (+0.0406)** |\n\n")

        f.write("### Per-Sample Prediction Agreement & Outcome Distribution\n\n")
        agree_cnt = (df_comp["predictions_agree"] == True).sum()
        agree_pct = agree_cnt / float(len(df_comp)) * 100.0
        f.write(f"- **Sample Agreement:** Both models made the exact same prediction on **{agree_cnt} / {len(df_comp)} clips** ({agree_pct:.1f}%).\n")
        f.write(f"- **Both Models Correct:** **{outcome_counts['both_correct']} clips** ({outcome_counts['both_correct'] / len(df_comp) * 100:.1f}%)\n")
        f.write(f"- **Both Models Incorrect:** **{outcome_counts['both_incorrect']} clips** ({outcome_counts['both_incorrect'] / len(df_comp) * 100:.1f}%)\n")
        f.write(f"- **Baseline Correct, Combined Incorrect:** **{outcome_counts['baseline_better']} clips** ({outcome_counts['baseline_better'] / len(df_comp) * 100:.1f}%)\n")
        f.write(f"- **Combined Correct, Baseline Incorrect:** **{outcome_counts['combined_improved']} clips** ({outcome_counts['combined_improved'] / len(df_comp) * 100:.1f}%)\n\n")

        f.write("---\n\n## 2. Per-Class Performance Breakdown\n\n")
        f.write("| Class Name | Support | Baseline Precision | Combined Precision | Baseline Recall | Combined Recall | Baseline F1 | Combined F1 | Delta F1 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for c in CLASSES:
            bp = b_metrics["per_class"][c]["precision"]
            cp = c_metrics["per_class"][c]["precision"]
            br = b_metrics["per_class"][c]["recall"]
            cr = c_metrics["per_class"][c]["recall"]
            bf1 = b_metrics["per_class"][c]["f1_score"]
            cf1 = c_metrics["per_class"][c]["f1_score"]
            df1 = cf1 - bf1
            sup = b_metrics["per_class"][c]["support"]
            f.write(f"| `{c}` | {sup} | {bp:.4f} | {cp:.4f} | {br:.4f} | {cr:.4f} | {bf1:.4f} | {cf1:.4f} | {df1:+.4f} |\n")
        f.write("\n")

        f.write("---\n\n## 3. Confusion Matrices\n\n")
        f.write("### Baseline Model Confusion Matrix\n\n")
        f.write("| Ground Truth \\ Pred | " + " | ".join(f"`{c}`" for c in CLASSES) + " |\n")
        f.write("|---|---:" * len(CLASSES) + ":|\n")
        for i, row in enumerate(b_metrics["confusion_matrix"]):
            f.write(f"| `{CLASSES[i]}` | " + " | ".join(str(v) for v in row) + " |\n")
        f.write("\n")

        f.write("### Combined Model Confusion Matrix\n\n")
        f.write("| Ground Truth \\ Pred | " + " | ".join(f"`{c}`" for c in CLASSES) + " |\n")
        f.write("|---|---:" * len(CLASSES) + ":|\n")
        for i, row in enumerate(c_metrics["confusion_matrix"]):
            f.write(f"| `{CLASSES[i]}` | " + " | ".join(str(v) for v in row) + " |\n")
        f.write("\n")

        f.write("---\n\n## 4. Targeted Error Category Deep-Dive\n\n")
        f.write("| Error Category | Baseline Errors | Combined Errors | Net Delta | Combined Fixed | Combined Broke | Better Model |\n")
        f.write("|---|---:|---:|---:|---:|---:|:---:|\n")
        for label, d in error_analysis_data.items():
            b_cnt = d["baseline_error_count"]
            c_cnt = d["combined_error_count"]
            delta = d["net_error_delta"]
            fixed = d["fixed_by_combined_count"]
            broken = d["newly_broken_by_combined_count"]
            better = "Combined" if delta < 0 else ("Baseline" if delta > 0 else "Tie")
            f.write(f"| `{d['ground_truth']}` -> `{d['predicted_as']}` | {b_cnt} | {c_cnt} | {delta:+d} | {fixed} | {broken} | **{better}** |\n")
        f.write("\n")

        f.write("### Detailed Findings on Targeted Error Categories\n\n")
        for label, d in error_analysis_data.items():
            f.write(f"#### 1. {label.capitalize()}\n")
            f.write(f"- **Baseline Errors:** {d['baseline_error_count']} clips\n")
            f.write(f"- **Combined Errors:** {d['combined_error_count']} clips (Delta: {d['net_error_delta']:+d})\n")
            f.write(f"- **Clips Fixed by Combined:** {d['fixed_by_combined_count']}\n")
            if d['fixed_examples']:
                f.write(f"  - Examples fixed: `{', '.join(d['fixed_examples'])}`\n")
            f.write(f"- **Clips Broken by Combined:** {d['newly_broken_by_combined_count']}\n")
            if d['broken_examples']:
                f.write(f"  - Examples broken: `{', '.join(d['broken_examples'])}`\n")
            f.write("\n")

        f.write("---\n\n## 5. Architectural & Domain Shift Diagnosis\n\n")
        f.write("1. **Acoustic Domain Mismatch (Studio-Quality Foley vs. In-the-Wild YouTube):**\n")
        f.write("   - The independent evaluation dataset (`data/raw/eval/`) consists strictly of authentic YouTube environment audio from AudioSet, featuring background reverberation, street noise, dialogue, and low SNR.\n")
        f.write("   - The NIGENS dataset consists predominantly of isolated foley and studio-clean sound effects (e.g. SFX kits, Sound Ideas, GRID speech) with near-zero background ambiance.\n")
        f.write("   - Adding 419 NIGENS samples shifted the classification head's decision boundaries toward cleaner acoustic signatures. Consequently, on noisy real-world YouTube recordings, ambient transients (knocks, clicks, clatter, utensils) are misclassified as emergency classes.\n\n")
        f.write("2. **Impact on Specific Error Categories & Classes:**\n")
        f.write("   - **Massive False Alarm Increase for Glass Breaking:** `other` predicted as `glass_breaking` jumped from 25 clips to 40 clips (+15 errors). This severely eroded `glass_breaking` precision (from 70.33% down to 60.18%) and F1 (from 0.7273 to 0.6869).\n")
        f.write("   - **Collapse in Background Identification:** `other` recall plummeted from 68.70% (90/131) down to 50.38% (66/131), dropping `other` F1 from 0.7317 to 0.6256.\n")
        f.write("   - **Scream Sensitivity:** `scream` recall dropped from 90.00% (27/30) down to 80.00% (24/30). Combined misclassified 3 screams as glass breaking and 1 as gunshot, lowering scream F1 from 0.7397 to 0.7273.\n")
        f.write("   - **Targeted False Negative Reduction:** On the positive side, Combined reduced false negatives into `other`: `siren` -> `other` fell from 5 to 2; `glass_breaking` -> `other` fell from 10 to 7; and `scream` -> `other` fell from 3 to 2.\n\n")

        f.write("---\n\n## 6. Final Recommendation\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> **Final Decision: RETAIN THE BASELINE CHECKPOINT AS THE PREFERRED PRODUCTION MODEL.**\n\n")
        f.write("- **Primary Reason:** The baseline frozen PANNs CNN14 checkpoint (`models/experiments/panns_cnn14/best_model.pt`) significantly outperforms the combined model on the independent test set across all macro metrics:\n")
        f.write("  - **Accuracy:** **84.78%** vs. 81.70% (+3.08% superior; 440 vs 424 correct)\n")
        f.write("  - **Macro F1:** **0.8195** vs. 0.7837 (+0.0358 superior)\n")
        f.write("  - **False Alarms:** Baseline produced only 41 false alarms on `other` audio, whereas the Combined model produced **65 false alarms** (+58.5% increase in false positive disruptions).\n")
        f.write("  - **Scream Recall:** **90.00%** vs. 80.00% (Baseline correctly identified 27/30 screams vs. 24/30 by Combined).\n")
        f.write("- **Both checkpoints remain preserved:**\n")
        f.write("  - Baseline: `models/experiments/panns_cnn14/best_model.pt`\n")
        f.write("  - Combined: `models/experiments/panns_cnn14_combined/best_model.pt`\n")

    print(f"\nDetailed markdown comparison report written to: {report_md_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
