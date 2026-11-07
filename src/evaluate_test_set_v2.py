"""Independent Benchmark Evaluation Pipeline for Test Set v2.

Evaluates trained Phase 1 models (E1 Baseline, E2 Matched, E3 General across seeds 42, 123, 2026)
on the untouched Test Set v2 benchmark (N = 750).

Calculates:
- Standard multi-class metrics (Accuracy, Macro Precision/Recall/F1).
- Per-class metrics (F1, Precision, Recall, Support).
- Operational Emergency metrics:
  * Emergency Recall (Macro and Exact)
  * Emergency Precision (Macro)
  * Other Recall
  * Overall False Emergency Rate (FER_overall)
- Stratified False Emergency Rates:
  * FER_confusers (Targeted confusers: shouting, metal, door slams, alarms, fireworks, whistles)
  * FER_benign (Benign background: speech, music, animals, household, traffic, environmental)
- Subdomain-level error breakdown.
- Statistical significance tests:
  * 95% Wilson Score Confidence Intervals.
  * Paired McNemar's tests (E2 vs E3, E2 vs E1).
  * 10,000-sample bootstrap difference distributions and 95% CIs.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.class_config import CLASSES, CLASS_TO_ID, NUM_CLASSES
from src.dataset import AudioSetDataset, load_split_manifest
from src.train import compute_classification_metrics
from src.model import PannsCNN14

TEST_SET_V2_MANIFEST = ROOT / "data" / "processed" / "test_set_v2_manifest.csv"
PHASE1_MODELS_DIR = ROOT / "models" / "experiments" / "phase1_main"

CONDITIONS = {
    "e1_baseline": "E1 Baseline (422 train)",
    "e2_matched": "E2 Matched (634 train)",
    "e3_general": "E3 General (634 train)",
}

SEEDS = [42, 123, 2026]


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval with continuity correction for binomial proportions."""
    if total == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total

    # Continuity corrected Wilson score interval
    denominator = 1 + z**2 / total
    centre_adjusted_probability = p + z**2 / (2 * total)
    adjusted_limits = (
        z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    )

    lower = max(0.0, (centre_adjusted_probability - adjusted_limits) / denominator)
    upper = min(1.0, (centre_adjusted_probability + adjusted_limits) / denominator)
    return round(float(lower), 4), round(float(upper), 4)


def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict[str, Any]:
    """Perform paired McNemar's test with continuity correction between two models."""
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    # Contingency table
    # b: model 1 correct, model 2 wrong
    # c: model 1 wrong, model 2 correct
    b = int(np.sum(correct1 & ~correct2))
    c = int(np.sum(~correct1 & correct2))
    a = int(np.sum(correct1 & correct2))
    d = int(np.sum(~correct1 & ~correct2))

    total_discordant = b + c
    if total_discordant == 0:
        chi2 = 0.0
        p_val = 1.0
    else:
        # Edwards continuity correction
        chi2 = float((abs(b - c) - 1.0)**2 / total_discordant) if abs(b - c) > 0 else 0.0
        p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))

    return {
        "contingency_table": {"both_correct": a, "m1_correct_m2_wrong": b, "m1_wrong_m2_correct": c, "both_wrong": d},
        "discordant_pairs": total_discordant,
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_val, 6),
        "statistically_significant": bool(p_val < 0.05),
    }


def mcnemar_false_emergency_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict[str, Any]:
    """Perform McNemar's test specifically on false emergency classifications for 'other' class."""
    other_idx = 4
    mask = (y_true == other_idx)

    # False emergency: predicted is in {0, 1, 2, 3}
    fe1 = (y_pred1[mask] != other_idx)
    fe2 = (y_pred2[mask] != other_idx)

    # b: fe1=False (correctly avoided FA), fe2=True (false alarmed)
    b = int(np.sum(~fe1 & fe2))
    # c: fe1=True (false alarmed), fe2=False (correctly avoided FA)
    c = int(np.sum(fe1 & ~fe2))
    a = int(np.sum(~fe1 & ~fe2))
    d = int(np.sum(fe1 & fe2))

    total_discordant = b + c
    if total_discordant == 0:
        chi2 = 0.0
        p_val = 1.0
    else:
        chi2 = float((abs(b - c) - 1.0)**2 / total_discordant) if abs(b - c) > 0 else 0.0
        p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))

    return {
        "contingency_table": {"both_quiet": a, "m1_quiet_m2_alarm": b, "m1_alarm_m2_quiet": c, "both_alarm": d},
        "discordant_pairs": total_discordant,
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_val, 6),
        "statistically_significant": bool(p_val < 0.05),
    }


def bootstrap_metric_difference(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    is_confuser: np.ndarray,
    is_benign: np.ndarray,
    n_bootstraps: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute bootstrap distribution and 95% CI of metric differences between Model 1 and Model 2."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    other_idx = 4

    delta_macro_f1 = []
    delta_fer_overall = []
    delta_fer_confusers = []
    delta_fer_benign = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        p1 = y_pred1[idx]
        p2 = y_pred2[idx]

        # Macro F1
        m1 = compute_classification_metrics(yt, p1, CLASSES)
        m2 = compute_classification_metrics(yt, p2, CLASSES)
        delta_macro_f1.append(m1["macro_f1"] - m2["macro_f1"])

        # FER overall (m2 - m1: positive means m1 has lower FER)
        other_mask = (yt == other_idx)
        if np.sum(other_mask) > 0:
            fer1 = np.sum((p1[other_mask] != other_idx)) / np.sum(other_mask)
            fer2 = np.sum((p2[other_mask] != other_idx)) / np.sum(other_mask)
            delta_fer_overall.append(fer2 - fer1)

        # FER confusers
        conf_mask = is_confuser[idx]
        if np.sum(conf_mask) > 0:
            fer1_c = np.sum((p1[conf_mask] != other_idx)) / np.sum(conf_mask)
            fer2_c = np.sum((p2[conf_mask] != other_idx)) / np.sum(conf_mask)
            delta_fer_confusers.append(fer2_c - fer1_c)

        # FER benign
        ben_mask = is_benign[idx]
        if np.sum(ben_mask) > 0:
            fer1_b = np.sum((p1[ben_mask] != other_idx)) / np.sum(ben_mask)
            fer2_b = np.sum((p2[ben_mask] != other_idx)) / np.sum(ben_mask)
            delta_fer_benign.append(fer2_b - fer1_b)

    def ci(arr):
        if len(arr) == 0:
            return [0.0, 0.0]
        return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]

    def safe_mean(arr):
        return round(float(np.mean(arr)), 4) if len(arr) > 0 else 0.0

    def safe_superior(arr):
        return round(float(np.mean(np.array(arr) > 0)), 4) if len(arr) > 0 else 0.0

    return {
        "delta_macro_f1": {
            "mean": safe_mean(delta_macro_f1),
            "ci_95": ci(delta_macro_f1),
            "p_superior": safe_superior(delta_macro_f1),
        },
        "delta_fer_overall_reduction": {
            "mean": safe_mean(delta_fer_overall),
            "ci_95": ci(delta_fer_overall),
            "p_superior": safe_superior(delta_fer_overall),
        },
        "delta_fer_confusers_reduction": {
            "mean": safe_mean(delta_fer_confusers),
            "ci_95": ci(delta_fer_confusers),
            "p_superior": safe_superior(delta_fer_confusers),
        },
        "delta_fer_benign_reduction": {
            "mean": safe_mean(delta_fer_benign),
            "ci_95": ci(delta_fer_benign),
            "p_superior": safe_superior(delta_fer_benign),
        },
    }


def evaluate_model_on_test_set_v2(ckpt_path: Path, manifest_path: Path, device: torch.device):
    """Evaluate a single checkpoint on Test Set v2 and extract predictions."""
    manifest_df = pd.read_csv(manifest_path)

    model = PannsCNN14(num_classes=NUM_CLASSES, freeze_backbone=True)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()

    samples = load_split_manifest(manifest_path, root_dir=ROOT)
    dataset = AudioSetDataset(manifest_path.parent, samples=samples, return_raw=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    rows = []
    y_true = []
    y_pred = []
    sample_idx = 0

    with torch.no_grad():
        for batch_audio, batch_labels, _ in loader:
            batch_audio = batch_audio.to(device)
            logits = model(batch_audio)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            for b in range(len(batch_labels)):
                fpath, cid, cname = samples[sample_idx]
                orig_row = manifest_df.iloc[sample_idx]
                sample_idx += 1

                t_id = int(batch_labels[b].item())
                p_id = int(preds[b])
                y_true.append(t_id)
                y_pred.append(p_id)

                row_dict = {
                    "sample_index": sample_idx,
                    "video_id": orig_row.get("video_id", ""),
                    "file_path": str(fpath.relative_to(ROOT)).replace("\\", "/"),
                    "target_class": cname,
                    "class_id": cid,
                    "subdomain": orig_row.get("subdomain", ""),
                    "matched_label": orig_row.get("matched_label", ""),
                    "true_id": t_id,
                    "true_class": CLASSES[t_id],
                    "pred_id": p_id,
                    "pred_class": CLASSES[p_id],
                    "is_correct": bool(t_id == p_id),
                    "is_false_emergency": bool(t_id == 4 and p_id != 4),
                    "is_missed_emergency": bool(t_id != 4 and p_id == 4),
                }
                for c_i, c_n in enumerate(CLASSES):
                    row_dict[f"prob_{c_n}"] = float(probs[b, c_i])

                rows.append(row_dict)

    pred_df = pd.DataFrame(rows)
    return pred_df, np.array(y_true), np.array(y_pred)


def compute_test_set_v2_metrics(pred_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute complete metric battery for Test Set v2."""
    base_metrics = compute_classification_metrics(y_true, y_pred, CLASSES)
    cm = np.array(base_metrics["confusion_matrix"])

    emerg_indices = [0, 1, 2, 3]
    other_idx = 4

    # Emergency metrics
    emerg_recalls = [base_metrics["per_class"][CLASSES[i]]["recall"] for i in emerg_indices]
    emerg_precisions = [base_metrics["per_class"][CLASSES[i]]["precision"] for i in emerg_indices]

    total_emerg_support = sum(base_metrics["per_class"][CLASSES[i]]["support"] for i in emerg_indices)
    exact_emerg_correct = sum(cm[i, i] for i in emerg_indices)
    emerg_recall_exact = float(exact_emerg_correct / total_emerg_support) if total_emerg_support > 0 else 0.0

    detected_emerg = sum(cm[i, j] for i in emerg_indices for j in emerg_indices)
    emerg_detection_rate = float(detected_emerg / total_emerg_support) if total_emerg_support > 0 else 0.0

    emerg_recall_ci = wilson_score_interval(exact_emerg_correct, total_emerg_support)

    # Other metrics
    other_support = base_metrics["per_class"]["other"]["support"]
    other_correct = cm[other_idx, other_idx]
    other_recall = float(other_correct / other_support) if other_support > 0 else 0.0
    other_pred_emerg = int(sum(cm[other_idx, j] for j in emerg_indices))
    false_emergency_rate = float(other_pred_emerg / other_support) if other_support > 0 else 0.0
    fer_ci = wilson_score_interval(other_pred_emerg, other_support)

    # Stratified FER on 'other'
    confuser_df = pred_df[pred_df["subdomain"].str.startswith("confuser")]
    benign_df = pred_df[pred_df["subdomain"].str.startswith("benign")]

    confuser_total = len(confuser_df)
    confuser_fe = int(confuser_df["is_false_emergency"].sum())
    fer_confusers = float(confuser_fe / confuser_total) if confuser_total > 0 else 0.0
    fer_confusers_ci = wilson_score_interval(confuser_fe, confuser_total)

    benign_total = len(benign_df)
    benign_fe = int(benign_df["is_false_emergency"].sum())
    fer_benign = float(benign_fe / benign_total) if benign_total > 0 else 0.0
    fer_benign_ci = wilson_score_interval(benign_fe, benign_total)

    # Subdomain breakdowns
    subdomain_metrics = {}
    for sub, grp in pred_df.groupby("subdomain"):
        sub_n = len(grp)
        sub_correct = int(grp["is_correct"].sum())
        sub_fe = int(grp["is_false_emergency"].sum())
        subdomain_metrics[sub] = {
            "support": sub_n,
            "correct": sub_correct,
            "accuracy": round(sub_correct / sub_n, 4) if sub_n > 0 else 0.0,
            "false_emergencies": sub_fe,
            "false_emergency_rate": round(sub_fe / sub_n, 4) if sub_n > 0 else 0.0,
        }

    return {
        "accuracy": base_metrics["accuracy"],
        "macro_precision": base_metrics["macro_precision"],
        "macro_recall": base_metrics["macro_recall"],
        "macro_f1": base_metrics["macro_f1"],
        "emergency_recall_macro": round(float(np.mean(emerg_recalls)), 4),
        "emergency_precision_macro": round(float(np.mean(emerg_precisions)), 4),
        "emergency_recall_exact": round(emerg_recall_exact, 4),
        "emergency_recall_ci_95": emerg_recall_ci,
        "emergency_detection_rate": round(emerg_detection_rate, 4),
        "other_recall": round(other_recall, 4),
        "false_emergency_rate_overall": round(false_emergency_rate, 4),
        "fer_overall_ci_95": fer_ci,
        "other_predicted_as_emerg": other_pred_emerg,
        "other_support": other_support,
        "fer_confusers": round(fer_confusers, 4),
        "fer_confusers_ci_95": fer_confusers_ci,
        "fer_confusers_count": f"{confuser_fe}/{confuser_total}",
        "fer_benign": round(fer_benign, 4),
        "fer_benign_ci_95": fer_benign_ci,
        "fer_benign_count": f"{benign_fe}/{benign_total}",
        "per_class": base_metrics["per_class"],
        "confusion_matrix": base_metrics["confusion_matrix"],
        "subdomain_metrics": subdomain_metrics,
    }


def run_full_test_set_v2_evaluation():
    print("=" * 80)
    print("  TEST SET V2 INDEPENDENT BENCHMARK EVALUATION (N = 750)")
    print("=" * 80)

    if not TEST_SET_V2_MANIFEST.exists():
        raise FileNotFoundError(f"Test Set v2 manifest not found: {TEST_SET_V2_MANIFEST}")

    manifest_df = pd.read_csv(TEST_SET_V2_MANIFEST)
    print(f"Loaded manifest: {TEST_SET_V2_MANIFEST.name} ({len(manifest_df)} samples)")
    print("Class distribution:")
    for c, cnt in manifest_df["target_class"].value_counts().items():
        print(f"  {c:<18}: {cnt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}\n")

    all_results = {}
    condition_preds = {}

    for cond_key, cond_name in CONDITIONS.items():
        all_results[cond_key] = {"display_name": cond_name, "runs": {}}
        condition_preds[cond_key] = {}

        for seed in SEEDS:
            run_name = f"{cond_key}_seed{seed}"
            ckpt_path = PHASE1_MODELS_DIR / run_name / "best_model.pt"

            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

            print(f"Evaluating {run_name}...")
            pred_df, y_true, y_pred = evaluate_model_on_test_set_v2(ckpt_path, TEST_SET_V2_MANIFEST, device)

            # Save per-sample predictions
            pred_csv_path = PHASE1_MODELS_DIR / run_name / "test_set_v2_predictions.csv"
            pred_df.to_csv(pred_csv_path, index=False)

            # Compute metrics
            metrics = compute_test_set_v2_metrics(pred_df, y_true, y_pred)
            metrics["seed"] = seed
            metrics["condition"] = cond_key

            # Save run test metrics json
            run_metrics_path = PHASE1_MODELS_DIR / run_name / "test_set_v2_metrics.json"
            with open(run_metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            all_results[cond_key]["runs"][str(seed)] = metrics
            condition_preds[cond_key][seed] = {"df": pred_df, "y_true": y_true, "y_pred": y_pred}

            print(f"  Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Emerg Prec: {metrics['emergency_precision_macro']:.4f}")
            print(f"  FER Overall: {metrics['false_emergency_rate_overall']*100:.2f}% | FER Confusers: {metrics['fer_confusers']*100:.2f}% ({metrics['fer_confusers_count']}) | FER Benign: {metrics['fer_benign']*100:.2f}% ({metrics['fer_benign_count']})")

    # Aggregated Summary Stats
    print("\n" + "=" * 80)
    print("  AGGREGATED RESULTS ACROSS SEEDS (MEAN +/- STD)")
    print("=" * 80)

    summary_stats = {}
    primary_metrics = [
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "emergency_recall_macro", "emergency_precision_macro",
        "other_recall", "false_emergency_rate_overall",
        "fer_confusers", "fer_benign"
    ]

    for cond_key, cond_data in all_results.items():
        summary_stats[cond_key] = {"display_name": cond_data["display_name"], "metrics": {}}
        runs = cond_data["runs"]

        for m in primary_metrics:
            vals = [runs[str(s)][m] for s in SEEDS]
            summary_stats[cond_key]["metrics"][m] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "runs": vals,
            }

        s_m = summary_stats[cond_key]["metrics"]
        print(f"\nCondition: {cond_data['display_name']}")
        print(f"  Accuracy:      {s_m['accuracy']['mean']:.4f} +/- {s_m['accuracy']['std']:.4f}")
        print(f"  Macro F1:      {s_m['macro_f1']['mean']:.4f} +/- {s_m['macro_f1']['std']:.4f}")
        print(f"  Emerg Prec:    {s_m['emergency_precision_macro']['mean']:.4f} +/- {s_m['emergency_precision_macro']['std']:.4f}")
        print(f"  FER Overall:   {s_m['false_emergency_rate_overall']['mean']*100:.2f}% +/- {s_m['false_emergency_rate_overall']['std']*100:.2f}%")
        print(f"  FER Confusers: {s_m['fer_confusers']['mean']*100:.2f}% +/- {s_m['fer_confusers']['std']*100:.2f}%")
        print(f"  FER Benign:    {s_m['fer_benign']['mean']*100:.2f}% +/- {s_m['fer_benign']['std']*100:.2f}%")

    # Statistical Significance Testing (Seed 42 primary pair, and ensemble pair)
    print("\n" + "=" * 80)
    print("  STATISTICAL SIGNIFICANCE TESTING (E2 MATCHED vs E3 GENERAL)")
    print("=" * 80)

    # Use seed 42 as representative run for paired hypothesis tests
    ref_seed = 42
    y_true_ref = condition_preds["e2_matched"][ref_seed]["y_true"]
    y_pred_e2 = condition_preds["e2_matched"][ref_seed]["y_pred"]
    y_pred_e3 = condition_preds["e3_general"][ref_seed]["y_pred"]
    y_pred_e1 = condition_preds["e1_baseline"][ref_seed]["y_pred"]

    pred_df_ref = condition_preds["e2_matched"][ref_seed]["df"]
    is_confuser = pred_df_ref["subdomain"].str.startswith("confuser").values
    is_benign = pred_df_ref["subdomain"].str.startswith("benign").values

    # McNemar tests
    mcnemar_e2_vs_e3 = mcnemar_test(y_true_ref, y_pred_e2, y_pred_e3)
    mcnemar_e2_vs_e1 = mcnemar_test(y_true_ref, y_pred_e2, y_pred_e1)

    fe_mcnemar_e2_vs_e3 = mcnemar_false_emergency_test(y_true_ref, y_pred_e2, y_pred_e3)
    fe_mcnemar_e2_vs_e1 = mcnemar_false_emergency_test(y_true_ref, y_pred_e2, y_pred_e1)

    # Bootstrap differences (10,000 resamples)
    bootstrap_e2_vs_e3 = bootstrap_metric_difference(
        y_true_ref, y_pred_e2, y_pred_e3, is_confuser, is_benign, n_bootstraps=10000, seed=42
    )

    stat_comparisons = {
        "mcnemar_overall_e2_vs_e3": mcnemar_e2_vs_e3,
        "mcnemar_overall_e2_vs_e1": mcnemar_e2_vs_e1,
        "mcnemar_false_emergency_e2_vs_e3": fe_mcnemar_e2_vs_e3,
        "mcnemar_false_emergency_e2_vs_e1": fe_mcnemar_e2_vs_e1,
        "bootstrap_differences_e2_minus_e3": bootstrap_e2_vs_e3,
    }

    print("McNemar's Test (Overall Classification, E2 vs E3):")
    print(f"  Contingency: {mcnemar_e2_vs_e3['contingency_table']}")
    print(f"  Chi2 = {mcnemar_e2_vs_e3['chi2_statistic']}, p = {mcnemar_e2_vs_e3['p_value']} (Sig: {mcnemar_e2_vs_e3['statistically_significant']})")

    print("\nMcNemar's Test (False Emergency Avoidance, E2 vs E3):")
    print(f"  Contingency: {fe_mcnemar_e2_vs_e3['contingency_table']}")
    print(f"  Chi2 = {fe_mcnemar_e2_vs_e3['chi2_statistic']}, p = {fe_mcnemar_e2_vs_e3['p_value']} (Sig: {fe_mcnemar_e2_vs_e3['statistically_significant']})")

    print("\nBootstrap Difference Distributions (10,000 resamples, E2 vs E3):")
    print(f"  Delta Macro F1:              {bootstrap_e2_vs_e3['delta_macro_f1']['mean']:+.4f} (95% CI: {bootstrap_e2_vs_e3['delta_macro_f1']['ci_95']})")
    print(f"  Delta FER Overall Reduction: {bootstrap_e2_vs_e3['delta_fer_overall_reduction']['mean']:+.4f} (95% CI: {bootstrap_e2_vs_e3['delta_fer_overall_reduction']['ci_95']})")
    print(f"  Delta FER Confuser Reduction:{bootstrap_e2_vs_e3['delta_fer_confusers_reduction']['mean']:+.4f} (95% CI: {bootstrap_e2_vs_e3['delta_fer_confusers_reduction']['ci_95']})")
    print(f"  Delta FER Benign Reduction:  {bootstrap_e2_vs_e3['delta_fer_benign_reduction']['mean']:+.4f} (95% CI: {bootstrap_e2_vs_e3['delta_fer_benign_reduction']['ci_95']})")

    # Save master benchmark evaluation JSON
    master_results = {
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_manifest": str(TEST_SET_V2_MANIFEST.name),
        "total_benchmark_samples": len(manifest_df),
        "conditions": all_results,
        "summary_statistics": summary_stats,
        "statistical_hypothesis_tests": stat_comparisons,
    }

    out_json_path = PHASE1_MODELS_DIR / "test_set_v2_evaluation_summary.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    print(f"\nSaved master benchmark evaluation summary to: {out_json_path}")
    return master_results


if __name__ == "__main__":
    run_full_test_set_v2_evaluation()
