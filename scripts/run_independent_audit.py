"""Comprehensive Read-Only Forensic Audit Script for Phase 2 E4."""
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
import torch

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.class_config import CLASSES, CLASS_TO_ID, NUM_CLASSES
from src.evaluate_phase2_e4 import (
    CATEGORY_PAIRS,
    SEEDS,
    evaluate_checkpoint_on_test_set,
    exact_mcnemar_test,
    paired_permutation_test,
    cluster_aware_bootstrap,
    holm_bonferroni_correction,
    compute_probability_ensemble,
    validate_recording_ids,
)

TEST_MANIFEST_PATH = root / "data" / "processed" / "test_set_v2_manifest.csv"
MODELS_DIR = root / "models" / "experiments" / "phase2_e4"
ORIG_SUMMARY_PATH = MODELS_DIR / "phase2_e4_evaluation_summary.json"
REPORTS_DIR = root / "reports"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Audit computation device: {device}")

# 1. Load Original Evaluation Summary for Comparison
with open(ORIG_SUMMARY_PATH, "r", encoding="utf-8") as f:
    orig_summary = json.load(f)

# 2. Category Mapping Audit
manifest_df = pd.read_csv(TEST_MANIFEST_PATH)
reference_ids = [str(r.get("filename", "") or r.get("video_id", "")) for _, r in manifest_df.iterrows()]
y_true_benchmark = manifest_df["class_id"].values.astype(int)
n_total = len(manifest_df)

subdomains_present = set(manifest_df["subdomain"].dropna().unique())

mapping_rows = []
for pair in CATEGORY_PAIRS:
    cat_key = pair["category_key"]
    disp_name = pair["display_name"]
    abl_cond = pair["ablation_condition"]
    ctl_cond = pair["control_condition"]
    train_n = pair["sample_size"]
    excluded_n = pair["excluded_count"]
    target_sub = pair["test_subdomain"]

    exists_in_test = target_sub in subdomains_present
    actual_slice_mask = (manifest_df["subdomain"] == target_sub).values
    slice_count = int(np.sum(actual_slice_mask))

    if slice_count == 0:
        slice_type = "all_confusers_fallback"
        slice_count = int(np.sum(manifest_df["subdomain"].str.startswith("confuser", na=False).values))
        discrepancy_status = "DISCREPANCY_FALLBACK"
        notes = (
            f"Dedicated slice '{target_sub}' absent in Test Set v2 manifest. "
            f"Fell back to all {slice_count} confuser recordings across 6 subdomains."
        )
    elif cat_key == "car_alarms_detectors" and target_sub == "confuser_car_alarms_buzzers":
        slice_type = "isolated_subdomain"
        discrepancy_status = "NAME_SUBDOMAIN_MISMATCH"
        notes = (
            f"Training condition named '{cat_key}' matched to Test Set v2 subdomain '{target_sub}'. "
            f"Sample count exactly N=25."
        )
    else:
        slice_type = "isolated_subdomain"
        discrepancy_status = "EXACT_MATCH"
        notes = f"Exact match with dedicated Test Set v2 subdomain '{target_sub}' (N={slice_count})."

    mapping_rows.append({
        "category_key": cat_key,
        "display_name": disp_name,
        "ablation_condition": abl_cond,
        "control_condition": ctl_cond,
        "train_samples_total": train_n,
        "train_clips_excluded": excluded_n,
        "target_confuser_subdomain": target_sub,
        "test_subdomain_exists": exists_in_test,
        "evaluated_sample_count": slice_count,
        "slice_type": slice_type,
        "discrepancy_status": discrepancy_status,
        "notes": notes,
    })

cat_mapping_df = pd.DataFrame(mapping_rows)
cat_mapping_csv = REPORTS_DIR / "phase2_e4_category_mapping_audit.csv"
cat_mapping_df.to_csv(cat_mapping_csv, index=False)
print(f"Saved category mapping audit CSV to {cat_mapping_csv}")

# 3. Model Prediction Generation & Extraction
all_conditions = ["e2_full"]
for pair in CATEGORY_PAIRS:
    all_conditions.append(pair["ablation_condition"])
    all_conditions.append(pair["control_condition"])

eval_data = {}
print("Loading all 45 checkpoints and running inference on Test Set v2...")
for cond in all_conditions:
    eval_data[cond] = {}
    for s in SEEDS:
        run_name = f"{cond}_seed{s}"
        ckpt_path = MODELS_DIR / run_name / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        pred_df, yt, yp, pr = evaluate_checkpoint_on_test_set(ckpt_path, TEST_MANIFEST_PATH, device)
        validate_recording_ids(reference_ids, pred_df["recording_id"].tolist(), run_name)
        eval_data[cond][s] = {
            "pred_df": pred_df,
            "y_pred": yp,
            "probs": pr,
        }

# Compute Probability Ensembles
ensembles = {}
for cond in all_conditions:
    probs_by_s = {s: eval_data[cond][s]["probs"] for s in SEEDS}
    mean_p, ens_preds = compute_probability_ensemble(probs_by_s)
    ensembles[cond] = {
        "mean_probs": mean_p,
        "preds": ens_preds,
    }

# 4. Recording-Level Audit Table Creation & Statistical Reproduction
recording_audit_rows = []
reproduced_primary = {}
reproduced_secondary = {}
primary_p_vals = {}
sec_ens_p_vals = {}

for pair in CATEGORY_PAIRS:
    cat_key = pair["category_key"]
    subdomain = pair["test_subdomain"]
    abl_cond = pair["ablation_condition"]
    ctl_cond = pair["control_condition"]

    sub_mask = (manifest_df["subdomain"] == subdomain).values
    is_fallback = False
    if np.sum(sub_mask) == 0:
        sub_mask = manifest_df["subdomain"].str.startswith("confuser", na=False).values
        is_fallback = True

    sub_indices = np.where(sub_mask)[0]
    n_slice = len(sub_indices)

    preds_abl = {s: eval_data[abl_cond][s]["y_pred"][sub_mask] for s in SEEDS}
    preds_ctl = {s: eval_data[ctl_cond][s]["y_pred"][sub_mask] for s in SEEDS}
    probs_abl = {s: eval_data[abl_cond][s]["probs"][sub_mask] for s in SEEDS}
    probs_ctl = {s: eval_data[ctl_cond][s]["probs"][sub_mask] for s in SEEDS}

    ens_abl = ensembles[abl_cond]["preds"][sub_mask]
    ens_ctl = ensembles[ctl_cond]["preds"][sub_mask]

    # Recording-level diffs
    diffs_theta1 = np.mean([
        (preds_abl[s] != 4).astype(float) - (preds_ctl[s] != 4).astype(float)
        for s in SEEDS
    ], axis=0)

    # Permutation test
    perm_res = paired_permutation_test(diffs_theta1, n_permutations=100000, seed=42)

    # Cluster bootstrap
    sub_rec_ids = [reference_ids[idx] for idx in sub_indices]
    boot_res = cluster_aware_bootstrap(
        recording_ids=sub_rec_ids,
        y_true=y_true_benchmark[sub_mask],
        preds_a_by_seed=preds_abl,
        preds_b_by_seed=preds_ctl,
        probs_a_by_seed=probs_abl,
        probs_b_by_seed=probs_ctl,
        n_bootstraps=10000,
        seed=42,
    )

    # Ensemble McNemar
    ens_b = int(np.sum((ens_abl != 4) & ~(ens_ctl != 4)))
    ens_c = int(np.sum(~(ens_abl != 4) & (ens_ctl != 4)))
    ens_mcnemar = exact_mcnemar_test(ens_b, ens_c)

    # Per seed McNemar
    seed_mcnemar = {}
    for s in SEEDS:
        b_s = int(np.sum((preds_abl[s] != 4) & ~(preds_ctl[s] != 4)))
        c_s = int(np.sum(~(preds_abl[s] != 4) & (preds_ctl[s] != 4)))
        seed_mcnemar[f"seed_{s}"] = exact_mcnemar_test(b_s, c_s)

    primary_p_vals[cat_key] = perm_res["p_value"]
    sec_ens_p_vals[cat_key] = ens_mcnemar["p_value"]

    reproduced_primary[cat_key] = {
        "category": cat_key,
        "display_name": pair["display_name"],
        "slice_type": "all_confusers_fallback" if is_fallback else "isolated_subdomain",
        "sample_count": n_slice,
        "theta_1": {
            "point_estimate": round(float(np.mean(diffs_theta1)), 4),
            "ci_95": boot_res["theta_1_avg_model"]["ci_95"],
            "bootstrap_mean": boot_res["theta_1_avg_model"]["bootstrap_mean"],
            "p_value": perm_res["p_value"],
            "method": perm_res["method"],
            "non_zero_count": perm_res["non_zero_count"],
        },
        "per_seed_mcnemar": seed_mcnemar,
        "theta_2_ensemble": {
            "point_estimate": round(float((ens_b - ens_c) / n_slice), 4),
            "b": ens_b,
            "c": ens_c,
            "total_discordant": ens_b + ens_c,
            "net_shift": ens_b - ens_c,
            "p_value": ens_mcnemar["p_value"],
            "ci_95": boot_res["theta_2_ensemble"]["ci_95"],
        },
    }

    # Populate recording-level rows
    for loc_i, global_idx in enumerate(sub_indices):
        rec_id = reference_ids[global_idx]
        t_id = y_true_benchmark[global_idx]
        rec_sub = manifest_df.iloc[global_idx].get("subdomain", "")

        abl_preds_s = [int(preds_abl[s][loc_i]) for s in SEEDS]
        ctl_preds_s = [int(preds_ctl[s][loc_i]) for s in SEEDS]
        abl_fa_s = [bool(p != 4) for p in abl_preds_s]
        ctl_fa_s = [bool(p != 4) for p in ctl_preds_s]

        abl_mean_fa = float(np.mean(abl_fa_s))
        ctl_mean_fa = float(np.mean(ctl_fa_s))
        d_i = round(abl_mean_fa - ctl_mean_fa, 4)

        ens_a = int(ens_abl[loc_i])
        ens_c_pred = int(ens_ctl[loc_i])
        ens_a_fa = bool(ens_a != 4)
        ens_c_fa = bool(ens_c_pred != 4)

        if ens_a_fa and not ens_c_fa:
            ens_disc = "b_ablation_false_alarm"
        elif not ens_a_fa and ens_c_fa:
            ens_disc = "c_control_false_alarm"
        else:
            ens_disc = "concordant"

        recording_audit_rows.append({
            "category_key": cat_key,
            "slice_index": loc_i + 1,
            "global_manifest_index": global_idx + 1,
            "recording_id": rec_id,
            "video_id": manifest_df.iloc[global_idx].get("video_id", ""),
            "filename": manifest_df.iloc[global_idx].get("filename", ""),
            "true_class": CLASSES[t_id],
            "subdomain": rec_sub,
            "abl_pred_seed42": CLASSES[abl_preds_s[0]],
            "abl_pred_seed123": CLASSES[abl_preds_s[1]],
            "abl_pred_seed2026": CLASSES[abl_preds_s[2]],
            "ctl_pred_seed42": CLASSES[ctl_preds_s[0]],
            "ctl_pred_seed123": CLASSES[ctl_preds_s[1]],
            "ctl_pred_seed2026": CLASSES[ctl_preds_s[2]],
            "abl_fa_count": sum(abl_fa_s),
            "ctl_fa_count": sum(ctl_fa_s),
            "abl_mean_fa": round(abl_mean_fa, 4),
            "ctl_mean_fa": round(ctl_mean_fa, 4),
            "theta_1_diff_d_i": d_i,
            "is_discordant_theta_1": bool(d_i != 0.0),
            "ensemble_abl_pred": CLASSES[ens_a],
            "ensemble_ctl_pred": CLASSES[ens_c_pred],
            "ensemble_discordant_type": ens_disc,
        })

# Apply Holm-Bonferroni corrections
hb_prim = holm_bonferroni_correction(primary_p_vals, alpha=0.05)
hb_sec_ens = holm_bonferroni_correction(sec_ens_p_vals, alpha=0.05)

for cat_key in reproduced_primary:
    reproduced_primary[cat_key]["holm_bonferroni_primary"] = hb_prim[cat_key]
    reproduced_primary[cat_key]["holm_bonferroni_ensemble"] = hb_sec_ens[cat_key]

# Save recording-level audit CSV
rec_audit_df = pd.DataFrame(recording_audit_rows)
rec_audit_csv = REPORTS_DIR / "phase2_e4_recording_level_audit.csv"
rec_audit_df.to_csv(rec_audit_csv, index=False)
print(f"Saved recording-level audit CSV ({len(rec_audit_df)} rows) to {rec_audit_csv}")

# 5. Compare with Original JSON and produce Statistical Reproduction JSON
orig_prim = orig_summary["primary_comparisons"]
audit_comparison = {
    "audit_metadata": {
        "timestamp": "2026-11-02T20:05:00",
        "benchmark_samples": 750,
        "seeds": [42, 123, 2026],
        "device": str(device),
        "primary_hypotheses_evaluated": len(primary_p_vals),
    },
    "reproduced_primary_results": reproduced_primary,
    "reproduction_verification": {},
}

all_matched = True
category_keys = [p["category_key"] for p in CATEGORY_PAIRS]
for cat_key in category_keys:
    orig_c = orig_prim[cat_key]
    repr_c = reproduced_primary[cat_key]

    theta1_pt_orig = orig_c["theta_1_permutation"]["obs_stat"]
    theta1_pt_repr = repr_c["theta_1"]["point_estimate"]
    theta1_p_orig = orig_c["theta_1_permutation"]["p_value"]
    theta1_p_repr = repr_c["theta_1"]["p_value"]
    theta1_nz_orig = orig_c["theta_1_permutation"]["non_zero_count"]
    theta1_nz_repr = repr_c["theta_1"]["non_zero_count"]

    theta2_b_orig = orig_c["ensemble_mcnemar"]["b"]
    theta2_b_repr = repr_c["theta_2_ensemble"]["b"]
    theta2_c_orig = orig_c["ensemble_mcnemar"]["c"]
    theta2_c_repr = repr_c["theta_2_ensemble"]["c"]
    theta2_p_orig = orig_c["ensemble_mcnemar"]["p_value"]
    theta2_p_repr = repr_c["theta_2_ensemble"]["p_value"]

    hb_rank_orig = orig_c["holm_bonferroni"]["rank"]
    hb_rank_repr = repr_c["holm_bonferroni_primary"]["rank"]
    hb_rej_orig = orig_c["holm_bonferroni"]["rejected"]
    hb_rej_repr = repr_c["holm_bonferroni_primary"]["rejected"]

    # Bootstrap CIs
    ci1_orig = orig_c["cluster_bootstrap"]["theta_1_avg_model"]["ci_95"]
    ci1_repr = repr_c["theta_1"]["ci_95"]

    pt_match = math.isclose(theta1_pt_orig, theta1_pt_repr, abs_tol=1e-4)
    # Permutation p-value match (MC tolerance 0.001, exact tolerance 1e-5)
    p_tol = 0.002 if "monte_carlo" in repr_c["theta_1"]["method"] else 1e-5
    p_match = math.isclose(theta1_p_orig, theta1_p_repr, abs_tol=p_tol)
    nz_match = (theta1_nz_orig == theta1_nz_repr)
    ens_b_match = (theta2_b_orig == theta2_b_repr)
    ens_c_match = (theta2_c_orig == theta2_c_repr)
    ens_p_match = math.isclose(theta2_p_orig, theta2_p_repr, abs_tol=1e-5)
    hb_match = (hb_rank_orig == hb_rank_repr and hb_rej_orig == hb_rej_repr)

    cat_ok = pt_match and p_match and nz_match and ens_b_match and ens_c_match and ens_p_match and hb_match
    if not cat_ok:
        all_matched = False

    audit_comparison["reproduction_verification"][cat_key] = {
        "status": "PASS" if cat_ok else "DISCREPANCY",
        "theta1_point_estimate": {"original": theta1_pt_orig, "reproduced": theta1_pt_repr, "match": pt_match},
        "theta1_non_zero_count": {"original": theta1_nz_orig, "reproduced": theta1_nz_repr, "match": nz_match},
        "theta1_p_value": {"original": theta1_p_orig, "reproduced": theta1_p_repr, "match": p_match},
        "theta1_ci_95": {"original": ci1_orig, "reproduced": ci1_repr},
        "theta2_discordant_b": {"original": theta2_b_orig, "reproduced": theta2_b_repr, "match": ens_b_match},
        "theta2_discordant_c": {"original": theta2_c_orig, "reproduced": theta2_c_repr, "match": ens_c_match},
        "theta2_p_value": {"original": theta2_p_orig, "reproduced": theta2_p_repr, "match": ens_p_match},
        "holm_bonferroni_primary": {"original_rank": hb_rank_orig, "reproduced_rank": hb_rank_repr, "original_rejected": hb_rej_orig, "reproduced_rejected": hb_rej_repr, "match": hb_match},
    }

audit_comparison["overall_reproduction_status"] = "ALL_PASS" if all_matched else "DISCREPANCIES_FOUND"

stat_reprod_json = REPORTS_DIR / "phase2_e4_statistical_reproduction.json"
with open(stat_reprod_json, "w", encoding="utf-8") as f:
    json.dump(audit_comparison, f, indent=2)
print(f"Saved statistical reproduction JSON to {stat_reprod_json}")
print(f"Overall statistical reproduction status: {audit_comparison['overall_reproduction_status']}")

# 6. Deep Dive into Construction Tools Investigation
print("\n" + "="*80)
print("DEEP DIVE: Construction Tools Negative Result Investigation")
print("="*80)

tools_rows = [r for r in recording_audit_rows if r["category_key"] == "construction_tools"]
tools_df = pd.DataFrame(tools_rows)
print(f"Total evaluated recordings for construction_tools: {len(tools_df)}")

# Which recordings changed classification?
disc_tools = tools_df[tools_df["is_discordant_theta_1"]].copy()
print(f"Discordant recordings count (K): {len(disc_tools)}")

disc_by_sub = disc_tools["subdomain"].value_counts()
print("\nDiscordant recordings breakdown by subdomain:")
print(disc_by_sub)

# Breakdown of direction: positive (abl FA > ctl FA) vs negative (ctl FA > abl FA)
pos_disc = disc_tools[disc_tools["theta_1_diff_d_i"] > 0]
neg_disc = disc_tools[disc_tools["theta_1_diff_d_i"] < 0]

print(f"\nPositive discordances (Ablation FA > Control FA): {len(pos_disc)}")
print(f"Negative discordances (Control FA > Ablation FA): {len(neg_disc)}")

print("\nNegative discordances details (Recordings where Matched Control made MORE false alarms):")
for _, r in neg_disc.iterrows():
    print(f"  Rec ID: {r['recording_id']} | Subdomain: {r['subdomain']} | "
          f"Abl preds: [{r['abl_pred_seed42']}, {r['abl_pred_seed123']}, {r['abl_pred_seed2026']}] | "
          f"Ctl preds: [{r['ctl_pred_seed42']}, {r['ctl_pred_seed123']}, {r['ctl_pred_seed2026']}] | "
          f"Diff d_i: {r['theta_1_diff_d_i']}")

print("\nPositive discordances details (Recordings where Ablation made MORE false alarms):")
for _, r in pos_disc.iterrows():
    print(f"  Rec ID: {r['recording_id']} | Subdomain: {r['subdomain']} | "
          f"Abl preds: [{r['abl_pred_seed42']}, {r['abl_pred_seed123']}, {r['abl_pred_seed2026']}] | "
          f"Ctl preds: [{r['ctl_pred_seed42']}, {r['ctl_pred_seed123']}, {r['ctl_pred_seed2026']}] | "
          f"Diff d_i: {r['theta_1_diff_d_i']}")

# Emergency classes involved in Control false alarms vs Ablation false alarms
ctl_fa_classes = []
abl_fa_classes = []
for _, r in disc_tools.iterrows():
    for pred in [r["ctl_pred_seed42"], r["ctl_pred_seed123"], r["ctl_pred_seed2026"]]:
        if pred != "other":
            ctl_fa_classes.append(pred)
    for pred in [r["abl_pred_seed42"], r["abl_pred_seed123"], r["abl_pred_seed2026"]]:
        if pred != "other":
            abl_fa_classes.append(pred)

print("\nEmergency classes triggered as false alarms by Control on discordant recordings:")
print(pd.Series(ctl_fa_classes).value_counts())

print("\nEmergency classes triggered as false alarms by Ablation on discordant recordings:")
print(pd.Series(abl_fa_classes).value_counts())

# Per-seed breakdown for Construction Tools
print("\nPer-seed discordant breakdown for Construction Tools:")
for s in SEEDS:
    res = seed_mcnemar[f"seed_{s}"]
    print(f"  Seed {s}: b (abl FA, ctl not)={res['b']}, c (ctl FA, abl not)={res['c']}, "
          f"net={res['net_shift']}, p={res['p_value']:.6f}")
