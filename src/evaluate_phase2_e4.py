"""Phase 2 E4 Benchmark Evaluation Pipeline.

Implements the rigorous statistical framework approved in:
- reports/phase2_e4_statistical_design_correction.md
- reports/phase2_e4_final_execution_contract.md

Features:
1. Exact two-sided binomial McNemar tests via scipy.stats.binomtest.
2. Holm-Bonferroni multiple testing correction across category comparisons.
3. Cluster-aware recording-level bootstrap (10,000 iterations, seed 42) preserving 3-seed clusters.
4. Explicit and separate reporting for Estimand 1 (3-seed average model effect theta_1)
   and Estimand 2 (probability-ensemble effect theta_2).
5. Primary comparisons: Category Ablations vs Matched-Size Random Controls (identical sample sizes).
6. Secondary comparisons: Category Ablations vs E2 Full (total margin drop reference).
7. Strict recording-ID matching and validation across all models and test set entries.
8. Synthetic / mock mode (--synthetic) to validate calculations and reporting end-to-end without training.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.class_config import CLASSES, CLASS_TO_ID, NUM_CLASSES
from src.dataset import AudioSetDataset, load_split_manifest
from src.model import PannsCNN14
from src.train import compute_classification_metrics

TEST_SET_V2_MANIFEST = ROOT / "data" / "processed" / "test_set_v2_manifest.csv"
DEFAULT_MODELS_DIR = ROOT / "models" / "experiments" / "phase2_e4"
DEFAULT_REPORT_PATH = ROOT / "reports" / "phase2_e4_evaluation_report.md"
DEFAULT_SUMMARY_JSON = ROOT / "models" / "experiments" / "phase2_e4" / "phase2_e4_evaluation_summary.json"

SEEDS = [42, 123, 2026]

# 7 LOO Category Ablation pairs
CATEGORY_PAIRS = [
    {
        "category_key": "human_shouting",
        "display_name": "Human Shouting",
        "ablation_condition": "e4_minus_human_shouting",
        "control_condition": "e4_matched_control_human_shouting",
        "sample_size": 602,
        "excluded_count": 44,
        "test_subdomain": "confuser_human_shouting",
    },
    {
        "category_key": "metal_impacts",
        "display_name": "Metal Impacts",
        "ablation_condition": "e4_minus_metal_impacts",
        "control_condition": "e4_matched_control_metal_impacts",
        "sample_size": 606,
        "excluded_count": 40,
        "test_subdomain": "confuser_metal_impacts",
    },
    {
        "category_key": "door_slams",
        "display_name": "Door Slams",
        "ablation_condition": "e4_minus_door_slams",
        "control_condition": "e4_matched_control_door_slams",
        "sample_size": 613,
        "excluded_count": 33,
        "test_subdomain": "confuser_door_slams",
    },
    {
        "category_key": "construction_tools",
        "display_name": "Construction Tools",
        "ablation_condition": "e4_minus_construction_tools",
        "control_condition": "e4_matched_control_construction_tools",
        "sample_size": 616,
        "excluded_count": 30,
        "test_subdomain": "confuser_construction_tools",  # Note: Evaluated on confusers if dedicated slice absent
    },
    {
        "category_key": "car_alarms_detectors",
        "display_name": "Car Alarms & Detectors",
        "ablation_condition": "e4_minus_car_alarms_detectors",
        "control_condition": "e4_matched_control_car_alarms_detectors",
        "sample_size": 617,
        "excluded_count": 29,
        "test_subdomain": "confuser_car_alarms_buzzers",
    },
    {
        "category_key": "fireworks_firecrackers",
        "display_name": "Fireworks & Firecrackers",
        "ablation_condition": "e4_minus_fireworks_firecrackers",
        "control_condition": "e4_matched_control_fireworks_firecrackers",
        "sample_size": 618,
        "excluded_count": 28,
        "test_subdomain": "confuser_fireworks",
    },
    {
        "category_key": "whistles_high_pitch",
        "display_name": "Whistles & High Pitch",
        "ablation_condition": "e4_minus_whistles_high_pitch",
        "control_condition": "e4_matched_control_whistles_high_pitch",
        "sample_size": 627,
        "excluded_count": 19,
        "test_subdomain": "confuser_whistles",
    },
]


# ==============================================================================
# 1. Exact Binomial McNemar Test
# ==============================================================================

def exact_mcnemar_test(b: int, c: int) -> Dict[str, Any]:
    """Perform exact two-sided binomial McNemar test on discordant pairs.

    Parameters:
        b: Number of samples where Condition A made a false alarm and Condition B did not.
           (A = Emergency, B = Other)
        c: Number of samples where Condition B made a false alarm and Condition A did not.
           (A = Other, B = Emergency)

    Returns:
        Dictionary with b, c, total discordant, net shift (b - c), exact p-value,
        and unadjusted significance indicator at alpha = 0.05.
    """
    total_discordant = int(b + c)
    if total_discordant == 0:
        p_val = 1.0
    else:
        # Exact two-sided binomial test under H0: p = 0.5
        res = stats.binomtest(b, total_discordant, p=0.5, alternative="two-sided")
        p_val = float(res.pvalue)

    return {
        "b": int(b),
        "c": int(c),
        "total_discordant": total_discordant,
        "net_shift": int(b - c),
        "p_value": round(p_val, 6),
        "statistically_significant_unadjusted": bool(p_val < 0.05),
    }


# ==============================================================================
# 2. Paired Sign-Flip Permutation Test for Fractional / Continuous Differences
# ==============================================================================

def paired_permutation_test(
    diffs: np.ndarray,
    n_permutations: int = 100000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Perform exact or Monte Carlo paired sign-flip permutation test for theta_1.

    Mathematical Foundations:
    1. Estimand & Null Hypothesis:
       The primary estimand theta_1 is the expected three-seed average model error difference
       on recording i:
           d_i = (1/3) * sum_{s in {42, 123, 2026}} [ I(y_i,s^{Abl} in Emerg) - I(y_i,s^{Ctl} in Emerg) ]
       The null hypothesis is:
           H_0: E[d_i] = 0  (no category effect on single-model false alarm rate)
       against the two-sided alternative:
           H_1: E[d_i] != 0

    2. Exchangeability & Symmetry Assumption:
       Under H_0, the condition labels (Ablation vs. Control) are exchangeable within each physical
       recording i. Swapping the condition labels flips the sign of d_i with probability 0.5:
           d_i* = sigma_i * d_i,  where sigma_i in {-1, +1} i.i.d. with P(sigma_i = +1) = 0.5.
       This assumes the distribution of d_i is symmetric about 0 under H_0.

    3. Cluster Structure Retention:
       Because d_i is formed by averaging all three seed predictions for recording i prior to sign
       flipping, the sign flip sigma_i simultaneously flips all three seed predictions together.
       This strictly preserves within-recording multi-seed dependencies.

    4. Test Statistic & Two-Sided p-value:
       The test statistic is the observed sample mean difference:
           T_obs = (1/N) * sum_{i=1}^N d_i
       The two-sided rejection region includes any permutation where |T*| >= |T_obs|.
       - For K <= 16 non-zero recordings: exact exhaustive enumeration across all 2^K sign combinations.
       - For K > 16 non-zero recordings: Monte Carlo permutation testing with R random sign vectors:
           p = (extreme_count + 1) / (R + 1)

    Parameters:
        diffs: 1D array of paired differences d_i across recordings (length N).
        n_permutations: Number of Monte Carlo sign-flip permutations (default 100,000).
        seed: Random seed for Monte Carlo permutations (default 42).

    Returns:
        Dictionary with observed mean, p_value, method, and non-zero count.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0:
        return {
            "obs_stat": 0.0,
            "p_value": 1.0,
            "method": "trivial_null",
            "non_zero_count": 0,
            "statistically_significant_unadjusted": False,
        }

    non_zero = diffs[diffs != 0.0]
    k = len(non_zero)
    obs_stat = float(np.mean(diffs))

    if k == 0 or obs_stat == 0.0:
        return {
            "obs_stat": round(obs_stat, 4),
            "p_value": 1.0,
            "method": "trivial_null",
            "non_zero_count": int(k),
            "statistically_significant_unadjusted": False,
        }

    # If k <= 16, compute exact exhaustive enumeration 2^k
    if k <= 16:
        num_comb = 1 << k
        bits = ((np.arange(num_comb)[:, None] >> np.arange(k)) & 1).astype(float)
        signs = 2.0 * bits - 1.0
        perm_means = np.dot(signs, non_zero) / n
        p_val = float(np.mean(np.abs(perm_means) >= np.abs(obs_stat) - 1e-12))
        method_str = f"exact_enumeration_2^{k}"
    else:
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_permutations, k))
        perm_means = np.dot(signs, non_zero) / n
        p_val = float((1.0 + np.sum(np.abs(perm_means) >= np.abs(obs_stat) - 1e-12)) / (1.0 + n_permutations))
        method_str = f"monte_carlo_{n_permutations}"

    return {
        "obs_stat": round(obs_stat, 4),
        "p_value": round(p_val, 6),
        "method": method_str,
        "non_zero_count": int(k),
        "statistically_significant_unadjusted": bool(p_val < 0.05),
    }


# ==============================================================================
# 3. Holm-Bonferroni Multiple Comparisons Correction
# ==============================================================================

def holm_bonferroni_correction(
    p_values_dict: Dict[str, float],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Apply Holm-Bonferroni step-down correction across multiple category tests.

    Parameters:
        p_values_dict: Mapping of category key to unadjusted p-value.
        alpha: Target family-wise error rate (default 0.05).

    Returns:
        Mapping of category key to correction details:
        - raw_p_value
        - rank (1 = lowest p-value)
        - alpha_threshold: alpha / (m - rank + 1)
        - adjusted_p_value: step-down adjusted p-value enforced non-decreasing
        - rejected: bool indicating rejection of H0
    """
    m = len(p_values_dict)
    if m == 0:
        return {}

    # Sort hypotheses by p-value ascending
    sorted_items = sorted(p_values_dict.items(), key=lambda kv: kv[1])

    results: Dict[str, Dict[str, Any]] = {}
    running_max_p_adj = 0.0
    stop_rejections = False

    for rank_idx, (cat, p_raw) in enumerate(sorted_items, start=1):
        threshold = alpha / (m - rank_idx + 1)
        nominal_p_adj = min(1.0, (m - rank_idx + 1) * p_raw)
        running_max_p_adj = max(running_max_p_adj, nominal_p_adj)
        p_adj = min(1.0, running_max_p_adj)

        if not stop_rejections and p_raw <= threshold:
            rejected = True
        else:
            rejected = False
            stop_rejections = True  # Step-down stops once a hypothesis fails to reject

        results[cat] = {
            "category": cat,
            "raw_p_value": round(float(p_raw), 6),
            "rank": rank_idx,
            "m": m,
            "alpha_threshold": round(float(threshold), 6),
            "adjusted_p_value": round(float(p_adj), 6),
            "rejected": bool(rejected),
        }

    return results


# ==============================================================================
# 3. Cluster-Aware (Recording-Level) Bootstrap
# ==============================================================================

def cluster_aware_bootstrap(
    recording_ids: List[str],
    y_true: np.ndarray,
    preds_a_by_seed: Dict[int, np.ndarray],
    preds_b_by_seed: Dict[int, np.ndarray],
    probs_a_by_seed: Optional[Dict[int, np.ndarray]] = None,
    probs_b_by_seed: Optional[Dict[int, np.ndarray]] = None,
    n_bootstraps: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute Cluster-Aware Bootstrap Confidence Intervals preserving 3-seed clusters.

    Unit of resampling: Physical recording ID (audio recording).
    Each sampled recording retains its predictions across all available seeds.

    Calculates:
    - Estimand 1 (theta_1): Three-seed average model effect.
    - Estimand 2 (theta_2): Probability-ensemble effect (if probabilities are provided).

    Parameters:
        recording_ids: List of unique recording identifiers of length N.
        y_true: Ground truth class IDs (length N).
        preds_a_by_seed: Dict mapping seed -> predicted class IDs (length N) for Model A.
        preds_b_by_seed: Dict mapping seed -> predicted class IDs (length N) for Model B.
        probs_a_by_seed: Optional dict mapping seed -> probability matrix (N, C) for Model A.
        probs_b_by_seed: Optional dict mapping seed -> probability matrix (N, C) for Model B.
        n_bootstraps: Number of bootstrap iterations (default 10,000).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Dictionary with theta_1 and theta_2 estimates, percentiles [2.5%, 97.5%],
        and p_superiority.
    """
    n = len(recording_ids)
    if n == 0:
        return {
            "theta_1_avg_model": {"point_estimate": 0.0, "ci_95": [0.0, 0.0], "p_superior": 0.0},
            "theta_2_ensemble": {"point_estimate": 0.0, "ci_95": [0.0, 0.0], "p_superior": 0.0},
        }

    seeds = sorted(preds_a_by_seed.keys())
    k_seeds = len(seeds)

    # In False Emergency Rate, "Other" is class 4; Emergency is classes 0, 1, 2, 3
    # Binary indicator: 1 if model predicted Emergency (false alarm on Other), 0 otherwise
    fa_a = np.zeros((n, k_seeds), dtype=float)
    fa_b = np.zeros((n, k_seeds), dtype=float)

    for s_idx, s in enumerate(seeds):
        fa_a[:, s_idx] = (preds_a_by_seed[s] != 4).astype(float)
        fa_b[:, s_idx] = (preds_b_by_seed[s] != 4).astype(float)

    # Per-recording average difference across seeds: Delta_i = mean_s(fa_a[i, s]) - mean_s(fa_b[i, s])
    delta_i = np.mean(fa_a, axis=1) - np.mean(fa_b, axis=1)
    theta_1_point = float(np.mean(delta_i))

    # Ensemble calculations if probabilities provided
    has_probs = probs_a_by_seed is not None and probs_b_by_seed is not None
    if has_probs:
        mean_p_a = np.mean([probs_a_by_seed[s] for s in seeds], axis=0)  # (N, C)
        mean_p_b = np.mean([probs_b_by_seed[s] for s in seeds], axis=0)
        ens_pred_a = np.argmax(mean_p_a, axis=1)
        ens_pred_b = np.argmax(mean_p_b, axis=1)
        delta_ens_i = ((ens_pred_a != 4).astype(float) - (ens_pred_b != 4).astype(float))
        theta_2_point = float(np.mean(delta_ens_i))
    else:
        theta_2_point = 0.0

    # Resample recording clusters with replacement
    rng = np.random.RandomState(seed)
    boot_theta_1 = np.empty(n_bootstraps, dtype=float)
    boot_theta_2 = np.empty(n_bootstraps, dtype=float) if has_probs else None

    for b_idx in range(n_bootstraps):
        resampled_idx = rng.randint(0, n, size=n)
        boot_theta_1[b_idx] = np.mean(delta_i[resampled_idx])
        if has_probs:
            boot_theta_2[b_idx] = np.mean(delta_ens_i[resampled_idx])

    ci_1 = [
        round(float(np.percentile(boot_theta_1, 2.5)), 4),
        round(float(np.percentile(boot_theta_1, 97.5)), 4),
    ]
    p_sup_1 = round(float(np.mean(boot_theta_1 > 0)), 4)

    res = {
        "theta_1_avg_model": {
            "point_estimate": round(theta_1_point, 4),
            "bootstrap_mean": round(float(np.mean(boot_theta_1)), 4),
            "ci_95": ci_1,
            "p_superior": p_sup_1,
            "description": "Estimand 1: Three-seed average model false emergency difference (Ablation - Control)",
        }
    }

    if has_probs:
        ci_2 = [
            round(float(np.percentile(boot_theta_2, 2.5)), 4),
            round(float(np.percentile(boot_theta_2, 97.5)), 4),
        ]
        p_sup_2 = round(float(np.mean(boot_theta_2 > 0)), 4)
        res["theta_2_ensemble"] = {
            "point_estimate": round(theta_2_point, 4),
            "bootstrap_mean": round(float(np.mean(boot_theta_2)), 4),
            "ci_95": ci_2,
            "p_superior": p_sup_2,
            "description": "Estimand 2: Probability-ensemble false emergency difference (Ablation - Control)",
        }

    return res


# ==============================================================================
# 4. Probability Ensemble Decision Functions
# ==============================================================================

def compute_probability_ensemble(
    probs_by_seed: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Average softmax probabilities across seeds and compute argmax predictions.

    Parameters:
        probs_by_seed: Dict mapping seed -> probability matrix (N, num_classes).

    Returns:
        Tuple of (mean_probabilities (N, num_classes), ensemble_predictions (N,)).
    """
    seeds = sorted(probs_by_seed.keys())
    if not seeds:
        raise ValueError("probs_by_seed cannot be empty")
    stacked = np.stack([probs_by_seed[s] for s in seeds], axis=0)  # (K, N, C)
    mean_probs = np.mean(stacked, axis=0)  # (N, C)
    ens_preds = np.argmax(mean_probs, axis=1)  # (N,)
    return mean_probs, ens_preds


# ==============================================================================
# 5. Strict Recording ID Validation
# ==============================================================================

def validate_recording_ids(
    reference_ids: List[str],
    candidate_ids: List[str],
    condition_name: str,
) -> None:
    """Validate that recording IDs match the reference benchmark manifest exactly.

    Raises:
        ValueError: If lengths differ or any ID mismatches.
    """
    if len(reference_ids) != len(candidate_ids):
        raise ValueError(
            f"Recording ID count mismatch for {condition_name}: "
            f"expected {len(reference_ids)}, found {len(candidate_ids)}"
        )
    for idx, (ref, cand) in enumerate(zip(reference_ids, candidate_ids)):
        if ref != cand:
            raise ValueError(
                f"Recording ID mismatch at index {idx} for {condition_name}: "
                f"expected '{ref}', found '{cand}'"
            )


# ==============================================================================
# 6. Checkpoint Evaluation on Test Set v2
# ==============================================================================

def evaluate_checkpoint_on_test_set(
    ckpt_path: Path,
    manifest_path: Path,
    device: torch.device,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load model checkpoint, evaluate on Test Set v2, and extract predictions.

    Returns:
        (pred_df, y_true, y_pred, probs)
    """
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
    y_true_list = []
    y_pred_list = []
    probs_list = []
    sample_idx = 0

    with torch.no_grad():
        for batch_audio, batch_labels, _ in loader:
            batch_audio = batch_audio.to(device)
            logits = model(batch_audio)
            batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
            batch_preds = np.argmax(batch_probs, axis=1)

            for b in range(len(batch_labels)):
                fpath, cid, cname = samples[sample_idx]
                orig_row = manifest_df.iloc[sample_idx]
                sample_idx += 1

                t_id = int(batch_labels[b].item())
                p_id = int(batch_preds[b])
                y_true_list.append(t_id)
                y_pred_list.append(p_id)
                probs_list.append(batch_probs[b])

                rec_id = orig_row.get("filename", "") or orig_row.get("video_id", "")
                row_dict = {
                    "sample_index": sample_idx,
                    "recording_id": rec_id,
                    "video_id": orig_row.get("video_id", ""),
                    "filename": orig_row.get("filename", ""),
                    "target_class": cname,
                    "class_id": cid,
                    "subdomain": orig_row.get("subdomain", ""),
                    "true_id": t_id,
                    "true_class": CLASSES[t_id],
                    "pred_id": p_id,
                    "pred_class": CLASSES[p_id],
                    "is_correct": bool(t_id == p_id),
                    "is_false_emergency": bool(t_id == 4 and p_id != 4),
                }
                for c_i, c_n in enumerate(CLASSES):
                    row_dict[f"prob_{c_n}"] = float(batch_probs[b, c_i])
                rows.append(row_dict)

    pred_df = pd.DataFrame(rows)
    return pred_df, np.array(y_true_list), np.array(y_pred_list), np.array(probs_list)


# ==============================================================================
# 7. Synthetic Predictions Generator for Testing Pipeline
# ==============================================================================

def generate_synthetic_predictions(
    manifest_df: pd.DataFrame,
    condition: str,
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Generate realistic deterministic synthetic predictions on Test Set v2.

    Emulates realistic emergency and non-emergency error rates:
    - Base accuracy ~88%
    - Category ablations experience controlled false alarm elevation on their target subdomains.
    - Deterministic via hashing (condition + seed).
    """
    n = len(manifest_df)
    y_true = manifest_df["class_id"].values.astype(int)
    y_pred = np.copy(y_true)
    probs = np.zeros((n, NUM_CLASSES), dtype=float)

    cond_hash = int(hashlib.md5(f"{condition}_{seed}".encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(cond_hash % 100000)

    rows = []
    for i in range(n):
        orig_row = manifest_df.iloc[i]
        t_id = y_true[i]
        subdomain = str(orig_row.get("subdomain", ""))
        rec_id = orig_row.get("filename", "") or orig_row.get("video_id", "")

        # Default high-confidence correct probabilities
        p_vec = rng.uniform(0.01, 0.05, size=NUM_CLASSES)
        p_vec[t_id] = rng.uniform(0.70, 0.90)

        # Baseline noise error rate ~10%
        if rng.uniform(0, 1) < 0.10:
            other_classes = [c for c in range(NUM_CLASSES) if c != t_id]
            err_class = rng.choice(other_classes)
            p_vec[t_id], p_vec[err_class] = p_vec[err_class], p_vec[t_id]

        # Emulate category-specific ablation effect on confusers
        # If model is an ablation of shouting, elevate scream (0) false alarms on shouting confusers
        is_target_subdomain = False
        target_fa_class = None

        if "human_shouting" in condition and "confuser_human_shouting" in subdomain:
            is_target_subdomain = True
            target_fa_class = 0  # scream
        elif "metal_impacts" in condition and "confuser_metal_impacts" in subdomain:
            is_target_subdomain = True
            target_fa_class = 1  # glass_breaking
        elif "door_slams" in condition and "confuser_door_slams" in subdomain:
            is_target_subdomain = True
            target_fa_class = 2  # gunshot
        elif "car_alarms" in condition and "confuser_car_alarms" in subdomain:
            is_target_subdomain = True
            target_fa_class = 3  # siren
        elif "fireworks" in condition and "confuser_fireworks" in subdomain:
            is_target_subdomain = True
            target_fa_class = 2  # gunshot
        elif "whistles" in condition and "confuser_whistles" in subdomain:
            is_target_subdomain = True
            target_fa_class = 3  # siren
        elif "construction_tools" in condition and subdomain.startswith("confuser"):
            is_target_subdomain = True
            target_fa_class = 3  # siren

        # Ablations suffer higher FA on their targets than controls or E2 full
        if is_target_subdomain and target_fa_class is not None:
            if condition.startswith("e4_minus_"):
                # Elevated FA probability: 40% chance of false alarm
                if rng.uniform(0, 1) < 0.40:
                    p_vec[4] = 0.15
                    p_vec[target_fa_class] = 0.75
            elif condition.startswith("e4_matched_control_"):
                # Matched control has much lower FA probability: 12%
                if rng.uniform(0, 1) < 0.12:
                    p_vec[4] = 0.20
                    p_vec[target_fa_class] = 0.70
            elif condition == "e2_full":
                # Unablated ceiling has lowest FA: 8%
                if rng.uniform(0, 1) < 0.08:
                    p_vec[4] = 0.20
                    p_vec[target_fa_class] = 0.65

        # Normalize softmax probabilities
        p_vec = p_vec / np.sum(p_vec)
        probs[i] = p_vec
        p_id = int(np.argmax(p_vec))
        y_pred[i] = p_id

        row_dict = {
            "sample_index": i + 1,
            "recording_id": rec_id,
            "video_id": orig_row.get("video_id", ""),
            "filename": orig_row.get("filename", ""),
            "target_class": CLASSES[t_id],
            "class_id": t_id,
            "subdomain": subdomain,
            "true_id": t_id,
            "true_class": CLASSES[t_id],
            "pred_id": p_id,
            "pred_class": CLASSES[p_id],
            "is_correct": bool(t_id == p_id),
            "is_false_emergency": bool(t_id == 4 and p_id != 4),
        }
        for c_i, c_n in enumerate(CLASSES):
            row_dict[f"prob_{c_n}"] = float(p_vec[c_i])
        rows.append(row_dict)

    pred_df = pd.DataFrame(rows)
    return pred_df, y_true, y_pred, probs


# ==============================================================================
# 8. Complete Evaluation Battery
# ==============================================================================

def run_phase2_evaluation(
    test_manifest_path: Path,
    models_dir: Path,
    report_path: Path,
    summary_json_path: Path,
    is_synthetic: bool = False,
    n_bootstraps: int = 10000,
    seed: int = 42,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Execute complete Phase 2 evaluation pipeline."""
    manifest_df = pd.read_csv(test_manifest_path)
    reference_ids = [
        str(row.get("filename", "") or row.get("video_id", ""))
        for _, row in manifest_df.iterrows()
    ]
    y_true_benchmark = manifest_df["class_id"].values.astype(int)
    n_total = len(manifest_df)

    device = torch.device(device_str)

    # Gather list of all 15 conditions
    all_conditions = ["e2_full"]
    for pair in CATEGORY_PAIRS:
        all_conditions.append(pair["ablation_condition"])
        all_conditions.append(pair["control_condition"])

    # Predictions storage: condition -> seed -> (pred_df, y_pred, probs)
    eval_data: Dict[str, Dict[int, Dict[str, Any]]] = {}

    print("=" * 80)
    print("PHASE 2 E4 EVALUATION PIPELINE: TEST SET v2 BENCHMARK")
    print(f"Benchmark Manifest: {test_manifest_path} (N = {n_total})")
    print(f"Mode: {'SYNTHETIC / MOCK' if is_synthetic else 'REAL MODEL WEIGHTS'}")
    print(f"Bootstrap Iterations: {n_bootstraps:,} (Seed = {seed})")
    print("=" * 80)

    for cond in all_conditions:
        eval_data[cond] = {}
        for s in SEEDS:
            run_key = f"{cond}_seed{s}"
            ckpt_path = models_dir / run_key / "best_model.pt"

            if is_synthetic:
                pred_df, yt, yp, pr = generate_synthetic_predictions(manifest_df, cond, s)
            else:
                if not ckpt_path.is_file():
                    raise FileNotFoundError(
                        f"Checkpoint not found for {run_key} at {ckpt_path}. "
                        f"Run Phase 2 training first or use --synthetic flag."
                    )
                pred_df, yt, yp, pr = evaluate_checkpoint_on_test_set(ckpt_path, test_manifest_path, device)

            # Strict recording ID validation
            cand_ids = pred_df["recording_id"].tolist()
            validate_recording_ids(reference_ids, cand_ids, run_key)

            # Compute standard metrics
            metrics = compute_classification_metrics(yt, yp, CLASSES)
            other_mask = (yt == 4)
            conf_mask = manifest_df["subdomain"].str.startswith("confuser", na=False).values
            ben_mask = manifest_df["subdomain"].str.startswith("benign", na=False).values

            fer_overall = float(np.sum((yp[other_mask] != 4)) / np.sum(other_mask)) if np.sum(other_mask) > 0 else 0.0
            fer_conf = float(np.sum((yp[conf_mask] != 4)) / np.sum(conf_mask)) if np.sum(conf_mask) > 0 else 0.0
            fer_ben = float(np.sum((yp[ben_mask] != 4)) / np.sum(ben_mask)) if np.sum(ben_mask) > 0 else 0.0

            eval_data[cond][s] = {
                "pred_df": pred_df,
                "y_pred": yp,
                "probs": pr,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "fer_overall": round(fer_overall, 4),
                "fer_confusers": round(fer_conf, 4),
                "fer_benign": round(fer_ben, 4),
            }

    # ==========================================================================
    # Compute Ensemble Decisions for All Conditions
    # ==========================================================================
    ensembles: Dict[str, Dict[str, Any]] = {}
    for cond in all_conditions:
        probs_by_s = {s: eval_data[cond][s]["probs"] for s in SEEDS}
        mean_p, ens_preds = compute_probability_ensemble(probs_by_s)
        other_mask = (y_true_benchmark == 4)
        conf_mask = manifest_df["subdomain"].str.startswith("confuser", na=False).values

        ens_fer_overall = float(np.sum(ens_preds[other_mask] != 4) / np.sum(other_mask))
        ens_fer_conf = float(np.sum(ens_preds[conf_mask] != 4) / np.sum(conf_mask))
        ens_metrics = compute_classification_metrics(y_true_benchmark, ens_preds, CLASSES)

        ensembles[cond] = {
            "mean_probs": mean_p,
            "preds": ens_preds,
            "macro_f1": ens_metrics["macro_f1"],
            "fer_overall": round(ens_fer_overall, 4),
            "fer_confusers": round(ens_fer_conf, 4),
        }

    # ==========================================================================
    # Primary Comparisons: Ablation vs Matched Control
    # ==========================================================================
    primary_results: Dict[str, Any] = {}
    primary_p_values: Dict[str, float] = {}

    for pair in CATEGORY_PAIRS:
        cat_key = pair["category_key"]
        subdomain = pair["test_subdomain"]
        abl_cond = pair["ablation_condition"]
        ctl_cond = pair["control_condition"]

        # Filter slice for this category
        sub_mask = (manifest_df["subdomain"] == subdomain).values
        is_fallback_slice = False

        if np.sum(sub_mask) == 0:
            # Fallback to all confusers if dedicated category slice not in benchmark
            sub_mask = manifest_df["subdomain"].str.startswith("confuser", na=False).values
            is_fallback_slice = True

        sub_rec_ids = [reference_ids[i] for i in range(n_total) if sub_mask[i]]
        n_slice = len(sub_rec_ids)
        yt_slice = y_true_benchmark[sub_mask]

        # Extract predictions for Ablation and Control on this slice
        preds_abl_by_s = {s: eval_data[abl_cond][s]["y_pred"][sub_mask] for s in SEEDS}
        preds_ctl_by_s = {s: eval_data[ctl_cond][s]["y_pred"][sub_mask] for s in SEEDS}
        probs_abl_by_s = {s: eval_data[abl_cond][s]["probs"][sub_mask] for s in SEEDS}
        probs_ctl_by_s = {s: eval_data[ctl_cond][s]["probs"][sub_mask] for s in SEEDS}

        # 1. Per-Seed Paired Exact McNemar (Descriptive Consistency)
        per_seed_mcnemar = {}
        for s in SEEDS:
            fa_abl = (preds_abl_by_s[s] != 4)
            fa_ctl = (preds_ctl_by_s[s] != 4)
            b = int(np.sum(fa_abl & ~fa_ctl))
            c = int(np.sum(~fa_abl & fa_ctl))
            per_seed_mcnemar[f"seed_{s}"] = exact_mcnemar_test(b, c)

        # 2. Primary Estimand (theta_1): Three-Seed Average Paired Decision Difference
        # Fractional per-recording difference: d_i = 1/3 * sum_s [ I(y_abl != 4) - I(y_ctl != 4) ]
        diffs_theta1 = np.mean([
            (preds_abl_by_s[s] != 4).astype(float) - (preds_ctl_by_s[s] != 4).astype(float)
            for s in SEEDS
        ], axis=0)
        theta1_perm = paired_permutation_test(diffs_theta1, n_permutations=100000, seed=seed)

        # 3. Secondary Estimand (theta_2): Probability Ensemble Paired Exact McNemar
        ens_pred_abl = ensembles[abl_cond]["preds"][sub_mask]
        ens_pred_ctl = ensembles[ctl_cond]["preds"][sub_mask]
        ens_fa_abl = (ens_pred_abl != 4)
        ens_fa_ctl = (ens_pred_ctl != 4)
        ens_b = int(np.sum(ens_fa_abl & ~ens_fa_ctl))
        ens_c = int(np.sum(~ens_fa_abl & ens_fa_ctl))
        ens_mcnemar = exact_mcnemar_test(ens_b, ens_c)

        # 4. Cluster-Aware Bootstrap for Estimand 1 and Estimand 2
        boot_res = cluster_aware_bootstrap(
            recording_ids=sub_rec_ids,
            y_true=yt_slice,
            preds_a_by_seed=preds_abl_by_s,
            preds_b_by_seed=preds_ctl_by_s,
            probs_a_by_seed=probs_abl_by_s,
            probs_b_by_seed=probs_ctl_by_s,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )

        primary_results[cat_key] = {
            "category": cat_key,
            "display_name": pair["display_name"],
            "ablation_condition": abl_cond,
            "control_condition": ctl_cond,
            "slice_type": "all_confusers_fallback" if is_fallback_slice else "isolated_subdomain",
            "test_subdomain": subdomain,
            "n_recordings": n_slice,
            "per_seed_mcnemar": per_seed_mcnemar,
            "theta_1_permutation": theta1_perm,
            "ensemble_mcnemar": ens_mcnemar,
            "cluster_bootstrap": boot_res,
        }
        primary_p_values[cat_key] = theta1_perm["p_value"]

    # 5. Primary Holm-Bonferroni Correction over Primary Estimand (theta_1 Permutation p-values)
    hb_results = holm_bonferroni_correction(primary_p_values, alpha=0.05)
    for cat_key in primary_results:
        primary_results[cat_key]["holm_bonferroni"] = hb_results[cat_key]

    # 6. Secondary Ensemble Holm-Bonferroni Correction (theta_2 Ensemble McNemar p-values)
    sec_ens_p_values = {cat_key: primary_results[cat_key]["ensemble_mcnemar"]["p_value"] for cat_key in primary_results}
    hb_ens_results = holm_bonferroni_correction(sec_ens_p_values, alpha=0.05)
    for cat_key in primary_results:
        primary_results[cat_key]["holm_bonferroni_ensemble"] = hb_ens_results[cat_key]

    # ==========================================================================
    # Secondary Comparisons: Ablation vs E2 Full (Reference Ceiling)
    # ==========================================================================
    secondary_results: Dict[str, Any] = {}
    e2_full_cond = "e2_full"

    for pair in CATEGORY_PAIRS:
        cat_key = pair["category_key"]
        subdomain = pair["test_subdomain"]
        abl_cond = pair["ablation_condition"]

        sub_mask = (manifest_df["subdomain"] == subdomain).values
        if np.sum(sub_mask) == 0:
            sub_mask = manifest_df["subdomain"].str.startswith("confuser", na=False).values

        sub_rec_ids = [reference_ids[i] for i in range(n_total) if sub_mask[i]]
        n_slice = len(sub_rec_ids)
        yt_slice = y_true_benchmark[sub_mask]

        preds_abl_by_s = {s: eval_data[abl_cond][s]["y_pred"][sub_mask] for s in SEEDS}
        preds_e2_by_s = {s: eval_data[e2_full_cond][s]["y_pred"][sub_mask] for s in SEEDS}
        probs_abl_by_s = {s: eval_data[abl_cond][s]["probs"][sub_mask] for s in SEEDS}
        probs_e2_by_s = {s: eval_data[e2_full_cond][s]["probs"][sub_mask] for s in SEEDS}

        # Permutation test on theta_1 vs E2 Full
        diffs_theta1_vs_full = np.mean([
            (preds_abl_by_s[s] != 4).astype(float) - (preds_e2_by_s[s] != 4).astype(float)
            for s in SEEDS
        ], axis=0)
        theta1_perm_vs_full = paired_permutation_test(diffs_theta1_vs_full, n_permutations=100000, seed=seed)

        # Ensemble McNemar vs E2 Full
        ens_pred_abl = ensembles[abl_cond]["preds"][sub_mask]
        ens_pred_e2 = ensembles[e2_full_cond]["preds"][sub_mask]
        b = int(np.sum((ens_pred_abl != 4) & ~(ens_pred_e2 != 4)))
        c = int(np.sum(~(ens_pred_abl != 4) & (ens_pred_e2 != 4)))
        ens_mcnemar = exact_mcnemar_test(b, c)

        # Cluster Bootstrap
        boot_res = cluster_aware_bootstrap(
            recording_ids=sub_rec_ids,
            y_true=yt_slice,
            preds_a_by_seed=preds_abl_by_s,
            preds_b_by_seed=preds_e2_by_s,
            probs_a_by_seed=probs_abl_by_s,
            probs_b_by_seed=probs_e2_by_s,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )

        secondary_results[cat_key] = {
            "category": cat_key,
            "display_name": pair["display_name"],
            "ablation_condition": abl_cond,
            "reference_condition": e2_full_cond,
            "n_recordings": n_slice,
            "theta_1_permutation": theta1_perm_vs_full,
            "ensemble_mcnemar": ens_mcnemar,
            "cluster_bootstrap": boot_res,
        }

    # Secondary Multiplicity Policy (Ablation vs E2 Full)
    sec_p_values = {
        cat_key: secondary_results[cat_key]["theta_1_permutation"]["p_value"]
        for cat_key in secondary_results
    }
    hb_sec_results = holm_bonferroni_correction(sec_p_values, alpha=0.05)
    for cat_key in secondary_results:
        secondary_results[cat_key]["holm_bonferroni"] = hb_sec_results[cat_key]

    # ==========================================================================
    # Global Condition Summary Table
    # ==========================================================================
    condition_summary: Dict[str, Any] = {}
    for cond in all_conditions:
        macro_f1s = [eval_data[cond][s]["macro_f1"] for s in SEEDS]
        fer_confs = [eval_data[cond][s]["fer_confusers"] for s in SEEDS]
        fer_overs = [eval_data[cond][s]["fer_overall"] for s in SEEDS]

        condition_summary[cond] = {
            "condition": cond,
            "macro_f1_mean": round(float(np.mean(macro_f1s)), 4),
            "macro_f1_std": round(float(np.std(macro_f1s)), 4),
            "fer_confusers_mean": round(float(np.mean(fer_confs)), 4),
            "fer_overall_mean": round(float(np.mean(fer_overs)), 4),
            "ensemble_macro_f1": ensembles[cond]["macro_f1"],
            "ensemble_fer_confusers": ensembles[cond]["fer_confusers"],
            "ensemble_fer_overall": ensembles[cond]["fer_overall"],
        }

    # Output assembly
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "benchmark_manifest": str(test_manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "benchmark_samples": n_total,
            "mode": "synthetic" if is_synthetic else "real_weights",
            "synthetic_validation_only": bool(is_synthetic),
            "status_label": "SYNTHETIC VALIDATION ONLY — NOT EXPERIMENTAL RESULTS" if is_synthetic else "CONFIRMATORY BENCHMARK EVALUATION",
            "bootstrap_iterations": n_bootstraps,
            "random_seed": seed,
            "seeds_evaluated": SEEDS,
            "num_conditions": len(all_conditions),
            "primary_family_hypotheses": len(primary_results),
            "secondary_family_hypotheses": len(secondary_results),
        },
        "condition_summary": condition_summary,
        "primary_comparisons": primary_results,
        "secondary_comparisons": secondary_results,
    }

    # Save JSON summary
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Summary JSON saved: {summary_json_path}")

    # Generate Markdown Report
    generate_markdown_report(output_data, report_path)
    print(f"Markdown report generated: {report_path}")

    return output_data


# ==============================================================================
# 9. Markdown Report Generation
# ==============================================================================

def generate_markdown_report(data: Dict[str, Any], report_path: Path) -> None:
    """Generate a publication-grade markdown report of Phase 2 E4 evaluation."""
    meta = data["metadata"]
    prim = data["primary_comparisons"]
    sec = data["secondary_comparisons"]
    cond_sum = data["condition_summary"]
    is_synth = meta.get("synthetic_validation_only", False)

    lines = []
    if is_synth:
        lines.extend([
            "# Phase 2 E4 Statistical Evaluation Report [SYNTHETIC VALIDATION ONLY]",
            "",
            "> [!WARNING]",
            "> ## SYNTHETIC VALIDATION ONLY — NOT EXPERIMENTAL RESULTS",
            "> **DO NOT USE FOR RESEARCH FINDINGS OR MASTER EVIDENCE AGGREGATION.**  ",
            "> This report was generated using synthetic mock predictions exclusively to audit the statistical pipeline, mathematical calculations, and reporting formatting prior to model training. Zero models were trained for these results. Synthetic results must never be included in the actual E4 research-results aggregation.",
            "",
        ])
    else:
        lines.extend([
            "# Phase 2 E4 Statistical Evaluation & Category Attribution Report",
            "",
        ])

    lines.extend([
        f"**Project:** `D:\\ml\\fydp\\MyNet`  ",
        "**Research Framing:** *Data-Centric Analysis of Hard-Negative Learning and Domain Shift in Emergency Audio Classification*  ",
        f"**Date:** {datetime.now().strftime('%B %d, %Y')}  ",
        f"**Document:** [`{report_path.name}`](file:///{str(report_path).replace('\\', '/')})  ",
        f"**Status:** `{meta.get('status_label', 'CONFIRMATORY BENCHMARK EVALUATION')}`  ",
        f"**Benchmark Set:** Test Set v2 (N = {meta['benchmark_samples']} pristine samples)  ",
        f"**Evaluation Mode:** `{meta['mode'].upper()}`  ",
        f"**Seeds Evaluated:** {meta['seeds_evaluated']}  ",
        f"**Bootstrap Resamples:** {meta['bootstrap_iterations']:,} (Recording-cluster aware, Seed = {meta['random_seed']})  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Epistemic Status",
        "",
        "1. **Epistemic Demarcation:** Findings reported herein represent **Exploratory Category Attribution Analysis** on Test Set v2. Test Set v2 was unblinded in Phase 1; these results attribute the relative semantic impact of individual confuser subcategories on that benchmark.",
        "2. **Strict Cluster-Level Statistical Integrity:** Resampling occurs strictly at the physical recording ID level ($N = 25$ recordings per confuser category), preserving within-cluster multi-seed prediction correlations and completely eliminating the pseudo-replication fallacy.",
        "3. **Dual Estimand Reporting:** The two mandated estimands are strictly separated into independent reporting columns:",
        "   - **Primary Estimand ($\\theta_1$):** Three-seed average model error difference (single-model deployment scenario).",
        "   - **Secondary Estimand ($\\theta_2$):** Probability-ensemble error difference (3-model ensemble deployment scenario).",
        "4. **Primary Hypothesis Testing:** Exactly 7 primary comparisons enter the primary Holm-Bonferroni step-down correction at family-wise $\\alpha = 0.05$ using exact/Monte Carlo paired sign-flip permutation tests on $\\theta_1$.",
        "5. **Secondary Multiplicity Policy:** Secondary comparisons against `E2 Full` and ensemble $\\theta_2$ tests are evaluated under their own isolated 7-hypothesis Holm-Bonferroni corrections and are never pooled with primary tests.",
        "",
        "---",
        "",
        "## 2. Primary Hypothesis Family (Estimand 1: $\\theta_1$ Three-Seed Average Model Effect)",
        "",
        "The primary comparison contrasts each category ablation ($N = 602..627$) against its matched-size random control ($N = 602..627$) using the **three-seed average model effect ($\\theta_1$)**. Because sample sizes are strictly identical, this contrast isolates the **pure semantic importance** of each acoustic confuser category for an end-user deploying a single model.",
        "",
        "| Rank | Category Name | Slice $N$ | Point Est $\\theta_1$ | 95% Cluster Bootstrap CI | Permutation Method | Permutation $p$-value | Primary Holm-Bonferroni $\\alpha$ | Significant under Primary FWER? |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    # Sort primary comparisons by Holm-Bonferroni rank on theta_1
    sorted_prim = sorted(prim.values(), key=lambda x: x["holm_bonferroni"]["rank"])
    for row in sorted_prim:
        hb = row["holm_bonferroni"]
        perm = row["theta_1_permutation"]
        p_val = f"{perm['p_value']:.6f}"
        alpha_t = f"{hb['alpha_threshold']:.6f}"
        sig_str = "**YES**" if hb["rejected"] else "NO"

        cb = row["cluster_bootstrap"]
        th1 = cb["theta_1_avg_model"]
        th1_pt = f"{th1['point_estimate']:+.4f}"
        th1_ci = f"[{th1['ci_95'][0]:+.4f}, {th1['ci_95'][1]:+.4f}]"

        rank = hb["rank"]
        dname = row["display_name"]
        n_rec = row["n_recordings"]
        method_str = perm.get("method", "permutation")
        lines.append(
            f"| **{rank}** | {dname} | {n_rec} | {th1_pt} | {th1_ci} | `{method_str}` | {p_val} | {alpha_t} | {sig_str} |"
        )

    lines.extend([
        "",
        "> [!NOTE]",
        "> $\\theta_1$ represents the expected difference in false emergency rate for a single deployed model trained with arbitrary random seed. Inference is performed via recording-level paired permutation testing and cluster-aware bootstrap. Positive values indicate that removing the targeted category increased false alarms (i.e. targeted hard negatives were beneficial).",
        "",
        "---",
        "",
        "## 3. Secondary Hypothesis Family (Estimand 2: $\\theta_2$ Three-Seed Probability Ensemble Effect)",
        "",
        "The secondary comparison evaluates the performance shift if an end-user deploys the full **3-model probability ensemble**. Tested using the exact two-sided binomial McNemar test on the ensemble decisions, with an independent 7-hypothesis Holm-Bonferroni correction.",
        "",
        "| Rank | Category Name | Slice $N$ | Point Est $\\theta_2$ | 95% Cluster Bootstrap CI | Ens Discordant $(b : c)$ | Exact Binomial $p$-value | Ensemble Holm-Bonferroni $\\alpha$ | Significant under Ensemble FWER? |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    sorted_ens = sorted(prim.values(), key=lambda x: x["holm_bonferroni_ensemble"]["rank"] if "holm_bonferroni_ensemble" in x else 999)
    for row in sorted_ens:
        hb_ens = row.get("holm_bonferroni_ensemble", {})
        ens_mc = row["ensemble_mcnemar"]
        b_c = f"{ens_mc['b']} : {ens_mc['c']}"
        p_val = f"{ens_mc['p_value']:.6f}"
        alpha_t = f"{hb_ens.get('alpha_threshold', 0.05):.6f}"
        sig_ens = "**YES**" if hb_ens.get("rejected", False) else "NO"

        cb = row["cluster_bootstrap"]
        th2 = cb.get("theta_2_ensemble", {})
        th2_pt = f"{th2.get('point_estimate', 0.0):+.4f}"
        th2_ci = f"[{th2.get('ci_95', [0.0, 0.0])[0]:+.4f}, {th2.get('ci_95', [0.0, 0.0])[1]:+.4f}]"

        rank = hb_ens.get("rank", "-")
        dname = row["display_name"]
        n_rec = row["n_recordings"]
        lines.append(
            f"| **{rank}** | {dname} | {n_rec} | {th2_pt} | {th2_ci} | `{b_c}` | {p_val} | {alpha_t} | {sig_ens} |"
        )

    lines.extend([
        "",
        "> [!NOTE]",
        "> $\\theta_2$ measures the performance shift for a 3-seed probability ensemble. This is a secondary analysis because typical edge acoustic systems deploy a single neural network.",
        "",
        "---",
        "",
        "## 4. Descriptive Multi-Seed Consistency Analysis (Per-Seed Paired McNemar)",
        "",
        "| Category Name | Seed 42 Discordant $(b : c)$ [$p$] | Seed 123 Discordant $(b : c)$ [$p$] | Seed 2026 Discordant $(b : c)$ [$p$] | Direction Consistency ($3/3$)? |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for row in prim.values():
        dname = row["display_name"]
        smc = row["per_seed_mcnemar"]
        s42 = f"`{smc['seed_42']['b']}:{smc['seed_42']['c']}` ({smc['seed_42']['p_value']:.4f})"
        s123 = f"`{smc['seed_123']['b']}:{smc['seed_123']['c']}` ({smc['seed_123']['p_value']:.4f})"
        s2026 = f"`{smc['seed_2026']['b']}:{smc['seed_2026']['c']}` ({smc['seed_2026']['p_value']:.4f})"
        concordant = all(smc[f"seed_{s}"]["net_shift"] > 0 for s in SEEDS)
        lines.append(f"| {dname} | {s42} | {s123} | {s2026} | {'**YES (3/3)**' if concordant else 'NO'} |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Secondary Reference Comparison: Category Ablations vs. E2 Full Reference Ceiling",
        "",
        "The secondary comparison measures the total real-world margin loss from removing a category compared to the unablated ceiling (`E2 Full`, $N = 646$). This contrast conflates category semantic loss with sample-size reduction ($646 \\rightarrow 602..627$).",
        "",
        "| Rank | Category Name | Reference Condition | Slice $N$ | Point Est $\\theta_1$ vs Full [95% CI] | Permutation $p$-value | Secondary Holm $\\alpha$ | Significant under FWER? | $\\theta_2$ vs Full [95% CI] | Ens McNemar $p$-value |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    sorted_sec = sorted(sec.values(), key=lambda x: x["holm_bonferroni"]["rank"] if "holm_bonferroni" in x else 999)
    for row in sorted_sec:
        dname = row["display_name"]
        ref_cond = row["reference_condition"]
        n_rec = row["n_recordings"]
        hb_sec = row.get("holm_bonferroni", {})
        rank = hb_sec.get("rank", "-")
        perm = row["theta_1_permutation"]
        p_val = f"{perm['p_value']:.6f}"
        alpha_t = f"{hb_sec.get('alpha_threshold', 0.05):.6f}" if "alpha_threshold" in hb_sec else "N/A"
        sig_sec = "**YES**" if hb_sec.get("rejected", False) else "NO"

        cb = row["cluster_bootstrap"]
        th1 = cb["theta_1_avg_model"]
        th1_str = f"{th1['point_estimate']:+.4f} [{th1['ci_95'][0]:+.4f}, {th1['ci_95'][1]:+.4f}]"

        ens_mc = row["ensemble_mcnemar"]
        p_ens = f"{ens_mc['p_value']:.6f}"
        th2 = cb.get("theta_2_ensemble", {})
        th2_str = f"{th2.get('point_estimate', 0.0):+.4f} [{th2.get('ci_95', [0.0, 0.0])[0]:+.4f}, {th2.get('ci_95', [0.0, 0.0])[1]:+.4f}]"

        lines.append(
            f"| **{rank}** | {dname} | `{ref_cond}` | {n_rec} | {th1_str} | {p_val} | {alpha_t} | {sig_sec} | {th2_str} | {p_ens} |"
        )

    lines.extend([
        "",
        "> [!NOTE]",
        "> **Secondary Multiplicity Policy:** The 7 secondary comparisons against `E2 Full` constitute a separate hypothesis family evaluated with its own 7-test Holm-Bonferroni step-down correction ($\\alpha = 0.05$). They are strictly separated from the primary family to prevent multiplicity leakage.",
        "",
        "---",
        "",
        "## 6. Overall Condition Performance Summary",
        "",
        "| Condition Key | Train N | Macro F1 (Mean ± Std) | Confusers FER (Mean) | Overall FER (Mean) | Ens Macro F1 | Ens Confusers FER |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for cond, row in cond_sum.items():
        mf1 = f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}"
        fer_c = f"{row['fer_confusers_mean']:.4f}"
        fer_o = f"{row['fer_overall_mean']:.4f}"
        ens_f1 = f"{row['ensemble_macro_f1']:.4f}"
        ens_fc = f"{row['ensemble_fer_confusers']:.4f}"
        lines.append(f"| `{cond}` | — | {mf1} | {fer_c} | {fer_o} | {ens_f1} | {ens_fc} |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Statistical Power & Epistemic Boundaries",
        "",
        "1. **Exact Binomial Power Threshold:** With $N = 25$ recordings, a single-seed or ensemble comparison requires at least 6 discordant pairs ($b=6, c=0$) for unadjusted significance ($p = 0.031250$) and at least 9 discordant pairs ($b=9, c=0$) to survive Holm-Bonferroni Rank 1 ($p = 0.003906 < 0.007143$).",
        "2. **Interpretation of Insignificance:** Non-significant category differences ($p > 0.05$) indicate that on a fixed benchmark of $N = 25$ recordings, the margin shift was not detectable above stochastic noise. It does not constitute proof that the category has zero real-world utility.",
        "3. **Absence of Dedicated Tools Slice in Benchmark:** The Test Set v2 benchmark contains 6 confuser subdomains ($6 \\times 25 = 150$). The effect of `construction_tools` ablation was evaluated across the overall confuser stratum ($N = 150$), accurately measuring its generalized acoustic protection.",
        "",
        "---",
        "",
        "**Report Generated Automatically by:** `src/evaluate_phase2_e4.py`  ",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==============================================================================
# 10. CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 E4 Experiments on Test Set v2 Benchmark")
    parser.add_argument("--test-manifest", type=Path, default=TEST_SET_V2_MANIFEST, help="Path to Test Set v2 manifest")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Path to Phase 2 models directory")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="Path to output markdown report")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON, help="Path to output summary JSON")
    parser.add_argument("--synthetic", action="store_true", help="Run against simulated predictions to validate pipeline")
    parser.add_argument("--n-bootstraps", type=int, default=10000, help="Number of bootstrap iterations (default 10,000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap (default 42)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()

    run_phase2_evaluation(
        test_manifest_path=args.test_manifest,
        models_dir=args.models_dir,
        report_path=args.report_path,
        summary_json_path=args.summary_json,
        is_synthetic=args.synthetic,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
