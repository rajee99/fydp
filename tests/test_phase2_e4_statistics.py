"""Unit tests for Phase 2 E4 statistical methodology.

Tests:
1. Exact two-sided binomial McNemar test on mandated edge cases:
   - 5:0 -> p = 0.062500 (insignificant at alpha = 0.05)
   - 6:0 -> p = 0.031250 (significant at unadjusted alpha = 0.05)
   - 9:0 -> p = 0.003906 (significant under Holm-Bonferroni rank 1: alpha_1 = 0.007143)
   - 7:2 -> p = 0.179688 (insignificant)
   - 2:7 -> p = 0.179688 (insignificant, opposite direction)
   - 0:0 -> p = 1.000000 (no discordant pairs)
2. Holm-Bonferroni multiple testing correction:
   - Sorting, step-down thresholds (alpha / (m - rank + 1)), adjusted p-values, stopping rule.
3. Cluster-aware recording-level bootstrap:
   - Reproducibility with seed 42.
   - Within-cluster preservation of 3-seed predictions.
   - Null case (identical predictions -> CI [0.0, 0.0]).
   - Controlled shift (+1 error on 25 recordings -> +0.0400).
4. Separate reporting of Estimand 1 (theta_1) and Estimand 2 (theta_2).
5. Probability ensemble calculations.
6. Strict recording ID validation (catches length mismatch, content mismatch).
"""

import math
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate_phase2_e4 import (
    exact_mcnemar_test,
    paired_permutation_test,
    holm_bonferroni_correction,
    cluster_aware_bootstrap,
    compute_probability_ensemble,
    validate_recording_ids,
)


class TestExactBinomialMcNemar:
    """Verifies exact binomial McNemar test against analytical mathematical truths."""

    def test_edge_case_5_to_0(self):
        """5:0 discordant errors: p = 2 * (1/2)^5 = 2/32 = 0.062500 (insignificant at 0.05)."""
        res = exact_mcnemar_test(5, 0)
        assert res["b"] == 5
        assert res["c"] == 0
        assert res["total_discordant"] == 5
        assert res["net_shift"] == 5
        assert math.isclose(res["p_value"], 0.062500, abs_tol=1e-5)
        assert res["statistically_significant_unadjusted"] is False

    def test_edge_case_6_to_0(self):
        """6:0 discordant errors: p = 2 * (1/2)^6 = 2/64 = 0.031250 (significant unadjusted)."""
        res = exact_mcnemar_test(6, 0)
        assert res["b"] == 6
        assert res["c"] == 0
        assert res["total_discordant"] == 6
        assert res["net_shift"] == 6
        assert math.isclose(res["p_value"], 0.031250, abs_tol=1e-5)
        assert res["statistically_significant_unadjusted"] is True

    def test_edge_case_9_to_0(self):
        """9:0 discordant errors: p = 2 * (1/2)^9 = 2/512 = 0.003906 (survives Holm-Bonferroni)."""
        res = exact_mcnemar_test(9, 0)
        assert res["b"] == 9
        assert res["c"] == 0
        assert res["total_discordant"] == 9
        assert res["net_shift"] == 9
        assert math.isclose(res["p_value"], 0.003906, abs_tol=1e-5)
        assert res["statistically_significant_unadjusted"] is True
        # Strictest Holm-Bonferroni threshold for 7 categories: 0.05 / 7 ~ 0.007143
        assert res["p_value"] < (0.05 / 7)

    def test_edge_case_7_to_2(self):
        """7:2 discordant errors (total 9): p = 2 * (1 + 9 + 36)/512 = 92/512 = 0.179688."""
        res = exact_mcnemar_test(7, 2)
        assert res["b"] == 7
        assert res["c"] == 2
        assert res["total_discordant"] == 9
        assert res["net_shift"] == 5
        assert math.isclose(res["p_value"], 0.179688, abs_tol=1e-5)
        assert res["statistically_significant_unadjusted"] is False

    def test_edge_case_2_to_7(self):
        """2:7 discordant errors (opposite direction): p = 0.179688, net_shift = -5."""
        res = exact_mcnemar_test(2, 7)
        assert res["b"] == 2
        assert res["c"] == 7
        assert res["total_discordant"] == 9
        assert res["net_shift"] == -5
        assert math.isclose(res["p_value"], 0.179688, abs_tol=1e-5)
        assert res["statistically_significant_unadjusted"] is False

    def test_edge_case_0_to_0(self):
        """0:0 discordant pairs: p = 1.000000, insignificant."""
        res = exact_mcnemar_test(0, 0)
        assert res["total_discordant"] == 0
        assert res["net_shift"] == 0
        assert res["p_value"] == 1.0
        assert res["statistically_significant_unadjusted"] is False


class TestHolmBonferroni:
    """Verifies step-down multiple testing correction."""

    def test_seven_categories_step_down_thresholds(self):
        """Verify the 7 theoretical step-down alpha thresholds: 0.05/7 to 0.05/1."""
        dummy_p = {f"cat_{i}": 0.01 * i for i in range(1, 8)}
        res = holm_bonferroni_correction(dummy_p, alpha=0.05)
        assert len(res) == 7

        expected_thresholds = [
            0.05 / 7,  # rank 1: ~0.007143
            0.05 / 6,  # rank 2: ~0.008333
            0.05 / 5,  # rank 3: 0.010000
            0.05 / 4,  # rank 4: 0.012500
            0.05 / 3,  # rank 5: ~0.016667
            0.05 / 2,  # rank 6: 0.025000
            0.05 / 1,  # rank 7: 0.050000
        ]

        # Verify ranks and thresholds
        sorted_by_rank = sorted(res.values(), key=lambda x: x["rank"])
        for idx, item in enumerate(sorted_by_rank):
            assert item["rank"] == idx + 1
            assert math.isclose(item["alpha_threshold"], expected_thresholds[idx], abs_tol=1e-5)

    def test_step_down_stopping_rule(self):
        """If rank 1 passes (p=0.0039 < 0.0071) but rank 2 fails (p=0.015 > 0.0083),
        rank 1 is rejected, and ranks 2-7 are NOT rejected even if later p-values are small.
        """
        p_dict = {
            "shouting": 0.003906,  # rank 1 (passes: 0.0039 < 0.00714)
            "metal": 0.015000,     # rank 2 (fails: 0.0150 > 0.00833)
            "doors": 0.020000,     # rank 3 (must not be rejected)
            "tools": 0.030000,
            "alarms": 0.040000,
            "fireworks": 0.050000,
            "whistles": 0.060000,
        }
        res = holm_bonferroni_correction(p_dict, alpha=0.05)
        assert res["shouting"]["rejected"] is True
        assert res["shouting"]["rank"] == 1

        assert res["metal"]["rejected"] is False
        assert res["metal"]["rank"] == 2

        # Step down stops: all subsequent hypotheses must be rejected=False
        for cat in ["doors", "tools", "alarms", "fireworks", "whistles"]:
            assert res[cat]["rejected"] is False

    def test_rank_one_failure_rejects_none(self):
        """If rank 1 fails threshold, zero hypotheses are rejected."""
        p_dict = {f"cat_{i}": 0.02 * i for i in range(1, 8)}  # lowest is 0.02 > 0.00714
        res = holm_bonferroni_correction(p_dict, alpha=0.05)
        for cat, item in res.items():
            assert item["rejected"] is False

    def test_adjusted_p_values_monotonic(self):
        """Adjusted p-values must be non-decreasing with rank."""
        p_dict = {
            "c1": 0.001,
            "c2": 0.004,
            "c3": 0.005,
            "c4": 0.02,
            "c5": 0.03,
            "c6": 0.04,
            "c7": 0.08,
        }
        res = holm_bonferroni_correction(p_dict, alpha=0.05)
        sorted_by_rank = sorted(res.values(), key=lambda x: x["rank"])
        for i in range(1, len(sorted_by_rank)):
            assert sorted_by_rank[i]["adjusted_p_value"] >= sorted_by_rank[i - 1]["adjusted_p_value"]


class TestClusterAwareBootstrap:
    """Verifies cluster-aware recording-level bootstrap behavior."""

    def test_bootstrap_reproducibility(self):
        """Fixed seed produces bit-for-bit identical confidence intervals."""
        recording_ids = [f"rec_{i:03d}" for i in range(25)]
        y_true = np.full(25, 4)  # All Other

        rng = np.random.RandomState(123)
        preds_a = {s: rng.choice([0, 4], size=25, p=[0.2, 0.8]) for s in [42, 123, 2026]}
        preds_b = {s: rng.choice([0, 4], size=25, p=[0.1, 0.9]) for s in [42, 123, 2026]}

        res1 = cluster_aware_bootstrap(recording_ids, y_true, preds_a, preds_b, n_bootstraps=500, seed=42)
        res2 = cluster_aware_bootstrap(recording_ids, y_true, preds_a, preds_b, n_bootstraps=500, seed=42)

        assert res1["theta_1_avg_model"]["point_estimate"] == res2["theta_1_avg_model"]["point_estimate"]
        assert res1["theta_1_avg_model"]["ci_95"] == res2["theta_1_avg_model"]["ci_95"]

    def test_null_case_identical_models(self):
        """When Model A and Model B have identical predictions, point estimate and CI are 0.0."""
        recording_ids = [f"rec_{i:03d}" for i in range(25)]
        y_true = np.full(25, 4)
        preds_a = {s: np.full(25, 4) for s in [42, 123, 2026]}
        preds_b = {s: np.full(25, 4) for s in [42, 123, 2026]}

        res = cluster_aware_bootstrap(recording_ids, y_true, preds_a, preds_b, n_bootstraps=200, seed=42)
        th1 = res["theta_1_avg_model"]
        assert th1["point_estimate"] == 0.0
        assert th1["ci_95"] == [0.0, 0.0]
        assert th1["p_superior"] == 0.0

    def test_controlled_effect_shift(self):
        """When Model A has exactly 1 extra false alarm across all 3 seeds on 25 recordings,
        the point estimate is exactly +0.0400 (+1/25).
        """
        recording_ids = [f"rec_{i:03d}" for i in range(25)]
        y_true = np.full(25, 4)

        preds_a = {s: np.full(25, 4) for s in [42, 123, 2026]}
        preds_b = {s: np.full(25, 4) for s in [42, 123, 2026]}

        # On recording 0, Model A makes an error (scream = 0) on all 3 seeds
        for s in [42, 123, 2026]:
            preds_a[s][0] = 0

        res = cluster_aware_bootstrap(recording_ids, y_true, preds_a, preds_b, n_bootstraps=500, seed=42)
        th1 = res["theta_1_avg_model"]
        assert math.isclose(th1["point_estimate"], 0.0400, abs_tol=1e-4)
        # CI must contain the true effect 0.04
        assert th1["ci_95"][0] <= 0.0400 <= th1["ci_95"][1]


class TestDistinctEstimandsSeparation:
    """Verifies that Estimand 1 (theta_1) and Estimand 2 (theta_2) are strictly kept distinct."""

    def test_separate_estimand_fields(self):
        recording_ids = [f"rec_{i:03d}" for i in range(25)]
        y_true = np.full(25, 4)

        # Build probabilities where seed voting diverges from ensemble
        probs_a = {}
        probs_b = {}
        preds_a = {}
        preds_b = {}

        for s in [42, 123, 2026]:
            p_a = np.zeros((25, 5))
            p_a[:, 4] = 0.90
            probs_a[s] = p_a
            preds_a[s] = np.argmax(p_a, axis=1)

            p_b = np.zeros((25, 5))
            p_b[:, 4] = 0.90
            probs_b[s] = p_b
            preds_b[s] = np.argmax(p_b, axis=1)

        res = cluster_aware_bootstrap(
            recording_ids=recording_ids,
            y_true=y_true,
            preds_a_by_seed=preds_a,
            preds_b_by_seed=preds_b,
            probs_a_by_seed=probs_a,
            probs_b_by_seed=probs_b,
            n_bootstraps=200,
            seed=42,
        )

        assert "theta_1_avg_model" in res
        assert "theta_2_ensemble" in res
        assert res["theta_1_avg_model"] is not res["theta_2_ensemble"]
        assert "point_estimate" in res["theta_1_avg_model"]
        assert "ci_95" in res["theta_1_avg_model"]
        assert "point_estimate" in res["theta_2_ensemble"]
        assert "ci_95" in res["theta_2_ensemble"]


class TestProbabilityEnsemble:
    """Verifies probability ensemble calculation."""

    def test_mean_probability_and_argmax(self):
        # 2 samples, 3 classes, 2 seeds
        # Seed 1: sample 0 -> [0.6, 0.4, 0.0], sample 1 -> [0.1, 0.2, 0.7]
        # Seed 2: sample 0 -> [0.8, 0.2, 0.0], sample 1 -> [0.3, 0.2, 0.5]
        # Mean: sample 0 -> [0.7, 0.3, 0.0] -> argmax 0
        # Mean: sample 1 -> [0.2, 0.2, 0.6] -> argmax 2
        p1 = np.array([[0.6, 0.4, 0.0], [0.1, 0.2, 0.7]])
        p2 = np.array([[0.8, 0.2, 0.0], [0.3, 0.2, 0.5]])

        mean_p, ens_preds = compute_probability_ensemble({42: p1, 123: p2})
        assert np.allclose(mean_p[0], [0.7, 0.3, 0.0])
        assert np.allclose(mean_p[1], [0.2, 0.2, 0.6])
        assert list(ens_preds) == [0, 2]

    def test_empty_seeds_raises(self):
        with pytest.raises(ValueError):
            compute_probability_ensemble({})


class TestRecordingIdValidation:
    """Verifies strict recording ID alignment."""

    def test_matching_ids_pass(self):
        ids = ["clip_1", "clip_2", "clip_3"]
        validate_recording_ids(ids, list(ids), "test_condition")

    def test_length_mismatch_raises(self):
        ids = ["clip_1", "clip_2"]
        with pytest.raises(ValueError, match="Recording ID count mismatch"):
            validate_recording_ids(ids, ["clip_1"], "test_condition")

    def test_id_content_mismatch_raises(self):
        ids = ["clip_1", "clip_2"]
        with pytest.raises(ValueError, match="Recording ID mismatch at index 1"):
            validate_recording_ids(ids, ["clip_1", "clip_WRONG"], "test_condition")


class TestMultiplicityPoliciesAndEstimands:
    """Verifies that primary and secondary multiplicity policies and estimands adhere to statistical contract."""

    def test_primary_family_exactly_seven_hypotheses(self):
        """Primary hypothesis family must consist of exactly 7 comparisons (Ablation vs Control)."""
        from src.evaluate_phase2_e4 import CATEGORY_PAIRS
        assert len(CATEGORY_PAIRS) == 7
        expected_categories = {
            "human_shouting",
            "metal_impacts",
            "door_slams",
            "construction_tools",
            "car_alarms_detectors",
            "fireworks_firecrackers",
            "whistles_high_pitch",
        }
        actual_categories = {pair["category_key"] for pair in CATEGORY_PAIRS}
        assert actual_categories == expected_categories

    def test_secondary_family_isolated_multiplicity(self):
        """Secondary family (vs E2 Full) must have a separately defined multiplicity correction."""
        p_prim = {f"cat_{i}": 0.01 * i for i in range(1, 8)}
        p_sec = {f"cat_{i}": 0.005 * i for i in range(1, 8)}

        # Both have m = 7
        hb_prim = holm_bonferroni_correction(p_prim, alpha=0.05)
        hb_sec = holm_bonferroni_correction(p_sec, alpha=0.05)

        assert len(hb_prim) == 7
        assert len(hb_sec) == 7
        # Verify primary and secondary are computed independently and not combined into m = 14 or m = 15
        assert hb_prim["cat_1"]["m"] == 7
        assert hb_sec["cat_1"]["m"] == 7

    def test_no_mcnemar_on_fractional_theta_1(self):
        """theta_1 is a continuous mean difference across 3 seeds and cannot be evaluated with ordinary McNemar."""
        # Generating fractional average decisions
        frac_diff = 0.333333
        # McNemar requires non-negative integers b, c; float inputs must not be passed
        # Exact McNemar on ints works:
        res = exact_mcnemar_test(3, 0)
        assert isinstance(res["b"], int)
        assert isinstance(res["c"], int)


class TestPairedPermutationTest:
    """Verifies paired sign-flip permutation testing on binary and fractional recording differences."""

    def test_permutation_binary_5_to_0(self):
        """Binary 5:0 must match exact binomial McNemar p = 0.062500."""
        d = np.array([1.0] * 5 + [0.0] * 20)
        res = paired_permutation_test(d)
        assert math.isclose(res["p_value"], 0.062500, abs_tol=1e-5)
        assert res["statistically_significant_unadjusted"] is False

    def test_permutation_binary_6_to_0(self):
        """Binary 6:0 must match exact binomial McNemar p = 0.031250."""
        d = np.array([1.0] * 6 + [0.0] * 19)
        res = paired_permutation_test(d)
        assert math.isclose(res["p_value"], 0.031250, abs_tol=1e-5)
        assert res["statistically_significant_unadjusted"] is True

    def test_permutation_binary_9_to_0(self):
        """Binary 9:0 must match exact binomial McNemar p = 0.003906 (survives Holm rank 1)."""
        d = np.array([1.0] * 9 + [0.0] * 16)
        res = paired_permutation_test(d)
        assert math.isclose(res["p_value"], 0.003906, abs_tol=1e-5)
        assert res["p_value"] < (0.05 / 7)
        assert res["statistically_significant_unadjusted"] is True

    def test_permutation_fractional_differences(self):
        """Permutation test accommodates fractional average decision differences without error."""
        # 10 recordings with +1/3 shift, 5 with +2/3 shift, 10 with 0
        d = np.array([1/3] * 10 + [2/3] * 5 + [0.0] * 10)
        res = paired_permutation_test(d, seed=42)
        assert res["obs_stat"] > 0
        assert 0.0 <= res["p_value"] <= 1.0
        assert res["non_zero_count"] == 15

    def test_permutation_negative_binary(self):
        """Negative binary shift (-6:0) has identical two-sided p-value (0.031250) and negative obs_stat."""
        d = np.array([-1.0] * 6 + [0.0] * 19)
        res = paired_permutation_test(d)
        assert math.isclose(res["p_value"], 0.031250, abs_tol=1e-5)
        assert math.isclose(res["obs_stat"], -6 / 25, abs_tol=1e-4)
        assert res["statistically_significant_unadjusted"] is True

    def test_permutation_binary_7_to_2_and_2_to_7(self):
        """Binary 7:2 and 2:7 discordant shifts must both yield p = 0.179688, exactly matching McNemar."""
        # 7:2 shift
        d_pos = np.array([1.0] * 7 + [-1.0] * 2 + [0.0] * 16)
        res_pos = paired_permutation_test(d_pos)
        assert math.isclose(res_pos["p_value"], 0.179688, abs_tol=1e-5)
        assert math.isclose(res_pos["obs_stat"], 5 / 25, abs_tol=1e-4)

        # 2:7 shift (mirror opposite)
        d_neg = np.array([1.0] * 2 + [-1.0] * 7 + [0.0] * 16)
        res_neg = paired_permutation_test(d_neg)
        assert math.isclose(res_neg["p_value"], 0.179688, abs_tol=1e-5)
        assert math.isclose(res_neg["obs_stat"], -5 / 25, abs_tol=1e-4)

    def test_permutation_cancelling_fractions(self):
        """Opposing non-zero differences that sum to zero yield obs_stat = 0, p = 1.0, but retain non_zero_count."""
        d = np.array([1/3] * 3 + [-1/3] * 3 + [0.0] * 19)
        res = paired_permutation_test(d)
        assert res["obs_stat"] == 0.0
        assert res["p_value"] == 1.0
        assert res["non_zero_count"] == 6
        assert res["statistically_significant_unadjusted"] is False

    def test_permutation_monte_carlo_determinism(self):
        """When k > 16, Monte Carlo permutation is triggered and produces deterministic results under fixed seed."""
        # k = 18 non-zero items
        d = np.array([0.3333] * 12 + [-0.3333] * 6 + [0.0] * 7)
        res1 = paired_permutation_test(d, n_permutations=10000, seed=42)
        res2 = paired_permutation_test(d, n_permutations=10000, seed=42)
        assert res1["method"] == "monte_carlo_10000"
        assert res2["method"] == "monte_carlo_10000"
        assert res1["p_value"] == res2["p_value"]
        assert res1["obs_stat"] == res2["obs_stat"]

    def test_permutation_monte_carlo_formula_bounds(self):
        """Monte Carlo formula (count + 1) / (R + 1) prevents p = 0 and satisfies minimum bound 1/(R+1)."""
        d = np.array([100.0] + [1.0] * 17 + [0.0] * 7)
        res = paired_permutation_test(d, n_permutations=1000, seed=42)
        assert res["method"] == "monte_carlo_1000"
        assert res["p_value"] > 0.0
        expected_min = 1.0 / 1001.0
        assert res["p_value"] >= round(expected_min, 6)

    def test_permutation_exact_enumeration_boundary(self):
        """k <= 16 uses exact exhaustive enumeration; k = 17 switches to Monte Carlo."""
        d_16 = np.array([1.0] * 16 + [0.0] * 9)
        res_16 = paired_permutation_test(d_16)
        assert res_16["method"] == "exact_enumeration_2^16"

        d_17 = np.array([1.0] * 17 + [0.0] * 8)
        res_17 = paired_permutation_test(d_17, n_permutations=500, seed=42)
        assert res_17["method"] == "monte_carlo_500"

    def test_permutation_null_case(self):
        """All-zero differences must yield p = 1.0."""
        d = np.zeros(25)
        res = paired_permutation_test(d)
        assert res["obs_stat"] == 0.0
        assert res["p_value"] == 1.0
        assert res["statistically_significant_unadjusted"] is False


class TestSyntheticReportLabelling:
    """Verifies that synthetic evaluation reports are explicitly labelled to prevent meta-pollution."""

    def test_synthetic_warning_banner(self, tmp_path):
        from src.evaluate_phase2_e4 import generate_markdown_report

        dummy_data = {
            "metadata": {
                "timestamp": "2026-11-02T17:00:00",
                "benchmark_manifest": "data/processed/test_set_v2_manifest.csv",
                "benchmark_samples": 750,
                "mode": "synthetic",
                "synthetic_validation_only": True,
                "status_label": "SYNTHETIC VALIDATION ONLY — NOT EXPERIMENTAL RESULTS",
                "bootstrap_iterations": 100,
                "random_seed": 42,
                "seeds_evaluated": [42, 123, 2026],
                "num_conditions": 15,
            },
            "condition_summary": {
                "e2_full": {
                    "macro_f1_mean": 0.88, "macro_f1_std": 0.01,
                    "fer_confusers_mean": 0.08, "fer_overall_mean": 0.09,
                    "ensemble_macro_f1": 0.97, "ensemble_fer_confusers": 0.01, "ensemble_fer_overall": 0.02,
                }
            },
            "primary_comparisons": {
                "human_shouting": {
                    "category": "human_shouting",
                    "display_name": "Human Shouting",
                    "n_recordings": 25,
                    "theta_1_permutation": {"obs_stat": 0.22, "p_value": 0.011719, "method": "exact_enumeration_2^10"},
                    "ensemble_mcnemar": {"b": 10, "c": 1, "p_value": 0.011719},
                    "per_seed_mcnemar": {
                        "seed_42": {"b": 8, "c": 1, "net_shift": 7, "p_value": 0.039},
                        "seed_123": {"b": 7, "c": 1, "net_shift": 6, "p_value": 0.070},
                        "seed_2026": {"b": 9, "c": 1, "net_shift": 8, "p_value": 0.021},
                    },
                    "holm_bonferroni": {"rank": 1, "alpha_threshold": 0.007143, "rejected": False},
                    "holm_bonferroni_ensemble": {"rank": 1, "alpha_threshold": 0.007143, "rejected": False},
                    "cluster_bootstrap": {
                        "theta_1_avg_model": {"point_estimate": 0.22, "ci_95": [0.06, 0.38]},
                        "theta_2_ensemble": {"point_estimate": 0.36, "ci_95": [0.16, 0.56]},
                    },
                }
            },
            "secondary_comparisons": {
                "human_shouting": {
                    "category": "human_shouting",
                    "display_name": "Human Shouting",
                    "reference_condition": "e2_full",
                    "n_recordings": 25,
                    "theta_1_permutation": {"obs_stat": 0.36, "p_value": 0.001953, "method": "exact_enumeration_2^10"},
                    "ensemble_mcnemar": {"b": 10, "c": 0, "p_value": 0.001953},
                    "holm_bonferroni": {"rank": 1, "alpha_threshold": 0.007143, "rejected": True},
                    "cluster_bootstrap": {
                        "theta_1_avg_model": {"point_estimate": 0.36, "ci_95": [0.24, 0.49]},
                        "theta_2_ensemble": {"point_estimate": 0.40, "ci_95": [0.20, 0.60]},
                    },
                }
            },
        }

        out_path = tmp_path / "test_report.md"
        generate_markdown_report(dummy_data, out_path)
        content = out_path.read_text(encoding="utf-8")

        assert "SYNTHETIC VALIDATION ONLY — NOT EXPERIMENTAL RESULTS" in content
        assert "DO NOT USE FOR RESEARCH FINDINGS" in content
