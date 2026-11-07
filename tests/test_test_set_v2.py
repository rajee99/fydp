"""Unit tests for Test Set v2 evaluation metrics and statistical testing functions."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from src.evaluate_test_set_v2 import (
    wilson_score_interval,
    mcnemar_test,
    mcnemar_false_emergency_test,
    bootstrap_metric_difference,
)


def test_wilson_score_interval():
    # 0 successes
    low, high = wilson_score_interval(0, 100)
    assert low == 0.0
    assert 0.0 < high < 0.1

    # 100 successes out of 100
    low, high = wilson_score_interval(100, 100)
    assert 0.9 < low < 1.0
    assert high == 1.0

    # 50 out of 100
    low, high = wilson_score_interval(50, 100)
    assert 0.35 < low < 0.50
    assert 0.50 < high < 0.65
    assert low < 0.5 < high


def test_mcnemar_test_identical():
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred1 = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred2 = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    res = mcnemar_test(y_true, y_pred1, y_pred2)
    assert res["discordant_pairs"] == 0
    assert res["p_value"] == 1.0
    assert not res["statistically_significant"]


def test_mcnemar_test_discordant():
    # 20 samples where model 1 gets 15 right and model 2 gets 5 right
    y_true = np.zeros(20, dtype=int)
    y_pred1 = np.zeros(20, dtype=int)
    y_pred1[15:] = 1  # 15 correct, 5 wrong

    y_pred2 = np.zeros(20, dtype=int)
    y_pred2[5:] = 1   # 5 correct, 15 wrong

    res = mcnemar_test(y_true, y_pred1, y_pred2)
    assert res["discordant_pairs"] > 0
    assert res["contingency_table"]["m1_correct_m2_wrong"] == 10
    assert res["contingency_table"]["m1_wrong_m2_correct"] == 0
    assert res["p_value"] < 0.05
    assert res["statistically_significant"]


def test_mcnemar_false_emergency_test():
    y_true = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    # Model 1 predicts 4 (other) for all: 0 false emergencies
    y_pred1 = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    # Model 2 predicts 0 (scream) for all: 10 false emergencies
    y_pred2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    res = mcnemar_false_emergency_test(y_true, y_pred1, y_pred2)
    assert res["discordant_pairs"] == 10
    assert res["contingency_table"]["m1_quiet_m2_alarm"] == 10
    assert res["contingency_table"]["m1_alarm_m2_quiet"] == 0
    assert res["p_value"] < 0.05
    assert res["statistically_significant"]


def test_bootstrap_metric_difference():
    n = 100
    y_true = np.random.randint(0, 5, size=n)
    y_pred1 = y_true.copy()  # perfect model
    y_pred2 = np.random.randint(0, 5, size=n)  # random model

    is_confuser = (y_true == 4)
    is_benign = np.zeros(n, dtype=bool)

    res = bootstrap_metric_difference(
        y_true, y_pred1, y_pred2, is_confuser, is_benign, n_bootstraps=200, seed=42
    )

    assert "delta_macro_f1" in res
    assert res["delta_macro_f1"]["mean"] > 0.3
    assert res["delta_macro_f1"]["p_superior"] > 0.95


def test_test_set_v2_manifest_structure():
    import pandas as pd
    manifest_path = ROOT / "data" / "processed" / "test_set_v2_manifest.csv"
    assert manifest_path.exists(), "test_set_v2_manifest.csv does not exist"

    df = pd.read_csv(manifest_path)
    assert len(df) == 750, f"Expected 750 rows, got {len(df)}"

    # Class breakdown
    vc = df["target_class"].value_counts().to_dict()
    assert vc.get("scream") == 100
    assert vc.get("glass_breaking") == 100
    assert vc.get("gunshot") == 100
    assert vc.get("siren") == 150
    assert vc.get("other") == 300

    # Confuser vs benign breakdown
    confusers = df[df["subdomain"].str.startswith("confuser")]
    benign = df[df["subdomain"].str.startswith("benign")]
    assert len(confusers) == 150
    assert len(benign) == 150

    # Zero duplicate video IDs
    assert len(df["video_id"].unique()) == 750


def test_test_set_v2_zero_leakage():
    import pandas as pd
    manifest_path = ROOT / "data" / "processed" / "test_set_v2_manifest.csv"
    df = pd.read_csv(manifest_path)
    test_vids = set(df["video_id"].dropna().str.strip())

    check_files = [
        "e1_train_split_manifest.csv",
        "shared_val_split_manifest.csv",
        "e2_matched_train_split_manifest.csv",
        "e2_train_split_manifest.csv",
        "e3_train_split_manifest.csv",
        "eval_manifest.csv",
    ]

    for fname in check_files:
        p = ROOT / "data" / "processed" / fname
        if p.exists():
            h_df = pd.read_csv(p)
            h_vids = set(h_df["video_id"].dropna().astype(str).str.strip())
            overlap = test_vids.intersection(h_vids)
            assert len(overlap) == 0, f"Leakage detected with {fname}: {overlap}"

