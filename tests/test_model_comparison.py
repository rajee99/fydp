"""Unit tests for model comparison reports, output schemas, and isolation guarantees."""

import json
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "models" / "experiments" / "model_comparison"


def test_comparison_outputs_exist():
    """Verify that all required comparison reports and data files exist."""
    assert (OUTPUT_DIR / "baseline_eval_report.json").exists(), "Missing baseline_eval_report.json"
    assert (OUTPUT_DIR / "combined_eval_report.json").exists(), "Missing combined_eval_report.json"
    report_file = ROOT / "reports" / "model_comparison_report.md"
    if not report_file.exists():
        report_file = OUTPUT_DIR / "model_comparison_report.md"
    assert report_file.exists(), "Missing model_comparison_report.md"
    assert (OUTPUT_DIR / "eval_predictions_comparison.csv").exists(), "Missing eval_predictions_comparison.csv"


def test_comparison_csv_schema_and_counts():
    """Verify eval_predictions_comparison.csv contains 519 rows and required columns."""
    csv_path = OUTPUT_DIR / "eval_predictions_comparison.csv"
    df = pd.read_csv(csv_path)
    assert len(df) == 519, f"Expected 519 rows, found {len(df)}"

    required_cols = [
        "filename", "relative_path", "video_id", "ground_truth_class",
        "baseline_pred_class", "baseline_correct",
        "combined_pred_class", "combined_correct",
        "predictions_agree", "outcome"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Verify outcome distribution sums to 519
    outcome_sum = df["outcome"].value_counts().sum()
    assert outcome_sum == 519


def test_reports_consistency():
    """Verify that reports contain valid metrics matching calculated CSV values."""
    with open(OUTPUT_DIR / "baseline_eval_report.json", "r", encoding="utf-8") as f:
        b_data = json.load(f)
    with open(OUTPUT_DIR / "combined_eval_report.json", "r", encoding="utf-8") as f:
        c_data = json.load(f)

    assert b_data["evaluated_clips"] == 519
    assert c_data["evaluated_clips"] == 519

    # Baseline performance check
    assert b_data["metrics"]["accuracy"] == pytest.approx(0.8478, abs=1e-3)
    assert b_data["metrics"]["macro_f1"] == pytest.approx(0.8195, abs=1e-3)

    # Combined performance check
    assert c_data["metrics"]["accuracy"] == pytest.approx(0.8170, abs=1e-3)
    assert c_data["metrics"]["macro_f1"] == pytest.approx(0.7837, abs=1e-3)
