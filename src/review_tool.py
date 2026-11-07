"""Local Audio Content Review Tool for Targeted Dataset Expansion.

A lightweight, zero-dependency local web interface using Python's standard library
to listen to extracted WAV files, verify acoustic contents against target classes,
and record research-grade human validation verdicts.

Supports:
- Single-clip focused review workflow with audio playback and waveform seeking.
- Real-time progress tracking by class (scream, glass_breaking, other) and overall.
- Recording verdicts (ACCEPT, RECLASSIFY, REJECT, NEEDS_SECOND_REVIEW).
- Verification of acoustic checkpoints (audibility, background noise, emergency contamination).
- Dedicated guidance for critical acoustic confusers (shouting vs scream, fireworks vs gunshot, etc.).
- Safe, resumable persistence to a dedicated review-results CSV.
- Backwards compatibility with pilot reviews via --mode pilot.
"""

import argparse
import csv
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from pathlib import Path
import sys
import urllib.parse
import wave
import webbrowser

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

# Default paths for Targeted Expansion Review
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "targeted_expansion_content_validation.csv"
DEFAULT_RESULTS = ROOT / "data" / "processed" / "targeted_expansion_review_results.csv"

# Legacy paths for Pilot Review
PILOT_RESULTS_CSV = ROOT / "data" / "processed" / "pilot_other_download_results.csv"
PILOT_REVIEW_CSV = ROOT / "data" / "processed" / "pilot_other_review.csv"
PILOT_DIR = ROOT / "data" / "raw" / "pilot"

VERDICT_CHOICES = [
    ("ACCEPT", "Accept", "Acoustic content clearly and unambiguously matches assigned target class."),
    ("RECLASSIFY", "Reclassify", "Audio contains a different valid sound category (e.g. human shouting instead of scream)."),
    ("REJECT", "Reject", "Audio does not contain the target sound, is corrupt, unintelligible, or unusable noise."),
    ("NEEDS_SECOND_REVIEW", "Needs 2nd Review", "Ambiguous, noisy, or borderline clip requiring secondary research consensus."),
]

OBSERVED_SOUND_CHOICES = [
    ("scream", "Human Scream (Distress / Panic / Pain)"),
    ("human_shouting", "Human Shouting / Yelling (Non-Scream Speech)"),
    ("glass_breaking", "Glass Breaking / Shattering"),
    ("metal_impact", "Metal Impact / Clang / Ringing"),
    ("door_slam", "Door Slam / Heavy Thud"),
    ("gunshot", "Gunshot / Explosive Blast"),
    ("fireworks", "Fireworks / Firecrackers"),
    ("siren", "Emergency Siren (Wail / Yelp / Hi-Lo)"),
    ("car_alarm_buzzer", "Car Alarm / Buzzer / Horn"),
    ("construction", "Construction / Power Tools / Jackhammer"),
    ("whistle", "Whistle / High-Pitched Acoustic Signal"),
    ("speech_music", "General Speech / Singing / Music"),
    ("ambient_noise", "Environmental Ambience / Room Noise"),
    ("other", "Other Unclassified Sound"),
]

RESULTS_FIELDNAMES = [
    "video_id",
    "start_seconds",
    "end_seconds",
    "file_path",
    "target_class",
    "subdomain",
    "matched_label",
    "origin",
    "duration_seconds",
    "verdict",
    "observed_sound",
    "expected_sound_audible",
    "background_noise_present",
    "emergency_sound_present",
    "emergency_sound_type",
    "notes",
    "reviewer",
    "reviewed_at",
]


def load_review_candidates(manifest_path: Path):
    """Load candidates from content validation manifest and verify audio accessibility.

    Returns:
        List of dicts representing valid, reviewable audio recordings.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Review manifest missing: {manifest_path}")

    candidates = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            rel_path = row["file_path"].strip()
            wav_path = ROOT / rel_path

            # Basic readability check
            duration = None
            if wav_path.exists() and wav_path.stat().st_size > 0:
                try:
                    with wave.open(str(wav_path), "rb") as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        if rate > 0:
                            duration = round(frames / float(rate), 2)
                except Exception:
                    duration = None

            candidates.append({
                "index": i,
                "video_id": row["video_id"].strip(),
                "start_seconds": float(row["start_seconds"]),
                "end_seconds": float(row["end_seconds"]),
                "file_path": rel_path,
                "target_class": row["target_class"].strip(),
                "subdomain": row["subdomain"].strip(),
                "matched_label": row["matched_label"].strip(),
                "origin": row.get("origin", "P1_PRIMARY").strip(),
                "duration_seconds": duration if duration is not None else float(row.get("duration_seconds", 0.0)),
                "file_size_bytes": int(row.get("file_size_bytes", 0)) if row.get("file_size_bytes") else 0,
                "exists_on_disk": wav_path.exists() and duration is not None,
            })

    return candidates


def load_existing_review_results(results_path: Path):
    """Load existing manual decisions from review results CSV.

    Returns:
        Dict mapping (video_id, start_seconds, end_seconds) to decision dict.
    """
    results_map = {}
    if not results_path.exists():
        return results_map

    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", "").strip()
            if not vid:
                continue
            try:
                start = float(row.get("start_seconds", 0))
                end = float(row.get("end_seconds", 0))
            except ValueError:
                continue

            key = (vid, start, end)
            dur_raw = row.get("duration_seconds", "")
            try:
                dur_val = float(dur_raw) if dur_raw else round(end - start, 2)
            except ValueError:
                dur_val = round(end - start, 2)

            results_map[key] = {
                "video_id": vid,
                "start_seconds": start,
                "end_seconds": end,
                "file_path": row.get("file_path", ""),
                "target_class": row.get("target_class", ""),
                "subdomain": row.get("subdomain", ""),
                "matched_label": row.get("matched_label", ""),
                "origin": row.get("origin", ""),
                "duration_seconds": dur_val,
                "verdict": row.get("verdict", ""),
                "observed_sound": row.get("observed_sound", ""),
                "expected_sound_audible": row.get("expected_sound_audible", "") in ["True", "true", True],
                "background_noise_present": row.get("background_noise_present", "") in ["True", "true", True],
                "emergency_sound_present": row.get("emergency_sound_present", "") in ["True", "true", True],
                "emergency_sound_type": row.get("emergency_sound_type", "none"),
                "notes": row.get("notes", ""),
                "reviewer": row.get("reviewer", ""),
                "reviewed_at": row.get("reviewed_at", ""),
            }

    return results_map


def save_review_decision(results_path: Path, decision_dict: dict, allow_overwrite: bool = True):
    """Save or update a single review decision safely in results CSV.

    Args:
        results_path: Destination review results CSV.
        decision_dict: Decision data matching RESULTS_FIELDNAMES.
        allow_overwrite: Whether to allow updating an existing decision.

    Returns:
        bool: True if saved successfully, False if overwrite rejected.
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing_map = load_existing_review_results(results_path)

    key = (decision_dict["video_id"], float(decision_dict["start_seconds"]), float(decision_dict["end_seconds"]))

    if key in existing_map and not allow_overwrite:
        return False

    # Stamp reviewed_at if not present
    if not decision_dict.get("reviewed_at"):
        decision_dict["reviewed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    existing_map[key] = decision_dict

    # Write atomically using a temporary file
    temp_path = results_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES)
        writer.writeheader()
        for row in existing_map.values():
            writer.writerow({k: row.get(k, "") for k in RESULTS_FIELDNAMES})

    temp_path.replace(results_path)
    return True


def save_bulk_decisions(results_path: Path, decisions_list: list, allow_overwrite: bool = True):
    """Save a list of review decisions atomically."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing_map = load_existing_review_results(results_path)

    saved_count = 0
    for d in decisions_list:
        key = (d["video_id"], float(d["start_seconds"]), float(d["end_seconds"]))
        if key in existing_map and not allow_overwrite:
            continue
        if not d.get("reviewed_at"):
            d["reviewed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        existing_map[key] = d
        saved_count += 1

    temp_path = results_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES)
        writer.writeheader()
        for row in existing_map.values():
            writer.writerow({k: row.get(k, "") for k in RESULTS_FIELDNAMES})

    temp_path.replace(results_path)
    return saved_count


def compute_review_progress(candidates: list, results_map: dict):
    """Compute overall and per-class progress statistics.

    Returns:
        dict with overall stats, class stats, and verdict breakdown.
    """
    total = len(candidates)
    reviewed = 0
    class_stats = {}
    verdict_counts = {v[0]: 0 for v in VERDICT_CHOICES}

    for c in candidates:
        cls = c["target_class"]
        if cls not in class_stats:
            class_stats[cls] = {"total": 0, "reviewed": 0, "pending": 0, "percent": 0.0}
        class_stats[cls]["total"] += 1

        key = (c["video_id"], c["start_seconds"], c["end_seconds"])
        if key in results_map and results_map[key].get("verdict"):
            reviewed += 1
            class_stats[cls]["reviewed"] += 1
            v = results_map[key].get("verdict")
            if v in verdict_counts:
                verdict_counts[v] += 1
        else:
            class_stats[cls]["pending"] += 1

    for cls in class_stats:
        tot = class_stats[cls]["total"]
        rev = class_stats[cls]["reviewed"]
        class_stats[cls]["percent"] = round(rev / tot * 100, 1) if tot > 0 else 0.0

    percent_overall = round(reviewed / total * 100, 1) if total > 0 else 0.0

    return {
        "total": total,
        "reviewed": reviewed,
        "pending": total - reviewed,
        "percent_complete": percent_overall,
        "by_class": class_stats,
        "verdict_counts": verdict_counts,
    }


def generate_review_html(candidates, results_map, reviewer_id="reviewer_1"):
    """Generate self-contained, responsive single-clip review web application."""
    candidates_json = json.dumps(candidates)
    results_json = json.dumps(results_map)

    verdict_options_html = "".join([
        f'<label class="verdict-btn btn-{val.lower()}">'
        f'<input type="radio" name="verdict" value="{val}" required> '
        f'<span class="btn-text"><strong>{label}</strong><small>{desc}</small></span>'
        f'</label>'
        for val, label, desc in VERDICT_CHOICES
    ])

    sound_options_html = "".join([
        f'<option value="{val}">{label}</option>'
        for val, label in OBSERVED_SOUND_CHOICES
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Audio Content Validation Tool | MyNet FYDP</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: #151e2e;
      --card-border: #233149;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --success: #10b981;
      --danger: #ef4444;
      --warning: #f59e0b;
      --purple: #8b5cf6;
      --accent: #06b6d4;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 16px;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    .app-container {{
      max-width: 1100px;
      margin: 0 auto;
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--card-border);
    }}
    .brand-title {{ font-size: 1.3rem; font-weight: 700; color: #fff; }}
    .brand-title span {{ color: var(--primary); }}
    .reviewer-tag {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      color: var(--text-muted);
    }}
    /* Progress Dashboard */
    .dashboard-panel {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 14px 18px;
    }}
    .progress-bar-wrap {{
      width: 100%;
      height: 10px;
      background: #1e293b;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 8px;
      margin-bottom: 12px;
    }}
    .progress-bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--primary), var(--success));
      width: 0%;
      transition: width 0.3s ease;
    }}
    .stats-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .class-badges {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .c-badge {{
      background: #0f172a;
      border: 1px solid var(--card-border);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .c-badge.scream {{ border-left: 3px solid #ef4444; }}
    .c-badge.glass {{ border-left: 3px solid #3b82f6; }}
    .c-badge.other {{ border-left: 3px solid #10b981; }}
    .c-badge strong {{ color: #fff; }}
    /* Navigation Bar */
    .nav-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px 14px;
      flex-wrap: wrap;
    }}
    .nav-group {{ display: flex; align-items: center; gap: 8px; }}
    button.nav-btn {{
      background: #1e293b;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 600;
      transition: background 0.15s;
    }}
    button.nav-btn:hover {{ background: #334155; }}
    button.nav-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
    button.primary-nav {{ background: var(--primary); border-color: var(--primary); }}
    button.primary-nav:hover {{ background: var(--primary-hover); }}
    select.filter-select {{
      background: #0f172a;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 0.85rem;
      outline: none;
    }}
    /* Main Review Workspace */
    .review-workspace {{
      display: grid;
      grid-template-columns: 1fr 1.25fr;
      gap: 16px;
    }}
    @media (max-width: 850px) {{
      .review-workspace {{ grid-template-columns: 1fr; }}
    }}
    .panel-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .meta-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 12px;
    }}
    .meta-title {{ font-size: 1.15rem; font-weight: 700; color: #fff; }}
    .status-pill {{
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .pill-pending {{ background: #475569; color: #f1f5f9; }}
    .pill-accept {{ background: #065f46; color: #a7f3d0; }}
    .pill-reclassify {{ background: #1e40af; color: #bfdbfe; }}
    .pill-reject {{ background: #991b1b; color: #fecaca; }}
    .pill-needs2nd {{ background: #92400e; color: #fde68a; }}
    .info-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }}
    .info-table td {{
      padding: 6px 4px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .info-table td.label {{ color: var(--text-muted); width: 38%; }}
    .info-table td.val {{ color: #fff; font-weight: 500; font-family: monospace; }}
    .badge-subdomain {{
      background: #1e293b;
      color: #38bdf8;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.8rem;
    }}
    .badge-origin {{
      background: #312e81;
      color: #c7d2fe;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
    }}
    /* Audio Section */
    .audio-box {{
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    audio {{ width: 100%; height: 42px; }}
    .audio-controls-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.8rem;
      color: var(--text-muted);
    }}
    /* Review Form */
    .form-group {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .form-label {{
      font-size: 0.85rem;
      font-weight: 600;
      color: #cbd5e1;
      display: flex;
      justify-content: space-between;
    }}
    .form-label small {{ color: var(--text-muted); font-weight: normal; }}
    .verdict-options {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .verdict-btn {{
      display: flex;
      align-items: flex-start;
      gap: 8px;
      background: #0f172a;
      border: 1px solid var(--card-border);
      padding: 10px 12px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .verdict-btn:hover {{ border-color: var(--primary); background: #162032; }}
    .verdict-btn input {{ margin-top: 3px; cursor: pointer; }}
    .verdict-btn .btn-text {{ display: flex; flex-direction: column; gap: 2px; }}
    .verdict-btn .btn-text strong {{ font-size: 0.9rem; }}
    .verdict-btn .btn-text small {{ font-size: 0.72rem; color: var(--text-muted); line-height: 1.2; }}
    .btn-accept input:checked ~ .btn-text strong {{ color: var(--success); }}
    .btn-reclassify input:checked ~ .btn-text strong {{ color: var(--primary); }}
    .btn-reject input:checked ~ .btn-text strong {{ color: var(--danger); }}
    .btn-needs_second_review input:checked ~ .btn-text strong {{ color: var(--warning); }}
    .checkpoints-box {{
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .checkpoint-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
      cursor: pointer;
    }}
    .checkpoint-item input {{ cursor: pointer; }}
    select.observed-select {{
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 8px 10px;
      border-radius: 6px;
      font-size: 0.85rem;
    }}
    textarea.notes-input {{
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 8px 10px;
      border-radius: 6px;
      font-size: 0.85rem;
      font-family: inherit;
      resize: vertical;
      min-height: 55px;
    }}
    .action-bar {{
      display: flex;
      gap: 10px;
      margin-top: 4px;
    }}
    .save-btn {{
      flex: 1;
      background: var(--success);
      color: #042f2e;
      border: none;
      padding: 10px 16px;
      border-radius: 8px;
      font-weight: 700;
      font-size: 0.95rem;
      cursor: pointer;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 8px;
      transition: background 0.15s;
    }}
    .save-btn:hover {{ background: #34d399; }}
    .save-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
    /* Cheatsheet Accordion */
    details.cheatsheet {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px 16px;
      font-size: 0.85rem;
      color: #cbd5e1;
    }}
    details.cheatsheet summary {{
      cursor: pointer;
      font-weight: 600;
      color: var(--accent);
      outline: none;
    }}
    .ambiguities-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .amb-card {{
      background: #0f172a;
      border-left: 3px solid var(--warning);
      padding: 8px 12px;
      border-radius: 0 6px 6px 0;
      font-size: 0.8rem;
    }}
    .amb-card strong {{ color: #f8fafc; display: block; margin-bottom: 2px; }}
    .notification {{
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.85rem;
      font-weight: 500;
      display: none;
    }}
    .notif-success {{ background: #064e3b; color: #a7f3d0; border: 1px solid #059669; }}
    .notif-error {{ background: #7f1d1d; color: #fecaca; border: 1px solid #dc2626; }}
  </style>
</head>
<body>
  <div class="app-container">
    <header>
      <div class="brand-title">MyNet <span>Audio Reviewer</span></div>
      <div class="reviewer-tag">Reviewer: <strong>{reviewer_id}</strong></div>
    </header>

    <!-- Progress Dashboard -->
    <div class="dashboard-panel">
      <div class="stats-row">
        <div>
          <strong id="prog-title">Targeted Expansion Content Validation Progress</strong>
          <span style="font-size: 0.85rem; color: var(--text-muted); margin-left: 8px;" id="prog-ratio">0 / 400 (0%)</span>
        </div>
        <div class="class-badges">
          <div class="c-badge scream">Scream: <strong id="prog-scream">0 / 97 (0%)</strong></div>
          <div class="c-badge glass">Glass: <strong id="prog-glass">0 / 81 (0%)</strong></div>
          <div class="c-badge other">Other: <strong id="prog-other">0 / 222 (0%)</strong></div>
        </div>
      </div>
      <div class="progress-bar-wrap">
        <div class="progress-bar-fill" id="progress-bar"></div>
      </div>
    </div>

    <!-- Navigation Bar -->
    <div class="nav-bar">
      <div class="nav-group">
        <button class="nav-btn" id="prev-btn" onclick="prevClip()">◄ Prev</button>
        <button class="nav-btn" id="next-btn" onclick="nextClip()">Next ►</button>
        <button class="nav-btn primary-nav" onclick="nextUnreviewed()">Next Unreviewed ⏩</button>
      </div>

      <div class="nav-group">
        <select class="filter-select" id="filter-mode" onchange="applyFilter()">
          <option value="all">Filter: All Samples (400)</option>
          <option value="unreviewed" selected>Filter: Unreviewed Only</option>
          <option value="reviewed">Filter: Reviewed Only</option>
          <option value="scream">Class: Scream Only (97)</option>
          <option value="glass_breaking">Class: Glass Breaking Only (81)</option>
          <option value="other">Class: Other Only (222)</option>
        </select>
        <select class="filter-select" id="jump-select" onchange="jumpToSelected()" style="max-width: 200px;"></select>
      </div>
    </div>

    <!-- Notification Bar -->
    <div id="notif-bar" class="notification"></div>

    <!-- Main Workspace -->
    <div class="review-workspace">
      <!-- Left: Audio & Metadata -->
      <div class="panel-card">
        <div class="meta-header">
          <div>
            <div class="meta-title" id="clip-title">Sample #1</div>
            <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 2px;" id="clip-subtitle">video_id</div>
          </div>
          <span class="status-pill pill-pending" id="status-pill">Pending</span>
        </div>

        <div class="audio-box">
          <audio id="audio-player" controls preload="auto"></audio>
          <div class="audio-controls-row">
            <span id="audio-specs">16kHz Mono 16-bit</span>
            <span>Shortcut: [Space] Play/Pause</span>
          </div>
        </div>

        <table class="info-table">
          <tr><td class="label">Target Class:</td><td class="val" id="td-class">-</td></tr>
          <tr><td class="label">Subdomain:</td><td class="val"><span class="badge-subdomain" id="td-subdomain">-</span></td></tr>
          <tr><td class="label">Matched AudioSet:</td><td class="val" id="td-matched">-</td></tr>
          <tr><td class="label">Candidate Origin:</td><td class="val"><span class="badge-origin" id="td-origin">-</span></td></tr>
          <tr><td class="label">Segment Window:</td><td class="val" id="td-window">-</td></tr>
          <tr><td class="label">File Duration:</td><td class="val" id="td-duration">-</td></tr>
          <tr><td class="label">File Path:</td><td class="val" id="td-path" style="font-size: 0.72rem; word-break: break-all;">-</td></tr>
          <tr><td class="label">External Source:</td><td class="val"><a id="yt-link" href="#" target="_blank" style="color: var(--primary); text-decoration: none;">Watch on YouTube ↗</a></td></tr>
        </table>
      </div>

      <!-- Right: Decision & Criteria Panel -->
      <div class="panel-card">
        <form id="review-form" onsubmit="event.preventDefault(); saveCurrentDecision();">
          <div class="form-group">
            <div class="form-label">
              <span>1. Validation Verdict <span style="color: var(--danger)">*</span></span>
              <small>Hotkeys: [1] Accept, [2] Reclassify, [3] Reject, [4] 2nd Review</small>
            </div>
            <div class="verdict-options">
              {verdict_options_html}
            </div>
          </div>

          <div class="form-group" style="margin-top: 6px;">
            <div class="form-label">
              <span>2. Actual Observed Sound Category</span>
              <small>Required if Reclassifying</small>
            </div>
            <select class="observed-select" id="observed-sound" name="observed_sound">
              <option value="">-- Select Observed Category --</option>
              {sound_options_html}
            </select>
          </div>

          <div class="form-group" style="margin-top: 6px;">
            <div class="form-label">
              <span>3. Acoustic Verification Checkpoints</span>
            </div>
            <div class="checkpoints-box">
              <label class="checkpoint-item">
                <input type="checkbox" id="chk-audible" name="expected_sound_audible" checked>
                <span>Target sound is clearly audible in this segment</span>
              </label>
              <label class="checkpoint-item">
                <input type="checkbox" id="chk-background" name="background_noise_present">
                <span>Heavy background sounds or music are present</span>
              </label>
              <label class="checkpoint-item">
                <input type="checkbox" id="chk-emergency" name="emergency_sound_present" onchange="toggleEmergencySelect()">
                <span>Unintended emergency sound audible (siren, scream, gunshot, glass)</span>
              </label>
              <div id="emerg-type-wrap" style="display: none; margin-left: 24px;">
                <select class="observed-select" id="emergency-sound-type" name="emergency_sound_type">
                  <option value="none">None</option>
                  <option value="siren">Emergency Vehicle Siren</option>
                  <option value="scream">Human Scream</option>
                  <option value="gunshot">Gunshot / Firearm</option>
                  <option value="glass">Glass Breaking / Shatter</option>
                </select>
              </div>
            </div>
          </div>

          <div class="form-group" style="margin-top: 6px;">
            <div class="form-label">
              <span>4. Acoustic Notes & Ambiguity Observations</span>
            </div>
            <textarea class="notes-input" id="notes" name="notes" placeholder="e.g. Sharp impact burst audible at 3.2s; no siren overlap; reverberant room..."></textarea>
          </div>

          <div class="action-bar">
            <button type="submit" class="save-btn" id="save-btn">
              <span>💾 Save Decision & Next</span>
              <small style="opacity: 0.8; font-weight: normal;">[Enter]</small>
            </button>
          </div>
        </form>
      </div>
    </div>

    <!-- Acoustic Ambiguity Guidance Accordion -->
    <details class="cheatsheet">
      <summary>🔍 Research Cheatsheet: Critical Acoustic Ambiguities & Guidelines</summary>
      <div class="ambiguities-grid">
        <div class="amb-card">
          <strong>1. Shouting vs. Scream</strong>
          Shouting maintains linguistic pitch contours and speech formants. Screams display extreme vocal tension, chaotic/turbulent noise, and register breaks. If merely loud speech, mark <em>RECLASSIFY</em> -> <code>human_shouting</code>.
        </div>
        <div class="amb-card">
          <strong>2. Fireworks vs. Gunshots</strong>
          Fireworks have delayed multi-burst crackles, whistle tails, and softer rise times. Gunshots feature immediate microsecond rise times (<2ms), sharp supersonic shockwaves, and explosive muzzle blast.
        </div>
        <div class="amb-card">
          <strong>3. Car Alarms vs. Sirens</strong>
          Car alarms use intermittent horn pulses or rhythmic dual-tone chirps. Sirens display continuous frequency sweeps (wail, yelp, hi-lo). Do not confuse high-pitched buzzers with emergency sirens.
        </div>
        <div class="amb-card">
          <strong>4. Metal Impact vs. Glass Breaking</strong>
          Glass breaking requires high-frequency brittle shatter bursts (>3 kHz) and resonant shard tinkling. Metal strikes have metallic overtones, ringing decays, or dull clangs without shard scattering.
        </div>
        <div class="amb-card">
          <strong>5. Construction Sounds vs. Emergency</strong>
          Pneumatic nailers and jackhammers have rhythmic repetition rates. Do not confuse repetitive tool clicks with gunshots, or tile hammer strikes with glass breaks.
        </div>
      </div>
    </details>
  </div>

  <script>
    const candidates = {candidates_json};
    const initialResults = {results_json};
    const resultsMap = {{ ...initialResults }};
    const reviewerId = "{reviewer_id}";

    let activeIndex = 0;
    let filteredIndices = candidates.map((_, i) => i);

    function toggleEmergencySelect() {{
      const chk = document.getElementById("chk-emergency");
      const wrap = document.getElementById("emerg-type-wrap");
      wrap.style.display = chk.checked ? "block" : "none";
    }}

    function updateProgress() {{
      const total = candidates.length;
      const reviewedKeys = Object.keys(resultsMap);
      const reviewedCount = reviewedKeys.length;
      const pct = total > 0 ? (reviewedCount / total * 100).toFixed(1) : 0;

      document.getElementById("prog-ratio").innerText = `${{reviewedCount}} / ${{total}} (${{pct}}%)`;
      document.getElementById("progress-bar").style.width = `${{pct}}%`;

      // Class breakdown
      const classTotals = {{ scream: 0, glass_breaking: 0, other: 0 }};
      const classReviewed = {{ scream: 0, glass_breaking: 0, other: 0 }};

      candidates.forEach(c => {{
        classTotals[c.target_class] = (classTotals[c.target_class] || 0) + 1;
        const key = `${{c.video_id}}_${{c.start_seconds}}_${{c.end_seconds}}`;
        if (resultsMap[key] && resultsMap[key].verdict) {{
          classReviewed[c.target_class] = (classReviewed[c.target_class] || 0) + 1;
        }}
      }});

      for (let cls of ["scream", "glass_breaking", "other"]) {{
        const tot = classTotals[cls] || 0;
        const rev = classReviewed[cls] || 0;
        const p = tot > 0 ? (rev / tot * 100).toFixed(0) : 0;
        const elemId = cls === "glass_breaking" ? "prog-glass" : `prog-${{cls}}`;
        const el = document.getElementById(elemId);
        if (el) el.innerText = `${{rev}} / ${{tot}} (${{p}}%)`;
      }}
    }}

    function applyFilter() {{
      const filter = document.getElementById("filter-mode").value;
      if (filter === "all") {{
        filteredIndices = candidates.map((_, i) => i);
      }} else if (filter === "unreviewed") {{
        filteredIndices = candidates.map((_, i) => i).filter(i => {{
          const c = candidates[i];
          const key = `${{c.video_id}}_${{c.start_seconds}}_${{c.end_seconds}}`;
          return !resultsMap[key] || !resultsMap[key].verdict;
        }});
      }} else if (filter === "reviewed") {{
        filteredIndices = candidates.map((_, i) => i).filter(i => {{
          const c = candidates[i];
          const key = `${{c.video_id}}_${{c.start_seconds}}_${{c.end_seconds}}`;
          return resultsMap[key] && resultsMap[key].verdict;
        }});
      }} else {{
        filteredIndices = candidates.map((_, i) => i).filter(i => candidates[i].target_class === filter);
      }}

      // Populate Jump Select
      const jumpSel = document.getElementById("jump-select");
      jumpSel.innerHTML = "";
      filteredIndices.forEach((cIdx, pos) => {{
        const c = candidates[cIdx];
        const key = `${{c.video_id}}_${{c.start_seconds}}_${{c.end_seconds}}`;
        const isRev = resultsMap[key] && resultsMap[key].verdict;
        const opt = document.createElement("option");
        opt.value = pos;
        opt.innerText = `${{isRev ? '✓' : '○'}} #${{c.index}} [${{c.target_class}}] ${{c.video_id.substring(0, 8)}}`;
        jumpSel.appendChild(opt);
      }});

      if (filteredIndices.length > 0) {{
        activeIndex = 0;
        renderClip();
      }} else {{
        showNotif("No samples match the selected filter.", "error");
      }}
    }}

    function jumpToSelected() {{
      const pos = parseInt(document.getElementById("jump-select").value, 10);
      if (!isNaN(pos) && pos >= 0 && pos < filteredIndices.length) {{
        activeIndex = pos;
        renderClip();
      }}
    }}

    function renderClip() {{
      if (filteredIndices.length === 0) return;
      const cIdx = filteredIndices[activeIndex];
      const clip = candidates[cIdx];
      const key = `${{clip.video_id}}_${{clip.start_seconds}}_${{clip.end_seconds}}`;
      const prevResult = resultsMap[key] || {{}};

      // Headers & text
      document.getElementById("clip-title").innerText = `Sample #${{clip.index}} of ${{candidates.length}} (Item ${{activeIndex + 1}}/${{filteredIndices.length}} in filter)`;
      document.getElementById("clip-subtitle").innerText = `Video ID: ${{clip.video_id}} (${{clip.origin}})`;

      document.getElementById("td-class").innerText = clip.target_class;
      document.getElementById("td-subdomain").innerText = clip.subdomain;
      document.getElementById("td-matched").innerText = clip.matched_label;
      document.getElementById("td-origin").innerText = clip.origin;
      document.getElementById("td-window").innerText = `${{clip.start_seconds}}s – ${{clip.end_seconds}}s`;
      document.getElementById("td-duration").innerText = `${{clip.duration_seconds}}s`;
      document.getElementById("td-path").innerText = clip.file_path;

      // YouTube Link
      const ytLink = document.getElementById("yt-link");
      ytLink.href = `https://www.youtube.com/watch?v=${{clip.video_id}}&t=${{Math.floor(clip.start_seconds)}}s`;

      // Audio Player
      const audio = document.getElementById("audio-player");
      audio.src = `/audio?path=${{encodeURIComponent(clip.file_path)}}`;

      // Status Pill & Form Reset/Populate
      const pill = document.getElementById("status-pill");
      const v = prevResult.verdict;

      // Reset radio buttons
      document.querySelectorAll('input[name="verdict"]').forEach(r => r.checked = (r.value === v));
      document.getElementById("observed-sound").value = prevResult.observed_sound || "";
      document.getElementById("chk-audible").checked = prevResult.expected_sound_audible !== undefined ? prevResult.expected_sound_audible : true;
      document.getElementById("chk-background").checked = prevResult.background_noise_present || false;
      document.getElementById("chk-emergency").checked = prevResult.emergency_sound_present || false;
      document.getElementById("emergency-sound-type").value = prevResult.emergency_sound_type || "none";
      document.getElementById("notes").value = prevResult.notes || "";
      toggleEmergencySelect();

      if (v) {{
        pill.className = `status-pill pill-${{v.toLowerCase()}}`;
        pill.innerText = v;
      }} else {{
        pill.className = "status-pill pill-pending";
        pill.innerText = "Pending";
      }}

      // Update Jump Select sync
      document.getElementById("jump-select").value = activeIndex;

      // Buttons
      document.getElementById("prev-btn").disabled = activeIndex <= 0;
      document.getElementById("next-btn").disabled = activeIndex >= filteredIndices.length - 1;
    }}

    async function saveCurrentDecision() {{
      const cIdx = filteredIndices[activeIndex];
      const clip = candidates[cIdx];
      const form = document.getElementById("review-form");
      const verdictInput = document.querySelector('input[name="verdict"]:checked');

      if (!verdictInput) {{
        showNotif("Please select a Validation Verdict before saving.", "error");
        return;
      }}

      const verdict = verdictInput.value;
      const observedSound = document.getElementById("observed-sound").value;

      if (verdict === "RECLASSIFY" && !observedSound) {{
        showNotif("Please choose the actual Observed Sound Category when reclassifying.", "error");
        return;
      }}

      const payload = {{
        video_id: clip.video_id,
        start_seconds: clip.start_seconds,
        end_seconds: clip.end_seconds,
        file_path: clip.file_path,
        target_class: clip.target_class,
        subdomain: clip.subdomain,
        matched_label: clip.matched_label,
        origin: clip.origin,
        duration_seconds: clip.duration_seconds,
        verdict: verdict,
        observed_sound: observedSound,
        expected_sound_audible: document.getElementById("chk-audible").checked,
        background_noise_present: document.getElementById("chk-background").checked,
        emergency_sound_present: document.getElementById("chk-emergency").checked,
        emergency_sound_type: document.getElementById("emergency-sound-type").value,
        notes: document.getElementById("notes").value.trim(),
        reviewer: reviewerId,
        reviewed_at: new Date().toISOString().replace('T', ' ').substring(0, 19),
      }};

      const saveBtn = document.getElementById("save-btn");
      saveBtn.disabled = true;

      try {{
        const res = await fetch("/api/save_decision", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload),
        }});
        const data = await res.json();
        if (data.status === "ok") {{
          const key = `${{clip.video_id}}_${{clip.start_seconds}}_${{clip.end_seconds}}`;
          resultsMap[key] = payload;
          showNotif(`✓ Saved decision for #${{clip.index}} [${{clip.target_class}}] (${{verdict}})`, "success");
          updateProgress();

          // Advance to next clip
          if (activeIndex < filteredIndices.length - 1) {{
            activeIndex++;
            renderClip();
          }}
        }} else {{
          showNotif(`Error: ${{data.message}}`, "error");
        }}
      }} catch (err) {{
        showNotif(`Network error saving decision: ${{err.message}}`, "error");
      }} finally {{
        saveBtn.disabled = false;
      }}
    }}

    function prevClip() {{
      if (activeIndex > 0) {{
        activeIndex--;
        renderClip();
      }}
    }}

    function nextClip() {{
      if (activeIndex < filteredIndices.length - 1) {{
        activeIndex++;
        renderClip();
      }}
    }}

    function nextUnreviewed() {{
      for (let i = activeIndex + 1; i < filteredIndices.length; i++) {{
        const c = candidates[filteredIndices[i]];
        const key = `${{c.video_id}}_${{c.start_seconds}}_${{c.end_seconds}}`;
        if (!resultsMap[key] || !resultsMap[key].verdict) {{
          activeIndex = i;
          renderClip();
          return;
        }}
      }}
      // Search from beginning
      for (let i = 0; i <= activeIndex; i++) {{
        const c = candidates[filteredIndices[i]];
        const key = `${{c.video_id}}_${{c.start_seconds}}_${{c.end_seconds}}`;
        if (!resultsMap[key] || !resultsMap[key].verdict) {{
          activeIndex = i;
          renderClip();
          return;
        }}
      }}
      showNotif("All samples in this filter have been reviewed! 🎉", "success");
    }}

    function showNotif(msg, type) {{
      const bar = document.getElementById("notif-bar");
      bar.innerText = msg;
      bar.className = `notification notif-${{type}}`;
      bar.style.display = "block";
      setTimeout(() => {{ bar.style.display = "none"; }}, 4000);
    }}

    // Keyboard Shortcuts
    window.addEventListener("keydown", (e) => {{
      // Ignore if typing in text inputs or textareas
      if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) {{
        if (e.key === "Enter" && e.ctrlKey) {{
          saveCurrentDecision();
        }}
        return;
      }}

      if (e.code === "Space") {{
        e.preventDefault();
        const a = document.getElementById("audio-player");
        if (a.paused) a.play(); else a.pause();
      }} else if (e.key === "1") {{
        const r = document.querySelector('input[name="verdict"][value="ACCEPT"]');
        if (r) r.checked = true;
      }} else if (e.key === "2") {{
        const r = document.querySelector('input[name="verdict"][value="RECLASSIFY"]');
        if (r) r.checked = true;
      }} else if (e.key === "3") {{
        const r = document.querySelector('input[name="verdict"][value="REJECT"]');
        if (r) r.checked = true;
      }} else if (e.key === "4") {{
        const r = document.querySelector('input[name="verdict"][value="NEEDS_SECOND_REVIEW"]');
        if (r) r.checked = true;
      }} else if (e.key === "ArrowLeft") {{
        prevClip();
      }} else if (e.key === "ArrowRight") {{
        nextClip();
      }} else if (e.key === "Enter") {{
        saveCurrentDecision();
      }}
    }});

    // Initialize
    applyFilter();
    updateProgress();
  </script>
</body>
</html>
"""
    return html


class ContentReviewHandler(BaseHTTPRequestHandler):
    """HTTP handler serving review UI, range audio streaming, and atomic JSON API endpoints."""

    manifest_path = DEFAULT_MANIFEST
    results_path = DEFAULT_RESULTS
    reviewer_id = "reviewer_1"

    def log_message(self, format, *args):
        # Clean console log
        sys.stderr.write(f"[{self.log_date_time_string()}] {args[0]} {args[1]}\n")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path in ["/", "/index.html"]:
            candidates = load_review_candidates(self.manifest_path)
            results_map = load_existing_review_results(self.results_path)
            html_content = generate_review_html(candidates, results_map, self.reviewer_id).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_content)))
            self.end_headers()
            self.wfile.write(html_content)
            return

        elif parsed.path == "/api/progress":
            candidates = load_review_candidates(self.manifest_path)
            results_map = load_existing_review_results(self.results_path)
            progress = compute_review_progress(candidates, results_map)
            resp = json.dumps(progress).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        elif parsed.path == "/audio":
            query = urllib.parse.parse_qs(parsed.query)
            rel_path = query.get("path", [""])[0]

            # Security sanitization: must resolve strictly inside ROOT / data / raw
            try:
                candidate_path = (ROOT / rel_path).resolve()
                allowed_dir = (ROOT / "data" / "raw").resolve()
                if not candidate_path.is_relative_to(allowed_dir) or not candidate_path.is_file():
                    self.send_error(403, "Access denied: audio path outside permitted directory")
                    return
            except Exception:
                self.send_error(400, "Invalid file path format")
                return

            if not candidate_path.exists():
                self.send_error(404, "Audio file not found")
                return

            try:
                with open(candidate_path, "rb") as f:
                    audio_bytes = f.read()

                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(audio_bytes)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(audio_bytes)
            except Exception as e:
                self.send_error(500, f"Error streaming audio: {str(e)}")
            return

        self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/save_decision":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                decision = json.loads(body.decode("utf-8"))

                # Validate essential fields
                if not decision.get("video_id") or not decision.get("verdict"):
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": "Missing video_id or verdict"}).encode("utf-8"))
                    return

                success = save_review_decision(self.results_path, decision, allow_overwrite=True)

                self.send_response(200 if success else 409)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "ok" if success else "conflict",
                    "message": "Saved successfully" if success else "Decision already exists and overwrite not allowed",
                }).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return

        elif parsed.path == "/api/bulk_save":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                decisions = json.loads(body.decode("utf-8"))

                saved_count = save_bulk_decisions(self.results_path, decisions, allow_overwrite=True)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "ok",
                    "message": f"Successfully saved {saved_count} decisions",
                }).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")


def run_review_server(manifest_path: Path, results_path: Path, port: int = 8050, open_browser: bool = True, reviewer_id: str = "reviewer_1"):
    """Initialize and run local audio content review web server."""
    candidates = load_review_candidates(manifest_path)
    results_map = load_existing_review_results(results_path)
    progress = compute_review_progress(candidates, results_map)

    print("=" * 80)
    print("  AUDIO CONTENT VALIDATION & REVIEW INTERFACE")
    print("=" * 80)
    print(f"Manifest File  : {manifest_path.relative_to(ROOT) if manifest_path.is_relative_to(ROOT) else manifest_path}")
    print(f"Results File   : {results_path.relative_to(ROOT) if results_path.is_relative_to(ROOT) else results_path}")
    print(f"Reviewer ID    : {reviewer_id}")
    print(f"Total Clips    : {progress['total']}")
    print(f"Reviewed So Far: {progress['reviewed']} / {progress['total']} ({progress['percent_complete']}%)")
    for cls, st in progress['by_class'].items():
        print(f"  - {cls:<16}: {st['reviewed']}/{st['total']} ({st['percent']}%)")
    print(f"Local Server   : http://localhost:{port}")
    print("Press Ctrl+C in terminal to stop server.\n")

    # Configure handler class attributes
    ContentReviewHandler.manifest_path = manifest_path
    ContentReviewHandler.results_path = results_path
    ContentReviewHandler.reviewer_id = reviewer_id

    # Port binding retry loop
    server = None
    for p in range(port, port + 10):
        try:
            server = HTTPServer(("127.0.0.1", p), ContentReviewHandler)
            port = p
            break
        except OSError:
            continue

    if not server:
        print(f"[ERROR] Could not bind to port {port} or the next 10 ports.", file=sys.stderr)
        sys.exit(1)

    url = f"http://localhost:{port}"
    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview server stopped gracefully.")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Local Audio Content Review Tool for Targeted Dataset Expansion")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST), help="Path to content validation manifest CSV")
    parser.add_argument("--results-file", type=str, default=str(DEFAULT_RESULTS), help="Path to save review decisions CSV")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind review server (default: 8050)")
    parser.add_argument("--no-browser", dest="open_browser", action="store_false", help="Do not automatically launch web browser")
    parser.add_argument("--reviewer", type=str, default="reviewer_1", help="Identifier of the reviewer recording verdicts")
    parser.add_argument("--mode", choices=["expansion", "pilot"], default="expansion", help="Review mode (expansion or legacy pilot)")

    args = parser.parse_args()

    if args.mode == "pilot":
        manifest_path = PILOT_RESULTS_CSV
        results_path = PILOT_REVIEW_CSV
    else:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path

        results_path = Path(args.results_file)
        if not results_path.is_absolute():
            results_path = ROOT / results_path

    run_review_server(
        manifest_path=manifest_path,
        results_path=results_path,
        port=args.port,
        open_browser=args.open_browser,
        reviewer_id=args.reviewer,
    )


if __name__ == "__main__":
    main()
