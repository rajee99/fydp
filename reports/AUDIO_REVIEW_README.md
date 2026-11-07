# Audio Content Validation & Review Workflow Guide

**Tool Script:** [`src/review_tool.py`](file:///d:/ml/fydp/MyNet/src/review_tool.py)  
**Input Review Manifest:** [`data/processed/targeted_expansion_content_validation.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_content_validation.csv) (400 items)  
**Output Results File:** [`data/processed/targeted_expansion_review_results.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_review_results.csv)  
**Audio Directory:** [`data/raw/targeted_expansion/`](file:///d:/ml/fydp/MyNet/data/raw/targeted_expansion/)  

---

## 1. Quick Start

Launch the local review interface using standard Python (zero external dependencies required):

```powershell
python src/review_tool.py
```

This will start a local HTTP server and automatically open the modern review dashboard in your default browser at:
`http://localhost:8050`

### Optional Command-Line Arguments

| Flag | Default | Description |
|---|---|---|
| `--port <int>` | `8050` | Port to bind local server (automatically tries next 10 if busy). |
| `--no-browser` | `False` | Run server without automatically launching browser window. |
| `--reviewer <id>` | `reviewer_1` | Identifier or name of the reviewer for audit tracking. |
| `--manifest <path>` | `targeted_expansion_content_validation.csv` | Custom input validation CSV. |
| `--results-file <path>` | `targeted_expansion_review_results.csv` | Custom destination CSV for saved verdicts. |
| `--mode pilot` | `expansion` | Switch to legacy pilot review mode. |

---

## 2. Review Workflow & Interface Features

The tool is engineered specifically for single-clip focused listening:

1. **Dashboard & Real-Time Progress Bar:**
   - Displays overall completion percentage ($X / 400$).
   - Displays real-time breakdown across all 3 classes:
     - `Scream`: $N / 97$
     - `Glass Breaking`: $N / 81$
     - `Other`: $N / 222$
2. **Audio Playback:**
   - Native HTML5 audio player streaming directly from `data/raw/targeted_expansion/`.
   - Supports range seeking, loop, and spacebar play/pause.
3. **Comprehensive Metadata Inspection:**
   - Target class and acoustic subdomain badge.
   - Candidate origin (`P1_PRIMARY` or `P2_BACKUP`).
   - Matched AudioSet label and original YouTube timestamp window.
   - One-click link to original YouTube video at the exact start timestamp for external context checks.
4. **Validation Decisions (Required):**
   - **`ACCEPT`**: Acoustic content clearly and unambiguously matches the target class.
   - **`RECLASSIFY`**: Clip contains an audible valid sound, but belongs to another class (e.g. human shouting instead of scream).
   - **`REJECT`**: Clip is corrupt, empty, inaudible, or completely wrong.
   - **`NEEDS_SECOND_REVIEW`**: Ambiguous, high-noise, or borderline recording requiring team discussion.
5. **Observed Sound Dropdown:**
   - Required when reclassifying; tracks whether the sound is `human_shouting`, `fireworks`, `metal_impact`, `car_alarm_buzzer`, etc.
6. **Acoustic Checkpoints:**
   - `[x] Target sound is clearly audible`
   - `[ ] Heavy background sounds or music are present`
   - `[ ] Unintended emergency sound audible` (triggers sub-selector for `scream`, `siren`, `gunshot`, `glass`).
7. **Acoustic Notes:**
   - Free-form observations (e.g. "Echoey hall; distinct shatter burst at 3.2s; no background sirens").

---

## 3. Keyboard Shortcuts

For rapid, ergonomic reviewing without touching the mouse:

| Key | Action |
|---|---|
| **`Space`** | Play / Pause audio playback |
| **`1`** | Select **`ACCEPT`** |
| **`2`** | Select **`RECLASSIFY`** |
| **`3`** | Select **`REJECT`** |
| **`4`** | Select **`NEEDS_SECOND_REVIEW`** |
| **`Enter`** or **`Ctrl+Enter`** | Save decision and auto-advance to next sample |
| **`Left Arrow` (`◄`)** | Navigate to previous clip |
| **`Right Arrow` (`►`)** | Navigate to next clip |

---

## 4. Research Guidelines: Resolving Acoustic Ambiguities

Do not rely on the original AudioSet label. Evaluate only the actual acoustic audio:

### 1. Human Shouting versus Screaming
- **Shouting (`human_shouting`):** Maintains linguistic pitch contours, distinct vowel formants, and periodic vocal fold vibration. The speaker is speaking loudly or aggressively with intelligible words.
- **Screaming (`scream`):** Features chaotic/turbulent noise, extreme vocal effort, high fundamental frequency ($F_0 > 500\text{ Hz}$), register breaks, and startle/distress timbre.
- **Rule:** If the vocalization is loud yelling without scream timbre, mark **`RECLASSIFY` $\rightarrow$ `human_shouting`**.

### 2. Fireworks versus Gunshots
- **Fireworks (`fireworks`):** Gradual rise time, whoosh sounds, multi-burst crackles, whistle tails, softer low-end resonance.
- **Gunshots (`gunshot`):** Instantaneous rise time ($<2\text{ ms}$), supersonic crack, explosive muzzle blast, sharp reverberation tail.
- **Rule:** If a candidate contains celebratory firecrackers or whistling rockets, mark **`RECLASSIFY` $\rightarrow$ `fireworks`**.

### 3. Car Alarms versus Sirens
- **Car Alarms / Buzzers (`car_alarm_buzzer`):** Intermittent horn honks, pulsed repetitive beeps, stationary electronic buzzers.
- **Emergency Sirens (`siren`):** Continuous frequency-modulated pitch sweeps (wail, yelp, two-tone hi-lo).
- **Rule:** Car horns and buzzers are valid **`other`** confusers, but must **not** be confused with sirens.

### 4. Metal Impacts versus Glass Breaking
- **Metal Impacts (`metal_impact`):** Metallic resonance, harmonic ringing overtones, dull solid clangs.
- **Glass Breaking (`glass_breaking`):** High-frequency brittle burst ($>3\text{ kHz}$) followed by secondary scattering and tinkling of fragments.
- **Rule:** Solid pan thuds or metal gate clangs belong in **`other`**, not glass breaking.

### 5. Construction Sounds versus Emergency Events
- **Construction (`construction`):** Jackhammers and pneumatic nailers display strict rhythmic periodicity. Hammer strikes lack the brittle shatter of glass or the explosive pressure wave of firearms.

---

## 5. Resuming and Data Persistence

- All decisions are recorded atomically to [`data/processed/targeted_expansion_review_results.csv`](file:///d:/ml/fydp/MyNet/data/processed/targeted_expansion_review_results.csv).
- When reopening the tool, prior verdicts are loaded immediately:
  - Reviewed items show green/colored status pills.
  - The "Next Unreviewed" button automatically jumps directly to the first unreviewed clip.
- **Safety Guarantee:** Original manifests, training audio, and baseline model weights are strictly untouched.
