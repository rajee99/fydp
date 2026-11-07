# Targeted Dataset Expansion Analysis Report
**Model Baseline:** Frozen PANNs CNN14 (`models/experiments/panns_cnn14/best_model.pt`)  
**Evaluation Scope:** 519 Valid Held-Out AudioSet Evaluation Clips  
**Overall Evaluation Performance:** Accuracy: 84.78% | Macro Precision: 0.8044 | Macro Recall: 0.8460 | Macro F1: 0.8195  
**Date:** October 30, 2026  

---

## 1. Executive Summary & Problem Formulation

The frozen PANNs CNN14 model established a strong baseline on five-class emergency audio classification, improving Macro F1 from the initial CNN baseline of **0.4430** to **0.8195** on independent evaluation. However, detailed examination of the 79 misclassifications reveals that model errors are not evenly distributed; they cluster sharply around specific acoustic confusers and severe training data deficits:

1. **Weakest Classes in Evaluation:**
   - `glass_breaking`: F1 **0.7273** (Precision 0.7033, Recall 0.7529, Support 85)
   - `other`: F1 **0.7317** (Precision 0.7826, Recall 0.6870, Support 131)
   - `scream`: F1 **0.7397** (Precision 0.6279, Recall 0.9000, Support 30)
   *(In contrast, `gunshot` achieved F1 **0.9266** and `siren` achieved F1 **0.9723**).*

2. **Primary Root Cause:**
   The training dataset suffers from both **acute class volume imbalance** (`scream` has only 37 audio clips total; `glass_breaking` has 96) and **severe semantic omission in the negative (`other`) class**. The current 120 `other` training clips contain zero examples of shouting, door slams, power tools, fireworks, or alarm beeps, and only 10 clips of household impacts (cutlery/dishes/knocks). Consequently, the model defaults to false alarms whenever sharp impulse clatter or high-frequency vocal dynamics occur.

---

## 2. Current Inventory & Distribution Breakdown

### 2.1 Training and Validation Inventory (Pre-Expansion)
The current training inventory consists of 527 strictly validated WAV files (16 kHz, mono, 16-bit PCM).

| Class ID | Target Class | Train Split | Val Split | Total Inventory | Class Share (%) |
|:---:|:---|:---:|:---:|:---:|:---:|
| 0 | `scream` | 30 | 7 | **37** | 7.02% |
| 1 | `glass_breaking` | 77 | 19 | **96** | 18.22% |
| 2 | `gunshot` | 92 | 23 | **115** | 21.82% |
| 3 | `siren` | 127 | 32 | **159** | 30.17% |
| 4 | `other` | 96 | 24 | **120** | 22.77% |
| **Total** | **All Classes** | **422** | **105** | **527** | **100.0%** |

### 2.2 Subdomain Breakdown for `other` Training Data
The 120 `other` clips were gathered across six synthetic subdomains (20 clips each):

```
+-----------------------------------------------------------------------------------------+
| Subdomain     | Count | Matched Labels & MIDs                                           |
+---------------+-------+-----------------------------------------------------------------+
| Speech        |   20  | Speech (19), Conversation (1) [Zero shouting/screaming variants]|
| Music         |   20  | Music (18), Singing (2)                                         |
| Animals       |   20  | Bird (9), Dog (5), Bark (3), Cat (3)                            |
| Household     |   20  | Cutlery/silverware (7), Clock (6), Cupboard (3), Dishes (2)...  |
| Traffic       |   20  | Vehicle (20) [Zero car alarms, sirens, or train whistles]       |
| Environmental |   20  | Water (6), Rain (6), Wind (3), Stream (3), Waves (2)            |
+-----------------------------------------------------------------------------------------+
```

> [!WARNING]
> In the entire 527-sample training set, **`scream` accounts for only 7.0% of data**, and **hard-negative metallic impacts (`Cutlery` + `Dishes`) total just 9 clips**. This structural deficit explains the high false-alarm rates.

---

## 3. Empirical Error Mode Analysis (Held-Out Evaluation)

Evaluation of all 519 independent evaluation audio files against the frozen PANNs CNN14 checkpoint produced exactly **79 errors**. A complete cross-referencing with AudioSet metadata exposes the physical acoustic mechanisms driving these misclassifications:

```
                                PREDICTED CLASS
                 scream   glass_breaking   gunshot   siren   other   | Total
GT: scream         27           0             0        0       3     |   30
GT: glass_break     7          64             4        0      10     |   85
GT: gunshot         0           1           101        0       7     |  109
GT: siren           0           1             0      158       5     |  164
GT: other           9          25             4        3      90     |  131
---------------------------------------------------------------------+------
Predicted Total    43          91           109      161     115     |  519
```

### 3.1 Error Mode 1: False Glass Breaking from Metallic & Transient Clatter (`other` → `glass_breaking`)
- **Occurrences:** **25 clips** (representing 31.6% of all evaluation errors).
- **Contributing AudioSet Labels:**
  - `Cutlery, silverware` (`/m/023pjk`): **11 clips**
  - `Knock` (`/m/07r4wb8`): **3 clips**
  - `Door` (`/m/02dgv`): **3 clips**
  - `Speech` (`/m/09x0r`): **3 clips**
  - `Dishes, pots, and pans` (`/m/04brg2`): **2 clips**
  - `Water` (`/m/0838f`): **2 clips**
  - `Music` (`/m/04rlf`): **1 clip**
- **Acoustic Mechanism:** Cutlery drops, ceramic dish contact, and sharp door knocks produce high-frequency, non-harmonic, decaying transients (>2 kHz to 8 kHz) that share the temporal-spectral envelope of breaking glass. Because the model was exposed to only 9 training examples of cutlery and dishes, it learns that high-frequency burst transients denote glass breakage.

### 3.2 Error Mode 2: Non-Destructive Glass Missed as Background (`glass_breaking` → `other`)
- **Occurrences:** **10 clips**.
- **Contributing AudioSet Labels:**
  - `Glass` (`/m/039jq`): **10 clips** (100% of misses).
- **Acoustic Mechanism:** The broad AudioSet MID `/m/039jq` ("Glass") contains non-destructive handling (light clinking of drinking glasses, stirring with a spoon, setting a bottle on a table). These lack the high-energy dispersion and broadband impulse crash of window or structural shatter (`/m/07rn7sz`). The model correctly recognizes them as non-destructive, but penalizes recall against the coarse label `glass_breaking`.

### 3.3 Error Mode 3: High-Frequency Phonation & Resonant Screams (`other` → `scream` & `glass_breaking` → `scream`)
- **Occurrences:** **16 clips** (9 `other` → `scream`, 7 `glass_breaking` → `scream`).
- **Contributing AudioSet Labels:**
  - `other` false screams: `Speech` (3), `Singing` (2), `Bird vocalizations` (2), `Cat` (1), `Knock` (1).
  - `glass_breaking` false screams: `Shatter` (`/m/07rn7sz`, 5 clips), `Glass` (2 clips).
- **Acoustic Mechanism:** 
  1. Melodic singing vibrato and emphasized high-register speech phonemes exhibit strong harmonic formant tracks between 1 kHz and 4 kHz, mimicking the open vocal-tract resonant frequency of human distress screams.
  2. Resonant ringing modes of fractured glass and crystalline shatter produce sustained narrowband ringing frequencies that overlap with scream harmonic trajectories.
  3. The training pipeline excluded shouting, crying, and yelling MIDs from `other`, leaving the classifier unable to establish the boundary between emotional speech and genuine screaming.

### 3.4 Error Mode 4: Ambient Attenuation & Background Drowning (`scream` → `other`, `gunshot` → `other`, `siren` → `other`)
- **Occurrences:** **15 clips** (3 `scream`, 7 `gunshot`, 5 `siren`).
- **Acoustic Mechanism:** Low signal-to-noise ratio (SNR). Distant gunshots muffled by walls, sirens at the limit of audibility mixed with engine rumblings, or screams recorded far from the microphone lack sufficient salient feature activations, defaulting to `other`.

---

## 4. AudioSet Candidate Availability & Source Selection Audit

An audit of the primary AudioSet balanced catalog (`data/raw/balanced_train_segments.csv`, 22,160 segments) was conducted against the **1,482 forbidden video IDs** already assigned to the project's training (740) and held-out evaluation (742) splits.

### 4.1 Candidate Availability in `balanced_train_segments.csv`

```
+----------------------------------------+-------------------+--------------------+------------------------+
| Acoustic Category Group                | Total in Balanced | Fresh Candidates   | Unique Fresh Video IDs |
+----------------------------------------+-------------------+--------------------+------------------------+
| 1. Scream                              |                59 |                  0 |                      0 |
|    - Screaming (/m/03qc9zr)            |                59 |                  0 |                      0 |
+----------------------------------------+-------------------+--------------------+------------------------+
| 2. Glass Breaking                      |               240 |                119 |                    119 |
|    - Shatter (/m/07rn7sz)              |                60 |                  0 |                      0 |
|    - Glass (/m/039jq)                  |                60 |                  0 |                      0 |
|    - Smash, crash (/m/07pjjrj)         |                60 |                 60 |                     60 |
|    - Crack (/m/07qs1cx)                |                60 |                 59 |                     59 |
+----------------------------------------+-------------------+--------------------+------------------------+
| 3. Door Slams                          |               242 |                236 |                    200 |
|    - Slam (/m/07rjzl8)                 |                61 |                 60 |                     60 |
|    - Door (/m/02dgv)                   |                61 |                 58 |                     58 |
|    - Sliding door (/m/02y_763)         |                60 |                 58 |                     58 |
|    - Cupboard (/m/0642b4)              |                60 |                 60 |                     60 |
+----------------------------------------+-------------------+--------------------+------------------------+
| 4. Metal Impacts & Dropped Utensils    |               306 |                259 |                    228 |
|    - Cutlery, silverware (/m/023pjk)   |                60 |                 42 |                     42 |
|    - Dishes, pots, pans (/m/04brg2)    |                60 |                 41 |                     41 |
|    - Knock (/m/07r4wb8)                |                60 |                 50 |                     50 |
|    - Bell (/m/0395lw)                  |                60 |                 60 |                     60 |
|    - Chime (/m/0f8s22)                 |                66 |                 66 |                     66 |
+----------------------------------------+-------------------+--------------------+------------------------+
| 5. Construction & Tool Impacts         |               476 |                475 |                    323 |
|    - Tools (/m/07k1x)                  |               101 |                100 |                    100 |
|    - Power tool (/m/0_ksk)             |                75 |                 75 |                     75 |
|    - Hammer (/m/03l9g)                 |                60 |                 60 |                     60 |
|    - Drill (/m/01d380)                 |                60 |                 60 |                     60 |
|    - Sawing (/m/01b82r)                |                60 |                 60 |                     60 |
|    - Chainsaw (/m/01j4z9)              |                60 |                 60 |                     60 |
|    - Jackhammer (/m/03p19w)            |                60 |                 60 |                     60 |
+----------------------------------------+-------------------+--------------------+------------------------+
| 6. Shouting & Emotional Speech         |               419 |                418 |                    398 |
|    - Shout (/m/07p6fty)                |                60 |                 60 |                     60 |
|    - Yell (/m/07sr1lc)                 |                60 |                 60 |                     60 |
|    - Children shouting (/t/dd00135)    |                60 |                 60 |                     60 |
|    - Crying, sobbing (/m/0463cq4)      |                60 |                 59 |                     59 |
|    - Baby cry (/t/dd00002)             |                60 |                 60 |                     60 |
|    - Wail, moan (/m/07qw_06)           |                59 |                 59 |                     59 |
|    - Bellow (/m/07q4ntr)               |                60 |                 60 |                     60 |
+----------------------------------------+-------------------+--------------------+------------------------+
| 7. Whistles & High-Pitched Sounds      |               313 |                309 |                    290 |
|    - Steam whistle (/m/06hck5)         |                70 |                 69 |                     69 |
|    - Whistle (/m/0l156k)               |                61 |                 61 |                     61 |
|    - Train whistle (/m/04zmvq)         |                60 |                 60 |                     60 |
|    - Beep, bleep (/m/02fs_r)           |                62 |                 60 |                     60 |
|    - Buzzer (/m/030rvx)                |                60 |                 59 |                     59 |
+----------------------------------------+-------------------+--------------------+------------------------+
| 8. Fireworks & Firecrackers            |               240 |                238 |                    220 |
|    - Burst, pop (/m/07qsvvw)           |                60 |                 60 |                     60 |
|    - Firecracker (/g/122z_qxw)         |                60 |                 60 |                     60 |
|    - Bang (/m/07pws3f)                 |                60 |                 59 |                     59 |
|    - Fireworks (/m/0g6b5)              |                60 |                 59 |                     59 |
+----------------------------------------+-------------------+--------------------+------------------------+
| 9. Car Alarms & Detectors              |               422 |                400 |                    365 |
|    - Alarm clock (/m/046dlr)           |                62 |                 62 |                     62 |
|    - Smoke alarm (/m/01y3hg)           |                60 |                 60 |                     60 |
|    - Doorbell (/m/03wwcy)              |                60 |                 60 |                     60 |
|    - Fire alarm (/m/0c3f7m)            |                60 |                 60 |                     60 |
|    - Reversing beeps (/m/02rhddq)      |                60 |                 58 |                     58 |
|    - Alarm (/m/07pp_mv)                |                60 |                 56 |                     56 |
|    - Car alarm (/m/02mfyn)             |                60 |                 44 |                     44 |
+----------------------------------------+-------------------+--------------------+------------------------+
```

### 4.2 Source Exhaustion & Sourcing Strategy
1. **Critical Finding on `scream` and `glass_breaking`:**
   In `balanced_train_segments.csv`, all 59 segments of `/m/03qc9zr` (Screaming), all 60 segments of `/m/07rn7sz` (Shatter), and all 60 segments of `/m/039jq` (Glass) were consumed during previous download batches.
   - **Strategy for Scream:** Must tap the official AudioSet **unbalanced training set** (`unbalanced_train_segments.csv`), which contains **2,154** candidate segments for `/m/03qc9zr`, or supplement with curated open benchmarks (e.g., FSD50K, VGGSound).
   - **Strategy for Glass Breaking:** Utilize the 119 fresh balanced candidates under `/m/07pjjrj` (`Smash, crash`) and `/m/07qs1cx` (`Crack`), while sourcing true structural glass shatter from the AudioSet unbalanced set (**1,418** segments under `/m/07rn7sz`).
2. **Abundant Candidates for Hard-Negative `other`:**
   Over **2,000 fresh videos** are immediately available in `balanced_train_segments.csv` across the 7 critical confuser categories. None of these video IDs overlap with existing train or eval sets.

---

## 5. Recommended Expansion Quotas & Target Allocations

To eliminate false alarms, balance class representations, and reach an operational training set of approximately **1,000 clips**, the following targeted expansion is recommended:

```
+------------------+---------------+--------------------+------------------+------------------------------------+
| Target Class     | Current Train | Recommended Target | Target Inventory | Priority Acoustic Focus            |
+------------------+---------------+--------------------+------------------+------------------------------------+
| scream           |            37 |          +110      |       147        | Gender variety, high/low distance, |
|                  |               |                    |                  | short bursts, ambient noise mix    |
+------------------+---------------+--------------------+------------------+------------------------------------+
| glass_breaking   |            96 |           +85      |       181        | Structural shatter, bottle/plate   |
|                  |               |                    |                  | destruction, sharp impact crashes  |
+------------------+---------------+--------------------+------------------+------------------------------------+
| gunshot          |           115 |           +35      |       150        | Distant gunfire, echoing shots,    |
|                  |               |                    |                  | automatic bursts in reverb         |
+------------------+---------------+--------------------+------------------+------------------------------------+
| siren            |           159 |             0      |       159        | Sufficient representation          |
+------------------+---------------+--------------------+------------------+------------------------------------+
| other (Hard Neg) |           120 |          +240      |       360        | Hard negative confusers (detailed) |
+------------------+---------------+--------------------+------------------+------------------------------------+
| Total            |           527 |          +470      |       997        | Balanced 5-class production corpus |
+------------------+---------------+--------------------+------------------+------------------------------------+
```

### 5.1 Hard-Negative `other` Allocation Plan (+240 Clips)

```
+---------------------------------------+---------------+------------------------------------------------------+
| Subdomain Category                    | Quota Target  | Target AudioSet Labels                               |
+---------------------------------------+---------------+------------------------------------------------------+
| Metal Impacts & Dropped Utensils      |   +45 clips   | Cutlery/silverware (20), Dishes/pots/pans (15),      |
|                                       |               | Knock (10) [Directly targets 25 eval errors]         |
+---------------------------------------+---------------+------------------------------------------------------+
| Human Shouting & Emotional Speech     |   +45 clips   | Shout (15), Yell (15), Children shouting (10),       |
|                                       |               | Crying/sobbing (5) [Directly targets 9 eval errors]  |
+---------------------------------------+---------------+------------------------------------------------------+
| Door Slams & Heavy Closures           |   +35 clips   | Slam (15), Door (10), Sliding door (5), Cupboard (5) |
+---------------------------------------+---------------+------------------------------------------------------+
| Car Alarms & Smoke Detectors          |   +35 clips   | Smoke alarm (10), Fire alarm (10), Car alarm (10),   |
|                                       |               | Reversing beeps (5)                                  |
+---------------------------------------+---------------+------------------------------------------------------+
| Fireworks & Firecrackers              |   +30 clips   | Firecracker (10), Fireworks (10), Burst/pop (10)     |
+---------------------------------------+---------------+------------------------------------------------------+
| Construction & Tool Impacts           |   +30 clips   | Hammer (10), Power tool (10), Drill (5), Sawing (5)  |
+---------------------------------------+---------------+------------------------------------------------------+
| Whistles & High-Pitched Beeps         |   +20 clips   | Whistle (5), Steam whistle (5), Beep/buzzer (10)     |
+---------------------------------------+---------------+------------------------------------------------------+
| Total Hard Negative Additions         |  +240 clips   | Sourced exclusively from balanced_train_segments.csv |
+---------------------------------------+---------------+------------------------------------------------------+
```

---

## 6. Proposed Collection Manifest Schema & Governance

To prevent data contamination, ensure split isolation, and provide acoustic auditability, the expansion collection must adhere to a standardized manifest format.

### 6.1 Manifest Column Specifications

```csv
video_id,start_seconds,end_seconds,target_class,subdomain,acoustic_category,matched_label,audio_set_mid,source_catalog,signal_attributes
```

1. **`video_id`** *(string, 11 chars)*: Unique YouTube identifier.
2. **`start_seconds`** *(float)*: Segment start offset in seconds.
3. **`end_seconds`** *(float)*: Segment end offset in seconds.
4. **`target_class`** *(enum)*: One of `scream`, `glass_breaking`, `gunshot`, `siren`, `other`.
5. **`subdomain`** *(string)*: High-level category (e.g., `emergency`, `metal_impacts`, `emotional_speech`, `closures`, `alarms`, `pyrotechnics`, `construction`).
6. **`acoustic_category`** *(string)*: Detailed acoustic subgroup (e.g., `cutlery_silverware`, `shouting_adult`, `door_slam`, `power_drill`).
7. **`matched_label`** *(string)*: Ontology display name from AudioSet class map.
8. **`audio_set_mid`** *(string)*: Ontology machine ID (e.g., `/m/023pjk`).
9. **`source_catalog`** *(enum)*: `balanced_train` or `unbalanced_train`.
10. **`signal_attributes`** *(string)*: Metadata flags on acoustic properties (e.g., `near_field;high_snr`, `distant;reverberant`, `impulsive`, `continuous`).

### 6.2 Pre-Ingestion Integrity Rules
- **Cross-Split Isolation:** Mandatory automated check rejecting any `video_id` present in `data/processed/eval_manifest.csv` or existing training audio.
- **Audio Validation Standards:** 16,000 Hz sample rate, single channel (mono), 16-bit PCM WAV container, strictly 10.0-second clip duration (±0.5s tolerance).
- **Ambiguity Filter:** Segments carrying overlapping emergency labels (e.g., a clip labeled both `/m/03qc9zr` Scream and `/m/07rn7sz` Shatter) must either be reviewed manually or discarded to avoid multi-label training ambiguity.
