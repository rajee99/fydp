"""Central Class Configuration for Emergency Audio Classification.

This module is the single source of truth for target class definitions,
integer ID mappings, and label associations across the project.

Final target classes (exactly 5):
  0: scream
  1: glass_breaking
  2: gunshot
  3: siren
  4: other

The 'other' class represents all non-target acoustic events, including
ordinary background noise, speech, music, traffic, wind, and ambient room/environmental sound.
"""

from typing import Dict, List, Set

# Exactly five target classes in canonical order
CLASSES: List[str] = [
    "scream",
    "glass_breaking",
    "gunshot",
    "siren",
    "other",
]

# Strict 0-indexed integer ID mapping
CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASSES)}

# Reverse ID to class name mapping
ID_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASSES)}

NUM_CLASSES: int = len(CLASSES)

# Explicit separation between emergency targets and non-target sounds
EMERGENCY_CLASSES: List[str] = [
    "scream",
    "glass_breaking",
    "gunshot",
    "siren",
]

# Single unified non-target class name
NON_TARGET_CLASS: str = "other"

# AudioSet YAMNet display name mapping for emergency target events
EMERGENCY_TARGET_LABELS: Dict[str, Set[str]] = {
    "scream": {"Screaming"},
    "glass_breaking": {"Glass", "Shatter"},
    "gunshot": {"Gunshot, gunfire"},
    "siren": {
        "Siren",
        "Ambulance (siren)",
        "Police car (siren)",
        "Civil defense siren",
        "Fire engine, fire truck (siren)",
    },
}

# Exclusion list of AudioSet MIDs associated with emergency sound events.
# Non-target audio candidates for 'other' must be verified for the absence
# of these events before being assigned to the 'other' class.
EMERGENCY_MIDS: Dict[str, str] = {
    "/m/03qc9zr": "Screaming",
    "/m/07rn7sz": "Shatter",
    "/m/039jq": "Glass",
    "/m/032s66": "Gunshot, gunfire",
    "/m/04zjc": "Machine gun",
    "/m/02z32qm": "Fusillade",
    "/m/0_1c": "Artillery fire",
    "/m/073cg4": "Cap gun",
    "/m/012n7d": "Ambulance (siren)",
    "/m/03kmc9": "Siren",
    "/m/04qvtq": "Police car (siren)",
    "/m/012ndj": "Fire engine, fire truck (siren)",
    "/m/0dgbq": "Civil defense siren",
    "/m/02mfyn": "Car alarm",
    "/m/01y3hg": "Smoke detector, smoke alarm",
    "/m/0c3f7m": "Fire alarm",
    "/m/07pp_mv": "Alarm",
    "/m/014zdl": "Explosion",
    "/m/0g6b5": "Fireworks",
    "/g/122z_qxw": "Firecracker",
    "/m/07pjjrj": "Smash, crash",
}

# Ambiguous acoustic confusers excluded from 'other' (e.g. shouting, yelling, crying, wailing, bang)
AMBIGUOUS_MIDS: Dict[str, str] = {
    "/m/07p6fty": "Shout",
    "/m/07sr1lc": "Yell",
    "/t/dd00135": "Children shouting",
    "/m/0463cq4": "Crying, sobbing",
    "/t/dd00002": "Baby cry, infant cry",
    "/m/07qw_06": "Wail, moan",
    "/m/07s2xch": "Groan",
    "/m/07r4k75": "Grunt",
    "/m/07pws3f": "Bang",
    "/m/07qsvvw": "Burst, pop",
    "/m/07qf0zm": "Howl",
}

# Audited exact AudioSet MIDs for 6 diverse non-emergency sub-domains
OTHER_SUBDOMAINS: Dict[str, Dict[str, str]] = {
    "speech": {
        "/m/09x0r": "Speech",
        "/m/01h8n0": "Conversation",
        "/m/02qldy": "Narration, monologue",
        "/m/02rtxlg": "Whispering",
    },
    "music": {
        "/m/04rlf": "Music",
        "/m/042v_gx": "Acoustic guitar",
        "/m/05r5c": "Piano",
        "/m/015lz1": "Singing",
    },
    "animals": {
        "/m/0bt9lr": "Dog",
        "/m/05tny_": "Bark",
        "/m/01yrx": "Cat",
        "/m/07qrkrw": "Meow",
        "/m/015p6": "Bird",
        "/m/020bb7": "Bird vocalization, bird call, bird song",
    },
    "household": {
        "/m/04brg2": "Dishes, pots, and pans",
        "/m/02dgv": "Door",
        "/m/07r4wb8": "Knock",
        "/m/023pjk": "Cutlery, silverware",
        "/m/01x3z": "Clock",
        "/m/02jz0l": "Water tap, faucet",
        "/m/0642b4": "Cupboard open or close",
    },
    "traffic": {
        "/m/0btp2": "Traffic noise, roadway noise",
        "/m/07yv9": "Vehicle",
        "/m/0k4j": "Car",
        "/m/01bjv": "Bus",
        "/m/04_sv": "Motorcycle",
    },
    "environmental": {
        "/m/03m9d0z": "Wind",
        "/m/0838f": "Water",
        "/m/06mb1": "Rain",
        "/m/0j6m2": "Stream",
        "/m/05kq4": "Ocean",
        "/m/034srq": "Waves, surf",
    },
}


def get_all_exclusion_mids() -> Set[str]:
    """Return all MIDs to exclude when sampling 'other' (emergency MIDs + ambiguous confusers)."""
    return set(EMERGENCY_MIDS.keys()) | set(AMBIGUOUS_MIDS.keys())


def get_classes() -> List[str]:
    """Return the list of all 5 class names."""
    return list(CLASSES)


def get_class_to_id() -> Dict[str, int]:
    """Return class name to integer ID mapping."""
    return dict(CLASS_TO_ID)


def get_id_to_class() -> Dict[int, str]:
    """Return integer ID to class name mapping."""
    return dict(ID_TO_CLASS)


def is_emergency_class(class_name: str) -> bool:
    """Return True if class_name is one of the 4 emergency classes."""
    return class_name in EMERGENCY_CLASSES


def is_valid_class(class_name: str) -> bool:
    """Return True if class_name is one of the 5 defined classes."""
    return class_name in CLASS_TO_ID
