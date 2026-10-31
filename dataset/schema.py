"""
schema.py  -  the ONE place that defines the label format and the prompt format.

Used by label_with_teacher.py, split.py and (later) eval.py and the final
pipeline. If you change a label name or the prompt wording, change it HERE
and re-run split.py, otherwise training and inference will silently disagree.

What the fine-tuned model must output for every passage ("unit"):
    {"speaker": "ALICE", "gender": "female", "emotion": "wonder",
     "intensity": "medium", "pace": "medium"}

NOTE: tone_shift is NOT learned. It is derived in code at runtime
(emotion != previous emotion) - no point asking a model to learn a rule.
"""

import json
import re

NARRATOR = "narrator"

EMOTIONS = ["joy", "warmth", "grief", "tension", "anger",
            "wonder", "reflection", "urgency", "neutral"]
INTENSITIES = ["low", "medium", "high"]
PACES = ["slow", "medium", "fast"]
GENDERS = ["male", "female", "unknown", "none"]

LABEL_KEYS = ["speaker", "gender", "emotion", "intensity", "pace"]

MAX_KNOWN_CHARACTERS = 15   # how many recent character names go into the prompt
CONTEXT_WORDS = 80          # how much previous text goes into the prompt

# --- the teacher sometimes drifts outside the closed label set; map it back ---
EMOTION_ALIASES = {
    "sadness": "grief", "sorrow": "grief", "sad": "grief", "melancholy": "grief",
    "despair": "grief", "mourning": "grief", "gloom": "grief",
    "fear": "tension", "dread": "tension", "anxiety": "tension", "suspense": "tension",
    "unease": "tension", "fearful": "tension", "anxious": "tension",
    "nervous": "tension", "horror": "tension",
    "happy": "joy", "happiness": "joy", "amusement": "joy", "amused": "joy",
    "excitement": "joy", "excited": "joy", "playful": "joy", "delight": "joy",
    "cheerful": "joy",
    "tender": "warmth", "affection": "warmth", "love": "warmth", "comfort": "warmth",
    "gentle": "warmth", "kindness": "warmth",
    "rage": "anger", "fury": "anger", "irritation": "anger", "frustration": "anger",
    "annoyance": "anger", "irritated": "anger", "furious": "anger", "angry": "anger",
    "resentment": "anger",
    "curiosity": "wonder", "curious": "wonder", "awe": "wonder", "amazement": "wonder",
    "surprise": "wonder", "surprised": "wonder", "mystery": "wonder",
    "thoughtful": "reflection", "contemplative": "reflection", "solemn": "reflection",
    "pensive": "reflection", "nostalgia": "reflection", "weariness": "reflection",
    "boredom": "reflection",
    "panic": "urgency", "alarm": "urgency", "hurry": "urgency", "haste": "urgency",
    "frantic": "urgency", "urgent": "urgency",
    "flat": "neutral", "calm": "neutral", "none": "neutral", "informative": "neutral",
}
INTENSITY_ALIASES = {"moderate": "medium", "normal": "medium", "mid": "medium",
                     "strong": "high", "mild": "low", "weak": "low", "subtle": "low"}
PACE_ALIASES = {"normal": "medium", "moderate": "medium", "slowly": "slow",
                "quick": "fast", "rapid": "fast"}
GENDER_ALIASES = {"m": "male", "man": "male", "f": "female", "woman": "female",
                  "other": "unknown", "n/a": "none", "na": "none"}


def _norm(x):
    return "" if x is None else str(x).strip().lower()


def clean_speaker(raw):
    """Normalise a character name: UPPERCASE, no punctuation, 'MR. DARCY' -> 'MR DARCY'."""
    name = "" if raw is None else str(raw).strip()
    if name.lower() in ("", "narrator", "none", "null", "unknown", "n/a", "unclear"):
        return NARRATOR
    name = re.sub(r"\s+", " ", name.upper())
    name = re.sub(r"[^A-Z0-9' \-]", "", name).strip(" -")
    return name[:40] or NARRATOR


def clean_label(raw, kind="dialogue"):
    """
    Turn a raw dict (from the teacher LLM, the student model, or a CSV row)
    into a valid label dict, or None if it can't be repaired.
    `kind` is "narration" or "dialogue"; narration is ALWAYS the narrator.
    """
    if not isinstance(raw, dict):
        return None

    emotion = _norm(raw.get("emotion"))
    emotion = EMOTION_ALIASES.get(emotion, emotion)
    intensity = _norm(raw.get("intensity"))
    intensity = INTENSITY_ALIASES.get(intensity, intensity)
    pace = _norm(raw.get("pace"))
    pace = PACE_ALIASES.get(pace, pace)

    if emotion not in EMOTIONS or intensity not in INTENSITIES or pace not in PACES:
        return None

    speaker = NARRATOR if kind == "narration" else clean_speaker(raw.get("speaker"))

    if speaker == NARRATOR:
        gender = "none"
    else:
        gender = _norm(raw.get("gender"))
        gender = GENDER_ALIASES.get(gender, gender)
        if gender not in GENDERS or gender == "none":
            gender = "unknown"

    return {"speaker": speaker, "gender": gender, "emotion": emotion,
            "intensity": intensity, "pace": pace}


# ------------------------------------------------------------------ prompt ---

TASK_HEADER = (
    "You are the narration director of an audiobook. Label the TARGET passage so a "
    "text-to-speech system knows who is speaking and how to perform it. "
    "Reply with one JSON object with keys speaker, gender, emotion, intensity, pace."
)


def tail_words(text, n=CONTEXT_WORDS):
    words = (text or "").split()
    return " ".join(words[-n:])


def update_known(known, speaker):
    """Keep a most-recently-used list of character names (mutates `known`)."""
    if not speaker or speaker == NARRATOR:
        return
    if speaker in known:
        known.remove(speaker)
    known.append(speaker)
    del known[:-MAX_KNOWN_CHARACTERS]


def build_prompt(known_characters, prev_emotion, context, text, kind):
    """The exact text the student model sees - at training time AND at inference."""
    chars = ", ".join(known_characters) if known_characters else "(none yet)"
    return (
        f"{TASK_HEADER}\n\n"
        f"Known characters: {chars}\n"
        f"Previous emotion: {prev_emotion or 'none'}\n"
        f"Previous text: {context or '(start of section)'}\n"
        f"Type: {kind}\n"
        f"TARGET: {text}\n\n"
        f"JSON:"
    )


def format_completion(label):
    """The exact text the student model must produce."""
    return json.dumps({k: label[k] for k in LABEL_KEYS}, ensure_ascii=False,
                      separators=(",", ":"))


def parse_completion(text, kind="dialogue"):
    """Parse model output back into a clean label (None if unusable)."""
    match = re.search(r"\{.*?\}", text or "", re.S)
    if not match:
        return None
    try:
        return clean_label(json.loads(match.group(0)), kind)
    except json.JSONDecodeError:
        return None


def make_examples(records):
    """
    Replays ONE book's labeled units in order and yields training examples,
    rebuilding the running state (known characters, previous emotion/text)
    exactly the way the live pipeline will build it at inference time.

    records: list of dicts with id, book, kind, text, window_start, label.
    """
    known, prev_label, prev_text = [], None, None
    for rec in records:
        if rec.get("window_start"):
            prev_label, prev_text = None, None   # new excerpt: context resets,
                                                 # known characters persist
        context = tail_words(prev_text) if prev_text else ""
        prev_emotion = prev_label["emotion"] if prev_label else None
        label = rec.get("label")

        if label:
            yield {
                "id": rec["id"], "book": rec["book"], "kind": rec["kind"],
                "text": rec["text"], "context": context, "known": list(known),
                "prompt": build_prompt(known, prev_emotion, context, rec["text"], rec["kind"]),
                "completion": format_completion(label),
                "label": label,
            }
            update_known(known, label["speaker"])
            prev_label = label
        else:
            prev_label = None
        prev_text = rec["text"]
