"""
label_with_teacher.py  -  STEP 2 of the dataset pipeline.

    python dataset/label_with_teacher.py                       # every book in data/units/
    python dataset/label_with_teacher.py --book alice          # one book
    python dataset/label_with_teacher.py --book alice --limit-batches 2   # quick trial

A big LLM (the "teacher", through your existing llm_client.py with its Groq /
Cerebras / Gemini key rotation) labels each unit with
    speaker, gender, emotion, intensity, pace
in batches of consecutive units. The output is data/labeled/<slug>.jsonl,
one line per unit.

It is RESUMABLE: if the API quota runs out (or you close the laptop), just run
the same command again later - it continues exactly where it stopped.

Needs, inside <project>/pipeline/ : llm_client.py and config.py (copied from
your old project) and a .env with your API keys in <project>/.env
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "dataset"))

from schema import CONTEXT_WORDS, clean_label, tail_words, update_known  # noqa: E402

try:
    from llm_client import call_llm_with_fallback, strip_json_fence  # noqa: E402
except Exception as e:  # missing file OR missing pip package
    print("Could not import llm_client from the pipeline/ folder:\n  ", repr(e))
    print("Copy llm_client.py and config.py from your old project into pipeline/, and run:\n"
          "  pip install groq cerebras-cloud-sdk google-genai python-dotenv")
    sys.exit(1)

UNITS_DIR = ROOT / "data" / "units"
LABELED_DIR = ROOT / "data" / "labeled"

TEACHER_RULES = """You are an expert audiobook director labeling passages from a public-domain novel. \
Your labels will train a model that tells a text-to-speech system WHO is speaking and HOW to perform each passage.

For EACH numbered passage return one JSON object with exactly these keys:
- "speaker": passages tagged [narration] are ALWAYS "narrator". For [dialogue] passages, the character who \
speaks the quoted words, as a short UPPERCASE name (e.g. "ALICE", "WHITE RABBIT", "MR DARCY"). If KNOWN \
CHARACTERS already contains that person, reuse the exact same name. Unnamed characters get their role \
("FOOTMAN", "CATERPILLAR"). If you truly cannot tell, use "narrator".
- "gender": "male" or "female" for a character, "unknown" if there is no way to tell, "none" for the narrator.
- "emotion": exactly one of: joy, warmth, grief, tension, anger, wonder, reflection, urgency, neutral.
    joy = delight, amusement, excitement | warmth = tenderness, comfort, affection
    grief = sorrow, loss, gloom | tension = fear, dread, suspense, unease
    anger = rage, irritation, frustration | wonder = awe, curiosity, surprise, mystery
    reflection = thoughtful, solemn, weary, musing | urgency = hurry, panic, danger, action
    neutral = ONLY flat description or exposition where no feeling is conveyed
- "intensity": "low", "medium" or "high" - how strongly the emotion comes through.
- "pace": "slow" (sorrow, reflection, solemn or hushed suspense), "medium" (ordinary narrative), \
"fast" (action, hurry, excitement, heated speech).

Judge what the TEXT conveys. Most passages carry some feeling, but do not invent drama: use neutral when \
nothing is felt. The previous text is context only - do not label it."""


def build_teacher_prompt(batch, known, prev_text):
    chars = ", ".join(known) if known else "(none yet)"
    blocks = "\n\n".join(f"[{i + 1}] [{u['kind']}] {u['text']}" for i, u in enumerate(batch))
    return (
        TEACHER_RULES
        + f"\n\nKNOWN CHARACTERS so far in this book: {chars}"
        + f"\nPREVIOUS TEXT (context only): {prev_text or '(start of excerpt)'}"
        + f"\n\nPASSAGES:\n\n{blocks}"
        + f"\n\nRespond ONLY with a JSON array of exactly {len(batch)} objects, in passage order. "
          "No commentary, no markdown."
    )


def extract_json_array(raw):
    """Pull a JSON list out of an LLM reply, tolerating fences, chatter, wrapper dicts."""
    if not raw:
        return None
    text = strip_json_fence(raw).strip()
    candidates = [text]
    starts = [m.start() for m in re.finditer(r"\[", text)][:20]
    end = text.rfind("]")
    candidates += [text[s:end + 1] for s in starts if end > s]
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return v
    return None


def label_batch(batch, known, prev_text):
    """
    Returns a list of label dicts (or None for units that failed), same length as batch.
    If the model returns the wrong number of objects, the batch is halved and retried.
    A RuntimeError from llm_client (all providers exhausted) is NOT caught here.
    """
    raw = call_llm_with_fallback(build_teacher_prompt(batch, known, prev_text))
    arr = extract_json_array(raw)
    if arr is not None and len(arr) == len(batch):
        return [clean_label(obj, u["kind"]) for obj, u in zip(arr, batch)]

    if len(batch) == 1:
        return [None]
    mid = len(batch) // 2
    first = label_batch(batch[:mid], known, prev_text)
    known2 = list(known)
    for lab in first:
        if lab:
            update_known(known2, lab["speaker"])
    second = label_batch(batch[mid:], known2, tail_words(batch[mid - 1]["text"], CONTEXT_WORDS))
    return first + second


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def next_batch(units, pos, batch_size):
    """Consecutive units, never crossing into a new excerpt window."""
    end = pos + 1
    while end < len(units) and end - pos < batch_size and not units[end]["window_start"]:
        end += 1
    return units[pos:end]


def label_book(slug, batch_size, limit_batches, delay):
    units = read_jsonl(UNITS_DIR / f"{slug}.jsonl")
    out_path = LABELED_DIR / f"{slug}.jsonl"
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    done = read_jsonl(out_path) if out_path.exists() else []

    for i, rec in enumerate(done):
        if i >= len(units) or rec["id"] != units[i]["id"]:
            print(f"[{slug}] {out_path.name} does not match data/units/{slug}.jsonl "
                  f"(was build_dataset.py re-run?). Delete the labeled file to start over.")
            return "error"

    known = []
    for rec in done:
        if rec.get("label"):
            update_known(known, rec["label"]["speaker"])
    failed = sum(1 for r in done if not r.get("label"))
    pos, batches = len(done), 0

    if pos >= len(units):
        print(f"[{slug}] already complete ({len(units)} units)")
        return "complete"

    with open(out_path, "a", encoding="utf-8") as f:
        while pos < len(units):
            if limit_batches and batches >= limit_batches:
                return "limit"
            batch = next_batch(units, pos, batch_size)
            prev_text = ""
            if pos > 0 and not batch[0]["window_start"]:
                prev_text = tail_words(units[pos - 1]["text"], CONTEXT_WORDS)

            try:
                labels = label_batch(batch, list(known), prev_text)
            except RuntimeError as e:
                print(f"\n[{slug}] API providers exhausted: {e}\n"
                      f"Progress is saved ({pos}/{len(units)}). Run the same command later.")
                return "api_exhausted"

            for u, lab in zip(batch, labels):
                rec = dict(u)
                rec["label"] = lab
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if lab:
                    update_known(known, lab["speaker"])
                else:
                    failed += 1
            f.flush()

            pos += len(batch)
            batches += 1
            print(f"[{slug}] {pos}/{len(units)} units labeled  (unlabelable so far: {failed})")
            if delay:
                time.sleep(delay)
    return "complete"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", nargs="*", help="slug(s); default = every file in data/units/")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--limit-batches", type=int, default=0, help="stop after N batches per book (trial run)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to sleep between batches")
    args = ap.parse_args()

    slugs = args.book or sorted(p.stem for p in UNITS_DIR.glob("*.jsonl"))
    if not slugs:
        print("No unit files found. Run dataset/build_dataset.py first.")
        return 1

    for slug in slugs:
        if not (UNITS_DIR / f"{slug}.jsonl").exists():
            print(f"[{slug}] no units file, skipped")
            continue
        status = label_book(slug, args.batch_size, args.limit_batches, args.delay)
        if status == "api_exhausted":
            return 2
    print("\nLabeling pass finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
