import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dataset"))
from build_dataset import clean_text, split_long  # noqa: E402
from schema import clean_speaker  # noqa: E402

NOT_SPEAKERS = {"NOTE", "NOTES", "INT", "EXT", "SCENE", "ACT", "CHAPTER", "TITLE", "FADE", "CUT",
                "END", "SETTING", "TIME", "WARNING", "TIP", "EXAMPLE", "SUMMARY", "QUESTION",
                "ANSWER", "STEP", "DAY", "REMEMBER", "PLACE", "LOCATION", "CHARACTERS"}
UPPER = r"(?<![A-Za-z0-9'])[A-Z][A-Z'\-]+(?: [A-Z][A-Z'\-]+){0,2}"
TITLE = r"(?<=\n)[A-Z][a-z]{1,14}(?: [A-Z][a-z]{1,14})?"
MARKER = re.compile(rf"({UPPER}|{TITLE}):(?=\s)")
DIRECTION = re.compile(r"\(([^)]*)\)")
MIN_DIRECTION_WORDS = 4


def find_markers(text):
    found = []
    for m in MARKER.finditer("\n" + text):
        name = m.group(1)
        if name.split()[0].upper() not in NOT_SPEAKERS:
            found.append((m.start() - 1, m.end() - 1, name))
    return found


def looks_like_script(text):
    markers = find_markers(text)
    names = Counter(clean_speaker(name) for _, _, name in markers)
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return len(markers) >= 6 and len(names) >= 2 and len(markers) >= 0.25 * len(paragraphs)


def normalize(chunk):
    return " ".join(chunk.split())


def script_units(text):
    text = clean_text(text)
    marks = find_markers(text)
    units = []

    def narration(chunk):
        chunk = normalize(chunk)
        if len(chunk.split()) >= MIN_DIRECTION_WORDS:
            units.extend({"kind": "narration", "text": piece} for piece in split_long(chunk))

    def speech(chunk, speaker):
        chunk = normalize(chunk)
        if not chunk:
            return
        for piece in split_long(chunk):
            units.append({"kind": "dialogue", "text": piece, "speaker_hint": speaker,
                          "prompt_text": f"\u201c{piece}\u201d {speaker.title()} said."})

    narration(text[:marks[0][0]])
    for i, (_, end, name) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body, speaker, pos = text[end:stop], clean_speaker(name), 0
        for m in DIRECTION.finditer(body):
            speech(body[pos:m.start()], speaker)
            narration(m.group(1))
            pos = m.end()
        speech(body[pos:], speaker)
    return units
