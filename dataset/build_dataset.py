"""
build_dataset.py  -  STEP 1 of the dataset pipeline.

    python dataset/build_dataset.py                      # all books
    python dataset/build_dataset.py --books alice dracula
    python dataset/build_dataset.py --units-per-book 300 --window-size 60

What it does
  1. Downloads public-domain novels from Project Gutenberg into data/raw_books/
     (if a download is blocked, save the .txt there yourself as <slug>.txt).
  2. Strips the Gutenberg header/footer and cleans the text.
  3. Cuts the text into "units":
       - narration unit : text with no quoted speech
       - dialogue unit  : a quoted line (+ short tag like  said Alice,  attached)
     Long narration-after-a-quote ("Maren said, looking out at the grey harbour.")
     becomes its own narration unit, so a quote can be voiced by the character
     and the tag by the narrator.
  4. Picks several contiguous excerpts ("windows") spread across each book
     instead of labeling the whole book (keeps teacher-API cost small).
  5. Writes data/units/<slug>.jsonl

IMPORTANT: later, the live audiobook pipeline must call build_units() from this
file too, so training and inference see the same kind of chunks.
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw_books"
UNITS_DIR = ROOT / "data" / "units"

# slug -> Project Gutenberg ebook id. The script prints each book's real title
# after loading it, so if an id is wrong you will notice immediately.
BOOKS = {
    "alice": 11,
    "looking_glass": 12,
    "pride_prejudice": 1342,
    "sherlock_adventures": 1661,
    "dracula": 345,
    "little_women": 514,
    "tom_sawyer": 74,
    "huck_finn": 76,
    "wizard_of_oz": 55,
    "frankenstein": 84,
    "treasure_island": 120,
    "jane_eyre": 1260,
    "peter_pan": 16,
    "great_expectations": 1400,
    "christmas_carol": 46,
    "wuthering_heights": 768,
    "jungle_book": 236,
    "anne_green_gables": 45,
    "hound_baskervilles": 2852,
    "secret_garden": 113,
}

URL_PATTERNS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/ebooks/{id}.txt.utf-8",
]

# ---- chunking knobs ---------------------------------------------------------
MIN_TAG_WORDS = 7        # narration between/after quotes shorter than this stays
                         # attached to the quote ("said Alice,"); longer -> own unit
MIN_UNIT_WORDS = 25      # tiny narration units get merged with a neighbour...
MERGE_MAX_WORDS = 120    # ...as long as the merged unit stays below this
MAX_UNIT_WORDS = 140     # narration longer than this is split at sentence ends

QUOTE_SPAN = re.compile(r"\u201c[^\u201d]*\u201d|\"[^\"]*\"")
QUOTE_CHARS = "\"\u201c\u201d"

START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                      re.I | re.S)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
HEADING_RE = re.compile(r"^(chapter|book|part|volume|letter|act|scene|stave)\s+"
                        r"([ivxlcdm]+|\d+)\b", re.I)


# ------------------------------------------------------------------ download --

def download_book(slug, gid):
    path = RAW_DIR / f"{slug}.txt"
    if path.exists() and path.stat().st_size > 10_000:
        return path
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    last_err = None
    for pattern in URL_PATTERNS:
        url = pattern.format(id=gid)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "audiobook-genai-student-project"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read().decode("utf-8-sig", errors="replace")
            if len(data) > 10_000:
                data = data.replace("\r\n", "\n").replace("\r", "\n")
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(data)
                return path
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
    raise RuntimeError(f"could not download {slug} (id {gid}): {last_err}. "
                       f"Download it manually and save as {path}")


# --------------------------------------------------------------- cleaning -----

def load_raw(path):
    """Read a raw book with clean '\\n' line endings, even if the file was
    previously corrupted to '\\r\\r\\n' by a Windows text-mode write."""
    text = path.read_bytes().decode("utf-8-sig", errors="replace")
    return re.sub(r"\r+\n", "\n", text).replace("\r", "\n")


def strip_gutenberg(text):
    m = START_RE.search(text)
    if m:
        text = text[m.end():]
    m = END_RE.search(text)
    if m:
        text = text[:m.start()]
    return text


def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    text = re.sub(r"\[Illustration[^\]]*\]", "", text, flags=re.I | re.S)
    text = re.sub(r"_([^_\n]+?)_", r"\1", text)        # _italics_ -> italics
    text = text.replace("_", "")
    return text


def join_lines(par):
    return re.sub(r"\s+", " ", " ".join(l.strip() for l in par.split("\n") if l.strip())).strip()


def has_dialogue(text):
    return bool(QUOTE_SPAN.search(text))


def is_junk(par):
    words = par.split()
    if not words:
        return True
    if re.fullmatch(r"[\W_]+", par):                           # "* * * * *"
        return True
    if len(words) <= 8 and HEADING_RE.match(par):              # "CHAPTER XII."
        return True
    if len(words) <= 6 and par.isupper():                      # ALL-CAPS headings
        return True
    if re.fullmatch(r"[IVXLCDM]+\.?", par.strip()):            # roman numerals
        return True
    # short title-case line with no end punctuation = chapter title
    if len(words) <= 6 and not has_dialogue(par) and par[-1] not in ".!?\"\u201d'\u2019:;,":
        return True
    return False


# ------------------------------------------------------------ segmentation ----

def balanced_quotes(par):
    """True only if quotes open and close in a sane order (not just equal counts:
    'happen,\u201d she said, \u201cwhenever...' has one of each but is NOT balanced)."""
    if par.count('"') % 2:
        return False
    depth = 0
    for ch in par:
        if ch == "\u201c":
            if depth:
                return False
            depth = 1
        elif ch == "\u201d":
            if not depth:
                return False
            depth = 0
    return depth == 0


def split_long(text, max_words=MAX_UNIT_WORDS):
    if len(text.split()) <= max_words:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, cur, n = [], [], 0
    for s in sentences:
        w = len(s.split())
        if cur and n + w > max_words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


def paragraph_to_units(par):
    """One paragraph -> list of [kind, text]."""
    quoted = bool(QUOTE_SPAN.search(par))

    # unbalanced quotes = a speech that continues across paragraphs; keep whole
    if not balanced_quotes(par):
        return [["dialogue", par]]

    if not quoted:
        return [["narration", s] for s in split_long(par)]

    pieces, pos = [], 0
    for m in QUOTE_SPAN.finditer(par):
        if m.start() > pos:
            pieces.append(("narration", par[pos:m.start()].strip()))
        pieces.append(("dialogue", m.group(0)))
        pos = m.end()
    if pos < len(par):
        pieces.append(("narration", par[pos:].strip()))
    pieces = [(k, t) for k, t in pieces if t]

    units, prefix = [], ""
    for kind, t in pieces:
        if kind == "dialogue":
            units.append(["dialogue", (prefix + " " + t).strip()])
            prefix = ""
        else:
            short = len(t.split()) < MIN_TAG_WORDS
            if short and units:
                units[-1][1] += " " + t             # inline tag: stays with the quote
            elif short:
                prefix = t                          # leading tag: joins the next quote
            else:
                for s in split_long(t):
                    units.append(["narration", s])
    if prefix:
        units.append(["narration", prefix])
    return units


def merge_short_narration(units):
    out = []
    for kind, text in units:
        if (out and kind == "narration" and out[-1][0] == "narration"):
            a, b = len(out[-1][1].split()), len(text.split())
            if (a < MIN_UNIT_WORDS or b < MIN_UNIT_WORDS) and a + b <= MERGE_MAX_WORDS:
                out[-1][1] += " " + text
                continue
        out.append([kind, text])
    return out


def build_units(raw_text):
    """Raw Gutenberg text -> list of {"kind", "text"}.  Reused by the live pipeline."""
    body = clean_text(strip_gutenberg(raw_text))
    units = []
    for chunk in re.split(r"\n\s*\n", body):
        par = join_lines(chunk)
        if is_junk(par):
            continue
        units.extend(paragraph_to_units(par))
    return [{"kind": k, "text": t} for k, t in merge_short_narration(units)]


# ------------------------------------------------------------------ windows ---

def pick_windows(n_units, units_per_book, window_size, rng):
    """Evenly spread contiguous windows, skipping front/back matter."""
    lo, hi = int(n_units * 0.05), int(n_units * 0.95)
    n_windows = max(1, units_per_book // window_size)
    span = hi - lo - window_size
    if span <= 0:
        return [(lo, min(hi, lo + window_size))]
    slot = span / n_windows
    starts, prev_end = [], lo
    for i in range(n_windows):
        s = lo + int(i * slot + rng.random() * slot * 0.8)
        s = max(s, prev_end)
        if s + window_size > hi:
            break
        starts.append(s)
        prev_end = s + window_size
    return [(s, s + window_size) for s in starts]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def process_book(slug, gid, args):
    path = RAW_DIR / f"{slug}.txt"
    if not path.exists():
        if args.no_download:
            print(f"[{slug}] missing {path.name} (skipped, --no-download)")
            return None
        path = download_book(slug, gid)
        time.sleep(1.0)                                  # be polite to gutenberg.org

    raw = load_raw(path)
    title = re.search(r"^Title:\s*(.+)$", raw, re.M)
    units = build_units(raw)
    if len(units) < args.window_size * 2:
        print(f"[{slug}] only {len(units)} units - looks wrong, skipped")
        return None

    rng = random.Random(f"{args.seed}-{slug}")
    windows = pick_windows(len(units), args.units_per_book, args.window_size, rng)

    rows, k = [], 0
    for w, (a, b) in enumerate(windows):
        for j, u in enumerate(units[a:b]):
            rows.append({"id": f"{slug}-{k:05d}", "book": slug, "idx": k,
                         "orig_idx": a + j, "window": w, "window_start": j == 0,
                         "kind": u["kind"], "text": u["text"]})
            k += 1
    write_jsonl(UNITS_DIR / f"{slug}.jsonl", rows)

    n_dlg = sum(r["kind"] == "dialogue" for r in rows)
    mid = sum(r["text"][:1].islower() for r in rows)
    if mid > 0.05 * len(rows):
        print(f"[{slug}] WARNING: {100 * mid // len(rows)}% of units start mid-sentence - "
              f"the raw file probably has broken line endings. Check data/raw_books/{slug}.txt")
    print(f"[{slug}] '{title.group(1).strip() if title else '?'}': {len(units)} units total -> "
          f"{len(rows)} selected in {len(windows)} windows ({100 * n_dlg // max(1, len(rows))}% dialogue)")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="*", help="slugs to process (default: all in BOOKS)")
    ap.add_argument("--units-per-book", type=int, default=300)
    ap.add_argument("--window-size", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-download", action="store_true",
                    help="only use books already in data/raw_books/")
    args = ap.parse_args()

    slugs = args.books or list(BOOKS)
    total = 0
    for slug in slugs:
        if slug not in BOOKS and not (RAW_DIR / f"{slug}.txt").exists():
            print(f"[{slug}] unknown slug (add it to BOOKS or drop <slug>.txt into data/raw_books/)")
            continue
        try:
            n = process_book(slug, BOOKS.get(slug), args)
        except RuntimeError as e:
            print(f"[{slug}] {e}")
            continue
        total += n or 0
    print(f"\nDone. {total} units ready in {UNITS_DIR}")


if __name__ == "__main__":
    main()
