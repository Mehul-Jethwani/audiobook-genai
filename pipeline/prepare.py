import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dataset"))
from build_dataset import build_units  # noqa: E402
from script_mode import looks_like_script, script_units  # noqa: E402

TERMINAL = ".!?\"'\u201d\u2019:;)"


def pdf_blocks(doc, start_page=None, end_page=None):
    first = max(1, start_page or 1) - 1
    last = min(len(doc), end_page or len(doc))
    if first >= len(doc):
        raise ValueError(f"This PDF has only {len(doc)} pages.")
    pages = []
    for index in range(first, last):
        page = doc[index]
        height = page.rect.height
        blocks = []
        for b in page.get_text("blocks"):
            if b[6] != 0:
                continue
            text = " ".join(b[4].replace("-\n", "").split())
            if text:
                blocks.append((text, b[1] < 0.08 * height or b[3] > 0.92 * height))
        pages.append(blocks)
    return pages


def pdf_to_text(path, start_page=None, end_page=None):
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf
    with pymupdf.open(str(path)) as doc:
        pages = pdf_blocks(doc, start_page, end_page)

    def shape(text):
        return re.sub(r"\d+", "#", text)

    repeats = Counter(shape(t) for blocks in pages for t, margin in blocks
                      if margin and len(t.split()) <= 10)
    paragraphs = []
    for blocks in pages:
        for text, margin in blocks:
            if margin and (re.fullmatch(r"\d+|[ivxlcdm]+", text, re.I) or repeats[shape(text)] >= 3):
                continue
            if paragraphs and paragraphs[-1][-1] not in TERMINAL and text[0].islower():
                paragraphs[-1] += " " + text
            else:
                paragraphs.append(text)
    if not paragraphs:
        raise ValueError("No text found in this PDF. Scanned PDFs need OCR, which this pipeline does not include.")
    return "\n\n".join(paragraphs)


def read_text(path, start_page=None, end_page=None):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return pdf_to_text(path, start_page, end_page)
    raw = path.read_bytes().decode("utf-8-sig", errors="replace")
    return re.sub(r"\r+\n", "\n", raw).replace("\r", "\n")


def prepare(path, skip_units=0, max_units=None, start_page=None, end_page=None):
    text = read_text(path, start_page, end_page)
    units = (script_units(text) if looks_like_script(text) else build_units(text))[skip_units:]
    return units[:max_units] if max_units else units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="units.json")
    ap.add_argument("--skip-units", type=int, default=0)
    ap.add_argument("--max-units", type=int, default=None)
    ap.add_argument("--start-page", type=int, default=None)
    ap.add_argument("--end-page", type=int, default=None)
    args = ap.parse_args()

    units = prepare(args.input, args.skip_units, args.max_units, args.start_page, args.end_page)
    Path(args.out).write_text(json.dumps(units, ensure_ascii=False, indent=1), encoding="utf-8")
    dlg = sum(u["kind"] == "dialogue" for u in units)
    mode = "script" if any("speaker_hint" in u for u in units) else "prose"
    print(f"{len(units)} units ({dlg} dialogue, {mode} format) -> {args.out}")


if __name__ == "__main__":
    main()
