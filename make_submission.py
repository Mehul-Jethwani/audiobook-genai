import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FILES = ["README.md", "CODE_WALKTHROUGH.md", "requirements.txt", "app.py", "make_submission.py", "check_setup.py",
         "data/sample_labels.json", "data/sample_scene.txt"]
FOLDERS = {
    "dataset": {".py"},
    "pipeline": {".py"},
    "training": {".py", ".ipynb"},
    "results": None,
    "adapter_final": None,
    "data/labeled": {".jsonl"},
    "data/splits": {".jsonl", ".json"},
    "data/gold": None,
}
MODEL_FOLDER = "models/narration_director"
SKIP_PARTS = {"__pycache__", ".git", ".vscode", ".ipynb_checkpoints", "venv", ".venv", "node_modules",
              "cache_audio", "output"}
SKIP_NAMES = {".env"}
KEY_PATTERNS = [r"gsk_[A-Za-z0-9]{20,}", r"csk-[A-Za-z0-9]{20,}", r"AIza[0-9A-Za-z_\-]{30,}",
                r"sk-[A-Za-z0-9]{20,}", r"hf_[A-Za-z0-9]{20,}"]
TEXT_SUFFIXES = {".py", ".ipynb", ".md", ".json", ".txt", ".csv", ".jsonl"}
MAX_SCAN_BYTES = 20_000_000


def wanted(path):
    return (not (SKIP_PARTS & set(path.relative_to(ROOT).parts)) and path.name not in SKIP_NAMES
            and path.suffix.lower() != ".zip")


def collect(with_model):
    found = []
    for rel in FILES:
        if (ROOT / rel).is_file():
            found.append(ROOT / rel)
    folders = dict(FOLDERS)
    if with_model:
        folders[MODEL_FOLDER] = None
    for rel, suffixes in folders.items():
        base = ROOT / rel
        if base.is_dir():
            found += [p for p in sorted(base.rglob("*"))
                      if p.is_file() and wanted(p) and (suffixes is None or p.suffix.lower() in suffixes)]
    return found


def leaked_keys(paths):
    hits = []
    for p in paths:
        if p.suffix.lower() in TEXT_SUFFIXES and p.stat().st_size < MAX_SCAN_BYTES:
            text = p.read_text(encoding="utf-8", errors="ignore")
            hits += [(p, pat) for pat in KEY_PATTERNS if re.search(pat, text)]
    return hits


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "audiobook_genai_submission.zip"))
    ap.add_argument("--with-model", action="store_true", help="also include models/narration_director (about 3 GB)")
    args = ap.parse_args()

    paths = collect(args.with_model)
    hits = leaked_keys(paths)
    if hits:
        print("STOPPED: something that looks like an API key is inside these files:")
        for p, pat in hits:
            print("  ", p.relative_to(ROOT))
        print("Remove the key from the file (keys belong only in .env), then run this again.")
        sys.exit(1)

    out = Path(args.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, Path("audiobook_genai") / p.relative_to(ROOT))

    print(f"wrote {out} ({human(out.stat().st_size)}, {len(paths)} files)")
    groups = {}
    for p in paths:
        top = p.relative_to(ROOT).parts[0] if len(p.relative_to(ROOT).parts) > 1 else "(root)"
        groups.setdefault(top, []).append(p.stat().st_size)
    for name, sizes in sorted(groups.items()):
        print(f"  {name:<14}{len(sizes):>5} files  {human(sum(sizes)):>9}")
    for expected in ("adapter_final/adapter_config.json", "data/gold/gold.jsonl", "results"):
        if not (ROOT / expected).exists():
            print(f"WARNING: {expected} is missing, so the zip does not contain it")
    if not args.with_model:
        print("The merged model (models/narration_director) is not included. Share it as a Drive link, "
              "or add --with-model.")
    print("Not included on purpose: .env, caches, output audio, data/raw_books, data/units.")


if __name__ == "__main__":
    main()
