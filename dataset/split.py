"""
split.py  -  STEP 3 of the dataset pipeline.

    python dataset/split.py                  # make splits + gold review sheet
    (open data/gold/gold_review.csv in Excel, fix wrong labels, save as CSV)
    python dataset/split.py --build-gold     # turn the corrected sheet into gold.jsonl

Outputs
  data/splits/train.jsonl, val.jsonl, test.jsonl   (prompt + completion + label)
  data/splits/stats.json                           (label distributions, for the report)
  data/gold/gold_review.csv                        (sample of the TEST books to hand-correct)
  data/gold/gold.jsonl, gold_stats.json            (after --build-gold)

The split is BY BOOK, never by passage: if passages from the same book were in
both train and test, the model could memorise character names and the score
would be inflated. Say this in your viva.
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import (EMOTIONS, GENDERS, INTENSITIES, LABEL_KEYS, NARRATOR, PACES,  # noqa: E402
                    clean_label, format_completion, make_examples)

ROOT = Path(__file__).resolve().parent.parent
LABELED_DIR = ROOT / "data" / "labeled"
SPLITS_DIR = ROOT / "data" / "splits"
GOLD_DIR = ROOT / "data" / "gold"
MIN_LABELED_UNITS = 50


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_books():
    books = {}
    for p in sorted(LABELED_DIR.glob("*.jsonl")):
        examples = list(make_examples(read_jsonl(p)))
        if len(examples) < MIN_LABELED_UNITS:
            print(f"[{p.stem}] only {len(examples)} labeled units - skipped")
            continue
        books[p.stem] = examples
    return books


def choose_books(slugs, seed, test_arg, val_arg):
    slugs = sorted(slugs)
    if test_arg or val_arg:
        test = [s for s in (test_arg or []) if s in slugs]
        val = [s for s in (val_arg or []) if s in slugs]
    else:
        shuffled = slugs[:]
        random.Random(seed).shuffle(shuffled)
        n_test = max(1, round(0.15 * len(slugs)))
        n_val = max(1, round(0.10 * len(slugs)))
        test, val = shuffled[:n_test], shuffled[n_test:n_test + n_val]
    train = [s for s in slugs if s not in test and s not in val]
    return train, val, test


def distribution(examples):
    n = max(1, len(examples))
    labels = [e["label"] for e in examples]
    out = {"n": len(examples),
           "kind": dict(Counter(e["kind"] for e in examples))}
    for key in ("emotion", "intensity", "pace", "gender"):
        out[key] = {k: round(100 * v / n, 1) for k, v in Counter(l[key] for l in labels).most_common()}
    narr = sum(l["speaker"] == NARRATOR for l in labels)
    out["narrator_pct"] = round(100 * narr / n, 1)
    out["distinct_speakers"] = len({l["speaker"] for l in labels})
    return out


def print_distribution(name, d):
    print(f"\n== {name}: {d['n']} units | kinds {d['kind']} | narrator {d['narrator_pct']}% "
          f"| {d['distinct_speakers']} distinct speakers")
    for key in ("emotion", "intensity", "pace", "gender"):
        print(f"   {key:<10}", "  ".join(f"{k} {v}%" for k, v in d[key].items()))


# --------------------------------------------------------------------- gold ---

GOLD_COLS = ["id", "book", "kind", "known_characters", "context", "text"] + LABEL_KEYS


def sample_gold(test_examples, size, seed):
    rng = random.Random(seed)
    dlg = [e for e in test_examples if e["kind"] == "dialogue"]
    nar = [e for e in test_examples if e["kind"] == "narration"]
    rng.shuffle(dlg)
    rng.shuffle(nar)
    half = size // 2
    picked = dlg[:half] + nar[:size - min(half, len(dlg))]
    return sorted(picked[:size], key=lambda e: e["id"])


def write_gold_review(examples):
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLD_DIR / "gold_review.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GOLD_COLS)
        w.writeheader()
        for e in examples:
            row = {"id": e["id"], "book": e["book"], "kind": e["kind"],
                   "known_characters": ", ".join(e["known"]), "context": e["context"],
                   "text": e["text"]}
            row.update(e["label"])
            w.writerow(row)
    return path


def build_gold():
    review = GOLD_DIR / "gold_review.csv"
    test_path = SPLITS_DIR / "test.jsonl"
    if not review.exists() or not test_path.exists():
        print("Run `python dataset/split.py` first (needs gold_review.csv and test.jsonl).")
        return 1
    by_id = {e["id"]: e for e in read_jsonl(test_path)}

    gold, bad = [], []
    agree = Counter()
    with open(review, encoding="utf-8-sig", newline="") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            ex = by_id.get(row["id"])
            if ex is None:
                bad.append((line_no, "unknown id"))
                continue
            fixed = clean_label({k: row.get(k) for k in LABEL_KEYS}, ex["kind"])
            if fixed is None:
                bad.append((line_no, f"invalid label values in row {row['id']}"))
                continue
            for k in LABEL_KEYS:
                agree[k] += fixed[k] == ex["label"][k]
            agree["all_fields"] += fixed == ex["label"]
            rec = dict(ex)
            rec["teacher_label"] = ex["label"]
            rec["label"] = fixed
            rec["completion"] = format_completion(fixed)
            gold.append(rec)

    for line_no, msg in bad:
        print(f"  CSV line {line_no}: {msg}")
    if not gold:
        print("No valid gold rows.")
        return 1

    write_jsonl(GOLD_DIR / "gold.jsonl", gold)
    n = len(gold)
    stats = {"n": n, "teacher_agreement_pct": {k: round(100 * v / n, 1) for k, v in agree.items()}}
    (GOLD_DIR / "gold_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nWrote {n} gold examples ({len(bad)} bad rows skipped).")
    print("Teacher-vs-human agreement (this is the teacher's accuracy, put it in your report):")
    for k, v in stats["teacher_agreement_pct"].items():
        print(f"   {k:<11} {v}%")
    return 0


# --------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test", help="comma-separated book slugs to force into test")
    ap.add_argument("--val", help="comma-separated book slugs to force into val")
    ap.add_argument("--gold-size", type=int, default=400)
    ap.add_argument("--force", action="store_true", help="overwrite an existing gold_review.csv")
    ap.add_argument("--build-gold", action="store_true")
    args = ap.parse_args()

    if args.build_gold:
        return build_gold()

    books = load_books()
    if len(books) < 3:
        print(f"Need at least 3 labeled books for a book-level split, found {len(books)}.")
        return 1

    train_b, val_b, test_b = choose_books(
        list(books), args.seed,
        args.test.split(",") if args.test else None,
        args.val.split(",") if args.val else None)
    if not train_b or not val_b or not test_b:
        print("Split came out empty - check the --test/--val slugs.")
        return 1
    print(f"train books ({len(train_b)}): {', '.join(train_b)}")
    print(f"val   books ({len(val_b)}): {', '.join(val_b)}")
    print(f"test  books ({len(test_b)}): {', '.join(test_b)}")

    splits = {name: [e for b in names for e in books[b]]
              for name, names in (("train", train_b), ("val", val_b), ("test", test_b))}

    stats = {}
    for name, rows in splits.items():
        write_jsonl(SPLITS_DIR / f"{name}.jsonl", rows)
        stats[name] = distribution(rows)
        stats[name]["books"] = {"train": train_b, "val": val_b, "test": test_b}[name]
        print_distribution(name, stats[name])
    (SPLITS_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    for key, allowed in (("emotion", EMOTIONS), ("intensity", INTENSITIES),
                         ("pace", PACES), ("gender", GENDERS)):
        for v in allowed:
            share = stats["train"][key].get(v, 0)
            if share < 2 and not (key == "gender" and v == "none"):
                print(f"WARNING: {key}='{v}' is only {share}% of train - the model will rarely learn it.")

    review = GOLD_DIR / "gold_review.csv"
    if review.exists() and not args.force:
        print(f"\n{review.name} already exists - NOT overwritten (your edits are safe). "
              f"Use --force to regenerate it.")
    else:
        gold = sample_gold(splits["test"], args.gold_size, args.seed)
        path = write_gold_review(gold)
        print(f"\nWrote {len(gold)} rows to {path}\n"
              f"Open it in Excel, correct wrong labels in the speaker/gender/emotion/intensity/pace "
              f"columns, save as CSV, then run:  python dataset/split.py --build-gold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
