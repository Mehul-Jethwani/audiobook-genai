import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description="Book text/PDF -> labeled units -> narrated audiobook")
    ap.add_argument("--input", help=".pdf or .txt file")
    ap.add_argument("--adapter", help="model folder: the merged model (works on CPU, slow) or the LoRA adapter (needs an NVIDIA GPU)")
    ap.add_argument("--labels", help="existing labels.json: skips labeling, so no GPU is needed")
    ap.add_argument("--out", default="output/audiobook.mp3")
    ap.add_argument("--cache", default="cache_audio")
    ap.add_argument("--skip-units", type=int, default=0)
    ap.add_argument("--max-units", type=int, default=None)
    ap.add_argument("--start-page", type=int, default=None)
    ap.add_argument("--end-page", type=int, default=None)
    ap.add_argument("--also-flat", action="store_true", help="also render a single-voice version for comparison")
    args = ap.parse_args()

    from narrate import narrate
    out = Path(args.out)

    if args.labels:
        labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    else:
        if not args.input or not args.adapter:
            ap.error("give --input and --adapter, or --labels")
        from annotate import annotate
        from prepare import prepare
        print("1/3 preparing text")
        units = prepare(args.input, args.skip_units, args.max_units, args.start_page, args.end_page)
        print(f"    {len(units)} units")
        print("2/3 labeling with the fine-tuned model")
        labels = annotate(units, args.adapter)
        labels_path = out.with_name(out.stem + "_labels.json")
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"    saved {labels_path}")

    print("3/3 narrating")
    narrate(labels, out, args.cache)
    if args.also_flat:
        narrate(labels, out.with_name(out.stem + "_flat.mp3"), args.cache, flat=True)


if __name__ == "__main__":
    main()
