import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "dataset")]

from annotate import label_units, load_generator  # noqa: E402
from narrate import narrate  # noqa: E402
from prepare import prepare  # noqa: E402

SAMPLE = ROOT / "data" / "sample_scene.txt"
OUT_DIR = ROOT / "output" / "app"
HEADERS = ["kind", "speaker", "gender", "emotion", "intensity", "pace", "text"]
PLACEHOLDER = '{"speaker":"narrator","gender":"none","emotion":"neutral","intensity":"medium","pace":"medium"}'


def find_model(explicit):
    for candidate in ([explicit] if explicit else []) + [ROOT / "models" / "narration_director", ROOT / "adapter_final"]:
        folder = Path(candidate)
        if (folder / "config.json").exists() or (folder / "adapter_config.json").exists():
            return folder
    sys.exit("No model found. Put the merged model in models/narration_director or the adapter in adapter_final, "
             "or pass --model <folder>.")


def table(labels):
    return [[r["kind"], r["speaker"], r["gender"], r["emotion"], r["intensity"], r["pace"], r["text"][:90]]
            for r in labels]


def make_runner(generate):
    def run(file, start_page, end_page, max_units, use_sample):
        empty = ("", [], None, None)
        try:
            if use_sample:
                path, start, end = SAMPLE, None, None
            elif file is None:
                yield ("Choose a PDF or text file first, or tick the sample option.",) + empty[1:]
                return
            else:
                path = Path(getattr(file, "name", file))
                start, end = int(start_page or 1), int(end_page or 0) or None
            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem)
            yield ("Reading the text...",) + empty[1:]
            units = prepare(path, max_units=int(max_units), start_page=start, end_page=end)
            if not units:
                yield ("No text found in those pages. Scanned PDFs are not supported.",) + empty[1:]
                return
            mode = "script (NAME: lines)" if any("speaker_hint" in u for u in units) else "prose (quoted dialogue)"
            yield (f"{len(units)} units, {mode} format. Labeling with the fine-tuned model...",) + empty[1:]

            t0 = time.time()
            labels = label_units(units, generate, progress=False)
            rows = table(labels)
            status = f"Labeled {len(labels)} units in {time.time() - t0:.0f} s. Generating speech..."
            yield (status, rows, None, None)

            run_dir = OUT_DIR / f"{stem}_{int(time.time())}"
            expressive, flat = run_dir / "expressive.mp3", run_dir / "flat.mp3"
            try:
                narrate(labels, expressive, ROOT / "cache_audio")
                narrate(labels, flat, ROOT / "cache_audio", flat=True)
            except Exception as e:
                yield (f"Speech failed ({type(e).__name__}): check the internet connection. Labels are shown above.",
                       rows, None, None)
                return
            yield (f"Done. Files saved in {run_dir}", rows, str(expressive), str(flat))
        except Exception as e:
            yield (f"Error: {e}",) + empty[1:]
    return run


def build_app(generate, banner=""):
    import gradio as gr
    run = make_runner(generate)
    with gr.Blocks(title="Expressive Audiobook") as demo:
        gr.Markdown("# Expressive Audiobook\nA fine-tuned model reads each passage, decides who speaks and how, "
                    "and the narrator and characters are voiced accordingly.")
        if banner:
            gr.Markdown(f"**{banner}**")
        with gr.Row():
            with gr.Column():
                file = gr.File(label="PDF or text file", file_types=[".pdf", ".txt"])
                with gr.Row():
                    start = gr.Number(label="First page", value=1, precision=0)
                    end = gr.Number(label="Last page (0 = to the end)", value=3, precision=0)
                units = gr.Slider(5, 80, value=25, step=5, label="Maximum units to narrate")
                sample = gr.Checkbox(label="Use the built-in sample scene instead", value=False)
                go = gr.Button("Narrate", variant="primary")
            with gr.Column():
                status = gr.Textbox(label="Status", interactive=False)
                expressive = gr.Audio(label="Fine-tuned model: character voices and emotion", type="filepath")
                flat = gr.Audio(label="Single flat voice (for comparison)", type="filepath")
        labels = gr.Dataframe(headers=HEADERS, label="What the model decided", interactive=False, wrap=True)
        go.click(run, [file, start, end, units, sample], [status, labels, expressive, flat])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="merged model folder or adapter folder")
    ap.add_argument("--share", action="store_true", help="create a public link (useful in Colab)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--ui-test", action="store_true",
                    help="skip the model and use placeholder labels: only tests the page, PDF reading and speech")
    args = ap.parse_args()

    banner = ""
    if args.ui_test:
        generate = lambda prompt: PLACEHOLDER
        banner = "UI TEST MODE: placeholder labels, not the model's decisions."
    else:
        model_dir = find_model(args.model)
        print(f"loading model from {model_dir} ...")
        try:
            generate = load_generator(model_dir)
        except ImportError as e:
            sys.exit(f"Could not load the model ({e}). On a laptop without a GPU run: pip install torch transformers")
        except Exception as e:
            sys.exit(f"Could not load the model: {e}")
    build_app(generate, banner).queue().launch(share=args.share, server_port=args.port, inbrowser=not args.share)


if __name__ == "__main__":
    main()
