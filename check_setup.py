import json
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED = [
    "app.py", "README.md", "requirements.txt",
    "dataset/schema.py", "dataset/build_dataset.py", "dataset/label_with_teacher.py", "dataset/split.py",
    "pipeline/prepare.py", "pipeline/script_mode.py", "pipeline/annotate.py", "pipeline/voice_select.py",
    "pipeline/voice_map.py", "pipeline/narrate.py", "pipeline/run.py", "pipeline/llm_client.py", "pipeline/config.py",
    "training/finetune.ipynb", "training/eval.py", "training/export_model.ipynb",
    "training/make_figures.py", "training/demo.ipynb",
    "data/sample_labels.json", "data/sample_scene.txt", "data/gold/gold.jsonl",
    "data/splits/train.jsonl", "data/splits/val.jsonl", "data/splits/test.jsonl",
]
OPTIONAL = ["CODE_WALKTHROUGH.md", "make_submission.py", "results/eval_results.json",
            "results/figures/system_comparison.png", "results/figures/emotion_confusion.png",
            "results/figures/loss_curve.png"]
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]
MODEL_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "model.safetensors"]
EXPECTED_TENSORS = 338
MODULES = {"matplotlib": "matplotlib", "pymupdf": "pymupdf", "edge_tts": "edge-tts", "pydub": "pydub",
           "gender_guesser": "gender-guesser", "gradio": "gradio"}

results = []


def report(level, message):
    results.append(level)
    print(f"[{level}] {message}")


def safetensors_check(path):
    size = path.stat().st_size
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            return None, "file is too small to be a model"
        header_len = struct.unpack("<Q", raw)[0]
        if header_len > min(size - 8, 100_000_000):
            return None, "header is corrupt"
        header = json.loads(f.read(header_len))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    expected = 8 + header_len + max((v["data_offsets"][1] for v in tensors.values()), default=0)
    if expected != size:
        return len(tensors), f"file has {size:,} bytes but should have {expected:,}: the download is incomplete"
    return len(tensors), None


def check_files():
    for rel in REQUIRED:
        if (ROOT / rel).is_file():
            report("ok", rel)
        else:
            report("FAIL", f"missing: {rel}")
    for rel in OPTIONAL:
        if not (ROOT / rel).exists():
            report("warn", f"not there yet: {rel}")


def check_syntax():
    bad = 0
    for p in sorted(ROOT.rglob("*.py")):
        if {"__pycache__", "venv", ".venv"} & set(p.relative_to(ROOT).parts):
            continue
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")
        except SyntaxError as e:
            bad += 1
            report("FAIL", f"syntax error in {p.relative_to(ROOT)} line {e.lineno}: re-download this file")
    if not bad:
        report("ok", "every .py file compiles")


def check_adapter():
    folder = ROOT / "adapter_final"
    if not folder.is_dir():
        report("FAIL", "adapter_final folder is missing")
        return
    clutter = [p.name for p in folder.iterdir() if p.is_dir() or p.suffix.lower() == ".zip"]
    if clutter:
        report("warn", f"adapter_final has leftover download items: {clutter}. Move the files inside up one level and delete these.")
    for name in ADAPTER_FILES:
        if not (folder / name).is_file():
            report("FAIL", f"adapter_final/{name} is missing (it may be inside a sub-folder)")
    weights = folder / "adapter_model.safetensors"
    if weights.is_file():
        count, problem = safetensors_check(weights)
        report("FAIL" if problem else "ok", f"adapter weights: {problem or str(count) + ' tensors, file complete'}")


def check_model():
    folder = ROOT / "models" / "narration_director"
    loose = [p.name for p in (ROOT / "models").glob("*")] if (ROOT / "models").is_dir() else []
    loose = [n for n in loose if n != "narration_director"]
    if loose:
        report("warn", f"loose items in models/: {loose}. The model files belong inside models/narration_director/.")
    if not folder.is_dir():
        report("warn", "models/narration_director does not exist yet (optional: the adapter works without it)")
        return
    for name in MODEL_FILES:
        if not (folder / name).is_file():
            report("FAIL", f"models/narration_director/{name} is missing")
    weights = folder / "model.safetensors"
    if weights.is_file():
        count, problem = safetensors_check(weights)
        if problem:
            report("FAIL", f"merged model weights: {problem}")
        elif count != EXPECTED_TENSORS:
            report("warn", f"merged model has {count} tensors (expected {EXPECTED_TENSORS} for Qwen2.5-1.5B)")
        else:
            report("ok", f"merged model weights: {count} tensors, file complete ({weights.stat().st_size / 1024 ** 3:.2f} GB)")


def check_tools():
    for module, package in MODULES.items():
        try:
            __import__(module)
            report("ok", f"python package {package}")
        except ImportError:
            report("warn", f"python package missing: pip install {package}")
    report("ok" if shutil.which("ffmpeg") else "warn",
           "ffmpeg found" if shutil.which("ffmpeg") else "ffmpeg not found: audio joining will fail (winget install Gyan.FFmpeg)")
    report("ok" if (ROOT / ".env").exists() else "warn",
           ".env present (it is never included in the zip)" if (ROOT / ".env").exists() else ".env not found (only needed for dataset labeling)")


def main():
    print(f"checking {ROOT}\n")
    check_files()
    check_syntax()
    check_adapter()
    check_model()
    check_tools()
    fails, warns = results.count("FAIL"), results.count("warn")
    print(f"\n{results.count('ok')} ok, {warns} warnings, {fails} problems")
    print("Everything required is in place." if not fails else "Fix the FAIL lines above, then run this again.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
