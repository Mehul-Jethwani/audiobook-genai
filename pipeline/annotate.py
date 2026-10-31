import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dataset"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import (NARRATOR, build_prompt, parse_completion, tail_words,  # noqa: E402
                    update_known)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 48


def fallback_label(unit, known):
    if unit["kind"] == "narration" or not known:
        speaker, gender = NARRATOR, "none"
    else:
        speaker, gender = known[-1], "unknown"
    return {"speaker": speaker, "gender": gender, "emotion": "neutral",
            "intensity": "low", "pace": "medium"}


def label_units(units, generate, progress=True):
    known, prev_label, prev_text = [], None, None
    out, start, failed = [], time.time(), 0
    for i, unit in enumerate(units):
        shown = unit.get("prompt_text", unit["text"])
        prompt = build_prompt(known, prev_label["emotion"] if prev_label else None,
                              tail_words(prev_text) if prev_text else "", shown, unit["kind"])
        label = parse_completion(generate(prompt), unit["kind"])
        used_fallback = label is None
        if used_fallback:
            failed += 1
            label = fallback_label(unit, known)
        if unit.get("speaker_hint"):
            label["speaker"] = unit["speaker_hint"]
            if label["gender"] == "none":
                label["gender"] = "unknown"
        rec = dict(unit)
        rec.update(label)
        rec["tone_shift"] = bool(prev_label) and prev_label["emotion"] != label["emotion"]
        rec["fallback"] = used_fallback
        out.append(rec)
        update_known(known, label["speaker"])
        prev_label, prev_text = label, shown
        if progress and ((i + 1) % 10 == 0 or i + 1 == len(units)):
            rate = (i + 1) / (time.time() - start)
            print(f"  labeled {i + 1}/{len(units)}  ({rate:.1f} units/s, fallbacks so far: {failed})")
    from voice_select import speaker_genders
    resolved = speaker_genders(out)
    for rec in out:
        if rec["speaker"] != NARRATOR:
            rec["gender_model"] = rec["gender"]
            rec["gender"] = resolved[rec["speaker"]]
    return out


def load_generator(model_path, model_name=MODEL):
    import torch
    import transformers
    from packaging import version
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    path = str(model_path)
    has_adapter = os.path.exists(os.path.join(path, "adapter_config.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_key = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"

    try:
        tok = AutoTokenizer.from_pretrained(path)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if has_adapter:
        from peft import PeftModel
        if device != "cuda":
            raise RuntimeError("The adapter runs in 4-bit and needs a GPU. On a CPU use the merged model folder.")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb,
                                                    device_map={"": 0}, **{dtype_key: torch.float16})
        model = PeftModel.from_pretrained(base, path)
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(path, **{dtype_key: dtype}).to(device)
    model.eval()
    print(f"model loaded: {'base + LoRA adapter (4-bit)' if has_adapter else 'merged model'} on {device}")

    def generate(prompt):
        enc = tok(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def annotate(units, adapter_dir, model_name=MODEL):
    return label_units(units, load_generator(adapter_dir, model_name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("units")
    ap.add_argument("--adapter", required=True, help="adapter folder or merged model folder")
    ap.add_argument("--out", default="labels.json")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    units = json.loads(Path(args.units).read_text(encoding="utf-8"))
    labels = annotate(units, args.adapter, args.model)
    Path(args.out).write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(labels)} labeled units -> {args.out}")


if __name__ == "__main__":
    main()
