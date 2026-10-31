# Expressive Audiobook Narration with a Fine-Tuned Narrative-Direction Model

A generative-AI pipeline that turns a book (PDF or text) into a multi-voice, emotionally paced audiobook. A language model that we fine-tuned reads each passage and decides **who is speaking** and **how it should be performed**; a text-to-speech engine then voices every character differently and adjusts speed, pitch and volume to the emotion.

## Problem

Standard text-to-speech reads a whole book in one flat voice. Audiobooks need a different voice per character and delivery that follows the story: slower for grief, faster for danger. Doing this needs an understanding of narrative text, which is a language task rather than a speech task.

## Pipeline

```
book.pdf / .txt
   |  pipeline/prepare.py     extract text, remove headers/page numbers, split into units
   v                          (narration units and quoted-speech units)
units
   |  pipeline/annotate.py    FINE-TUNED MODEL: speaker, gender, emotion, intensity, pace
   v                          (one unit at a time, carrying character and emotion context)
labels.json
   |  pipeline/voice_select.py + voice_map.py + narrate.py
   v                          per-character voices, emotion -> rate/pitch/volume, pauses
audiobook.mp3 + manifest.json
```

## What is trained, and what is not

| Component | Source |
|---|---|
| Narrative-direction model (labels each passage) | **Trained by us**: Qwen2.5-1.5B-Instruct fine-tuned with 4-bit QLoRA |
| Training data | **Built by us**: 20 public-domain novels, labeled by a large teacher LLM, speaker/gender checked by hand |
| Speech synthesis | Pretrained neural voices (edge-tts). Not trained by us |

## Dataset

- 20 Project Gutenberg novels, 300 consecutive-excerpt units each (6000 units). Each unit is narration or one quoted line.
- **Teacher labels** from a large LLM (gpt-oss-120b via Groq, with Cerebras/Gemini fallbacks) for `speaker, gender, emotion, intensity, pace`. Closed label sets: 9 emotions, 3 intensities, 3 paces.
- **Split by book**, never by passage, so the model cannot memorise characters: 16 books train (4800 units), 1 book validation (300), 3 books test (900).
- **Gold set**: 250 units from the test books (125 dialogue). Speaker and gender were corrected by hand. Emotion, intensity and pace were not human-reviewed, so they measure fidelity to the teacher.
- The teacher's own accuracy against the hand corrections: **80%** speaker, **88%** gender (dialogue units).

## Model and training

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`, 4-bit NF4, LoRA r=16, alpha=32, dropout 0.05 on all attention and MLP projections.
- 2 epochs, learning rate 2e-4 (cosine), effective batch size 16, max length 768 tokens, loss on the output JSON only.
- 4798 training examples. About 52 minutes on a free Colab T4.

## Results (gold set, 250 units)

| System | Valid JSON | Emotion acc / macro-F1 | Intensity acc / macro-F1 | Pace acc / macro-F1 | Speaker acc* | Gender acc* |
|---|---|---|---|---|---|---|
| Majority baseline | 100% | 26.8 / 0.05 | 59.2 / 0.25 | 73.6 / 0.28 | 25.6 | 65.6 |
| Base model, zero-shot | 76.0% | 22.8 / 0.22 | 19.2 / 0.19 | 17.2 / 0.18 | 15.2 | 59.2 |
| Fine-tuned, adapter + 4-bit (as trained) | 100% | 62.8 / 0.61 | 74.8 / 0.69 | 76.0 / 0.64 | 44.8 | 65.6 |
| **Fine-tuned, frozen merged model (shipped)** | **100%** | **65.2 / 0.63** | **75.2 / 0.67** | **76.4 / 0.64** | **48.0** | **72.8** |
| LLM teacher (reference) | 100% | not scored | not scored | not scored | 80.0 | 88.0 |

\* dialogue units only. Emotion, intensity and pace are scored against unreviewed teacher labels. Pace is dominated by "medium" (about 74%), so macro-F1 is the fair measure there. Merging the adapter changed no score by more than noise (the gold set is small); 169 of 250 units get identical labels. Inference takes about 0.44 s per unit on a T4 (batch 8).

Figures for the report are produced by `training/make_figures.py`.

## Folder structure

```
app.py                 web app: upload a PDF, get an expressive audiobook
models/narration_director/   frozen model: one standalone folder (merged weights, about 3 GB)
adapter_final/               the same trained model as a small LoRA adapter (about 70 MB)
dataset/    schema.py            label format and prompt format (single source of truth)
            build_dataset.py     download books, clean, split into units, pick excerpts
            label_with_teacher.py  teacher-LLM labeling (resumable)
            split.py             book-level train/val/test split, gold-set sheet
training/   finetune.ipynb       QLoRA fine-tuning (Colab T4)
            eval.py              evaluation against baselines (Colab T4)
            export_model.ipynb   merge the adapter into the frozen model and verify it
            make_figures.py      loss curve, comparison chart, confusion matrix
            demo.ipynb           live demo in Colab: upload a PDF, hear the result
pipeline/   prepare.py           PDF/text to units (prose or script format)
            script_mode.py       screenplay-style "NAME: line" handling
            annotate.py          loads the frozen model and labels each unit
            voice_select.py      voices per character, gender from names
            voice_map.py         emotion, intensity, pace -> rate, pitch, volume, pauses
            narrate.py           speech synthesis and audio assembly
            run.py               command-line runner
            llm_client.py, config.py   LLM client used only by the labeling step
data/       raw_books/ units/ labeled/ splits/ gold/ sample files
results/    eval_results.json, figures/, audio samples
```

## The frozen model

Training produces the model once. After that nothing is retrained: every PDF is just input to the saved model.

| Form | What it is | Needs |
|---|---|---|
| `adapter_final/` | only the weights we learned (LoRA), about 70 MB, applied on top of Qwen2.5-1.5B-Instruct | NVIDIA GPU, base model downloaded from Hugging Face |
| `models/narration_director/` | adapter merged into the base model: one standalone folder with a model card | NVIDIA GPU, or CPU (slow) |

`training/export_model.ipynb` creates the second form and checks it against the evaluation of the first on the gold set.

## How to run

**1. Build the dataset (local)**
```
pip install -r requirements.txt
python dataset/build_dataset.py
python dataset/label_with_teacher.py
python dataset/split.py
python dataset/split.py --build-gold      # after correcting data/gold/gold_review.csv
```
**2. Train (Colab T4)**: upload `data/splits/train.jsonl` and `val.jsonl` to Drive and run `training/finetune.ipynb`.

**3. Evaluate (Colab T4)**: run `training/eval.py`, then `python training/make_figures.py --eval eval_results.json --state trainer_state.json`.

**4. Freeze the model (Colab T4, once)**: run `training/export_model.ipynb`, then download `models/narration_director` from Drive into the project.

**5. Use it**
```
# web app (GPU machine, or Colab with --share)
python app.py

# command line
python pipeline/run.py --input book.pdf --adapter models/narration_director --out output/book.mp3 --also-flat --max-units 100 --start-page 5 --end-page 8

# any machine without a GPU: narrate previously produced labels
python pipeline/run.py --labels output/book_labels.json --out output/book.mp3
```
Speech synthesis needs an internet connection. `training/demo.ipynb` runs the same pipeline in Colab and plays the flat and expressive versions side by side.

## Limitations

- Speaker attribution (44.8%) is the weakest part. It is better than baselines but well below the teacher (80%), so character voices are sometimes wrong.
- Emotion, intensity and pace labels come from an LLM and are not human-verified. Emotion boundaries such as tension / reflection / neutral are subjective.
- Labeling is sequential (each prompt depends on the previous prediction), so a full book takes a while.
- Only double-quote dialogue is detected. Scanned PDFs (no text layer) are not supported. English only.
- The gold set is small (250 units, 125 dialogue), so differences of a few points are within noise.

## Future work

Larger speaker-focused training data, a speaker-name memory with alias resolution, a trained prosody model in place of the rule table, and an open-source TTS to remove the online dependency.
