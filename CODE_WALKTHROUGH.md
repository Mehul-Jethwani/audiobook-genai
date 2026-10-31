# Code walkthrough and viva guide

How to use this: read section 1 and 2 first (10 minutes). Then read one file section at a time, with the file open next to it. Section 7 has likely viva questions. Section 8 lists claims you should NOT make.

---

## 1. The project in one minute (say this)

"Normal text-to-speech reads a whole book in one flat voice. I built a pipeline that makes an expressive audiobook. A language model that I fine-tuned reads each passage and decides who is speaking, their gender, the emotion, how strong it is, and the pace. Then a text-to-speech engine gives every character a different voice and changes speed, pitch and volume to match the emotion. The trained part is the labeling model: I built a dataset from 20 public-domain novels, labeled it with a large LLM, checked speaker and gender by hand on a gold set, fine-tuned a 1.5-billion-parameter model on it, and evaluated it on books it never saw."

---

## 2. The story: every step we took, in order

| # | Step | Files | Result |
|---|---|---|---|
| 1 | Idea: replace API calls by a model we train | - | Needed a dataset, training, evaluation |
| 2 | Define what the model must output (the "label schema") | `dataset/schema.py` | 5 fields: speaker, gender, emotion, intensity, pace |
| 3 | Download 20 novels, clean them, cut into "units" | `dataset/build_dataset.py` | 6000 units (300 per book) |
| 4 | A big LLM (the "teacher") labels every unit | `dataset/label_with_teacher.py` | `data/labeled/*.jsonl` |
| 5 | Split by book into train / validation / test, and make a gold set | `dataset/split.py` | 16 train books, 1 val, 3 test; 250 gold units |
| 6 | I corrected speaker and gender on the gold set by hand | `data/gold/gold_review.csv` | Teacher accuracy: 80% speaker, 88% gender |
| 7 | Fine-tune Qwen2.5-1.5B with QLoRA on Colab | `training/finetune.ipynb` | `adapter_final/` |
| 8 | Evaluate against baselines on the gold set | `training/eval.py` | Results table |
| 9 | Freeze the model: merge adapter into one standalone folder, verify | `training/export_model.ipynb` | `models/narration_director/` |
| 10 | Build the pipeline: PDF to units to labels to audio | `pipeline/*.py` | Working audiobook |
| 11 | Handle screenplays ("SAM: ..." lines) and name-based voice gender | `pipeline/script_mode.py`, `voice_select.py` | Distinct character voices |
| 12 | Live demo in Colab, and a web app | `training/demo.ipynb`, `app.py` | Demo for the evaluator |
| 13 | Package the submission safely | `make_submission.py` | Zip without keys |

Bugs we found and fixed along the way (good to mention, it shows real engineering):
- Windows line-ending bug: files downloaded on Windows turned every line into its own paragraph, so sentences were cut in half. Fixed in `download_book` and `load_raw`.
- PDF page numbers blocked paragraphs from joining across pages. Fixed with header/footer detection in `prepare.py`.
- New library versions (`warmup_ratio` removed, `torchao` clash, Microsoft speech 403). Fixed with version-safe code.
- The model's gender guess was no better than guessing, so voice gender now comes from a name lookup.

---

## 3. Big picture

### Training time (done once)
```
Gutenberg books -> units -> teacher LLM labels -> split by book -> fine-tune -> adapter_final/
                                                       |                          |
                                                  gold set (hand-checked)   merge -> models/narration_director/
                                                       |
                                                   evaluation
```

### Run time (every PDF)
```
PDF -> prepare.py (text, units) -> annotate.py (frozen model labels each unit)
    -> voice_select.py (who gets which voice) + voice_map.py (emotion -> speed/pitch/volume)
    -> narrate.py (speech for each unit, joined into one mp3)
```
Nothing is trained at run time. The PDF is only input to the saved model.

### Where things live
- **Code files**: instructions only, no model inside.
- **Base model** (Qwen2.5-1.5B-Instruct): pretrained by Alibaba, downloaded from Hugging Face when needed.
- **`adapter_final/`**: the weights WE learned (LoRA), about 70 MB.
- **`models/narration_director/`**: base plus our weights merged into one folder, about 3 GB. This is the "frozen model".

---

## 4. Concepts you must understand (plain words)

- **LLM (large language model)**: a neural network that predicts the next token (piece of a word). If you give it instructions as text, it can follow them.
- **Prompt**: the exact text we give the model. Ours lists known characters, the previous emotion, the previous text, the type (narration or dialogue) and the target passage, and ends with `JSON:`.
- **Fine-tuning**: continuing to train a pretrained model on our own examples so it learns our task.
- **LoRA**: instead of changing all 1.5 billion weights, we add small extra matrices (rank 16) next to the existing ones and train only those. About 18 million numbers (my estimate from the settings) instead of 1.5 billion. The result is the "adapter".
- **Quantization / QLoRA**: the base model is stored in 4-bit numbers while training so it fits on a free 16 GB T4 GPU. The small LoRA weights are trained in higher precision.
- **Adapter vs merged model**: the adapter is the small file that sits on top of the base model. Merging adds it into the base weights so one standalone model remains. Same behaviour, no extra step at load time.
- **Teacher-student (distillation)**: a big, strong model (the teacher, a 120-billion-parameter LLM through an API) creates labels. A small model (the student) learns to copy them. The student is cheaper, faster, offline and owned by us. The teacher is better at speaker detection: the student does not beat the teacher.
- **Closed label set**: emotion must be one of 9 words, intensity one of 3, pace one of 3. This makes the output checkable and the task a classification problem.
- **Loss masking**: during training, the loss counts only the JSON answer tokens, not the prompt. Otherwise the model would waste effort learning to repeat the prompt.
- **Split by book**: train, validation and test books are different books. If passages from one book were in both train and test, the model could memorise character names and look better than it is.
- **Gold set**: 250 test units where I corrected speaker and gender myself.
- **Baselines**: (1) always guess the most common label; (2) the untouched base model with the same prompt (zero-shot). Our model must beat both to show training helped.
- **Accuracy vs macro-F1**: accuracy is the share correct. If 74% of units are "medium pace", always saying "medium" scores 74% and learns nothing. Macro-F1 averages the F1 score of each class equally, so rare classes count. We report both.
- **Prosody**: speed (rate), pitch and volume of speech. The speech engine accepts them as numbers like `+10%`, `-3Hz`.

---

## 5. File by file

Imports that appear in many files:
- `json`: read and write JSON/JSONL data files.
- `re`: regular expressions (pattern matching in text).
- `sys`: `sys.path` lets a script import files from other folders; `sys.exit` stops with a message.
- `pathlib.Path`: file paths that work on Windows and Linux. `Path(__file__).resolve().parent.parent` is "the project folder".
- `argparse`: command-line options like `--books alice`.
- `time`: timing and pauses.
- `collections.Counter`: counts how often each item appears.

A **unit** is one piece of text we label: either narration (no quoted speech) or dialogue (one quoted line, possibly with a short "said X" tag).

### 5.1 `dataset/schema.py` (the single source of truth)

Purpose: defines the labels and the prompt. Training, evaluation and the live pipeline all import it, so they can never disagree about the format.

Imports: `json` (turn labels into JSON text), `re` (clean names, find the JSON in model output).

Constants:
- `EMOTIONS`, `INTENSITIES`, `PACES`, `GENDERS`: the allowed words.
- `LABEL_KEYS`: the fixed order of the five fields.
- `MAX_KNOWN_CHARACTERS = 15`, `CONTEXT_WORDS = 80`: how many recent names and how many previous words go in the prompt.
- `*_ALIASES`: maps words the teacher sometimes uses ("sadness", "moderate") to our labels ("grief", "medium").
- `TASK_HEADER`: the instruction sentence at the top of every prompt.

Functions:
- `_norm(x)`: lowercase, strip spaces, `None` becomes empty string.
- `clean_speaker(raw)`: turns a name into UPPERCASE with no punctuation ("Mr. Darcy" becomes "MR DARCY"). Empty or "unknown" becomes `narrator`. Max 40 characters.
- `clean_label(raw, kind)`: takes a messy dictionary and returns a valid label or `None`. Steps: (1) must be a dict; (2) normalise emotion, intensity, pace and map aliases; (3) if any is outside the allowed set, return `None`; (4) if the unit is narration the speaker is forced to `narrator`; (5) narrator gets gender `none`, characters get male/female/unknown; (6) return the five fields. Every label from the teacher, the student or the CSV goes through this.
- `tail_words(text, n)`: last `n` words of a text (the context shown to the model).
- `update_known(known, speaker)`: keeps a most-recently-used list of character names, capped at 15.
- `build_prompt(...)`: builds the exact prompt string. This is what the model reads.
- `format_completion(label)`: the exact answer string the model must write: compact JSON with fixed key order.
- `parse_completion(text, kind)`: finds the first `{...}` in the model output, parses it, passes it through `clean_label`. Returns `None` if unusable.
- `make_examples(records)`: replays one book in order and produces training examples. It rebuilds the same running state (known characters, previous emotion, previous text) that the live pipeline will have. At the start of each excerpt (`window_start`) the context resets but known characters persist.

Say in the viva: "Training and inference share one prompt builder, so there is no train-test mismatch in the format."

### 5.2 `dataset/build_dataset.py` (books to units)

Purpose: download novels, clean them, cut them into units, pick excerpts.

Imports: `random` (seeded random excerpt choice), `urllib.request` and `urllib.error` (download without extra libraries).

Key parts:
- `BOOKS`: 20 slugs mapped to Project Gutenberg ids. The script prints each book's real title so a wrong id is visible.
- `download_book`: tries 3 URL patterns, normalises line endings, writes the file with `newline="\n"`. This fixes the Windows bug.
- `load_raw`: reads bytes and converts `\r\r\n` and `\r\n` to `\n`, so even a damaged file is healed.
- `strip_gutenberg`: cuts off the Project Gutenberg header and licence using the START and END marker lines.
- `clean_text`: removes `[Illustration]` markers and underscore italics.
- `is_junk`: drops chapter headings, page ornaments, all-caps headings, roman numerals and short title lines.
- `balanced_quotes`: true only if opening and closing curly quotes alternate correctly, not merely equal in number.
- `split_long`: splits narration longer than 140 words at sentence ends.
- `paragraph_to_units` (the heart): finds quoted speech with a regex (`QUOTE_SPAN`). Each quote becomes a dialogue unit. Narration between quotes shorter than 7 words ("said Alice,") stays attached to the quote. Longer narration becomes its own narration unit. A speech that continues across paragraphs (unbalanced quotes) stays whole.
- `merge_short_narration`: tiny narration pieces (under 25 words) merge with neighbours up to 120 words.
- `build_units`: strip, clean, split into paragraphs, drop junk, make units. The live pipeline reuses this function.
- `pick_windows`: ignores the first and last 5% of a book (front matter), then picks evenly spread random windows of 60 consecutive units. 300 units per book means 5 windows. Consecutive units are needed so context makes sense.
- `process_book`: seeded random per book (reproducible), writes `data/units/<book>.jsonl`, warns if over 5% of units start mid-sentence (broken file).

### 5.3 `dataset/label_with_teacher.py` (teacher labeling)

Purpose: a big LLM labels every unit.

Imports: `schema` (clean and update); `llm_client.call_llm_with_fallback` from your earlier project (tries several free providers, waits when rate-limited).

Key parts:
- `TEACHER_RULES`: the long instruction telling the teacher the five fields and what each emotion word means.
- `build_teacher_prompt`: numbers the passages `[1] [dialogue] text`, adds known characters and the previous text, and asks for exactly N JSON objects.
- `extract_json_array`: pulls a JSON list out of the reply even with extra chatter or code fences.
- `label_batch`: sends a batch (12 units). If the reply has the wrong number of objects, it splits the batch in half and retries. If one unit still fails, it gets `None`.
- `next_batch`: a batch never crosses from one excerpt to the next.
- `label_book`: resumable. It appends to the output file after each batch, so if the free API quota runs out you rerun the same command and it continues. It checks ids match so a changed units file is detected.

Say: "The teacher is a limitation: the student can never be much better than the labels it learns from. I measured the teacher on the gold set: 80% speaker, 88% gender."

### 5.4 `dataset/split.py` (splits and gold set)

Purpose: split by book, print label statistics, build the gold set.

Imports: `csv` (Excel-readable sheet), `random`, `Counter`.

Key parts:
- `load_books`: turns each labeled book into training examples with `make_examples`.
- `choose_books`: random but seeded split, or explicit with `--test` and `--val`. We forced test books `huck_finn, hound_baskervilles, little_women` and validation `christmas_carol` so the test set has female speakers too.
- `distribution` and `print_distribution`: show the share of each label, to catch classes too rare to learn.
- `sample_gold`: 125 dialogue and 125 narration units from the test books.
- `write_gold_review`: writes `gold_review.csv` (UTF-8 with BOM so Excel opens it correctly). It never overwrites your edits unless `--force`.
- `build_gold`: reads the corrected sheet, cleans each row, compares to the teacher label, writes `gold.jsonl` and the teacher-vs-human agreement.

### 5.5 `training/finetune.ipynb` (training, cell by cell)

1. GPU check (`torch.cuda.is_available`).
2. Install: removes the old `torchao`, installs `peft` (LoRA), `bitsandbytes` (4-bit), `datasets`, `accelerate`.
3. Config: model `Qwen/Qwen2.5-1.5B-Instruct`, `MAX_LEN=768`, `EPOCHS=2`, `LR=2e-4`, batch 4 with 4 accumulation steps (effective 16). Paths on Google Drive. `SMOKE=True` runs a 10-step test first.
4. Data: `encode` makes `input_ids = prompt tokens + answer tokens + end token`, and `labels = -100 for prompt tokens (ignored) + answer tokens`. That is the loss masking. Examples longer than 768 tokens are dropped, not cut (2 of 4800 were dropped).
5. Model: `BitsAndBytesConfig` loads the base in 4-bit NF4 with double quantization and fp16 compute. `prepare_model_for_kbit_training` readies it. `LoraConfig(r=16, alpha=32, dropout=0.05)` on all attention and MLP projections (`q,k,v,o,gate,up,down`). `get_peft_model` attaches the adapter. Trainable weights are cast to float32 for stable fp16 training.
6. Training: `TrainingArguments` with cosine learning-rate schedule, 5% warmup, fp16, gradient checkpointing (saves memory), logging every 20 steps, evaluation and checkpoint every 100 steps. Settings the installed library does not accept are filtered out, so a version change cannot crash it. `get_last_checkpoint` resumes after a Colab disconnect. About 52 minutes on a T4.
7. Save the adapter and tokenizer to Drive.
8. Preview: generates answers for 6 validation passages and prints them next to the gold answers.

### 5.6 `training/eval.py` (evaluation)

Purpose: score four systems on the 250-unit gold set.

Key parts:
- `accuracy`, `macro_f1`: for each class, F1 = 2 TP / (2 TP + FP + FN); macro-F1 is the average over classes present in the gold labels.
- `pairs`: collects (gold, predicted) pairs; speaker and gender are scored on dialogue units only (narration is trivially "narrator"); an invalid model answer counts as wrong.
- `score`: valid-JSON rate, accuracy and macro-F1 for emotion, intensity, pace; accuracy for speaker; accuracy and F1 for gender.
- `majority_preds`: the "always guess the most common" baseline. For speaker it guesses the most recently seen character.
- `generate`: batched generation (batch 8) with left padding, parsed by `parse_completion`.
- `main`: loads the base model and adapter, runs the baseline, the teacher row, the zero-shot base (adapter switched off with `disable_adapter`), and our model; prints the table and emotion confusion matrix; saves `eval_results.json` and `eval_predictions.jsonl`.
- `HUMAN_VERIFIED = ["speaker","gender"]`: the teacher's scores on the other three fields are not shown (shown as `-`), because those labels were never human-checked.

### 5.7 `training/export_model.ipynb` (freeze the model)

1. Loads the base model in fp16 (not 4-bit), attaches the adapter, calls `merge_and_unload()` (adds the learned matrices into the base weights), saves with `save_pretrained`, and saves the tokenizer.
2. Reloads the saved folder and scores it on the gold set with the same functions as `eval.py`. Prints both tables side by side and the share of units with identical labels.
3. Writes `MODEL_CARD.md` into the folder.

Why verify: the adapter was trained against a 4-bit base, and merging into an fp16 base changes numbers slightly. Our result: merged model scores the same or slightly higher on every field (within noise), and 169 of 250 units get exactly identical labels (67.6%).

### 5.8 `training/make_figures.py`

Reads `eval_results.json` and `trainer_state.json`, draws three PNGs with matplotlib: system comparison bars, emotion confusion matrix, loss curve.

### 5.9 `pipeline/prepare.py` (PDF or text to units)

Imports: `build_units` and the script-mode functions; `pymupdf` (inside the function, to read PDFs).

- `pdf_blocks`: for each page, `get_text("blocks")` returns text blocks (roughly paragraphs) with positions. Image blocks are skipped. A block in the top or bottom 8% of the page is flagged as a margin block. A hyphen at a line end is joined.
- `pdf_to_text`: margin blocks that are page numbers, or short text repeated on 3 or more pages (running headers), are dropped. If a paragraph does not end in sentence punctuation and the next block starts lowercase, they are joined (paragraph continues across a page). Raises a clear error for scanned PDFs.
- `read_text`: PDF or plain text (fixing line endings).
- `prepare`: reads the text; if it looks like a screenplay it uses script mode, otherwise `build_units`. Then applies skip and maximum units.

### 5.10 `pipeline/script_mode.py` (screenplays)

For text like `SAM: No way! LEO: Hey.`
- `MARKER`: a regex for "NAME:" where the name is 1 to 3 UPPERCASE words anywhere, or a Title-case name at the start of a line. `NOT_SPEAKERS` excludes words like NOTE, SCENE, INT.
- `looks_like_script`: at least 6 markers, at least 2 distinct names, and markers in at least 25% of paragraphs. This stops normal prose with a stray "Note:" from triggering it.
- `script_units`: text before the first marker becomes narration. Each marker starts a speech until the next marker. `(stage directions)` of 4 or more words become narration, shorter ones are dropped. Each speech unit gets `speaker_hint` (the printed name) and `prompt_text` (`"text" Sam said.`), so the model sees the same shape it was trained on.

Say honestly: in script mode the speaker comes from rules, because the script prints it. The model supplies emotion, intensity and pace.

### 5.11 `pipeline/annotate.py` (the frozen model labels the units)

Imports: `schema` functions; `torch`, `transformers`, `peft` inside `load_generator`.

- `fallback_label`: a safe neutral label if the model's answer is unusable.
- `label_units(units, generate)`: the main loop. For each unit: build the prompt from the running state, call the model, parse the answer. If the unit has a `speaker_hint`, overwrite the speaker with it. Compute `tone_shift` (emotion differs from the previous unit). Update known characters and previous label. After the loop, gender is resolved per character (name lookup first, else majority of the model's guesses) so the table shows one gender per character; the raw guess is kept as `gender_model`.
- `load_generator(model_path)`: chooses by folder content. If it has `adapter_config.json`: base model in 4-bit plus adapter (needs a GPU). Otherwise: a merged model loaded in fp16 on GPU or float32 on CPU. Returns a function `generate(prompt)` that does greedy decoding (`do_sample=False`, max 48 new tokens).
- Why sequential: each prompt contains the previous prediction, so units cannot be batched.

### 5.12 `pipeline/voice_select.py` (who gets which voice)

Imports: `Counter`; `edge_tts` and `gender_guesser` inside functions.

- `fetch_pools`: asks Microsoft's voice list for English neural voices, split by gender.
- `build_pools`: preferred voices first, then the rest; the narrator's voice is removed from the character pools. If the list cannot be fetched, built-in voices are used.
- `name_gender`: titles first (MR, SIR give male; MRS, MISS give female), then a name lookup library (`gender-guesser`), accepting "mostly male/female".
- `speaker_genders`: name lookup first; if unknown, the majority of the model's male/female guesses; else unknown.
- `assign_voices`: characters get voices in order of first appearance, round-robin inside their gender pool, so two characters never share a voice (until the pool runs out). Unknown gender alternates male and female.

### 5.13 `pipeline/voice_map.py` (emotion to speech settings)

- `EMOTION_STYLE`: for each emotion, a base (rate %, pitch Hz, volume %). Examples: grief (-14, -6, -8) is slower, lower and quieter; urgency (16, 4, 8) is faster, higher and louder.
- `INTENSITY_SCALE`: low 0.6, medium 1.0, high 1.4 multiplies the effect.
- `PACE_RATE`: slow -8, fast +8 added to the rate.
- `prosody(label)`: combines them, clamps to rate ±40%, pitch ±20 Hz, volume ±30%, and formats as `+6%`, `-2Hz` for the speech engine.
- `pause_ms(prev, cur)`: silence between units. Base 200 ms, +150 if the speaker changes, +100 on an emotion shift, max 600. If the previous unit ends with a comma, colon or dash (a quote followed by its "said X" tag), only 120 ms.

Honest note: this table is rule-based and hand-tuned by ear, not learned. The learned part is the model that picks emotion, intensity and pace.

### 5.14 `pipeline/narrate.py` (speech and audio)

Imports: `asyncio` (many speech requests at once), `hashlib` (cache keys), `ThreadPoolExecutor` (runs async code safely inside Jupyter), `pydub` (join audio, needs ffmpeg), `edge_tts` (speech service).

- `synth`: calls `edge_tts.Communicate(text, voice, rate, pitch, volume).save(...)`. It writes to a `.part` file, then renames, so a crash never leaves a broken file.
- `synth_retry`: at most 4 requests at once (a semaphore), 3 attempts with growing waits.
- `plan`: for each unit, chooses voice and prosody and builds a cache key from a hash of text, voice, rate, pitch and volume. Flat mode uses the narrator voice and zero prosody.
- `synthesize_all`: only requests what is not already cached, so reruns are fast.
- `assemble`: joins the clips with silence gaps, records start and end times of every unit (the manifest), exports a 128 kbps mp3.
- `narrate`: orchestrates, writes `<name>.mp3` and `<name>.json`.

### 5.15 `pipeline/run.py`, `app.py`, `training/demo.ipynb`, `make_submission.py`

- `run.py`: command-line runner for the three stages. With `--labels` it skips labeling, so no GPU is needed.
- `app.py`: a Gradio web page: upload, pages, maximum units; shows status, the table of decisions, and both audio versions. It finds `models/narration_director` or `adapter_final` automatically.
- `demo.ipynb`: the Colab demo. Cells 1 to 4 install and load the model once. The "Narrate a PDF" cell uploads a PDF, runs the three stages, prints timing, the table and two audio players. If speech fails, it downloads the labels so narration can run on a laptop. The last cell is an optional web app.
- `make_submission.py`: zips the project, skipping `.env`, caches, output and raw books, and refuses to run if it finds something that looks like an API key.

### 5.16 `llm_client.py` and `config.py` (from your earlier project)

Used only by the labeling step. The client tries several free LLM providers and waits and retries when one is rate-limited (your logs show 429 messages and "cooling down 65s"). I did not re-read them for this document: explain them in your own words, or send them to me.

---

## 6. Results to remember

Gold set: 250 units from 3 books the model never saw (125 dialogue).

| System | Emotion acc / F1 | Intensity acc / F1 | Pace acc / F1 | Speaker | Gender |
|---|---|---|---|---|---|
| Majority baseline | 26.8 / 0.05 | 59.2 / 0.25 | 73.6 / 0.28 | 25.6 | 65.6 |
| Base model, zero-shot (valid JSON 76%) | 22.8 / 0.22 | 19.2 / 0.19 | 17.2 / 0.18 | 15.2 | 59.2 |
| Ours: adapter + 4-bit | 62.8 / 0.61 | 74.8 / 0.69 | 76.0 / 0.64 | 44.8 | 65.6 |
| Ours: frozen merged model | 65.2 / 0.63 | 75.2 / 0.67 | 76.4 / 0.64 | 48.0 | 72.8 |
| LLM teacher (reference) | not scored | not scored | not scored | 80.0 | 88.0 |

Also: dataset 20 books, 6000 units; train 4800 (16 books), validation 300, test 900. Training about 52 minutes on a T4. Inference 0.44 s per unit batched; the live demo labeled 25 units in 65 s (about 2.6 s per unit, one at a time).

How to read them: the trained model clearly beats both baselines on emotion, intensity and pace (macro-F1 about 0.6 against 0.05 to 0.28). Speaker attribution improves a lot over baselines but stays far below the teacher. The gold set is small, so differences of 2 to 3 points between the adapter and merged model are within noise.

---

## 7. Likely viva questions

1. **Is this just an API wrapper?** No. The big LLM was used once, to create training labels. The model that runs is our fine-tuned model. The speech voices are pretrained and I say so.
2. **What did you train?** A LoRA adapter on Qwen2.5-1.5B-Instruct, using 4800 examples from 16 novels, 2 epochs, then merged into one frozen model.
3. **Why not train a model from scratch?** It needs far more data and compute than I have. Fine-tuning a pretrained model is the standard way to get a specialised model with a small dataset.
4. **What is LoRA and why use it?** It trains small extra matrices instead of all weights: far less memory and time, and the saved adapter is only tens of megabytes.
5. **What is 4-bit quantization for?** It shrinks the base model so training fits on a free 16 GB T4 GPU.
6. **Why a closed label set?** It makes outputs checkable and scoreable, and a small model learns a fixed vocabulary reliably. Result: 100% valid JSON.
7. **Why split by book?** To avoid leakage. Passages from the same book share characters and style.
8. **Why a gold set, and how was it made?** The teacher's labels can be wrong. I corrected speaker and gender on 250 test units. Emotion, intensity and pace were not human-checked, so those scores measure agreement with the teacher.
9. **How good is the teacher?** 80% on speaker and 88% on gender against my corrections, on dialogue units.
10. **Why is the student worse than the teacher on speakers?** It is 1.5 billion parameters against about 120 billion, and it must infer names from short context. It is the weakest part and I report it as a limitation.
11. **Why use the student at all?** It runs without API limits (my labeling hit rate-limit messages constantly), is owned and reproducible, and works offline once saved.
12. **What is macro-F1 and why use it?** The average F1 over classes. It stops a majority-class guess from looking good (pace: 74% accuracy for always saying "medium", but F1 0.28).
13. **What is zero-shot?** Giving the untouched base model the same prompt without training. It got only 76% valid JSON and 17 to 23% accuracy on the three fields.
14. **What is the adapter versus the merged model?** Same learned weights, different packaging. The merged one is a standalone folder. I verified that it scores the same or slightly higher on the gold set.
15. **How does it handle new characters?** The prompt carries a list of the 15 most recent names, so the model can reuse names; unseen characters are labeled from context.
16. **What if the PDF is a screenplay?** Script mode detects "NAME:" lines and takes the speaker from the script.
17. **How are voices chosen?** Names get gender from titles and a name lookup, then different voices from the matching pool. The narrator has its own voice.
18. **How does emotion change the voice?** A table maps emotion, intensity and pace to speed, pitch and volume. The table is rule-based; the model chooses its inputs.
19. **What are the limits?** English fiction with double-quote dialogue or script format; no scanned PDFs; speaker accuracy is moderate; labeling is sequential and takes about 2.6 s per unit on a T4; speech needs internet.
20. **How did you avoid copyright problems?** All training books are public domain from Project Gutenberg.
21. **What was the hardest bug?** The Windows line-ending bug: every line became a paragraph, which cut sentences in half. I found it by inspecting the labeled units.
22. **How do you know it is not overfitting?** Validation loss was tracked during training, and the evaluation uses test books the model never saw.
23. **What would you improve?** More speaker-focused data, alias handling ("Mr Darcy" and "Darcy"), a learned prosody model, and an offline speech engine.
24. **How long does a book take?** About 2.6 s per unit for labeling plus speech time. A 25-unit demo is about 2 minutes.
25. **Can it run without a GPU?** The merged model can run on a CPU but slowly; I have not measured it. Narration from saved labels needs no GPU.

---

## 8. Do NOT claim

- Do not say the emotion, intensity or pace labels were human-verified. Only speaker and gender were.
- Do not say the model beats the teacher. It does not.
- Do not say the speech voices were trained by you.
- Do not say it works on any PDF. It targets English fiction, and scanned PDFs are not supported.
- Do not quote differences of 1 to 3 points as real improvements. The gold set is small.
- Do not say the app runs fast on your laptop. It has no NVIDIA GPU; the live demo runs in Colab.
