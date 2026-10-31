# Expressive AI Audiobook Narration

A fine-tuned language model that turns books into expressive, multi-character audiobooks.

Instead of reading an entire book in one voice, the system analyses individual passages, identifies speakers, predicts emotions and selects an appropriate delivery style. A text-to-speech engine then generates narration with different voices and adjusts speaking rate, pitch and pauses to reflect the story.

The project combines a fine-tuned Qwen model, a custom-built training dataset and a speech synthesis pipeline.

## Overview

Traditional text-to-speech systems generally focus on converting text into speech. They do not necessarily understand who is speaking or how a particular passage should sound.

This project addresses that problem by introducing a narrative-direction stage before speech synthesis.

For each passage, the model predicts:

* **Speaker:** Who is speaking, including the narrator.
* **Gender:** Used to help select an appropriate voice.
* **Emotion:** The emotional character of the passage.
* **Intensity:** How strongly the emotion should be expressed.
* **Pace:** Whether the passage should be delivered slowly, at a normal pace or quickly.

These predictions are then converted into voice and speech settings.

## How It Works

The system follows three main stages:

**1. Text preparation**

A PDF or text file is processed to extract readable content. The text is cleaned and divided into smaller units of narration and dialogue.

**2. Narrative analysis**

The fine-tuned Qwen model processes each unit and predicts its speaker, gender, emotion, intensity and pace. Previous character and emotion information is carried forward to maintain context between passages.

**3. Speech generation**

The predictions are used to select voices and adjust speech rate, pitch, volume and pauses. The final output is an expressive audiobook, with an optional flat-voice version for comparison.

```text
             PDF / TXT
                 |
                 v
       Text Extraction & Cleaning
          pipeline/prepare.py
                 |
                 v
           Text Segmentation
       Narration and Dialogue Units
                 |
                 v
        Fine-Tuned Qwen Model
          pipeline/annotate.py
                 |
                 v
          Narrative Labels
    Speaker | Emotion | Intensity
             Pace | Gender
                 |
                 v
          Voice Selection
       Voice Mapping & Controls
                 |
                 v
          Edge TTS Engine
                 |
                 v
        Expressive Audiobook
          + Flat Comparison
```

## Model and Technology

The narrative-direction model is based on **Qwen2.5-1.5B-Instruct**, fine-tuned using 4-bit QLoRA.

| Component               | Technology                                |
| ----------------------- | ----------------------------------------- |
| Base language model     | Qwen2.5-1.5B-Instruct                     |
| Fine-tuning             | LoRA / QLoRA                              |
| Quantization            | 4-bit NF4                                 |
| Model training          | PyTorch, Transformers, PEFT, BitsAndBytes |
| Dataset processing      | Python, JSONL                             |
| Speech synthesis        | Edge TTS                                  |
| PDF processing          | PyMuPDF                                   |
| Application interface   | Gradio                                    |
| Teacher-model labelling | Groq, Cerebras and Gemini APIs            |

**What was trained:** The narrative-direction model that analyses passages and generates structured labels.

**What was not trained:** The speech synthesis engine. The project uses pretrained Edge TTS voices and controls their delivery through the generated labels.

## Dataset Development

A custom dataset was prepared using 20 novels sourced from Project Gutenberg.

Each book contributed 300 selected units, giving a total of **6,000 units**.

The dataset preparation process involved:

1. Downloading and cleaning the original book text.
2. Removing unnecessary headers, footers and formatting.
3. Separating narration and quoted dialogue.
4. Selecting consecutive excerpts from different parts of each book.
5. Generating labels using a teacher LLM.
6. Reviewing speaker and gender labels manually.
7. Splitting the dataset at the book level.

### Dataset distribution

| Split      |  Books |     Units |
| ---------- | -----: | --------: |
| Training   |     16 |     4,800 |
| Validation |      1 |       300 |
| Testing    |      3 |       900 |
| **Total**  | **20** | **6,000** |

The split was performed by book rather than by individual passage. This helps reduce character-level information leakage between training and testing.

### Annotation

The teacher model generated five labels for each unit:

* Speaker
* Gender
* Emotion
* Intensity
* Pace

The annotation process used a large language model through Groq, with Cerebras and Gemini as fallback providers.

A separate gold evaluation set contains 250 units, including 125 dialogue units. Speaker and gender labels were corrected manually. Emotion, intensity and pace labels were not independently human-reviewed.

The teacher's accuracy against the manual corrections was:

| Label   | Accuracy |
| ------- | -------: |
| Speaker |      80% |
| Gender  |      88% |

These figures are calculated on dialogue units.

## Model Fine-Tuning

The base model was fine-tuned using QLoRA with 4-bit NF4 quantization.

| Parameter               | Configuration                 |
| ----------------------- | ----------------------------- |
| Base model              | Qwen2.5-1.5B-Instruct         |
| Quantization            | 4-bit NF4                     |
| LoRA rank               | 16                            |
| LoRA alpha              | 32                            |
| LoRA dropout            | 0.05                          |
| Target modules          | Attention and MLP projections |
| Epochs                  | 2                             |
| Learning rate           | 2e-4                          |
| Scheduler               | Cosine                        |
| Effective batch size    | 16                            |
| Maximum sequence length | 768                           |
| Training examples       | 4,798                         |

Training took approximately 52 minutes on a free Google Colab NVIDIA T4 GPU.

The trained LoRA adapter is included in this repository under `adapter_final/`.

The larger merged model checkpoint is not included because of its size. It can be exported using `training/export_model.ipynb`.

## Evaluation Results

The model was evaluated on the 250-unit gold set against a majority-class baseline, the original base model and the fine-tuned model.

| System                     | Valid JSON | Emotion Accuracy | Emotion Macro-F1 | Intensity Accuracy | Intensity Macro-F1 | Pace Accuracy | Pace Macro-F1 |
| -------------------------- | ---------: | ---------------: | ---------------: | -----------------: | -----------------: | ------------: | ------------: |
| Majority baseline          |       100% |            26.8% |             0.05 |              59.2% |               0.25 |         73.6% |          0.28 |
| Base model (zero-shot)     |      76.0% |            22.8% |             0.22 |              19.2% |               0.19 |         17.2% |          0.18 |
| Fine-tuned adapter (4-bit) |       100% |            62.8% |             0.61 |              74.8% |               0.69 |         76.0% |          0.64 |
| Fine-tuned merged model    |       100% |            65.2% |             0.63 |              75.2% |               0.67 |         76.4% |          0.64 |

Speaker and gender accuracy were evaluated separately on dialogue units:

| System                  | Speaker Accuracy | Gender Accuracy |
| ----------------------- | ---------------: | --------------: |
| Majority baseline       |            25.6% |           65.6% |
| Base model (zero-shot)  |            15.2% |           59.2% |
| Fine-tuned adapter      |            44.8% |           65.6% |
| Fine-tuned merged model |            48.0% |           72.8% |
| Teacher model           |            80.0% |           88.0% |

The merged model results were obtained from the locally exported checkpoint; that 3 GB checkpoint is not included in this repository.

**A note on evaluation:** Emotion, intensity and pace are measured against teacher-generated labels rather than independently human-verified labels. The gold set is relatively small, so small differences between model configurations should not be overinterpreted.

The evaluation figures are available in `results/figures/`, and the recorded metrics are in `results/eval_results.json`.

## Repository Structure

```text
audiobook-genai/
│
├── adapter_final/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── tokenizer files
│
├── data/
│   ├── raw_books/          Original book texts
│   ├── units/              Processed narration/dialogue units
│   ├── labeled/            Teacher-labelled data
│   ├── splits/             Train, validation and test sets
│   └── gold/               Gold evaluation set and review files
│
├── dataset/
│   ├── build_dataset.py
│   ├── label_with_teacher.py
│   ├── schema.py
│   └── split.py
│
├── pipeline/
│   ├── prepare.py          Text extraction and segmentation
│   ├── annotate.py         Model inference and labelling
│   ├── voice_select.py     Character voice selection
│   ├── voice_map.py        Emotion-to-speech mapping
│   ├── narrate.py          Speech generation
│   ├── script_mode.py      Script-style input handling
│   └── run.py              Command-line pipeline
│
├── training/
│   ├── finetune.ipynb      QLoRA fine-tuning
│   ├── export_model.ipynb  Adapter merging and export
│   ├── demo.ipynb          Colab demonstration
│   └── make_figures.py     Evaluation visualisations
│
├── results/
│   ├── eval_results.json
│   ├── trainer_state.json
│   └── figures/
│
├── app.py                  Gradio application
├── check_setup.py
├── make_submission.py
├── requirements.txt
├── CODE_WALKTHROUGH.md
└── README.md
```

The merged base model is intentionally excluded from the repository. The included adapter is applied to the original Qwen base model downloaded from Hugging Face when running inference.

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Mehul-Jethwani/audiobook-genai.git
cd audiobook-genai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For model inference, install a CUDA-compatible PyTorch build and the additional model dependencies:

```bash
pip install torch transformers peft accelerate bitsandbytes
```

Use a PyTorch build compatible with your NVIDIA GPU and CUDA environment.

### 4. Run the application

```bash
python app.py
```

The application automatically looks for the included adapter in `adapter_final/`. The base Qwen model is downloaded from Hugging Face when required.

The included 4-bit adapter requires an NVIDIA GPU with CUDA support.

### 5. Run from the command line

```bash
python pipeline/run.py --input book.pdf --adapter adapter_final --out output/book.mp3 --also-flat
```

You can also limit processing to particular pages or a maximum number of units:

```bash
python pipeline/run.py --input book.pdf --adapter adapter_final --out output/book.mp3 --max-units 100 --start-page 5 --end-page 8
```

To generate speech from labels that have already been produced, without loading the language model:

```bash
python pipeline/run.py --labels output/book_labels.json --out output/book.mp3
```

Speech synthesis requires an internet connection.

### Testing the interface

To test the Gradio interface without loading the fine-tuned model:

```bash
python app.py --ui-test
```

This mode uses placeholder labels. It is intended only for testing the interface and speech pipeline, not for evaluating the model.

## Dataset Labelling and API Configuration

The dataset labelling scripts use external LLM providers. They support the following environment variables:

```env
GROQ_API_KEY=
GROQ_API_KEYS=

CEREBRAS_API_KEY=
CEREBRAS_API_KEYS=

GEMINI_API_KEY=
GEMINI_API_KEYS=
```

You can configure a single key or multiple comma-separated keys for a provider.

Keep API keys in a local `.env` file. Never commit credentials to the repository.

These API keys are used for dataset annotation; normal inference with the saved fine-tuned adapter does not require them.

## Limitations

There are several areas where the system can be improved:

* **Speaker attribution:** Identifying the correct character remains the weakest evaluated component. The model performs better than the baseline but still makes mistakes.
* **Emotion labels:** Emotion, intensity and pace are teacher-generated and have not been independently human-verified.
* **Sequential inference:** Predictions use previous context, so processing a complete book can take time.
* **Dialogue detection:** The current text segmentation primarily detects double-quoted dialogue.
* **PDF support:** Scanned PDFs without an extractable text layer are not supported.
* **Language support:** The current pipeline is designed for English text.
* **Speech synthesis:** Edge TTS requires an internet connection.

## Future Improvements

* Improve character identification and speaker consistency across longer stories.
* Introduce character memory and alias resolution.
* Replace rule-based speech adjustments with a learned prosody model.
* Explore offline speech synthesis.
* Expand the manually reviewed evaluation dataset.

## Author

**Mehul Jethwani**

B.Tech – Computer Science and Engineering (AI & ML)

Adani University, Ahmedabad

GitHub: [Mehul-Jethwani](https://github.com/Mehul-Jethwani)

---

*The raw book texts were sourced from Project Gutenberg. Users should check the applicable copyright status and Project Gutenberg terms before redistributing those texts. This repository does not currently include a separate software licence.*
