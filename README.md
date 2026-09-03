# 🎙️ Magpie-TTS Arabic Dialects (Saudi & Emirati)

Fine-tuned text-to-speech models built on NVIDIA's **[Magpie-TTS Multilingual (357M)](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)** architecture, specialized for natural, expressive conversational **Saudi Arabic (`ar-SA`)** and **Emirati Arabic (`ar-AE`)**.

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Saudi%20Model-blue)](https://huggingface.co/AhmedEladl/Magpie-TTS-Saudi-Arabic)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Emirati%20Model-orange)](https://huggingface.co/AhmedEladl/Magpie-TTS-Emirates-Arabic)
[![NVIDIA NeMo](https://img.shields.io/badge/Framework-NVIDIA%20NeMo%203.0+-76B900)](https://github.com/NVIDIA-NeMo/NeMo)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## 🌟 Highlights

- **Native Regional Dialect Support:** Fine-tuned to capture natural pitch, cadence, and colloquial vocabulary of Saudi and Emirati Arabic.
- **High-Fidelity Discrete Neural Audio:** Powered by NVIDIA's neural audio codec **[NanoCodec 22kHz](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps)** (1.89 kbps, 21.5 fps).
- **Clean, Lightweight Inference:** Automated model downloading from Hugging Face Hub, in-memory caching, and CLI / Python execution.
- **Interactive Web App:** Ready-to-use Gradio interface with ZeroGPU compatibility for Hugging Face Spaces.
- **End-to-End Training Pipeline:** Full Standard GPU orchestration for data preparation, vocabulary alignment, and fine-tuning.

---

## 📦 Project Structure

```text
magpie-arabic-tts/
├── inference.py          # Run speech synthesis (CLI or standalone script)
├── app.py                # Gradio web application for interactive synthesis
├── prepare_dataset.py    # Resampling (22kHz), filtering & NeMo manifest generator
├── train.py              # Standard GPU fine-tuning pipeline
├── eval_prompts.py       # 72 standard benchmark prompts for evaluation
├── requirements.txt      # Pinned environment dependencies
├── .gitignore            # Git ignore configuration
└── README.md             # Project documentation
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone repository
git clone https://github.com/AhmedEladl/magpie-arabic-tts.git
cd magpie-arabic-tts

# Install dependencies
pip install -r requirements.txt
```

> **Requirements:** Python 3.10+, PyTorch 2.1+, NVIDIA GPU with CUDA recommended for fast synthesis (CPU fallback supported).

---

## 🚀 Inference

### Option 1: Run via Command Line

```bash
# Saudi Dialect
python inference.py --dialect saudi --text "صباح الخير، وش أخباركم اليوم؟" --output saudi.wav

# Emirati Dialect
python inference.py --dialect emirati --text "الجو اليوم وايد حار، خلنا نروح مكان بارد." --output emirati.wav
```

---

### Option 2: Run via Python Script

```python
import soundfile as sf
from huggingface_hub import hf_hub_download
from nemo.collections.tts.modules.magpietts_inference.utils import ModelLoadConfig, load_magpie_model

# 1. Download Model & Codec from Hugging Face Hub
print("Downloading models...")
model_path = hf_hub_download(repo_id="AhmedEladl/Magpie-TTS-Saudi-Arabic", filename="Magpie-TTS-Saudi-Female.nemo")
codec_path = hf_hub_download(repo_id="nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps", filename="nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo")

# 2. Load the fine-tuned model and codec
config = ModelLoadConfig(nemo_file=model_path, codecmodel_path=codec_path)
model, _ = load_magpie_model(config)
model.eval().cuda()

# 3. Generate Audio
prompt = "الذكاء الاصطناعي صار جزء أساسي من حياتنا اليومية، وتطوير نماذج تدعم لهجتنا خطوة جداً مهمة."
print("Generating audio...")
res = model.do_tts(transcript=prompt, language="ar-SA", apply_TN=False)

# 4. Save the output
audio = res[0].cpu().numpy()
if len(audio.shape) == 2:
    audio = audio[0]

sf.write("output.wav", audio, 22050)
print("✅ Audio saved successfully to output.wav")
```

_(For Emirati dialect, replace repo with `AhmedEladl/Magpie-TTS-Emirates-Arabic`, filename with `Magpie-TTS.nemo`, and use `language="ar-AE"`)._

---

### 4. Interactive Web UI (Gradio)

Launch the interactive web demo locally:

```bash
python app.py
```

The web UI provides dialect switching, custom text input with RTL rendering, sample prompts, and audio playback.

---

## 🧠 Technical Details & Tokenizer Fix

Inside NeMo's text-to-speech framework, `LANGUAGE_TOKENIZER_MAP` defines tokenizer routing for standard language codes. By default, NeMo maps generic `"ar"` but omits regional codes `"ar-SA"` and `"ar-AE"`.

When calling `model.do_tts(language="ar-SA")` without explicit mapping, NeMo defaults to the English phoneme tokenizer (`CMUDict`), which skips all Arabic characters and generates an empty audio array.

This repository automatically injects the proper tokenizer mapping:

```python
from nemo.collections.tts.parts.utils.tts_dataset_utils import LANGUAGE_TOKENIZER_MAP

LANGUAGE_TOKENIZER_MAP["ar-SA"] = ["arabic_SA_chartokenizer"]
LANGUAGE_TOKENIZER_MAP["ar-AE"] = ["arabic_SA_chartokenizer"]
```

---

## 🏋️ Fine-Tuning Pipeline

### Step 1: Prepare Training Data

Prepare any Arabic speech dataset from Hugging Face or local files:

```bash
# For Saudi Arabic
python prepare_dataset.py --dataset "AhmedEladl/saudi-dialect-speech-female" --output-dir "./data_saudi"

# For Emirati Arabic
python prepare_dataset.py --dataset "AhmedEladl/emirates-dialect-speech-male" --output-dir "./data_emirati"
```

This step:

- Resamples all audio to **22,050 Hz Mono** (matching NanoCodec requirements).
- Filters clips under 0.5 seconds or over 12 seconds.
- Removes unsupported numerals and problematic symbols.
- Randomly pairs reference context audio (`context_audio_filepath`) to satisfy Magpie's speaker conditioning.

### Step 2: GPU Training

Run fine-tuning locally or on a standard cloud GPU instance (e.g., RunPod, AWS, local server):

```bash
python train.py --dialect saudi --data-dir ./data --batch-size 2 --epochs 10
```

Training parameters used:

- **Optimizer:** AdamW (`lr=5e-6`)
- **Precision:** 32-bit float (FP32)
- **Batch Size:** 2 with 2 gradient accumulation steps (effective batch size = 4)
- **Context Duration:** 3.0s – 10.0s
- **Epochs:** 10

---

## 📚 References & Official Sources

- **[Magpie-TTS Finetuning Guide — NVIDIA NeMo Speech](https://docs.nvidia.com/nemo/speech/nightly/tts/magpietts-finetuning.html):** Official NVIDIA documentation for adapting Magpie-TTS to custom speakers, low-resource languages, and regional dialects.
- **[NVIDIA Magpie-TTS Multilingual (357M)](https://huggingface.co/nvidia/magpie_tts_multilingual_357m):** Pretrained multilingual base checkpoint on Hugging Face.
- **[NVIDIA NanoCodec 22kHz](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps):** Discrete neural audio codec model used for acoustic token decoding.
- **Foundational Research:**
  - _Improving Robustness of LLM-based Speech Synthesis by Learning Monotonic Alignment_ ([arXiv:2406.17957](https://arxiv.org/abs/2406.17957)).
  - _Align2Speak: Improving TTS for Low Resource Languages via ASR-Guided Preference Optimization_ ([arXiv:2509.21718](https://arxiv.org/abs/2509.21718)).

---
