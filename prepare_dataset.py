"""
Dataset Preparation Pipeline for Magpie-TTS Fine-Tuning.

Prepares raw audio datasets (local or Hugging Face) for NVIDIA NeMo Magpie-TTS:
1. Resamples audio to 22,050 Hz Mono (required by NanoCodec).
2. Cleans text and filters out unsupported numbers/symbols.
3. Filters clips by duration (e.g. 0.5s to 12.0s).
4. Assigns cross-reference context audio and context text for voice conditioning.
5. Shuffles and exports train.json and val.json manifests.
"""

import argparse
import json
import os
import random
import re
from typing import List, Dict, Any
import soundfile as sf

# Unsupported characters that can crash or cause tokenization issues during training
UNSUPPORTED_CHARS = [
    '٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
]


def clean_arabic_text(text: str) -> str:
    """Normalize Arabic text and remove stray non-Arabic characters."""
    if not text:
        return ""
    # Strip tatweel and excessive whitespace
    text = re.sub(r'ـ+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def prepare_dataset(
    dataset_name: str,
    output_dir: str = "data",
    split: str = "train",
    val_ratio: float = 0.1,
    min_duration: float = 0.5,
    max_duration: float = 12.0,
    text_column: str = "cleaned_dialectal_transcription",
    audio_column: str = "audio",
):
    """
    Loads dataset, processes audio and manifests.
    """
    from datasets import load_dataset, Audio

    os.makedirs(output_dir, exist_ok=True)
    audio_export_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_export_dir, exist_ok=True)

    print(f"📦 Loading dataset '{dataset_name}' (split: {split})...")
    ds = load_dataset(dataset_name, split=split)

    # Enforce 22,050 Hz resampling for NanoCodec
    print("🔄 Resampling audio to 22,050 Hz Mono...")
    ds = ds.cast_column(audio_column, Audio(sampling_rate=22050))

    entries: List[Dict[str, Any]] = []
    skipped_chars = 0
    skipped_duration = 0

    print(f"⚙️ Processing {len(ds)} samples...")
    for i, sample in enumerate(ds):
        # Extract and clean text (support both Saudi 'cleaned_...' and Emirati 'clean_...' columns)
        raw_text = (
            sample.get(text_column)
            or sample.get("cleaned_dialectal_transcription")
            or sample.get("clean_dialectal_transcription")
            or sample.get("dialectal_transcription")
            or sample.get("text")
            or sample.get("normalized_text", "")
        )
        text = clean_arabic_text(raw_text)

        if not text:
            continue

        # Skip rows containing numbers or problematic characters
        if any(char in text for char in UNSUPPORTED_CHARS):
            skipped_chars += 1
            continue

        audio_data = sample[audio_column]
        duration = len(audio_data["array"]) / audio_data["sampling_rate"]

        # Filter out bounds
        if duration < min_duration or duration > max_duration:
            skipped_duration += 1
            continue

        # Determine stable filename
        base_name = sample.get("filename")
        if base_name:
            base_name = os.path.splitext(os.path.basename(base_name))[0]
        else:
            base_name = f"sample_{i:05d}"

        wav_filename = f"{base_name}.wav"
        wav_filepath = os.path.join(audio_export_dir, wav_filename)

        # Write WAV file
        sf.write(wav_filepath, audio_data["array"], audio_data["sampling_rate"])

        # Relative path from dataset root for portability
        entries.append({
            "audio_filepath": os.path.join("audio", wav_filename).replace("\\", "/"),
            "text": text,
            "duration": round(duration, 3),
            "context_audio_duration": 0.0,
        })

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1} / {len(ds)} samples...")

    print(f"\n📊 Summary:")
    print(f"  Valid samples:     {len(entries)}")
    print(f"  Skipped (chars):   {skipped_chars}")
    print(f"  Skipped (duration):{skipped_duration}")

    if not entries:
        raise RuntimeError("No valid samples remaining after filtering! Check dataset column names.")

    # Assign reference context audio for voice conditioning
    print("🔗 Assigning context audio references for voice conditioning...")
    n = len(entries)
    for i in range(n):
        ctx_idx = random.randint(0, n - 1)
        while ctx_idx == i and n > 1:
            ctx_idx = random.randint(0, n - 1)

        entries[i]["context_audio_filepath"] = entries[ctx_idx]["audio_filepath"]
        entries[i]["context_text"] = entries[ctx_idx]["text"]

    # Shuffle and split
    random.seed(42)
    random.shuffle(entries)

    split_idx = int((1.0 - val_ratio) * len(entries))
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]

    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Dataset preparation complete!")
    print(f"  📁 Audio directory:  {audio_export_dir}")
    print(f"  📄 Train manifest:  {train_path} ({len(train_entries)} samples)")
    print(f"  📄 Val manifest:    {val_path} ({len(val_entries)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for NeMo Magpie-TTS fine-tuning.")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="AhmedEladl/saudi-dialect-speech-female",
        help="Hugging Face dataset identifier (e.g. AhmedEladl/saudi-dialect-speech-female or AhmedEladl/emirates-dialect-speech-male)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data",
        help="Output directory to store audio and manifests",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split proportion (default: 0.1)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="cleaned_dialectal_transcription",
        help="Column name containing transcription text",
    )

    args = parser.parse_args()
    prepare_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        text_column=args.text_column,
    )


if __name__ == "__main__":
    main()
