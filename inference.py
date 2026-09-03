"""
Inference script for Magpie-TTS Arabic Dialects (Saudi & Emirati).

Synthesize high-fidelity Arabic speech using fine-tuned NVIDIA Magpie-TTS models.
"""

import argparse
import os
import sys
from typing import Optional, Tuple
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

# Dialect registry with verified Hugging Face checkpoints
DIALECT_REGISTRY = {
    "saudi": {
        "name": "Saudi Arabic (Female)",
        "repo_id": "AhmedEladl/Magpie-TTS-Saudi-Arabic",
        "filename": "Magpie-TTS-Saudi-Female.nemo",
        "lang_code": "ar-SA",
        "default_sample": "الذكاء الاصطناعي صار جزء أساسي من حياتنا اليومية، وتطوير نماذج تدعم لهجتنا خطوة جداً مهمة.",
    },
    "emirati": {
        "name": "Emirati Arabic (Male)",
        "repo_id": "AhmedEladl/Magpie-TTS-Emirates-Arabic",
        "filename": "Magpie-TTS.nemo",
        "lang_code": "ar-AE",
        "default_sample": "مرحبا الساع، شحالك؟ عساك طيب ومستانس إن شاء الله.",
    },
}

CODEC_REPO = "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps"
CODEC_FILENAME = "nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo"


def setup_tokenizer_patch():
    """
    Applies the critical Arabic tokenizer patch to NeMo's global language-tokenizer map.
    NVIDIA NeMo natively maps standard 'ar' but omits dialect codes 'ar-SA' and 'ar-AE',
    which causes NeMo to fall back to English phonemes and produce empty/silent audio.
    """
    try:
        from nemo.collections.tts.parts.utils.tts_dataset_utils import LANGUAGE_TOKENIZER_MAP
        LANGUAGE_TOKENIZER_MAP["ar-SA"] = ["arabic_SA_chartokenizer"]
        LANGUAGE_TOKENIZER_MAP["ar-AE"] = ["arabic_SA_chartokenizer"]
        LANGUAGE_TOKENIZER_MAP["ar"] = ["arabic_SA_chartokenizer"]
    except ImportError:
        pass


class MagpieArabicTTS:
    """Manager class for loading and running Magpie-TTS Arabic dialect models."""

    def __init__(self, dialect: str = "saudi", device: Optional[str] = None):
        dialect_key = dialect.lower().strip()
        if dialect_key not in DIALECT_REGISTRY:
            # Check aliases
            if dialect_key in ["ar-sa", "sa"]:
                dialect_key = "saudi"
            elif dialect_key in ["ar-ae", "uae", "ae"]:
                dialect_key = "emirati"
            else:
                raise ValueError(
                    f"Unsupported dialect: '{dialect}'. Choose from: {list(DIALECT_REGISTRY.keys())}"
                )

        self.dialect_key = dialect_key
        self.dialect_info = DIALECT_REGISTRY[dialect_key]

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.sample_rate = 22050
        self._load_model()

    def _load_model(self):
        setup_tokenizer_patch()

        from nemo.collections.tts.modules.magpietts_inference.utils import (
            ModelLoadConfig,
            load_magpie_model,
        )

        print(f"📦 Downloading/Checking codec: {CODEC_REPO}...")
        codec_path = hf_hub_download(repo_id=CODEC_REPO, filename=CODEC_FILENAME)

        print(f"📦 Downloading/Checking {self.dialect_info['name']} model: {self.dialect_info['repo_id']}...")
        model_path = hf_hub_download(
            repo_id=self.dialect_info["repo_id"],
            filename=self.dialect_info["filename"],
        )

        print(f"🚀 Loading {self.dialect_info['name']} model into memory...")
        config = ModelLoadConfig(nemo_file=model_path, codecmodel_path=codec_path)
        model, _ = load_magpie_model(config)
        model.eval()

        if self.device == "cuda":
            model = model.cuda()

        self.model = model
        print("✅ Model loaded and ready for synthesis!")

    def synthesize(
        self,
        text: str,
        speaker_index: int = 0,
        apply_tn: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize Arabic dialect speech from text.

        Args:
            text: Input Arabic text prompt.
            speaker_index: Speaker index integer (default: 0).
            apply_tn: Whether to apply NeMo Text Normalization (default: False).

        Returns:
            Tuple of (audio_array: np.ndarray, sample_rate: int)
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        setup_tokenizer_patch()

        lang_code = self.dialect_info["lang_code"]
        res = self.model.do_tts(
            transcript=text.strip(),
            language=lang_code,
            apply_TN=apply_tn,
            speaker_index=speaker_index,
        )

        audio = res[0].detach().cpu().numpy()
        if len(audio.shape) == 2:
            audio = audio[0]

        return audio, self.sample_rate

    def save(self, audio: np.ndarray, output_path: str):
        """Save synthesized audio array to a .wav file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, audio, self.sample_rate)
        print(f"💾 Audio saved successfully to: {output_path}")


def synthesize(
    text: str,
    dialect: str = "saudi",
    output_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """
    Convenience function to synthesize speech directly.

    Example:
        audio, sr = synthesize("يا هلا والله ومسهلا", dialect="saudi", output_path="hello.wav")
    """
    tts = MagpieArabicTTS(dialect=dialect, device=device)
    audio, sr = tts.synthesize(text)
    if output_path:
        tts.save(audio, output_path)
    return audio, sr


def main():
    parser = argparse.ArgumentParser(
        description="Magpie-TTS Arabic Dialect Speech Synthesis (Saudi & Emirati)"
    )
    parser.add_argument(
        "--dialect",
        "-d",
        type=str,
        default="saudi",
        choices=["saudi", "emirati", "ar-sa", "ar-ae"],
        help="Target Arabic dialect (default: saudi)",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Arabic text prompt to synthesize. If omitted, uses default sample.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Destination path for the generated .wav file (default: output.wav)",
    )
    parser.add_argument(
        "--apply-tn",
        action="store_true",
        help="Enable NeMo Text Normalization for numbers and dates.",
    )

    args = parser.parse_args()

    dialect_key = "saudi" if "sa" in args.dialect.lower() else "emirati"
    text = args.text or DIALECT_REGISTRY[dialect_key]["default_sample"]

    print(f"\n==========================================")
    print(f"Dialect:  {DIALECT_REGISTRY[dialect_key]['name']}")
    print(f"Text:     {text}")
    print(f"Output:   {args.output}")
    print(f"==========================================\n")

    tts = MagpieArabicTTS(dialect=dialect_key)
    audio, _ = tts.synthesize(text, apply_tn=args.apply_tn)
    tts.save(audio, args.output)


if __name__ == "__main__":
    main()
