"""
Gradio Web Interface for Magpie-TTS Arabic Dialects (Saudi & Emirati).

Supports local execution as well as deployment to Hugging Face Spaces (including ZeroGPU).
"""

import os
import sys
import numpy as np
import gradio as gr

# Optional Spaces support for ZeroGPU environments
try:
    import spaces
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False

from inference import MagpieArabicTTS, DIALECT_REGISTRY

# In-memory model cache to avoid reloading weights repeatedly
LOADED_MODELS = {}

def get_or_load_model(dialect_key: str):
    """Retrieves model from cache or instantiates a new one."""
    if dialect_key not in LOADED_MODELS:
        print(f"Loading {dialect_key} model into cache...")
        LOADED_MODELS[dialect_key] = MagpieArabicTTS(dialect=dialect_key)
    return LOADED_MODELS[dialect_key]


def synthesize_speech(text: str, dialect_display: str, apply_tn: bool):
    """Gradio handler for synthesizing speech."""
    if not text or not text.strip():
        raise gr.Error("الرجاء إدخال نص باللغة العربية أولاً.")

    dialect_key = "saudi" if "سعودية" in dialect_display or "Saudi" in dialect_display else "emirati"
    
    tts = get_or_load_model(dialect_key)
    audio_array, sample_rate = tts.synthesize(text.strip(), apply_tn=apply_tn)

    # Normalize audio to prevent clipping
    max_val = np.max(np.abs(audio_array))
    if max_val > 0:
        audio_array = audio_array / max_val * 0.95

    return (sample_rate, (audio_array * 32767).astype(np.int16))


# Apply GPU decorator if running on Hugging Face ZeroGPU
if SPACES_AVAILABLE:
    synthesize_speech = spaces.GPU(synthesize_speech)


EXAMPLE_PROMPTS = [
    ["الذكاء الاصطناعي صار جزء أساسي من حياتنا اليومية، وتطوير نماذج تدعم لهجتنا خطوة جداً مهمة.", "اللهجة السعودية (Saudi Arabic)"],
    ["ترى الطريق جهة الرياض اليوم مرة زحمة، فإذا مستعجلين أحسن نطلع من الحين عشان نوصل قبل الموعد.", "اللهجة السعودية (Saudi Arabic)"],
    ["يا هلا والله ومسهلا! كيف حالك؟ عساك بخير وصحة وسلامة.", "اللهجة السعودية (Saudi Arabic)"],
    ["مرحبا الساع، شحالك؟ عساك طيب ومستانس إن شاء الله.", "اللهجة الإماراتية (Emirati Arabic)"],
    ["الجو اليوم وايد حار، خلنا نروح مكان بارد ونتقهوى هناك.", "اللهجة الإماراتية (Emirati Arabic)"],
    ["هي نعم، حتى عندنا في الإمارات أصدروا تأشيرات خاصة للمبدعين والعمل عن بعد.", "اللهجة الإماراتية (Emirati Arabic)"],
]

CUSTOM_CSS = """
.rtl-text textarea {
    direction: rtl !important;
    text-align: right !important;
    font-size: 1.15em !important;
    line-height: 1.7 !important;
}
"""

with gr.Blocks(title="Magpie-TTS Arabic Dialects", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ Magpie-TTS Arabic Dialects (Saudi & Emirati)
        
        Fine-tuned text-to-speech models built on NVIDIA's **Magpie-TTS Multilingual (357M)** architecture, 
        specialized for natural conversational **Saudi** and **Emirati** Arabic speech.
        
        * Models hosted on Hugging Face:
          - [AhmedEladl/Magpie-TTS-Saudi-Arabic](https://huggingface.co/AhmedEladl/Magpie-TTS-Saudi-Arabic)
          - [AhmedEladl/Magpie-TTS-Emirates-Arabic](https://huggingface.co/AhmedEladl/Magpie-TTS-Emirates-Arabic)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            dialect_choice = gr.Radio(
                choices=[
                    "اللهجة السعودية (Saudi Arabic)",
                    "اللهجة الإماراتية (Emirati Arabic)",
                ],
                value="اللهجة السعودية (Saudi Arabic)",
                label="اختر اللهجة / Select Dialect",
            )
            
            text_input = gr.Textbox(
                lines=4,
                placeholder="اكتب النص العربي هنا...",
                label="النص العربي / Arabic Prompt",
                elem_classes=["rtl-text"],
                value=EXAMPLE_PROMPTS[0][0],
            )
            
            apply_tn = gr.Checkbox(
                label="تفعيل المعالجة اللغوية للأرقام والرموز (Text Normalization)",
                value=False,
            )
            
            generate_btn = gr.Button(" توليد الصوت / Generate Speech", variant="primary", size="lg")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="الصوت المتولد / Generated Audio",
                type="numpy",
                autoplay=True,
            )

    gr.Examples(
        examples=EXAMPLE_PROMPTS,
        inputs=[text_input, dialect_choice],
        label="أمثلة جاهزة للتجربة / Sample Prompts",
    )

    generate_btn.click(
        fn=synthesize_speech,
        inputs=[text_input, dialect_choice, apply_tn],
        outputs=[audio_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
