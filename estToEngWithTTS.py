# estToEngWithTTS.py

from transformers import T5Tokenizer, T5ForConditionalGeneration
from TTS.api import TTS
import torch
import gradio as gr

# Load FLAN-T5 model and tokenizer
model_name = "google/flan-t5-xl"

tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load TTS model (English)
tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)

# Translation function
def translate_est_to_eng(text_est):
    if not text_est.strip():
        return "Please enter some Estonian text.", None

    prompt = f"Translate Estonian to English: {text_est}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
        num_beams=4,
        early_stopping=True
    )
    eng_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return eng_text, eng_text  # Return twice: once for display, once for TTS

# TTS function
def tts_wrapper(text):
    if not text or not text.strip():
        return (None, None)

    wav = tts.synthesize_wav(text)
    return ("audio.wav", wav)

# Gradio Interface
with gr.Blocks(title="📚 Est → Eng + Whisper/TTS") as demo:
    gr.Markdown("### 📚 Estonian Literature Translator + Audio Synthesizer\n"
                "Enter Estonian literary text below to translate to English and generate audio.")

    with gr.Row():
        est_input = gr.Textbox(label="Estonian Text", lines=10, placeholder="Paste Estonian literature here...")

    with gr.Row():
        eng_output = gr.Textbox(label="Translated English Text", lines=10)

    with gr.Row():
        audio_output = gr.Audio(label="Synthesized Speech", type="filepath")

    with gr.Row():
        translate_btn = gr.Button("📖 Translate & Generate Audio")

    translate_btn.click(
        fn=translate_est_to_eng,
        inputs=est_input,
        outputs=[eng_output, gr.State()]
    ).then(
        fn=tts_wrapper,
        inputs=eng_output,
        outputs=audio_output
    )

demo.launch()
