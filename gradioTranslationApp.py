# gradioTranslationApp.py

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import gradio as gr

# Load model and tokenizer
model_name = "google/flan-t5-xl"

tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# List of supported languages
languages = [
    "English", "Spanish", "Japanese", "Persian", "Hindi", "French", "Chinese", "Bengali", "Gujarati",
    "German", "Telugu", "Italian", "Arabic", "Polish", "Tamil", "Marathi", "Malayalam", "Oriya",
    "Punjabi", "Portuguese", "Urdu", "Galician", "Hebrew", "Korean", "Catalan", "Thai", "Dutch",
    "Indonesian", "Vietnamese", "Bulgarian", "Filipino", "Central Khmer", "Lao", "Turkish", "Russian",
    "Croatian", "Swedish", "Yoruba", "Kurdish", "Burmese", "Malay", "Czech", "Finnish", "Somali",
    "Tagalog", "Swahili", "Sinhala", "Kannada", "Zhuang", "Igbo", "Xhosa", "Romanian", "Haitian",
    "Estonian", "Slovak", "Lithuanian", "Greek", "Nepali", "Assamese", "Norwegian"
]

# Helper function to generate translation prompt
def translate(source_lang, target_lang, text):
    if not text.strip():
        return "Please enter text to translate."

    prompt = f"Translate {source_lang} to {target_lang}: {text}"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=100,
        num_beams=4,
        early_stopping=True
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# Gradio interface
with gr.Blocks(title="🌍 FLAN-T5 Zero-Shot Translator") as demo:
    gr.Markdown("### 🌍 FLAN-T5-XL Zero-Shot Translation Demo\n"
                "Select source and target language, then enter your question. Example: _What time is it?_")

    with gr.Row():
        src_lang = gr.Dropdown(choices=languages, label="Source Language")
        tgt_lang = gr.Dropdown(choices=languages, label="Target Language")

    text_input = gr.Textbox(lines=3, placeholder="Enter text here...", label="Text to Translate")
    output_box = gr.Textbox(label="Translated Text")

    translate_btn = gr.Button("🔄 Translate")

    translate_btn.click(fn=translate, inputs=[src_lang, tgt_lang, text_input], outputs=output_box)

# Launch app
demo.launch()
