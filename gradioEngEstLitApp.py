# gradioEngEstLitApp.py

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

# Translation function
def translate(source_lang, target_lang, text):
    if not text.strip():
        return "Please enter some text to translate."

    prompt = f"Translate {source_lang} to {target_lang}: {text}"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
        num_beams=4,
        early_stopping=True
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# Swap languages
def switch_languages(text_eng, text_est):
    return text_est, text_eng

# Gradio interface
with gr.Blocks(title="📚 Eng ↔ Est Literary Translator") as demo:
    gr.Markdown("### 📚 English ↔ Estonian Literary Text Translator\n"
                "Paste literary text below and translate between English and Estonian.")

    with gr.Row():
        swap_btn = gr.Button("🔄 Swap Languages", variant="secondary")

    # Hidden state placeholders for language directions
    src_lang = gr.Textbox(visible=False)
    tgt_lang = gr.Textbox(visible=False)

    with gr.Row():
        eng_box = gr.Textbox(label="English Text", lines=10, placeholder="Enter English literature here...")
        est_box = gr.Textbox(label="Estonian Translation", lines=10, placeholder="Translation will appear here...")

    translate_btn = gr.Button("📖 Translate")

    # Set correct source/target language pairs
    eng_to_est = [gr.Textbox("English", visible=False), gr.Textbox("Estonian", visible=False)]
    est_to_eng = [gr.Textbox("Estonian", visible=False), gr.Textbox("English", visible=False)]

    translate_btn.click(
        fn=translate,
        inputs=eng_to_est + [eng_box],
        outputs=est_box
    )

    translate_btn.click(
        fn=translate,
        inputs=est_to_eng + [est_box],
        outputs=eng_box
    )

    # Swap button behavior
    swap_btn.click(fn=switch_languages, inputs=[eng_box, est_box], outputs=[eng_box, est_box])

# Launch app
demo.launch()
