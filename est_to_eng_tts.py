# est_to_eng_tts.py

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import asyncio
import edge_tts

# Load model and tokenizer
model_name = "google/flan-t5-xl"

tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# Translation function
def translate_est_to_eng(text_est):
    prompt = f"Translate Estonian to English: {text_est}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
        num_beams=4,
        early_stopping=True
    )
    eng_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if not eng_text.strip():
        raise ValueError("Translation returned empty string.")
    return eng_text

# TTS function using edge-tts
async def text_to_speech(text, output_file="output.mp3", voice="en-US-AriaNeural"):
    if not text.strip():
        print("No text provided for speech synthesis.")
        return
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        print(f"✅ Saved audio to {output_file}")
    except Exception as e:
        print(f"❌ TTS Error: {e}")

# Main function
def main():
    print("📘 Enter Estonian text below:")
    est_text = input("> ")

    try:
        # Translate to English
        eng_text = translate_est_to_eng(est_text)
        print("\n📝 Translated English text:\n", eng_text)

        # Save translation
        with open("translation.txt", "w") as f:
            f.write(eng_text)
        print("✅ Saved translation to 'translation.txt'")

        # Generate speech
        asyncio.run(text_to_speech(eng_text))

    except Exception as e:
        print(f"🚫 Error: {e}")

if __name__ == "__main__":
    main()
