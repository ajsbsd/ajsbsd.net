from transformers import MarianTokenizer, MarianMTModel
from datasets import load_dataset
import json
import torch
import time  # <-- For timing

# Load model and tokenizer
model_name = "Helsinki-NLP/opus-mt-et-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load dataset
dataset = load_dataset("paulpall/legalese-sentences_estonian")

# List to store translations with timing
translations = []

# Translate examples
for idx, example in enumerate(dataset['train']):
    estonian_text = example['Corpus']

    # Tokenize input
    tokenized = tokenizer(estonian_text, return_tensors="pt", padding=True).to(device)

    # Start timing
    start_time = time.time()

    # Translate
    translated = model.generate(**tokenized)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    # End timing
    end_time = time.time()
    translation_time = end_time - start_time

    # Store result
    translations.append({
        "input": estonian_text,
        "output": translated_text,
        "time_seconds": round(translation_time, 4)  # Rounded for readability
    })

    print(f"Input:  {estonian_text}")
    print(f"Output: {translated_text}")
    print(f"Time:   {translation_time:.4f} seconds\n")

    if idx >= 20:
        break  # Remove or increase this limit as needed

# Save to file
with open("helsinki_nlp_estonian_translations_with_timing.json", "w", encoding="utf-8") as f:
    json.dump(translations, f, ensure_ascii=False, indent=2)

print("✅ Translations completed with timing info and saved to JSON.")
