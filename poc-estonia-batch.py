from datasets import load_dataset
from transformers import MarianTokenizer, MarianMTModel
import torch
import json

# Load model and tokenizer
model_name = "Helsinki-NLP/opus-mt-et-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load dataset
dataset = load_dataset("paulpall/legalese-sentences_estonian")
total_examples = len(dataset["train"])
print(f"Found {total_examples} examples. Starting translation...")

# Batch size (adjust based on your GPU RAM, e.g., 8–32)
batch_size = 8

translations = []

# Batched processing
for i in range(0, total_examples, batch_size):
    batch = dataset["train"][i:i + batch_size]
    inputs = batch["Corpus"]  # Or whatever key contains the Estonian text

    # Tokenize and translate
    translated = model.generate(**tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(device))
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    # Save input/output pairs
    for est_text, eng_text in zip(inputs, translated_texts):
        translations.append({
            "input": est_text,
            "output": eng_text
        })

    print(f"Translated {min(i + batch_size, total_examples)} / {total_examples}")

# Save to JSON file
with open("helsinki_nlp_full_estonian_translations.json", "w", encoding="utf-8") as f:
    json.dump(translations, f, ensure_ascii=False, indent=2)

print("✅ Full dataset translated and saved to JSON.")
