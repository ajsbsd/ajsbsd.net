from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from datasets import load_dataset

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model_path = "./flan-t5-xl"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

# Load dataset
dataset = load_dataset("paulpall/legalese-sentences_estonian")

# List to store translations
translations = []

# Translate examples
for idx, example in enumerate(dataset['train']):
    estonian_text = example['Corpus']  # Correct key from dataset
    
    # Improved prompt
    prompt = f"""
Translate the following Estonian legal text to English:

Estonian: {estonian_text}

Translation:
"""

    # Tokenize input and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate translation
    outputs = model.generate(**inputs, max_new_tokens=200, num_beams=4)
    
    # Decode result
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Append to list
    translations.append({
        "input": estonian_text,
        "output": translated_text
    })

    print(f"Input:  {estonian_text}")
    print(f"Output: {translated_text}\n")

    if idx >= 10:
        break  # Remove or increase this limit as needed

# Optional: Save translations to file
import json
with open("flan_t5_estonian_translations.json", "w", encoding="utf-8") as f:
    json.dump(translations, f, ensure_ascii=False, indent=2)

print("✅ Translations completed and saved to 'flan_t5_estonian_translations.json'")
