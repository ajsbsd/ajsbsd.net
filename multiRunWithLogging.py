# multiRunWithLogging.py

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import time
import csv
import os

# Load model and tokenizer
model_name = "google/flan-t5-xl"

tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# CSV setup
csv_file = "inference_log.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["timestamp", "input_text", "output_text", "generation_time_sec"])

    # List of prompts
    prompts = [
        "translate English to German: How old are you?",
        "summarize: The quick brown fox jumps over the lazy dog.",
        "explain: What is photosynthesis?",
        "generate a joke: Why did the chicken cross the road?",
        "answer: Who wrote 'To be or not to be'?"
    ]

    for prompt in prompts:
        start_time = time.time()

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            num_beams=4,
            early_stopping=True
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        elapsed = time.time() - start_time

        # Log to CSV
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), prompt, output_text, f"{elapsed:.2f}"])
        
        # Print result
        print(f"\nPrompt: {prompt}")
        print(f"Response: {output_text}")
        print(f"Time taken: {elapsed:.2f} seconds")

print(f"\n✅ All prompts processed. Logged to '{csv_file}'.")
