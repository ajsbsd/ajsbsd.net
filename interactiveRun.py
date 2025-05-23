# interactiveRun.py

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

    print("\n💬 FLAN-T5-XL Interactive Prompt Runner")
    print("Enter your prompts below. Type 'q' to quit.\n")

    while True:
        user_input = input("📝 Your prompt: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("\n👋 Exiting. Goodbye!")
            break

        start_time = time.time()

        # Tokenize and generate
        inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
            num_beams=4,
            early_stopping=True
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        elapsed = time.time() - start_time

        # Log to CSV
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            user_input,
            output_text,
            f"{elapsed:.2f}"
        ])

        # Print result
        print(f"🤖 Response: {output_text}")
        print(f"⏱️ Time taken: {elapsed:.2f} seconds\n")

print(f"\n✅ Session complete. Logged to '{csv_file}'.")
