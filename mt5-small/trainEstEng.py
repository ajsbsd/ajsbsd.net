# --- 1. Load Libraries ---
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    pipeline
)

# --- 2. Load Dataset ---
dataset = load_dataset("ajsbsd/legalese-sentences_estonian-english")
print("Dataset loaded successfully.")

# --- 3. Load Tokenizer and Model ---
model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.gradient_checkpointing_enable()

# 3.5
def filter_invalid(ex):
    return len(ex["input"].strip()) > 0 and len(ex["output"].strip()) > 0

	

# --- 4. Preprocess Dataset ---
def preprocess(examples):
    inputs = ["translate Estonian to English: " + doc for doc in examples["input"]]
    targets = examples["output"]

    # Filter out empty strings
    filtered_inputs = []
    filtered_targets = []

    for inp, out in zip(inputs, targets):
        if len(inp.strip()) == 0 or len(out.strip()) == 0:
            continue
        filtered_inputs.append(inp)
        filtered_targets.append(out)

    model_inputs = tokenizer(filtered_inputs, text_target=filtered_targets, max_length=128, truncation=True, padding="max_length")
    return model_inputs

# EOF

tokenized_datasets = dataset.map(preprocess, batched=True)
tokenized_datasets = tokenized_datasets.filter(filter_invalid)

#  This actually maps for training
tokenized_datasets = dataset.map(preprocess, batched=True)
print("Data tokenized and mapped.")

# --- 5. Set Up Training ---
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="mt5-finetuned-legal",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=5,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=10,
    report_to="none",
    push_to_hub=False,
    fp16=True,
    save_strategy="epoch",
    gradient_checkpointing=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    processing_class=tokenizer,
)

# --- 6. Start Training ---
print("Starting training...")
trainer.train()

# --- 7. Save Model and Tokenizer ---
print("Saving model and tokenizer...")
trainer.save_model("mt5-finetuned-legal")
tokenizer.save_pretrained("mt5-finetuned-legal")

# --- 8. Run Inference Test ---
#print("Running test inference...")
#translator = pipeline("translation", model="mt5-finetuned-legal", tokenizer="mt5-finetuned-legal")

#estonian_text = "Kujutis-, kombineeritud või ruumilise kaubamärgi korral esitatakse lisaks..."
#translated = translator(estonian_text, max_length=128)
#print(f"Translated Text: {translated[0]['translation_text']}")

# --- 9. (Optional) Push to Hugging Face Hub ---
# Uncomment and replace with your HF username/token if you want to upload
"""
from huggingface_hub import login
login(token="your-hf-token")

trainer.push_to_hub()
tokenizer.push_to_hub(repo_id="your-username/mt5-finetuned-legal")
"""
