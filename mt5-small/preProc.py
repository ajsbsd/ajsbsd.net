# inspect_dataset.py

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# --- Configs ---
model_name = "google/mt5-small"

# --- Load Dataset ---
dataset = load_dataset("ajsbsd/legalese-sentences_estonian-english")
print("Dataset loaded successfully.")

# --- Filter Empty Examples ---
def filter_invalid(ex):
    return len(ex["input"].strip()) > 0 and len(ex["output"].strip()) > 0

dataset = dataset.filter(filter_invalid)
print("Filtered out empty or invalid examples.")

# --- Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False)

# --- Preprocess Function ---
def preprocess(examples):
    inputs = ["translate Estonian to English: " + doc for doc in examples["input"]]
    targets = examples["output"]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    # Replace pad tokens in labels with -100
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess, batched=True)
print("Data tokenized and mapped.")

# --- Inspect One Batch ---
loader = DataLoader(tokenized_datasets["train"], batch_size=4)
batch = next(iter(loader))

print("\n--- Sample Input/Label Inspection ---\n")

for i in range(len(batch["input_ids"])):
    input_ids = batch["input_ids"][i]
    label_ids = batch["labels"][i]

    decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
    decoded_label = tokenizer.decode([x for x in label_ids if x != -100], skip_special_tokens=True)

    print(f"Sample {i+1}:")
    print("Input IDs:", input_ids)
    print("Decoded Input:", decoded_input)
    print("Label IDs:", label_ids)
    print("Decoded Label:", decoded_label)
    print("-" * 50)
