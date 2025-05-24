from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch
import pandas as pd
import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./flan-t5-large"
offload_folder = "offload"
os.makedirs(offload_folder, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare quantization config for 8-bit loading (new recommended way)
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# Load model with quantization config, device map, and offloading
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    quantization_config=quant_config,
    device_map="auto",
    offload_folder=offload_folder,
    offload_state_dict=True,
)

# Identify the device of model embeddings for input tensors placement
embedding_device = next(model.parameters()).device

# Sample dataset
data = {
    "source_text": ["This is a test.", "This is another test."],
    "target_text": ["Ceci est un test.", "Ceci est un autre test."]
}
df = pd.DataFrame(data)
df["input_text"] = "translate English to French: " + df["source_text"]

class TranslationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        source_text = self.df.iloc[idx]["input_text"]
        target_text = self.df.iloc[idx]["target_text"]

        source_encoded = self.tokenizer(
            source_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        target_encoded = self.tokenizer(
            target_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        return {
            "input_ids": source_encoded["input_ids"].squeeze(),
            "attention_mask": source_encoded["attention_mask"].squeeze(),
            "labels": target_encoded["input_ids"].squeeze()
        }

dataset = TranslationDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(embedding_device)
        attention_mask = batch['attention_mask'].to(embedding_device)
        labels = batch['labels'].to(embedding_device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({"loss": loss.item()})

    save_dir = f"translated_flan_t5_epoch_{epoch+1}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
