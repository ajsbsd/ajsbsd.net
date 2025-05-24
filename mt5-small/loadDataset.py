from datasets import load_dataset

dataset = load_dataset("ajsbsd/legalese-sentences_estonian-english")

from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False)

def preprocess(examples):
    inputs = ["translate Estonian to English: " + doc for doc in examples["input"]]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

tokenized_datasets = dataset.map(preprocess, batched=True)

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.gradient_checkpointing_enable()

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
    fp16=True,  # Remove if no GPU
    save_strategy="epoch",
    gradient_checkpointing=True

)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("mt5-finetuned-legal")
tokenizer.save_pretrained("mt5-finetuned-legal")

from transformers import pipeline

translator = pipeline("translation", model="mt5-finetuned-legal", tokenizer="mt5-finetuned-legal")

# Test inference
estonian_text = "Kujutis-, kombineeritud või ruumilise kaubamärgi korral esitatakse lisaks..."
translated = translator(estonian_text, max_length=128)
print(translated[0]['translation_text'])

#HF_TOKEN
#trainer.push_to_hub()
#tokenizer.push_to_hub(repo_id="your-username/mt5-finetuned-legal")
