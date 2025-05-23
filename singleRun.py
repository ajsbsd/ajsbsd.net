from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")

# Input task
input_text = "translate English to German: How old are you?"

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # Move full batch to GPU

# Generate output
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=50,
    num_beams=4,
    early_stopping=True
)

# Decode and print result
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
