from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

# Set up quantization config
quant_config = BitsAndBytesConfig(load_in_8bit=True)

# Load model and tokenizer
model_name = "./ajsbsd.net/flan-t5-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=quant_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# No need to call .to("cuda") — model is already on GPU!

# Try a quick inference
input_text = "Explain quantum computing in simple terms."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # You can still move inputs manually
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
