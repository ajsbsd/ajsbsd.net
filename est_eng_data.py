import json

with open("helsinki_nlp_estonian_translations_with_timing.json", "r") as f:
    data = json.load(f)

for entry in data[:5]:
    print("ESTONIAN:", entry["input"])
    print("ENGLISH: ", entry["output"])
    print("TIME:    ", entry["time_seconds"], "seconds\n")
