from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "mistral"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

print("Chat with your fine-tuned Mistral model (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("AI:", response.split(user_input)[-1].strip())
