from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "mistral-conversational-lora"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

tests = [
    "Hi there!",
    "Can you tell me a short poem about AI?",
    "Explain the difference between machine learning and deep learning.",
    "Tell me a joke about programmers.",
]

print("Testing Fine-Tuned Mistral Model\n")

for prompt in tests:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}\nResponse: {response.replace(prompt, '').strip()}\n{'-'*60}")

