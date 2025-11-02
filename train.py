from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files="dataset/conversations.jsonl")

def tokenize(example):
    text = example["instruction"] + "\n" + example["response"]
    return tokenizer(text, truncation=True, max_length=512)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

training_args = TrainingArguments(
    output_dir="./mistral-conversational-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    max_steps=500,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# Save both LoRA adapter and full merged model (optional)
model.save_pretrained("mistral-conversational-lora")
tokenizer.save_pretrained("mistral-conversational-lora")

print("✅ Training complete.")
