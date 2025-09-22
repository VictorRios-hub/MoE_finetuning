from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import re
import torch
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_model = "ibm-granite/granite-3.1-1b-a400m-instruct"
adapter_dir = "checkpoints"

tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    dtype="auto"
)

# Load LoRA weights
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()

# Load GSM8K dataset
dataset = load_dataset("gsm8k", "main")
test_data = dataset["test"]

def build_prompt(question: str) -> str:
    user_msg = {"role": "user", "content": question}
    return tokenizer.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)

num_re = re.compile(r"-?\d+")

def extract_last_number(text: str):
    nums = num_re.findall(text)
    return nums[-1] if nums else None

def evaluate(n=100):
    correct, total = 0, 0
    for example in test_data.select(range(n)):
        prompt = build_prompt(example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,  
                do_sample=False
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = extract_last_number(generated)
        true = extract_last_number(example["answer"])

        if pred == true:
            correct += 1
        total += 1

        logger.info(f"Question: {example['question']}")
        logger.info(f"Model: {generated}")
        logger.info(f"True: {example['answer']}")
        logger.info(f"Pred: {pred}, true: {true}, Correct: {pred==true}\n")

    acc = correct / total if total > 0 else 0
    logger.info(f"Accuracy on {total} samples: {acc:.2%}")
    return acc

evaluate(10)  # test on 10 problems
