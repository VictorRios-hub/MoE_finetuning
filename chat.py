import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "ibm-granite/granite-3.1-1b-a400m-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()

chat = [
    { "role": "user", "content": "Si Pierre a 12 pommes et en achète 7, combien en a-t-il ?" },
]
chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

input_tokens = tokenizer(chat, return_tensors="pt").to(device)

output = model.generate(**input_tokens, 
                        max_new_tokens=100)
output = tokenizer.batch_decode(output)

print(output)
