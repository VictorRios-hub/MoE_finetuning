import argparse
import csv
import os
import re
import time
from typing import Optional, List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

# -------------------------
# Evaluation utilities
# -------------------------
RE_LAST_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")

def extract_answer(text: str) -> Optional[str]:
    nums = RE_LAST_NUM.findall(text)
    return nums[-1] if nums else None

def build_prompt(tokenizer, question: str) -> str:
    user_msg = {"role": "user", "content": question}
    return tokenizer.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)

# -------------------------
# Load model (+ optional LoRA)
# -------------------------
def load_model(base_model: str, adapter_dir: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, device_map="auto")
    model.eval()

    if adapter_dir:
        if not HAS_PEFT:
            raise RuntimeError("peft is not installed, but --adapter_dir was provided.")
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()

    return model, tokenizer

# -------------------------
# Batched greedy generation
# -------------------------
@torch.no_grad()
def generate_batch(model, tokenizer, prompts: List[str], max_new_tokens: int = 200):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,              
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    latency = time.perf_counter() - start
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return texts, latency

# -------------------------
# Evaluation on GSM8K test
# -------------------------
def evaluate(base_model: str,
             adapter_dir: Optional[str],
             n_samples: int = 200,
             batch_size: int = 2,
             out_dir: str = "eval_outputs",
             max_new_tokens: int = 200):

    model, tokenizer = load_model(base_model, adapter_dir)

    ds = load_dataset("gsm8k", "main")["test"]
    if n_samples and n_samples > 0:
        n_samples = min(n_samples, len(ds))
        ds = ds.select(range(n_samples))

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "eval_results.csv")

    correct = 0
    total = len(ds)
    acc_curve = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "question", "gold_text", "pred_text", "gold_extracted", "pred_extracted", "correct"])

        for i in tqdm(range(0, total, batch_size), desc="Evaluating"):
            idxs = list(range(i, min(i + batch_size, total)))
            batch = ds.select(idxs)
            questions = [ex["question"] for ex in batch]
            gold_texts = [ex["answer"] for ex in batch]

            prompts = [build_prompt(tokenizer, q) for q in questions]
            preds, _ = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)

            for j, (q, gold, pred) in enumerate(zip(questions, gold_texts, preds)):
                gold_ans = extract_answer(gold)
                pred_ans = extract_answer(pred)
                is_ok = int(gold_ans is not None and pred_ans == gold_ans)
                correct += is_ok
                step = i + j + 1
                acc_curve.append(correct / step)
                writer.writerow([i + j, q, gold, pred, gold_ans, pred_ans, is_ok])

    accuracy = correct / total if total else 0.0
    print(f"\nAccuracy: {accuracy:.2%}  ({correct}/{total})")

    # Accuracy curve
    try:
        import matplotlib.pyplot as plt
        x = list(range(1, len(acc_curve) + 1))
        y = acc_curve
        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Sample index")
        plt.ylabel("Cumulative accuracy")
        plt.title(f"GSM8K test — cumulative accuracy (n={total})")
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(out_dir, "accuracy_curve.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {plot_path}")
    except Exception as e:
        print(f"(Plotting skipped): {e}")

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Samples: {total}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast GSM8K test evaluation for Granite-3.1-1B (finetuned)")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Path to your LoRA adapter dir (omit if merged model)")
    parser.add_argument("--n", type=int, default=200, help="Number of test samples (<=0 for full test)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="eval_outputs")
    args = parser.parse_args()

    BASE = "ibm-granite/granite-3.1-1b-a400m-instruct"
    evaluate(
        base_model=BASE,
        adapter_dir=args.adapter_dir,
        n_samples=None if args.n is None or args.n <= 0 else args.n,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
        max_new_tokens=args.max_new_tokens,
    )
