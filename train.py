"""
Training script to fine-tune Granite 3.1 1B A400M (MoE) using GRPO and
LoRA on the GSM8K dataset

This script is based on the **TRL** (Hugging Face) library and its
implementation of **Group Relative Policy Optimization**. It automatically
handles vLLM configuration, applies LoRA, and finetunes the attention layers
of the MoE model

"""

import os
import re
import random
import numpy as np
import torch
import logging
import time

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.integrations import WandbCallback
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# REPRODUCIBILITY
# -------------------------
SEED = int(os.environ.get("SEED", 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# MODEL & ENVIRONMENT CONFIGURATION
# -------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "ibm-granite/granite-3.1-1b-a400m-instruct")      # HF model path
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "checkpoints")                                    # directory to save checkpoints and results
USE_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8        # use bfloat16 if GPU supports it (since I'm on a RTX A3000)
USE_FP16 = not USE_BF16

PER_DEVICE_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1))                                # per device batch size
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM", 4))                                     # gradient accumulation steps
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", 4))                                 # number of samples generated for each prompt during training
GENERATION_BATCH_SIZE = int(os.environ.get("GENERATION_BATCH_SIZE", 4))                     # batch size for generation during training (should fit in VRAM)
LR = float(os.environ.get("LR", 5e-6))                                                      # small LR since we are fine-tuning a large model
MAX_STEPS = int(os.environ.get("MAX_STEPS", 1000))                                          # total number of training steps
LOG_STEPS = int(os.environ.get("LOG_STEPS", 10))                                            # log metrics every n steps
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", 200))                                         # save checkpoint every n steps

# vLLM
USE_VLLM = os.environ.get("USE_VLLM", "1") == "1"                                           # use vLLM if possible
VLLM_MODE = os.environ.get("VLLM_MODE", "colocate")                                         # vLLM mode: colocate or separate
VLLM_MEM_UTIL = float(os.environ.get("VLLM_MEM_UTIL", 0.30))                                # allocate ~30% VRAM to the model, 70% to vLLM cache (specs and 16GB GPU compatible)

# Generation (for testing)
# GEN_MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", 256))
# TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.3))
# TOP_P = float(os.environ.get("TOP_P", 0.95))

# -------------------------
# MODEL LOADING & TOKENIZER
# -------------------------

# Initialize model 
model_path = MODEL_NAME
logger.info(f"Loading model: {model_path}")

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    dtype = torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else torch.float32) # VRAM optimization
)

# Load tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_path) # default use_fast=True
logger.info(f"Model loaded on {torch.cuda.current_device()} with dtype {base_model.dtype}")

# Enable gradient checkpointing to reduce activation VRAM usage
# if os.environ.get("GRAD_CHKPT", "0") == "1":
#     model.gradient_checkpointing_enable()

# -------------------------
# Apply LoRA (Q, V projections) and unfreeze the MoE router
# -------------------------

# LoRA configuration for Q and V projection modules of attention heads
lora_config = LoraConfig(
    r=8,                                   # rank of matrices 
    lora_alpha=32,                          # scaling factor 
    lora_dropout=0.05,                      # dropout for layers
    bias="none",                            # do not train base model biases
    task_type="CAUSAL_LM",                  # task type (auto-regressive generative model)
    target_modules=["q_proj", "v_proj"]     # attention layers targeted
)

# Apply LoRA to the model
model = get_peft_model(base_model, lora_config)
logger.info("LoRA method applied")

# Unfreeze router parameters (possible names: router, gate, routing)
router_unfrozen = 0
for n, p in model.named_parameters():
    if any(k in n.lower() for k in ["router", "gate", "routing"]):
        p.requires_grad = True
        router_unfrozen += p.numel()
logger.info(f"Router params unfrozen (approx): {router_unfrozen}")

# -------------------------
# Dataset GSM8K
# -------------------------

# Load GSM8K dataset
logger.info("Loading GSM8K dataset ...")
data = load_dataset("gsm8k", "main")
train_data = data["train"]  

# Build the list of formatted prompts and answers
train_prompts = []
for sample in train_data:
    question = sample["question"]
    # Format chat for Granite Instruct
    user_msg = {"role": "user", "content": question}
    prompt_text = tokenizer.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
    train_prompts.append({"prompt": prompt_text, "answer": sample["answer"]})

# Convert to HuggingFace Dataset
train_dataset  = Dataset.from_list(train_prompts)
logger.info("Prompts formatted and dataset created")

# -------------------------
# Reward function: +1 if the final answer is correct, otherwise 0
# TRL passes additional columns (here "answer") to the reward function
# -------------------------

## Extract the last number in the answer
num_re = re.compile(r"-?\d+(?:\.\d+)?") # matches integers and decimals, including negative numbers

def extract_last_number(text: str):
    nums = num_re.findall(text)
    return nums[-1] if nums else None

def reward_function(prompts, completions, **kwargs):
    answers = kwargs.get("answer", [])
    rewards = []
    for comp, true_answer in zip(completions, answers):

        # Ensure both are strings
        generated_text = comp if isinstance(comp, str) else str(comp)
        true_answer = true_answer if isinstance(true_answer, str) else str(true_answer)

        # Extract the final answer (number) from the generated text
        final_answer = extract_last_number(generated_text)

        # Extract the expected answer (last number from the correct answer)
        expected = extract_last_number(true_answer)

        # Compare and grade
        if final_answer is not None and expected is not None and final_answer == expected:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# -------------------------
# Config GRPO
# -------------------------

# GRPO hyperparameters configuration
logger.info("Building GRPO config…")
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_generations=NUM_GENERATIONS,
    generation_batch_size=GENERATION_BATCH_SIZE,
    learning_rate=LR,
    max_steps=MAX_STEPS,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    save_safetensors=True,
    optim="adamw_torch",

    # Precision
    bf16=USE_BF16,
    fp16=USE_FP16,

    # Génération
    # max_new_tokens=GEN_MAX_TOKENS,
    # temperature=TEMPERATURE,
    # top_p=TOP_P,

    # vLLM
    use_vllm=USE_VLLM,
    vllm_mode=VLLM_MODE,                            
    vllm_gpu_memory_utilization=VLLM_MEM_UTIL,

    # Reporting
    report_to=["wandb"],                          
)

# -------------------------
# Trainer
# -------------------------

# Initialize the GRPO trainer
logger.info("Initializing GRPO trainer …")
trainer = GRPOTrainer(
    model=model,                        # our model with LoRA (PeftModel)
    args=grpo_config,                   # training config 
    train_dataset=train_dataset ,       # dataset containing "prompt" and "answer"
    reward_funcs=reward_function,       # reward function defined above
    processing_class=tokenizer          # provide the tokenizer
)

# -------------------------
# Training
# -------------------------

logger.info("Starting training…")
start_time = time.time()

# Resume from checkpoint if specified
if grpo_config.resume_from_checkpoint:
    logger.info(f"Resuming training from checkpoint {grpo_config.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=grpo_config.resume_from_checkpoint)
else:
    trainer.train()

end_time = time.time()
elapsed = end_time - start_time
logger.info(f"Training completed in {elapsed:.2f} seconds")

# -------------------------
# Final save (LoRA adapters)
# -------------------------
logger.info("Saving final adapter…")
trainer.save_model(OUTPUT_DIR) 
