# config.py
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig

MODEL_ID = "google/gemma-3-270m-it"
ADAPTER_PATH = "./myemoji-gemma-adapters"
DATASET_ID = "kr15t3n/text2emoji"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"],
)

TRAINING_ARGS = SFTConfig(
    output_dir=ADAPTER_PATH,
    num_train_epochs=3,  # Start with 3 epochs, monitor manually
    per_device_train_batch_size=4,
    logging_strategy="steps",
    logging_steps=50,  # Log more frequently to monitor progress
    save_strategy="epoch",
    save_total_limit=2,  # Keep only best 2 checkpoints
    learning_rate=5e-5,
    lr_scheduler_type="constant",
    max_length=256,
    gradient_checkpointing=False,
    packing=False,
    optim="adamw_torch_fused",
    report_to="tensorboard",
    weight_decay=0.01,
    fp16=True,  # Use fp16 instead of bf16 for broader compatibility
    bf16=False,  # Explicitly disable bf16
)
