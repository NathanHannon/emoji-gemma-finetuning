# train.py
import torch
import emoji
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
import config  # Import our configurations

# --- GPU Detection ---
print("=== GPU Detection ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
else:
    print("No GPU available - training will use CPU")
print("=====================\n")


# --- 1. Data Preparation ---
def prepare_dataset():
    dataset = load_dataset(config.DATASET_ID, split="train")

    def is_only_emoji(sample):
        emoji_string = sample["emoji"]
        return emoji_string and all(emoji.is_emoji(char) for char in emoji_string)

    dataset = dataset.filter(is_only_emoji)

    def translate_format(sample):
        return {
            "messages": [
                {"role": "system", "content": "Translate this text to emoji: "},
                {"role": "user", "content": f"{sample['text']}"},
                {"role": "assistant", "content": f"{sample['emoji']}"},
            ]
        }

    formatted_dataset = dataset.map(
        translate_format, remove_columns=list(dataset.features.keys())
    )
    return formatted_dataset.train_test_split(test_size=0.1, shuffle=True)


# --- 2. Model Loading ---
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_ID,
    quantization_config=config.BNB_CONFIG,
    device_map="auto",
    attn_implementation="eager",
)
model.config.pad_token_id = tokenizer.pad_token_id

# Check model device placement
print("=== Model Device Information ===")
print(f"Model device: {next(model.parameters()).device}")
if hasattr(model, "hf_device_map"):
    print(f"Device map: {model.hf_device_map}")
print("===============================\n")

# --- 3. Training ---
dataset_splits = prepare_dataset()

trainer = SFTTrainer(
    model=model,
    args=config.TRAINING_ARGS,
    train_dataset=dataset_splits["train"],
    eval_dataset=dataset_splits["test"],
    peft_config=config.LORA_CONFIG,
    processing_class=tokenizer,
)

print("Starting training...")
if torch.cuda.is_available():
    print(
        f"GPU memory before training: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reserved"
    )

trainer.train()

if torch.cuda.is_available():
    print(
        f"GPU memory after training: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reserved"
    )

trainer.save_model(config.ADAPTER_PATH)
print(f"Training complete. Adapters saved to {config.ADAPTER_PATH}")
