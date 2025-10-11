import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 1. Configuration ---
# Make sure these paths are correct
base_model_id = "google/gemma-3-270m-it"
adapter_path = "./myemoji-gemma-adapters"  # The folder where your adapters were saved

# --- 2. Load the Tokenizer and Base Model ---
print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, device_map="auto", torch_dtype=torch.bfloat16
)

# --- 3. Load the Fine-Tuned Model (with Adapters) ---
print("Loading fine-tuned model with LoRA adapters...")
model = PeftModel.from_pretrained(base_model, adapter_path)


# --- 4. Function to Get a Response ---
def get_emoji_translation(text_input):
    """Generates an emoji translation for the given text."""
    prompt_template = [
        {"role": "system", "content": "Translate this text to emoji:"},
        {"role": "user", "content": text_input},
    ]

    # Apply the chat template and encode for the model
    prompt = tokenizer.apply_chat_template(
        prompt_template, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # or "cpu"

    # Generate a response
    outputs = model.generate(**inputs, max_new_tokens=64)

    # Decode and clean up the output
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated part
    return response_text.split("model\n")[-1].strip()


# --- 5. Interactive Testing Loop ---
print("\n--- Emoji Translation Test ---")
print("Type a phrase to translate, or 'quit' to exit.")

while True:
    user_input = input("> ")
    if user_input.lower() == "quit":
        break

    emoji_output = get_emoji_translation(user_input)
    print(f"  🤖: {emoji_output}\n")
