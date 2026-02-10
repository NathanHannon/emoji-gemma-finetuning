from huggingface_hub import HfApi

# Your Repo ID
repo_id = "NathanHannon/emoji_gemma3.270m"

# Files that should ONLY be in the 'onnx/' folder now
# We keep README.md and .gitattributes at the root
files_to_delete = [
    "config.json",
    "generation_config.json",
    "model.onnx",
    "model_quantized.onnx",
    "ort_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]

api = HfApi()
print(f"Deleting duplicates from root of {repo_id}...")

# Delete files one by one (safest way)
for filename in files_to_delete:
    try:
        # Check if file exists first (optional, but delete_file throws if missing)
        api.delete_file(path_in_repo=filename, repo_id=repo_id)
        print(f"✅ Deleted {filename}")
    except Exception as e:
        # Ignore if file is already gone
        print(f"⚠️ Skipped {filename}: {e}")

print(
    "\nCleanup complete! Your root folder should now only contain README.md and the 'onnx' folder."
)
