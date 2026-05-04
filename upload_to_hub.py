"""
Upload the exported ONNX model to the Hugging Face Hub with the correct structure:

  <repo_root>/
    config.json              ← Transformers.js reads this from the root
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    generation_config.json
    chat_template.jinja      (if present)
    onnx/
      model.onnx             ← full-precision ONNX weights
      model_quantized.onnx   ← quantized weights (loaded by worker.js via dtype:'q8')

Usage:
    python upload_to_hub.py --repo NathanHannon/emoji_gemma3.270m --src onnx_output_dir
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, login

# Files that must live at the repository root so that Transformers.js can find
# them without a subfolder prefix.
ROOT_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
}


def upload(repo_id: str, src_dir: str, token: str | None = None):
    src = Path(src_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src}")

    if token:
        login(token=token)

    api = HfApi()

    # Create the repo if it does not already exist.
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"\nUploading to: https://huggingface.co/{repo_id}\n")

    # --- Upload root-level config / tokenizer files ---
    for name in ROOT_FILES:
        candidate = src / name
        if candidate.exists():
            print(f"  [root]   {name}")
            api.upload_file(
                path_or_fileobj=str(candidate),
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="model",
            )

    # --- Upload ONNX model files into the onnx/ subfolder ---
    onnx_src = src / "onnx"
    if not onnx_src.is_dir():
        raise FileNotFoundError(
            f"No 'onnx/' subfolder found inside {src}. "
            "Run export_helper.py and quantize.py first."
        )

    for onnx_file in sorted(onnx_src.iterdir()):
        if onnx_file.is_file():
            dest_path = f"onnx/{onnx_file.name}"
            print(f"  [onnx/]  {onnx_file.name}")
            api.upload_file(
                path_or_fileobj=str(onnx_file),
                path_in_repo=dest_path,
                repo_id=repo_id,
                repo_type="model",
            )

    print(f"\nDone! Model available at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload ONNX model to Hugging Face Hub")
    parser.add_argument(
        "--repo",
        required=True,
        help="Hugging Face repo ID, e.g. NathanHannon/emoji_gemma3.270m",
    )
    parser.add_argument(
        "--src",
        default="onnx_output_dir",
        help="Local directory produced by optimum-cli export onnx (default: onnx_output_dir)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face write token (or set HF_TOKEN env var / run `huggingface-cli login`)",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    upload(repo_id=args.repo, src_dir=args.src, token=token)


if __name__ == "__main__":
    main()
