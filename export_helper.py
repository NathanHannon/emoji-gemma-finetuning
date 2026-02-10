import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import MODEL_ID, ADAPTER_PATH


def merge_and_export():
    print(f"Loading base model: {MODEL_ID}")
    # Load base model in full precision (float32) or float16 for merging
    # Note: 4-bit models cannot be merged easily. We load the base model normally.
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    print(f"Loading adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    output_dir = "./merged_model"
    print(f"Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Merge complete.")

    print("\n" + "=" * 50)
    print("NEXT STEPS FOR WEB EXPORT:")
    print("=" * 50)
    print("1. Install optimum and required dependencies:")
    print("   pip install optimum[onnxruntime]")
    print("")
    print("2. Run the export command to create the web-ready ONNX model:")
    print(
        f"   optimum-cli export onnx --model {output_dir} --task text-generation-with-past --trust-remote-code onnx_output_dir"
    )
    print("")
    print("3. (Optional but recommended) Quantize the model for web (smaller size):")
    print(
        "   See Transformers.js documentation for 'quantize_model.py' or use optimum's quantization tools."
    )
    print("")
    print(
        "4. Upload the 'onnx_output_dir' to Hugging Face Hub (e.g., your-username/emoji-gemma-web)"
    )
    print("")
    print("5. Update 'docs/worker.js' with your new Hugging Face Model ID.")
    print("=" * 50)


if __name__ == "__main__":
    merge_and_export()
