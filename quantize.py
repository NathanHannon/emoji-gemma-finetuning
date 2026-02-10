from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from pathlib import Path
import os
import shutil


def quantize():
    model_dir = Path("onnx_output_dir/onnx")
    output_file = model_dir / "model_quantized.onnx"

    # Create onnx subdir if it doesn't exist
    model_dir.mkdir(exist_ok=True)

    print(f"Quantizing model from {model_dir}...")

    # Initialize quantizer
    quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="model.onnx")

    # Create configuration for Int8 dynamic quantization (best for CPU/Web)
    # Using 'arm64' config usually gives good defaults for dynamic quantization (u8/s8)
    dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

    # Quantize
    quantizer.quantize(
        save_dir=model_dir,
        quantization_config=dqconfig,
    )

    # Optimum saves as "model_quantized.onnx" by default usually, but sometimes "model.onnx"
    # Let's check and ensure it's named model_quantized.onnx
    matches = list(model_dir.glob("*quantized.onnx"))
    if matches:
        print(f"Success! Generated: {matches[0]}")
    elif (model_dir / "model.onnx").exists():
        # Ideally we don't overwrite the original if it was copied there,
        # but ORTQuantizer might have saved it as model.onnx
        print("Renaming output to model_quantized.onnx...")
        shutil.move(model_dir / "model.onnx", output_file)
        print(f"Success! Generated: {output_file}")

    # Also ensure config.json is in the onnx subdir for transformers.js to find it
    if not (model_dir / "config.json").exists():
        print("Copying config files to onnx subdirectory...")
        for file in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]:
            src = model_dir / file
            if src.exists():
                shutil.copy(src, model_dir / file)


if __name__ == "__main__":
    quantize()
