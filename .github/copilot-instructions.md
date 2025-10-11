# Copilot Instructions for Emoji Gemma Fine-tuning Project

## Project Overview
This is a specialized ML fine-tuning project that adapts Google's Gemma 3 270M model for text-to-emoji translation using LoRA (Low-Rank Adaptation) technique. The architecture follows a configuration-driven approach with separate concerns for model setup, training, and evaluation.

## Key Architecture Patterns

### Configuration-Driven Design
- **`config.py`** - Central configuration hub containing all hyperparameters, model configs, and training arguments
- Three main config objects: `BNB_CONFIG` (quantization), `LORA_CONFIG` (adapter), `TRAINING_ARGS` (training)
- Always modify hyperparameters in `config.py`, never hardcode in training scripts

### Data Pipeline Pattern
```python
# Standard flow in prepare_dataset():
load_dataset() → filter(is_only_emoji) → map(translate_format) → train_test_split()
```
- Dataset transforms to conversational format with system/user/assistant roles
- Emoji validation using `emoji.is_emoji()` to ensure clean data
- 90/10 train/test split is hardcoded

### Memory-Efficient Training Stack
- **4-bit quantization** via BitsAndBytesConfig (reduces VRAM from ~12GB to ~3GB)
- **LoRA adapters** instead of full fine-tuning (faster, less memory)
- **FP16 precision** for RTX 4070 compatibility (not BF16)
- **Gradient checkpointing disabled** to balance speed vs memory

## Critical Workflows

### GPU Setup Verification
```bash
python gpu_test.py  # Always run first - validates CUDA PyTorch installation
```
- Common issue: CPU-only PyTorch installed (`2.8.0+cpu`) instead of CUDA version
- Fix: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Training Execution
```bash
python train.py  # Main training script
tensorboard --logdir myemoji-gemma-adapters/runs  # View training progress
```
- Training automatically saves checkpoints to `myemoji-gemma-adapters/`
- TensorBoard logs written to `runs/` subdirectory
- GPU memory monitoring built into training script

### Model Inference (Post-Training)
```python
# Load fine-tuned adapter
from transformers import pipeline
pipe = pipeline("text-generation", model="./myemoji-gemma-adapters")
```

## Project-Specific Conventions

### Hardware Assumptions
- **RTX 4070 (12GB VRAM)** - all memory settings optimized for this specific GPU
- Batch size=4, max_length=256 chosen to fit within VRAM constraints
- FP16 used instead of BF16 due to GPU architecture limitations

### Training Strategy
- **Conservative approach**: 3 epochs, constant learning rate (5e-5), early stopping disabled
- **Monitoring**: Logs every 50 steps, saves every epoch, keeps only 2 best checkpoints
- **No evaluation during training** due to SFTConfig limitations in current TRL version

### Error Handling Patterns
- GPU detection at training start with fallback messaging
- Memory monitoring before/after training phases
- Automatic device placement with `device_map="auto"`

## Integration Points

### External Dependencies
- **HuggingFace Hub**: Model (`google/gemma-3-270m-it`) and dataset (`kr15t3n/text2emoji`)
- **TensorBoard**: Automatic logging integration via `report_to="tensorboard"`
- **Transformers + TRL**: Tight coupling - SFTTrainer uses `processing_class` not `tokenizer`

### File Dependencies
```
train.py → config.py (imports all configurations)
config.py → transformers, peft, trl (configuration objects)
myemoji-gemma-adapters/ ← train.py (output directory)
```

## Development Guidelines

### When Modifying Training
1. Always test GPU setup with `gpu_test.py` first
2. Adjust hyperparameters in `config.py`, not inline
3. Monitor TensorBoard for overfitting (validation loss increasing)
4. Check `.gitignore` includes model checkpoints (large files)

### Common Pitfalls
- **SFTConfig parameter compatibility**: Many standard training args not supported (eval_strategy, early_stopping_patience)
- **Memory management**: Increasing batch_size beyond 4 may cause OOM on RTX 4070
- **Data format**: Must use conversational format with exact role names ("system", "user", "assistant")

### Debugging Commands
```bash
# Verify environment
python -c "import torch; print(torch.cuda.is_available())"
python -c "from transformers import AutoTokenizer; print('TRL installed correctly')"

# Check model loading
python -c "from config import MODEL_ID; from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(MODEL_ID)"
```