# 🎯 Emoji Gemma Fine-tuning

A specialized machine learning project that fine-tunes Google's Gemma 3 270M model for text-to-emoji translation using LoRA (Low-Rank Adaptation) technique. Transform natural language into contextually appropriate emojis! 

## 🚀 Features

- **Memory-Efficient Training**: Uses 4-bit quantization and LoRA adapters to reduce VRAM usage from ~12GB to ~3GB
- **RTX 4070 Optimized**: All settings tuned for 12GB VRAM GPUs with FP16 precision
- **Real-time Monitoring**: TensorBoard integration for training visualization
- **Interactive Testing**: Built-in chat interface to test your fine-tuned model
- **Configuration-Driven**: Centralized hyperparameter management

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA RTX 4070 (12GB VRAM) or equivalent
- **RAM**: 16GB+ recommended
- **Storage**: 5GB+ free space

### Software
- **Python**: 3.8+
- **CUDA**: 12.1 (handled automatically by PyTorch)
- **OS**: Windows 10/11, Linux, or macOS

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/NathanHannon/emoji-gemma-finetuning.git
cd emoji-gemma-finetuning
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install transformers datasets trl peft bitsandbytes emoji tensorboard accelerate
```

### 4. Verify GPU Setup
```bash
python gpu_test.py
```

You should see output like:
```
=== GPU Test ===
PyTorch version: 2.8.0+cu121
CUDA available: True
GPU 0: NVIDIA GeForce RTX 4070
Memory: 12.0 GB
```

⚠️ **If you see `PyTorch version: 2.8.0+cpu`**, you have the CPU-only version. Reinstall with CUDA support.

## 🎮 Quick Start

### Training Your Model

1. **Configure settings** (optional):
   ```python
   # Edit config.py to adjust hyperparameters
   num_train_epochs = 3  # Adjust based on your needs
   per_device_train_batch_size = 4  # Increase if you have more VRAM
   learning_rate = 5e-5  # Conservative learning rate
   ```

2. **Start training**:
   ```bash
   python train.py
   ```

3. **Monitor progress**:
   ```bash
   # In a new terminal
   tensorboard --logdir myemoji-gemma-adapters/runs
   # Open http://localhost:6006 in your browser
   ```

### Testing Your Model

```bash
python test_model.py
```

Example interaction:
```
> I love programming
🤖: 💻❤️

> Beautiful sunset today
🤖: 🌅✨

> quit
```

## 📊 Training Details

### Dataset
- **Source**: `kr15t3n/text2emoji` from Hugging Face
- **Size**: ~50K text-emoji pairs
- **Format**: Conversational (system/user/assistant roles)
- **Split**: 90% training, 10% validation

### Model Architecture
- **Base Model**: Google Gemma 3 270M Instruct
- **Technique**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit NF4 with FP16 compute
- **Target Modules**: All linear layers

### Training Configuration
```python
# Key hyperparameters
Epochs: 3
Batch Size: 4 per device
Learning Rate: 5e-5 (constant)
Max Sequence Length: 256 tokens
LoRA Rank: 16
LoRA Alpha: 32
LoRA Dropout: 0.05
```

### Performance
- **Training Time**: ~20-30 minutes on RTX 4070
- **Memory Usage**: ~7-10GB VRAM
- **Model Size**: Base model (540MB) + Adapters (~32MB)

## 📁 Project Structure

```
emoji-gemma-finetuning/
├── config.py                 # Central configuration hub
├── train.py                  # Main training script
├── test_model.py             # Interactive model testing
├── gpu_test.py               # GPU setup verification
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── myemoji-gemma-adapters/   # Fine-tuned model output
│   ├── checkpoint-*/         # Training checkpoints
│   ├── runs/                 # TensorBoard logs
│   └── README.md             # Model card
└── .github/
    └── copilot-instructions.md  # AI agent guidelines
```

## 🔧 Configuration

All hyperparameters are centralized in `config.py`:

```python
# Model and dataset
MODEL_ID = "google/gemma-3-270m-it"
DATASET_ID = "kr15t3n/text2emoji"
ADAPTER_PATH = "./myemoji-gemma-adapters"

# Quantization settings
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# LoRA adapter settings
LORA_CONFIG = LoraConfig(
    r=16,                      # Rank
    lora_alpha=32,             # Alpha scaling
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# Training arguments
TRAINING_ARGS = SFTConfig(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    fp16=True,                 # Use FP16 for RTX 4070
    report_to="tensorboard"
)
```

## 🎯 Usage Examples

### Basic Text-to-Emoji Translation
```python
from transformers import AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM

# Load your fine-tuned model
model = AutoPeftModelForCausalLM.from_pretrained("./myemoji-gemma-adapters")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")

# Create prompt
messages = [
    {"role": "system", "content": "Translate this text to emoji:"},
    {"role": "user", "content": "Happy birthday!"}
]

# Generate emoji
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Batch Processing
```python
texts = [
    "I love coffee in the morning",
    "Working late tonight",
    "Beach vacation was amazing"
]

for text in texts:
    emoji = get_emoji_translation(text)
    print(f"{text} → {emoji}")
```

## 🐛 Troubleshooting

### Common Issues

1. **"CUDA not available" error**
   ```bash
   # Reinstall PyTorch with CUDA
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Out of memory error**
   ```python
   # Reduce batch size in config.py
   per_device_train_batch_size = 2  # or even 1
   ```

3. **SFTConfig parameter errors**
   - Many standard Transformers training arguments aren't supported by `SFTConfig`
   - Check the current `config.py` for working parameters
   - Use `processing_class` instead of `tokenizer` in SFTTrainer

4. **Model loading issues**
   ```bash
   # Verify model files exist
   ls myemoji-gemma-adapters/
   # Should show: adapter_config.json, adapter_model.safetensors, etc.
   ```

### Performance Optimization

1. **Increase batch size** (if you have VRAM headroom):
   ```python
   per_device_train_batch_size = 8  # Test incrementally
   ```

2. **Enable gradient accumulation**:
   ```python
   gradient_accumulation_steps = 2  # Effective batch size = 4 * 2 = 8
   ```

3. **Monitor GPU usage**:
   ```bash
   nvidia-smi  # Check VRAM usage during training
   ```

## 📈 Monitoring & Evaluation

### TensorBoard Visualization
```bash
tensorboard --logdir myemoji-gemma-adapters/runs
```

Key metrics to watch:
- **Training Loss**: Should decrease steadily
- **Learning Rate**: Constant at 5e-5
- **GPU Memory**: Should stay under 12GB

### Manual Evaluation
```bash
python test_model.py
```

Test various inputs:
- Emotions: "feeling happy", "so sad today"
- Activities: "cooking dinner", "watching movies"
- Objects: "my new car", "beautiful flowers"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit with clear messages: `git commit -m "Add feature description"`
5. Push and create a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install black flake8 pytest

# Format code
black .

# Run tests
python -m pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google** for the Gemma 3 270M model
- **Hugging Face** for Transformers, TRL, and PEFT libraries
- **kr15t3n** for the text2emoji dataset
- **Microsoft** for BitsAndBytesConfig quantization

## 📚 Related Projects

- [Hugging Face TRL](https://github.com/huggingface/trl) - Transformer Reinforcement Learning
- [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning
- [Gemma Models](https://huggingface.co/google/gemma-3-270m-it) - Google's Gemma family

## 📊 Changelog

### v1.0.0 (Current)
- Initial release with Gemma 3 270M fine-tuning
- LoRA adapter implementation
- RTX 4070 optimization
- TensorBoard integration
- Interactive testing interface

---

**Made with ❤️ for the emoji community! 🎉**