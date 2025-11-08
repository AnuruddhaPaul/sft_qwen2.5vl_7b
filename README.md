# Vision-Language Model Fine-Tuning on reasoning-10k-v2

A comprehensive project for fine-tuning Qwen 2.5 VL 7B model on the `reasoning-10k-v2` dataset using supervised fine-tuning (SFT) with evaluation on VMC-Bench and MathVista benchmarks.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements supervised fine-tuning (SFT) of the Qwen 2.5 VL 7B vision-language model on the `reasoning-10k-v2` dataset, which contains multi-modal reasoning tasks with images and text. The model is evaluated on:

- **VMC-Bench dev split**: Multi-choice visual reasoning benchmark
- **MathVista testmini**: Mathematical visual reasoning benchmark

The project uses **Unsloth** for memory-efficient training with 4-bit quantization and LoRA (Low-Rank Adaptation), enabling training on GPUs with 24GB+ VRAM.

## ✨ Features

- **4-bit Quantized Training**: Dramatically reduces VRAM requirements
- **LoRA Fine-tuning**: Efficient parameter-efficient training
- **Multi-modal Support**: Handles both image and text inputs
- **Checkpointing**: Resume training from interruptions
- **Comprehensive Evaluation**: Automated evaluation on standard benchmarks
- **Flexible Dataset Handling**: Base64 image decoding and preprocessing
- **Answer Extraction**: Robust LaTeX answer extraction with fallback methods

## 📦 Requirements

### Hardware Requirements

- **Minimum**: 24 GB VRAM (RTX 4090 or equivalent)
- **Recommended**: 40 GB VRAM (A100 or RTX 5090)
- **For Colab**: A100 40GB recommended (free T4 may encounter OOM errors)

### Software Requirements

- Python 3.8+
- CUDA 11.8+ or 12.x
- PyTorch 2.0+
- Transformers 4.35+
- Unsloth library

## 🔧 Installation

### Step 1: Install UV (Optional but Recommended)

UV is a fast Python package installer and resolver. Install it using:

```bash
# On Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

### Step 2: Create Virtual Environment

Using UV (recommended):
```bash
uv venv venv
source venv/bin/activate  # On Linux/macOS
# or
.\\venv\\Scripts\\activate  # On Windows
```

Using standard Python:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
.\\venv\\Scripts\\activate  # On Windows
```

### Step 3: Install Dependencies

Using UV:
```bash
uv pip install unsloth transformers datasets pillow torch torchvision trl peft accelerate bitsandbytes
```

Using pip:
```bash
pip install unsloth transformers datasets pillow torch torchvision trl peft accelerate bitsandbytes
```

### Step 4: Install Additional Dependencies

```bash
# For data processing
pip install pandas numpy base64 typing-extensions

# For evaluation
pip install scipy scikit-learn
```

### Platform-Specific Notes

**Windows Users:**
```bash
# Suppress PyTorch compilation errors (already included in code)
# Set environment variables if needed
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**Colab Users:**
```bash
# Install Unsloth directly in notebook
!pip install unsloth

# No need for CUDA setup, pre-configured
```

## 📊 Dataset

The project uses the **reasoning-10k-v2** dataset from Hugging Face:

- **Dataset**: `ArkaMukherjee/reasoning-10k-v2`
- **Size**: ~10,000 samples
- **Format**: Multi-modal (text + base64-encoded images)
- **Task**: Visual reasoning with detailed solutions

The dataset is automatically downloaded during execution.

### Dataset Structure

```python
{
    'question': str,      # Question text
    'solution': str,      # Detailed solution
    'answer': str,        # Final answer
    'image': str,         # Base64-encoded image
    'dataset_name': str,  # Source dataset name
    'uid': str,          # Unique identifier
    'metadata': dict     # Additional metadata
}
```

## 📁 Project Structure

```
project-root/
│
├── sft_qwen2.5vl_7b.py           # Main training script
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── data/
│   ├── checkpoint_unified_dataset.jsonl  # Checkpoint file
│   └── prepared_unified_10k_dataset.jsonl # Processed dataset
│
├── outputs/                       # Training outputs (if enabled)
│   ├── checkpoints/
│   └── logs/
│
└── results/
    ├── vmcbench_dev_results.json  # Evaluation results
    └── mathvista_results.json     # Evaluation results
```

## 🚀 Usage

### Quick Start

1. **Clone or download the project files**

2. **Install dependencies** (see Installation section)

3. **Run the main script**:

For Jupyter Notebook/Colab:
```python
# The script is designed as a Jupyter notebook
# Simply run cells sequentially
```

For Python script:
```bash
python sft_qwen2.5vl_7b.py
```

### Loading the Model

```python
from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct",
    load_in_4bit=True,  # Use 4-bit quantization
    use_gradient_checkpointing="unsloth",
)
```

### Preparing the Dataset

```python
# Load dataset from Hugging Face
dataset = load_unified_dataset("ArkaMukherjee/reasoning-10k-v2")

# Convert to conversation format
converted_dataset, failed_conversions = prepare_dataset(
    dataset,
    start_idx=0,
    save_interval=1000,
    checkpoint_path="checkpoint_unified_dataset.jsonl"
)
```

### Sample Inference (Before Training)

```python
perform_sample_inference(model, tokenizer, converted_dataset, sample_idx=0)
```

## 🏋️ Training

### Training Configuration

```python
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_ratio=0.03,
        num_train_epochs=1,
        max_grad_norm=2.0,
        learning_rate=2e-5,
        logging_steps=50,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="no",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=16384,
    ),
)
```

### Start Training

```python
trainer_stats = trainer.train()
```

### LoRA Configuration

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,              # LoRA rank
    lora_alpha=16,     # LoRA alpha
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
)
```

### Resume Training from Checkpoint

```python
# Set start index to resume from specific point
start_idx = 5000  # Resume from sample 5000

converted_dataset, failed_conversions = prepare_dataset(
    dataset,
    start_idx=start_idx,
    save_interval=1000,
    checkpoint_path="checkpoint_unified_dataset.jsonl"
)
```

## 📈 Evaluation

### VMC-Bench Evaluation

```python
# Evaluate on VMC-Bench dev split
results = evaluate_vmcbench_dev(
    model, 
    tokenizer, 
    max_samples=None,  # None for full evaluation
    verbose=True
)

# Results include:
# - Overall accuracy
# - Accuracy by category
# - Detailed per-sample results
```

### MathVista Evaluation

```python
# Evaluate on MathVista testmini
results = evaluate_mathvista_testmini(
    model, 
    tokenizer, 
    max_samples=None,  # None for full evaluation
    verbose=True
)

# Results include:
# - Overall accuracy
# - Multiple choice vs open-ended accuracy
# - Accuracy by question type
# - Accuracy by answer type
```

### Manual Inference

```python
from PIL import Image

# Load your own image
image = Image.open("path/to/your/image.png")
instruction = "Solve this problem and provide answer in \\\\boxed{answer} format"

# Run inference
FastVisionModel.for_inference(model)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=1280,
    use_cache=True,
    temperature=1.5,
    min_p=0.1
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 🏗️ Model Architecture

### Base Model
- **Name**: Qwen2.5-VL-7B-Instruct
- **Parameters**: 7 billion
- **Quantization**: 4-bit (using bitsandbytes)
- **Context Length**: 16,384 tokens

### LoRA Configuration
- **Rank (r)**: 16
- **Alpha**: 16
- **Dropout**: 0
- **Target Modules**: All vision and language layers

### Training Strategy
- **Method**: Supervised Fine-Tuning (SFT)
- **Optimizer**: AdamW (fused)
- **Learning Rate**: 2e-5
- **LR Scheduler**: Cosine
- **Gradient Checkpointing**: Enabled (Unsloth optimization)

## 📊 Results

Expected performance metrics:

### VMC-Bench Dev Split
- **Overall Accuracy**: ~XX.X% (to be filled after evaluation)
- **Category-wise Performance**: Varies by visual reasoning category

### MathVista Testmini
- **Overall Accuracy**: ~XX.X% (to be filled after evaluation)
- **Multiple Choice**: ~XX.X%
- **Open-ended**: ~XX.X%

## 🛠️ Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory (OOM)

**Solution**:
```python
# Reduce batch size
per_device_train_batch_size = 1
gradient_accumulation_steps = 1

# Enable more aggressive memory optimization
use_gradient_checkpointing = "unsloth"
```

#### 2. Windows PyTorch Errors

**Solution** (already in code):
```python
import torch
torch._dynamo.config.suppress_errors = True
torch._inductor.config.triton.cudagraphs = False
torch._dynamo.config.disable = True
```

#### 3. Dataset Loading Fails

**Solution**:
```bash
# Check internet connection
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Reload dataset
dataset = load_dataset("ArkaMukherjee/reasoning-10k-v2", split="train")
```

#### 4. Image Decoding Errors

**Solution**:
- Check if images are properly base64-encoded
- Verify PIL/Pillow installation: `pip install --upgrade pillow`
- Check image format compatibility (RGB, L modes supported)

#### 5. Checkpoint Loading Fails

**Solution**:
```python
# Delete corrupted checkpoint
import os
if os.path.exists("checkpoint_unified_dataset.jsonl"):
    os.remove("checkpoint_unified_dataset.jsonl")

# Restart from beginning
start_idx = 0
```

### Performance Optimization

#### For Limited VRAM:
```python
# Use smaller batch size
per_device_train_batch_size = 1
gradient_accumulation_steps = 4  # Effective batch size = 4

# Reduce max length
max_length = 8192  # Instead of 16384
```

#### For Faster Training:
```python
# Increase batch size if VRAM allows
per_device_train_batch_size = 2
gradient_accumulation_steps = 2

# Enable data loading optimizations
dataloader_num_workers = 4
dataloader_prefetch_factor = 2
```

### Memory Diagnostics

Check GPU memory usage:
```python
import torch

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
```

Check system RAM:
```python
import psutil, os

p = psutil.Process(os.getpid())
print(f"Memory (rss, vms) MB: {p.memory_info().rss/1024**2}, {p.memory_info().vms/1024**2}")
```

## 📝 Citation

If you use this code or dataset, please cite:

```bibtex
@misc{reasoning10k-v2,
  author = {Arka Mukherjee},
  title = {reasoning-10k-v2: A Multi-modal Reasoning Dataset},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/ArkaMukherjee/reasoning-10k-v2}
}

@misc{qwen2.5-vl,
  title = {Qwen2.5-VL: Vision-Language Model},
  author = {Qwen Team},
  year = {2024},
  publisher = {Alibaba},
}

@misc{unsloth,
  title = {Unsloth: Fast and Memory-Efficient LLM Training},
  author = {Unsloth Team},
  year = {2024},
  url = {https://github.com/unslothai/unsloth}
}
```

## 📄 License

This project uses:
- **Qwen2.5-VL-7B**: Check model card for license
- **reasoning-10k-v2 dataset**: Check dataset card for license
- **Unsloth**: Apache 2.0 License

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [Your email or contact info]

## 🙏 Acknowledgments

- Qwen Team for the base model
- Arka Mukherjee for the reasoning-10k-v2 dataset
- Unsloth team for memory-efficient training library
- VMC-Bench and MathVista teams for evaluation benchmarks

---

**Happy Fine-tuning! 🚀**
