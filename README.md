# LoRA Fine-Tuning for Causal Language Modeling

This repository contains a Jupyter notebook demonstrating how to fine-tune a 4-bit quantized large language model (LLM) using **LoRA (Low-Rank Adaptation)** via **PEFT (Parameter-Efficient Fine-Tuning)** and the **Hugging Face Transformers** and **Accelerate** libraries. It supports experiment tracking with **Weights & Biases (W&B)**.

---

## Overview

Traditional fine-tuning of large models is resource-intensive. LoRA offers a memory- and compute-efficient method to adapt large pre-trained models to specific downstream tasks by injecting a small number of trainable parameters into existing layers.

This project leverages:
- **Quantized base models** (4-bit) for low-memory footprint
- **LoRA adapters** for efficient fine-tuning
- **Hugging Face Trainer** for training orchestration
- **Weights & Biases** for experiment tracking

---

## Features

- Loads and fine-tunes a pre-trained quantized LLM (e.g., Mistral, LLaMA)
- Applies LoRA to specified target modules (e.g., `q_proj`, `v_proj`)
- Tracks training metrics via Weights & Biases
- Supports text generation pipeline with the fine-tuned model

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch transformers peft accelerate datasets bitsandbytes wandb
```

---

## Setup

### 1. Load or Clone This Repository

```bash
git clone https://github.com/your-username/lora-finetune-llm.git
cd lora-finetune-llm
```

### 2. Add Your W&B API Key

Save your API key in a file called `wandb_key.txt` (one line only):

```bash
echo "your_wandb_api_key" > wandb_key.txt
```

Make sure this file is ignored by Git:

```bash
echo "wandb_key.txt" >> .gitignore
```

---

## Usage

### Training

Open the notebook `model.ipynb` and run the cells step-by-step. It includes:

- Model and tokenizer loading
- LoRA configuration
- Dataset loading and tokenization
- PEFT integration
- Hugging Face `Trainer` setup
- Training loop with W&B logging

### Inference

After training, generate text using the final fine-tuned model:

```python
from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)
output = pipe("Once upon a time")[0]["generated_text"]
print(output)
```

---

## Notes

- The training uses 4-bit quantization via `bitsandbytes` for low memory usage
- LoRA reduces the number of trainable parameters to a small fraction
- Suitable for low-resource environments (e.g., Google Colab with 1 GPU)
- Make sure your model supports `bnb_4bit` loading

---

## Customization

To adapt to other models or tasks:

- Change `base_model_id` to another compatible model
- Update `target_modules` in `LoraConfig` if your base model has different attention modules
- Swap the dataset with your own using `load_dataset()` from `datasets`

---

## Acknowledgements

This project uses the following open-source libraries:

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [Weights & Biases](https://wandb.ai)
