# MusDis - Music Disentanglement for Controllable Generation

A PyTorch framework for music disentanglement enabling controllable music generation.

## Overview

MusDis aims to disentangle musical audio into representations:
- **Timbre**: Timbric characteristics (instrument family, brightness, etc.)
- **Structure**: Structural patterns (pitch, loudness, rhythm, etc.)

The framework supports various generative architectures trained on latent representations (currently using [music2latent](https://github.com/SonyCSLParis/music2latent)).

## Architecture

- **Encoders**: Extract timbre and structure representations from latents
- **Generative Model**: Configurable architecture (currently UNet-based diffusion)
- **Training**: Flexible training pipeline with classifier-free guidance support

## Installation

```bash
# Clone repository
git clone https://github.com/lauraibnz/MusDis.git
cd MusDis

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Usage

### Dataset Preparation

```bash
# Prepare SLAKH dataset with latents only (recommended)
python -m scripts.prepare_dataset \
    --input_dir /path/to/slakh \
    --output_dir ./experiments/dataset/slakh_latents \
    --latents_only \
    --sample_rate 44100
```

### Training

```bash
# Train with small model
python -m scripts.train \
    --dataset_path ./experiments/dataset/slakh_latents \
    --config small.gin \
    --name my_experiment \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --epochs 50
```

### Inference

See `experiments/notebooks/inference.ipynb` for audio reconstruction and disentanglement examples.

## Configuration

- `configs/small.gin`: Smaller model for experiments (timbre_dim=16, structure_dim=8)
- `configs/base.gin`: Full model for production (timbre_dim=24, structure_dim=12)

## Project Structure

```
MusDis/
├── src/musdis/           # Core library
│   ├── dataset/          # Data loading and processing
│   ├── pipeline/         # Models and training
│   └── configs/          # Model configurations
├── scripts/              # Training and data preparation
└── experiments/          # Outputs and notebooks
```

## Requirements

- PyTorch >= 2.0
- music2latent
- wandb (for logging)
- lmdb (for datasets)

## License

[Add your license here]
