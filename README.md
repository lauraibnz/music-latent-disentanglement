# Symbolic-Guided Music Latent Disentanglement for One-Shot Timbre Transfer and Controllable Generation

*Anonymous authors (paper currently under review)*

## Overview

This repository studies structure--timbre disentanglement for one-shot timbre transfer and controllable music generation. In contrast to approaches that define structure implicitly through pitch and tempo transformations, this project places structure at the center of the disentanglement process by guiding it with symbolic musical information, specifically MIDI.

The method combines:

- structure-preserving augmentations
- pitch supervision
- timbre triplet supervision
- a `DiffusionTransformer1D` trained under a `RectifiedFlow` formulation

All components operate in the latent space of a pretrained music autoencoder, `music2latent`, enabling efficient training and inference while supporting reconstruction, one-shot timbre transfer, and controllable generation.

![Music Disentanglement Overview](assets/music-disentanglement.png)

The figure above summarizes the proposed symbolic-guided method. Solid arrows denote components used during both training and inference, while dashed arrows indicate training-only components.

## Current Scope

At the moment, this repository is intentionally focused and relatively minimal:

- one main latent codec path: `music2latent`
- one main generative path: `RectifiedFlow + DiffusionTransformer1D`
- symbolic-guided structure supervision through aligned `MIDI`
- dataset preparation, training, and notebook-based inference/reconstruction utilities

The current baseline configuration is centered around:

- `src/mld/pipeline/configs/base.gin`

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Dataset Preparation

Example command for preparing a latent dataset with MIDI supervision and augmentation banks:

```bash
python -m scripts.prepare_dataset \
    --input_dir /path/to/dataset \
    --output_dir ./experiments/dataset/mld_latents \
    --emb_model music2latent \
    --latents_only \
    --save_midi \
    --save_struct_aug_latent \
    --save_timbre_aug_latent
```

## Training

Example training command:

```bash
python -m scripts.train \
    --dataset_path ./experiments/dataset/mld_latents \
    --config base.gin \
    --name mld_baseline \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --epochs 50
```

## Repository Layout

```text
src/mld/
  dataset/        dataset loading, parsing, MIDI utilities, augmentations
  pipeline/       models, networks, configs, training utilities
  autoencoder/    latent codec helpers
scripts/          dataset preparation and training entry points
experiments/      notebooks, outputs, and run artifacts
```

## Notes

- This README is intentionally concise for now and can be expanded later with fuller setup, evaluation, and inference details.
