import argparse
import os
import random
from pathlib import Path

import gin
import numpy as np
import torch
import wandb

import mld
from mld.dataset import BaseDataset
from mld.pipeline.utils import count_parameters, create_dataloaders, get_device


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_config_path(config_arg):
    config_path = Path(config_arg)
    mld_base = Path(mld.__file__).resolve().parent
    candidate_paths = [
        config_path,
        mld_base / config_path,
        mld_base / "pipeline" / "configs" / config_path,
    ]
    for path in candidate_paths:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"Config not found: {config_arg}. "
        f"Tried: {', '.join(str(path) for path in candidate_paths)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLD model")
    parser.add_argument("--dataset_path", required=True, help="Path to LMDB dataset")
    parser.add_argument("--config", required=True, help="Gin config name or path")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--output_dir", default="./experiments/runs", help="Base output directory")
    parser.add_argument("--device", default="cuda:0", help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--wandb_project", default="mld", help="W&B project name")
    parser.add_argument("--wandb_entity", default=None, help="Optional W&B entity/team")
    parser.add_argument("--wandb_name", default=None, help="W&B run name (defaults to --name)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to checkpoint_epoch_*.pt or best_model.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    resolved_config = resolve_config_path(args.config)

    print(f"Loading config: {resolved_config}")
    gin.parse_config_file(resolved_config)

    set_seed(args.seed)
    print(f"Set random seed to: {args.seed}")

    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")

    experiment_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")

    print("Loading dataset...")
    dataset = BaseDataset(args.dataset_path)
    print(f"Dataset size: {len(dataset)}")

    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {0 if val_loader is None else len(val_loader)}")
    if val_loader is None:
        print("Validation disabled: no eligible validation batches were found.")

    print("Creating model...")
    model_class_ref = gin.query_parameter("%model_class")
    model_class = model_class_ref.configurable.wrapped
    model = model_class().to(device)
    print(f"Model class: {model_class.__name__}")

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if not args.no_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name or args.name,
            config={
                "name": args.name,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "resume_from_checkpoint": args.resume_from_checkpoint,
                "dataset_size": len(dataset),
                "total_params": total_params,
                "trainable_params": trainable_params,
            },
        )

    model.fit(
        dataloader=train_loader,
        validloader=val_loader,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        wandb_logging=not args.no_wandb,
        experiment_dir=experiment_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
