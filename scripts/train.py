import argparse
import os
import torch
import gin
import wandb
import numpy as np
import random

# Import musdis to set up gin config paths
from musdis.dataset import BaseDataset  
from musdis.pipeline.utils import create_dataloaders, get_device, count_parameters


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train MusDis disentanglement model")
    parser.add_argument("--dataset_path", required=True, help="Path to LMDB dataset")
    parser.add_argument("--config", required=True, help="Gin config name (searches in musdis.pipeline.configs)")
    parser.add_argument("--name", required=True, help="Experiment name (creates subdirectory)")
    parser.add_argument("--output_dir", default="./experiments/runs", help="Base output directory")
    parser.add_argument("--device", default="cuda:0", help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--wandb_project", default="musdis", help="Wandb project name")
    parser.add_argument("--wandb_name", default=None, help="Wandb run name (defaults to --name)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load gin config (gin automatically finds it in musdis.pipeline.configs)
    print(f"Loading config: {args.config}")
    gin.parse_config_file(args.config)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Set random seed to: {args.seed}")
    
    # Setup device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create experiment directory
    experiment_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = BaseDataset(args.dataset_path)
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Using batch_size={args.batch_size}, num_workers={args.num_workers}")
    
    # Create model (using gin config to determine model class)
    print("Creating model...")
    from musdis.pipeline.model import Diffusion
    model = Diffusion()
    model = model.to(device)
    
    # Print model info
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize wandb
    if not args.no_wandb:
        wandb_name = args.wandb_name if args.wandb_name is not None else args.name
        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config={
                "name": args.name,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "dataset_size": len(dataset),
                "total_params": total_params,
                "trainable_params": trainable_params,
            }
        )
        # Note: gin config will be logged by model.fit() when config.gin is saved
    
    # Start training using model's fit method
    model.fit(
        dataloader=train_loader,
        validloader=val_loader,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        wandb_logging=not args.no_wandb,
        experiment_dir=experiment_dir
    )
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
