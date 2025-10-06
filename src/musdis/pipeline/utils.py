import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Set multiprocessing start method to avoid segfaults
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set


def collate_fn(batch):
    """
    Collate function for AudioExample dataset.
    
    Args:
        batch: List of dict objects from BaseDataset.__getitem__
               Each has keys: {'audio': tensor, 'latent': tensor, 'metadata': dict}
    
    Returns:
        Batched dict with torch tensors
    """
    # Separate the components
    audios = []
    latents = []
    metadatas = []
    
    for item in batch:
        # Get data from dict (not AudioExample anymore)
        if 'audio' in item:
            audios.append(item['audio'])
        if 'latent' in item:  
            latents.append(item['latent'])
        metadatas.append(item['metadata'])
    
    # Create batched result
    result = {
        'metadata': metadatas  # Keep as list of dicts
    }
    
    # Stack tensors if available
    if audios:
        # Audio is [160000] -> stack to [B, 160000]
        result['audio'] = torch.stack(audios)
    
    if latents:
        # Latents are [64, 38] -> stack to [B, 64, 38]  
        result['latent'] = torch.stack(latents)
    
    return result


def create_dataloaders(dataset, batch_size=16, train_split=0.95, num_workers=8):
    """
    Create train and validation dataloaders from a dataset.
    
    Args:
        dataset: LMDBDataset instance
        batch_size: Batch size for training
        train_split: Fraction of data for training (rest for validation)
        num_workers: Number of dataloader workers
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Try with multiprocessing first, fallback to single process if issues
    try:
        return _create_dataloaders_internal(dataset, batch_size, train_split, num_workers)
    except (RuntimeError, OSError) as e:
        print(f"Warning: DataLoader with {num_workers} workers failed: {e}")
        print("Falling back to single-process DataLoader...")
        return _create_dataloaders_internal(dataset, batch_size, train_split, 0)


def _create_dataloaders_internal(dataset, batch_size, train_split, num_workers):
    """
    Create train and validation dataloaders from a dataset.
    
    Args:
        dataset: LMDBDataset instance
        batch_size: Batch size for training
        train_split: Fraction of data for training (rest for validation)
        num_workers: Number of dataloader workers
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    # Random split
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders with robust multiprocessing settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,  # Fix: Only set prefetch_factor if multiprocessing
        drop_last=True  # Avoid uneven batch sizes that can cause issues
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,  # Fix: Only set prefetch_factor if multiprocessing
        drop_last=False  # Keep all validation data
    )
    
    return train_loader, val_loader


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps') 
    else:
        return torch.device('cpu')


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable