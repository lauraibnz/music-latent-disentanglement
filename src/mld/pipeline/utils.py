import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from functools import partial
import gin

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
    metadatas = [item['metadata'] for item in batch]
    result = {'metadata': metadatas}

    tensor_keys = sorted(
        {
            key
            for item in batch
            for key, value in item.items()
            if key != 'metadata' and isinstance(value, torch.Tensor)
        }
    )

    for key in tensor_keys:
        has_key = [key in item for item in batch]
        if any(has_key) and not all(has_key):
            raise ValueError(
                f"Inconsistent batch: some samples contain '{key}' and others do not."
            )
        values = [item[key] for item in batch]
        result[key] = torch.stack(values)

    return result


@gin.configurable
def create_dataloaders(dataset, batch_size=16, train_split=0.95, num_workers=8,
                      group_key=("track", "stem"), seed=42,
                      use_grouped_batching=False,
                      groups_per_batch=16, samples_per_group=2,
                      debug_batch_stats=False,
                      skip_empty_midi=False):
    """
    Create train and validation dataloaders from a dataset.
    
    Args:
        dataset: LMDBDataset instance
        batch_size: Batch size for regular sampling
        train_split: Fraction of data for training (used only if 'split' not in metadata)
        num_workers: Number of dataloader workers
        group_key: Metadata key(s) used to group chunks from the same stem
        seed: Random seed for reproducibility
        use_grouped_batching: Use grouped batching (G stems × K chunks each)
        groups_per_batch: Number of stem groups per grouped batch (G)
        samples_per_group: Number of chunks per stem group (K)
        debug_batch_stats: Whether to print batch statistics for debugging
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Try with multiprocessing first, fallback to single process if issues
    try:
        return _create_dataloaders_internal(
            dataset, batch_size, train_split, num_workers,
            group_key, seed, use_grouped_batching, groups_per_batch,
            samples_per_group, debug_batch_stats, skip_empty_midi
        )
    except (RuntimeError, OSError) as e:
        print(f"Warning: DataLoader with {num_workers} workers failed: {e}")
        print("Falling back to single-process DataLoader...")
        return _create_dataloaders_internal(
            dataset, batch_size, train_split, 0,
            group_key, seed, use_grouped_batching, groups_per_batch,
            samples_per_group, debug_batch_stats, skip_empty_midi
        )


def _create_dataloaders_internal(dataset, batch_size, train_split, num_workers,
                                group_key=("track", "stem"), seed=42,
                                use_grouped_batching=False,
                                groups_per_batch=16, samples_per_group=2,
                                debug_batch_stats=False,
                                skip_empty_midi=False):
    """
    Create train and validation dataloaders from a dataset.
    
    Args:
        dataset: LMDBDataset instance
        batch_size: Batch size for regular sampling
        train_split: Fraction of data for training (used only if 'split' not in metadata)
        num_workers: Number of dataloader workers
        group_key: Metadata key(s) for grouping chunks from the same stem
        seed: Random seed
        use_grouped_batching: Whether to use grouped batching
        groups_per_batch: G for grouped batching
        samples_per_group: K for grouped batching
        debug_batch_stats: Print batch statistics
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    grouped_enabled = bool(use_grouped_batching)
    allowed_indices = None

    if skip_empty_midi:
        print("Filtering out chunks with empty midi_roll...")
        allowed_indices = set(dataset.get_nonempty_midi_indices())
        dropped = len(dataset) - len(allowed_indices)
        print(
            f"Retaining {len(allowed_indices)}/{len(dataset)} chunks with non-empty midi_roll "
            f"(dropped {dropped})."
        )

    if grouped_enabled:
        from mld.dataset.samplers import GroupedStemBatchSampler, collate_stem_grouped

        print("Using grouped stem batching...")
        all_metadata = dataset.get_all_metadata()
        
        # Create samplers for train/val splits
        train_sampler = GroupedStemBatchSampler(
            all_metadata,
            allowed_indices=allowed_indices,
            groups_per_batch=groups_per_batch,
            samples_per_group=samples_per_group,
            group_key=group_key,
            split="train",
            drop_last=False,
            shuffle=True,
            seed=seed
        )
        
        val_sampler = GroupedStemBatchSampler(
            all_metadata,
            allowed_indices=allowed_indices,
            groups_per_batch=groups_per_batch,
            samples_per_group=samples_per_group,
            group_key=group_key,
            split="validation",
            drop_last=False,
            shuffle=False,
            seed=seed + 1
        )
        
        # Create dataloaders
        grouped_collate = partial(collate_stem_grouped, group_key=group_key)

        train_loader = DataLoader(
            dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=grouped_collate,
            pin_memory=True,
            persistent_workers=num_workers > 0
        )
        
        val_loader = None
        if len(val_sampler) > 0:
            val_loader = DataLoader(
                dataset,
                batch_sampler=val_sampler,
                num_workers=num_workers,
                collate_fn=grouped_collate,
                pin_memory=True,
                persistent_workers=num_workers > 0
            )
        else:
            print("Warning: grouped validation sampler has zero batches; validation will be skipped.")
        
        # Debug: print batch statistics for first few batches
        if debug_batch_stats:
            print("\n=== Grouped Stem Batch Statistics (first 3 train batches) ===")
            for i, batch in enumerate(train_loader):
                if i >= 3:
                    break
                metadatas = batch['metadata']
                group_ids = batch['group_ids'].tolist()
                
                from collections import Counter
                group_counts = Counter(group_ids)
                
                print(f"\nBatch {i + 1}:")
                print(f"  Groups: {len(group_counts)}")
                print(f"  Chunks per group: min={min(group_counts.values())}, "
                      f"mean={sum(group_counts.values()) / len(group_counts):.1f}, "
                      f"max={max(group_counts.values())}")
                example_meta = metadatas[0] if metadatas else {}
                print(
                    f"  Example stem: track={example_meta.get('track')} "
                    f"stem={example_meta.get('stem')}"
                )
                print(f"  Total samples: {len(group_ids)}")
        
        return train_loader, val_loader
        
    else:
        # Regular training - respect metadata 'split' field if present
        from torch.utils.data import Subset
        
        all_metadata = dataset.get_all_metadata()
        
        # Check if metadata contains 'split' field
        has_split_field = any(meta.get('split') is not None for meta in all_metadata)
        
        if has_split_field:
            # Use metadata-based splits (respects preprocessing splits)
            print("Using metadata-based train/validation splits...")
            train_indices = [
                i for i, meta in enumerate(all_metadata)
                if meta.get('split') == 'train'
                and (allowed_indices is None or i in allowed_indices)
            ]
            val_indices = [
                i for i, meta in enumerate(all_metadata)
                if meta.get('split') == 'validation'
                and (allowed_indices is None or i in allowed_indices)
            ]
            
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
        else:
            # Fallback to random split if no 'split' field in metadata
            print("No 'split' field in metadata, using random split...")
            full_indices = list(range(len(dataset)))
            if allowed_indices is not None:
                full_indices = [i for i in full_indices if i in allowed_indices]
            dataset_size = len(full_indices)
            train_size = int(train_split * dataset_size)
            val_size = dataset_size - train_size
            
            train_indices, val_indices = torch.utils.data.random_split(
                full_indices, [train_size, val_size]
            )
            train_dataset = Subset(dataset, list(train_indices))
            val_dataset = Subset(dataset, list(val_indices))
        
        # Create dataloaders with robust multiprocessing settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=False
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
