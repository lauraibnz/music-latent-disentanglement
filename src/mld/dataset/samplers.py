"""Grouped batch samplers for stem-level cross-timbre training."""

import random
from collections import defaultdict

import torch
from torch.utils.data import Sampler


def resolve_metadata_group(meta, group_key):
    """Resolve a sample group identifier from metadata."""
    if not isinstance(meta, dict):
        raise ValueError(f"Expected metadata dict, got {type(meta).__name__}")

    if isinstance(group_key, (list, tuple)):
        missing = [key for key in group_key if key not in meta]
        if missing:
            raise ValueError(
                f"Missing group keys {missing} in metadata. "
                f"Available keys: {list(meta.keys())}"
            )
        return tuple(meta[key] for key in group_key)

    if group_key not in meta:
        raise ValueError(
            f"Missing group key '{group_key}' in metadata. "
            f"Available keys: {list(meta.keys())}"
        )
    return meta[group_key]


class GroupedStemBatchSampler(Sampler):
    """
    Create batches with G stem-groups and K chunks per stem-group.

    Each group is typically identified by ``("track", "stem")``, so grouped
    batches contain multiple chunks from the same stem. This matches the
    cross-timbre setup where positives should come from different chunks of the
    same stem rather than from coarse program labels.
    """

    def __init__(
        self,
        metadata_list,
        allowed_indices=None,
        groups_per_batch=16,
        samples_per_group=2,
        group_key=("track", "stem"),
        split=None,
        drop_last=True,
        shuffle=True,
        seed=None,
    ):
        self.groups_per_batch = int(groups_per_batch)
        self.samples_per_group = int(samples_per_group)
        self.group_key = tuple(group_key) if isinstance(group_key, (list, tuple)) else str(group_key)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.split = split
        self.allowed_indices = set(allowed_indices) if allowed_indices is not None else None

        self.group_indices = defaultdict(list)
        for idx, meta in enumerate(metadata_list):
            if not meta:
                continue
            if self.allowed_indices is not None and idx not in self.allowed_indices:
                continue
            if split is not None and meta.get("split") != split:
                continue
            try:
                group_id = resolve_metadata_group(meta, self.group_key)
            except ValueError:
                continue
            self.group_indices[group_id].append(idx)

        self.valid_groups = [
            group_id
            for group_id, indices in self.group_indices.items()
            if len(indices) >= self.samples_per_group
        ]

        if len(self.valid_groups) < self.groups_per_batch:
            print(
                f"Warning: only {len(self.valid_groups)} groups have >= {self.samples_per_group} chunks. "
                f"Reducing groups_per_batch from {self.groups_per_batch}."
            )
            self.groups_per_batch = len(self.valid_groups)

        self.rng = random.Random(seed)
        self.epoch = 0

        print("\n=== Grouped Stem Batch Sampler ===")
        print(f"Split: {self.split if self.split is not None else 'all'}")
        print(f"group_key={self.group_key}")
        print(f"Valid groups (>={self.samples_per_group} chunks): {len(self.valid_groups)}")
        print(f"Batch config: G={self.groups_per_batch}, K={self.samples_per_group}")
        print(f"Effective batch size: {self.groups_per_batch * self.samples_per_group}")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if self.seed is not None:
            self.rng = random.Random(self.seed + self.epoch)

    def _sample_from_group(self, group_id):
        indices = self.group_indices[group_id]
        if len(indices) < self.samples_per_group:
            raise ValueError(
                f"Group {group_id} has only {len(indices)} chunks but requires "
                f"{self.samples_per_group}. This should have been filtered earlier."
            )
        return self.rng.sample(indices, k=self.samples_per_group)

    def __iter__(self):
        if self.groups_per_batch <= 0:
            self.epoch += 1
            return

        if self.seed is not None:
            self.rng = random.Random(self.seed + self.epoch)

        groups = list(self.valid_groups)
        if self.shuffle:
            self.rng.shuffle(groups)

        for start in range(0, len(groups), self.groups_per_batch):
            batch_groups = groups[start:start + self.groups_per_batch]
            if len(batch_groups) < self.groups_per_batch and self.drop_last:
                continue

            batch_indices = []
            for group_id in batch_groups:
                batch_indices.extend(self._sample_from_group(group_id))

            if self.shuffle:
                self.rng.shuffle(batch_indices)

            yield batch_indices

        self.epoch += 1

    def __len__(self):
        if self.groups_per_batch <= 0:
            return 0
        n_full = len(self.valid_groups) // self.groups_per_batch
        if (len(self.valid_groups) % self.groups_per_batch) and not self.drop_last:
            return n_full + 1
        return n_full


def collate_stem_grouped(batch, group_key=("track", "stem")):
    """Collate grouped stem batches and expose integer group ids."""
    group_key = tuple(group_key) if isinstance(group_key, (list, tuple)) else str(group_key)

    metadatas = []
    raw_group_ids = []
    for item in batch:
        meta = item.get("metadata", {})
        metadatas.append(meta)
        raw_group_ids.append(resolve_metadata_group(meta, group_key))

    group_to_idx = {}
    group_ids = []
    for group_id in raw_group_ids:
        if group_id not in group_to_idx:
            group_to_idx[group_id] = len(group_to_idx)
        group_ids.append(group_to_idx[group_id])

    result = {
        "metadata": metadatas,
        "group_ids": torch.tensor(group_ids, dtype=torch.long),
    }

    tensor_keys = sorted(
        {
            key
            for item in batch
            for key, value in item.items()
            if key != "metadata" and isinstance(value, torch.Tensor)
        }
    )

    for key in tensor_keys:
        has_key = [key in item for item in batch]
        if any(has_key) and not all(has_key):
            raise ValueError(
                f"Inconsistent grouped batch: some samples contain '{key}' and others do not."
            )
        values = [item[key] for item in batch]
        result[key] = torch.stack(values)

    return result
