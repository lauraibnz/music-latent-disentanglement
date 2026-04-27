import random

import lmdb
import torch
from torch.utils.data import Dataset
import gin

from .audio_example import AudioExample
from .utils import load_from_lmdb
from .synth import parse_program_exclusion_spec


def _normalize_program_exclusion_spec(spec):
    if spec is None:
        return frozenset()
    if isinstance(spec, str):
        return frozenset(parse_program_exclusion_spec(spec))
    if isinstance(spec, (list, tuple, set, frozenset)):
        return frozenset(int(x) for x in spec)
    return frozenset([int(spec)])


@gin.configurable
class BaseDataset(Dataset):
    """Dataset for loading preprocessed audio chunks from LMDB (lazy env init per worker)."""
    
    def __init__(self,
                 lmdb_path,
                 transform=None,
                 return_latents=True,
                 return_midi=True,
                 return_struct_aug_latent=True,
                 return_timbre_aug_latent=True,
                 extra_array_keys=(),
                 struct_aug_exclude_programs=None,
                 struct_aug_filter_fallback="original"):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.return_latents = return_latents
        self.return_midi = return_midi
        self.return_struct_aug_latent = return_struct_aug_latent
        self.return_timbre_aug_latent = return_timbre_aug_latent
        self.extra_array_keys = tuple(str(key) for key in extra_array_keys)
        self.struct_aug_exclude_programs = _normalize_program_exclusion_spec(struct_aug_exclude_programs)
        self.struct_aug_filter_fallback = str(struct_aug_filter_fallback).lower()
        if self.struct_aug_filter_fallback not in {"original", "all", "error"}:
            raise ValueError(
                "struct_aug_filter_fallback must be one of: original, all, error. "
                f"Got '{struct_aug_filter_fallback}'."
            )
        
        # Don't open LMDB here! Each worker will open its own environment
        self.env = None
        self._nonempty_midi_indices_cache = {}

        # Get length (open temporarily in main process)
        with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                self.length = txn.stat()['entries']

    def _init_env(self):
        """Lazily open LMDB environment inside each worker process."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,  # safer for multiple workers
                meminit=False
            )

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Ensure environment is initialized (works for multiprocessing)
        self._init_env()

        # Load AudioExample from LMDB
        key = f"{idx:08d}"
        example = load_from_lmdb(self.env, key)
        
        if example is None:
            raise IndexError(f"No data found for index {idx}")
        
        result = {'metadata': dict(example.get_metadata() or {})}
        
        # Get audio if available
        audio = example.get('audio', None)
        if audio is not None:
            if self.transform:
                audio = self.transform(audio)
            result['audio'] = audio
        
        # Add latents if available and requested
        if self.return_latents:
            latent = example.get('z', None)
            if latent is not None:
                result['latent'] = latent
            if self.return_struct_aug_latent:
                struct_aug_bank = example.get('z_struct_aug', None)
                if struct_aug_bank is not None:
                    struct_aug_programs = list(result['metadata'].get('struct_aug_programs') or [])
                    filtered_programs = struct_aug_programs

                    if hasattr(struct_aug_bank, "dim") and struct_aug_bank.dim() >= 1:
                        k = int(struct_aug_bank.shape[0])
                        candidate_choices = list(range(k))

                        if self.struct_aug_exclude_programs and struct_aug_programs:
                            candidate_choices = [
                                choice_idx for choice_idx in range(min(k, len(struct_aug_programs)))
                                if struct_aug_programs[choice_idx] is None
                                or int(struct_aug_programs[choice_idx]) not in self.struct_aug_exclude_programs
                            ]

                        if candidate_choices and len(candidate_choices) != k:
                            struct_aug_bank = struct_aug_bank[candidate_choices]
                            filtered_programs = [
                                struct_aug_programs[choice_idx]
                                for choice_idx in candidate_choices
                                if choice_idx < len(struct_aug_programs)
                            ]
                        elif not candidate_choices:
                            if self.struct_aug_filter_fallback == "original":
                                if latent is None:
                                    raise ValueError(
                                        "struct_aug_filter_fallback='original' requires base latent 'z' to exist."
                                    )
                                struct_aug_bank = latent.unsqueeze(0)
                                filtered_programs = [None]
                                result['metadata']['struct_aug_filter_fallback'] = "original"
                            elif self.struct_aug_filter_fallback == "all":
                                filtered_programs = struct_aug_programs
                                result['metadata']['struct_aug_filter_fallback'] = "all"
                            else:
                                raise ValueError(
                                    "All structure augmentations were filtered out for sample "
                                    f"{idx} with excluded programs={sorted(self.struct_aug_exclude_programs)}."
                                )

                    result['z_struct_aug'] = struct_aug_bank
                    if filtered_programs:
                        result['metadata']['struct_aug_programs'] = filtered_programs
            if self.return_timbre_aug_latent:
                timbre_aug_latent = example.get('z_timbre_aug', None)
                if timbre_aug_latent is not None:
                    result['z_timbre_aug'] = timbre_aug_latent

        # Add MIDI roll if available and requested
        if self.return_midi:
            midi_roll = example.get('midi_roll', None)
            if midi_roll is not None:
                result['midi_roll'] = midi_roll

        for key in self.extra_array_keys:
            if key in result:
                continue
            value = example.get(key, None)
            if value is not None:
                result[key] = value
        
        return result

    def get_all_metadata(self):
        """
        Load all metadata for the entire dataset.
        Used for building grouped training batches.
        
        Returns:
            list: List of metadata dicts, one per dataset index
        """
        # Use a temporary environment to avoid pickling issues
        # Don't use self.env to keep the dataset picklable
        import lmdb
        meta_list = []
        
        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as temp_env:
            with temp_env.begin() as txn:
                for idx in range(self.length):
                    key = f"{idx:08d}"
                    example = load_from_lmdb(temp_env, key)
                    if example is not None:
                        meta_list.append(example.get_metadata())
                    else:
                        meta_list.append({})
        
        return meta_list

    def get_nonempty_midi_indices(self, split=None):
        """
        Return dataset indices whose stored ``midi_roll`` exists and is not all-zero.

        Args:
            split: Optional metadata split filter (for example ``"train"`` or
                ``"validation"``). When provided, only indices from that split
                are returned.

        Returns:
            list[int]: Dataset indices with non-empty MIDI rolls.
        """
        cache_key = split if split is not None else "__all__"
        cached = self._nonempty_midi_indices_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        indices = []
        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, data in cursor:
                    example = AudioExample.from_bytes(bytes(data))
                    metadata = example.get_metadata() or {}
                    if split is not None and metadata.get("split") != split:
                        continue

                    midi_roll = example.get("midi_roll", None)
                    if midi_roll is None or midi_roll.numel() == 0:
                        continue
                    if torch.any(midi_roll != 0):
                        indices.append(int(key.decode()))

        self._nonempty_midi_indices_cache[cache_key] = tuple(indices)
        return list(indices)

    def __del__(self):
        """Ensure LMDB environment is closed."""
        if self.env is not None:
            self.env.close()
            self.env = None
