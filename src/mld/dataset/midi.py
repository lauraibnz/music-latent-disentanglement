import os
import numpy as np
import torch


def load_midi_file_cached(midi_path, midi_cache, missing_midi_paths, pretty_midi_module):
    """
    Load a MIDI file with caching.

    Returns:
        pretty_midi.PrettyMIDI or None if unavailable/invalid.
    """
    if midi_path in midi_cache:
        return midi_cache[midi_path]

    if midi_path is None or not os.path.exists(midi_path):
        if midi_path not in missing_midi_paths:
            print(f"Warning: MIDI not found, using empty roll: {midi_path}")
            missing_midi_paths.add(midi_path)
        midi_cache[midi_path] = None
        return None

    try:
        midi_obj = pretty_midi_module.PrettyMIDI(midi_path)
    except Exception as exc:
        if midi_path not in missing_midi_paths:
            print(f"Warning: Failed to parse MIDI '{midi_path}', using empty roll: {exc}")
            missing_midi_paths.add(midi_path)
        midi_obj = None

    midi_cache[midi_path] = midi_obj
    return midi_obj


def build_midi_roll_chunk(midi_obj, start_sec, end_sec, num_frames):
    """
    Build chunk-aligned binary MIDI roll [128, T] from PrettyMIDI.

    Args:
        midi_obj: pretty_midi.PrettyMIDI or None
        start_sec (float): chunk start time
        end_sec (float): chunk end time
        num_frames (int): number of temporal frames in output roll

    Returns:
        torch.Tensor of shape [128, num_frames] with dtype uint8
    """
    if num_frames <= 0:
        return torch.zeros((128, 0), dtype=torch.uint8)

    if midi_obj is None:
        return torch.zeros((128, num_frames), dtype=torch.uint8)

    duration = max(1e-6, float(end_sec) - float(start_sec))

    times = float(start_sec) + (
        (np.arange(num_frames, dtype=np.float32) + 0.5)
        * (duration / float(num_frames))
    )

    roll = midi_obj.get_piano_roll(times=times)
    roll_bin = (roll > 0).astype(np.uint8)

    return torch.from_numpy(roll_bin)
