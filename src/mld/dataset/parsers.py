import os

from tqdm import tqdm
import yaml


def slakh(input_dir, ext=".flac"):
    tracks = [
        os.path.join(input_dir, subfolder)
        for subfolder in sorted(os.listdir(input_dir))
    ]
    ban_list = [
        "Chromatic Percussion",
        "Drums",
        "Percussive",
        "Sound Effects",
        "Sound effects",
    ]

    audio_files = []
    metadata = []
    stem_count = 0
    for track in tqdm(tracks):
        metadata_path = os.path.join(track, "metadata.yaml")
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path, "r") as f:
            meta = yaml.safe_load(f)
        for stem_name, stem in meta["stems"].items():
            if stem["inst_class"] in ban_list:
                continue
            stem_path = os.path.join(track, "stems", stem_name + ext)
            if not os.path.exists(stem_path):
                continue
            midi_path = os.path.join(track, "MIDI", stem_name + ".mid")
            stem_meta = stem.copy()
            stem_meta["track"] = os.path.basename(track)
            stem_meta["stem"] = stem_name
            stem_meta["midi_path"] = midi_path if os.path.exists(midi_path) else None
            audio_files.append(stem_path)
            metadata.append(stem_meta)
            stem_count += 1
    
    print(f"Found {stem_count} stems in {len(tracks)} tracks.")

    return audio_files, metadata
