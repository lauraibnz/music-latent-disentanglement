import os
from tqdm import tqdm
import yaml


def slakh(input_dir, ext=".flac"):
    tracks = [
        os.path.join(input_dir, subfolder)
        for subfolder in os.listdir(input_dir)
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
            stem_path = os.path.join(track, 'stems', stem_name + ext)
            if not os.path.exists(stem_path):
                continue
            audio_files.append(stem_path)
            metadata.append(stem)
            stem_count += 1
    
    print(f"Found {stem_count} stems in {len(tracks)} tracks.")
        
    return audio_files, metadata


def get_parser(parser_name):
    if parser_name == "slakh":
        return slakh
    else:
        raise ValueError(f"Parser {parser_name} not available")
