import argparse
import lmdb
from tqdm import tqdm
import torch
import torchaudio
import numpy as np
from music2latent import EncoderDecoder

from musdis.dataset.parsers import get_parser
from musdis.dataset.audio_example import AudioExample
from musdis.dataset.utils import save_to_lmdb


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for MusDis training")
    parser.add_argument("--input_dir", required=True, help="Input audio directory")
    parser.add_argument("--output_dir", required=True, help="Output LMDB directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of samples per LMDB transaction")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--db_size", type=int, default=200, help="LMDB maximum size in GB")
    parser.add_argument("--parser", default="slakh", help="Dataset parser to use")
    parser.add_argument("--ext", default=".wav", help="Audio file extension")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--chunk_size", type=float, default=10.0, help="Chunk size in seconds")
    parser.add_argument("--silence_threshold", type=float, default=0.05, help="Silence threshold")
    parser.add_argument("--emb_model", default="music2latent", help="Embedding model to use")
    parser.add_argument("--latents_only", action="store_true", help="Only save latents, skip audio (saves space)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Configuration:")
    print(f"  Input dir: {args.input_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Latents only: {args.latents_only}")
    print(f"  DB size: {args.db_size} GB")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  Chunk size: {args.chunk_size} seconds")
    
    # Create LMDB environment
    env = lmdb.open(
        args.output_dir,
        map_size=args.db_size * 1024**3,  # Convert GB to bytes
        map_async=True,
        writemap=True,
        readahead=False,
    )

    # Initialize encoder model once
    model = None
    if args.emb_model == "music2latent":
        model = EncoderDecoder(device=args.device)

    audio_files, metadata = get_parser(args.parser)(args.input_dir, args.ext)

    chunks_buffer, metadatas_buffer = [], []
    chunk_id = 0

    for i, (file, meta) in enumerate(zip(tqdm(audio_files), metadata)):
        waveform, sr = torchaudio.load(file)
        if sr != args.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, args.sample_rate)(waveform)
            sr = args.sample_rate

        # Handle audio chunking with proper size handling
        chunk_samples = int(args.sample_rate * args.chunk_size)
        total_samples = waveform.shape[-1]
        
        # Truncate to fit exact chunks (discard remainder)
        num_full_chunks = total_samples // chunk_samples
        if num_full_chunks == 0:
            continue  # Skip files shorter than chunk_size
        
        truncated_samples = num_full_chunks * chunk_samples
        waveform_truncated = waveform[..., :truncated_samples]
        chunks = waveform_truncated.reshape(-1, chunk_samples)
        
        for j, chunk in enumerate(chunks):
            if chunk.abs().max() < args.silence_threshold:
                continue  # Skip silent chunks

            chunks_buffer.append(chunk)
            metadatas_buffer.append(meta)

            if len(chunks_buffer) == args.batch_size or (j == len(chunks) - 1 and i == len(audio_files) - 1):

                # Process batch
                chunks_tensor = torch.stack(chunks_buffer).to(args.device)
                
                # Encode if model available
                if model is not None:
                    z_batch = model.encode(chunks_tensor)
                else:
                    z_batch = [None] * len(chunks_buffer)

                # Save each chunk
                for k, (chunk, chunk_z, chunk_meta) in enumerate(zip(chunks_buffer, z_batch, metadatas_buffer)):
                    chunk_meta = chunk_meta.copy()  # Don't modify original
                    chunk_meta["chunk_index"] = j
                    
                    ae = AudioExample(output_type="torch")
                    
                    # Only save audio if not latents_only mode
                    if not args.latents_only:
                        ae.put_array("audio", chunk.cpu())
                    
                    ae.put_metadata(chunk_meta)
                    
                    if chunk_z is not None:
                        ae.put_array("z", chunk_z.cpu())

                    key = f"{chunk_id:08d}"
                    
                    # Debug: Print size info occasionally
                    if chunk_id % 1000 == 0:
                        example_bytes = ae.to_bytes()
                        print(f"Chunk {chunk_id}: {len(example_bytes)} bytes, latent shape: {chunk_z.shape if chunk_z is not None else 'None'}")
                    
                    save_to_lmdb(env, key, ae)
                    chunk_id += 1

                chunks_buffer, metadatas_buffer = [], []

    env.close()
    print(f"Dataset preparation complete. Total chunks saved: {chunk_id}")

if __name__ == "__main__":
    main()

