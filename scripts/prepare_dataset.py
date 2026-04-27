import argparse
import hashlib
import os
import random
import tempfile
from pathlib import Path

import lmdb
from tqdm import tqdm
import torch
import torchaudio

from mld.autoencoder import build_latent_codec, encode_latent_batch
from mld.dataset.audio_example import AudioExample
from mld.dataset.midi import build_midi_roll_chunk, load_midi_file_cached
from mld.dataset.parsers import slakh
from mld.dataset.synth import (
    align_waveform_to_target,
    parse_program_exclusion_spec,
    pick_random_programs,
    run_synth_render,
    write_program_shifted_midi,
)
from mld.dataset.transforms import build_timbre_augmented_chunk_bank
from mld.dataset.utils import save_many_to_lmdb

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYNTH_BINARY = REPO_ROOT / "midi2audio" / "target" / "release" / "midi2audio"
DEFAULT_SOUNDFONT = REPO_ROOT / "midi2audio" / "data" / "TimGM6mb.sf2"
DEFAULT_STRUCT_AUG_EXCLUDE_PROGRAMS = "96-127"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a Slakh LMDB dataset for MLD training")
    parser.add_argument("--input_dir", required=True, help="Input dataset directory with train/ and validation/")
    parser.add_argument("--output_dir", required=True, help="Output LMDB directory")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of chunks to encode and write per batch",
    )
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--db_size", type=int, default=200, help="LMDB maximum size in GB")
    parser.add_argument("--ext", default=".flac", help="Stem audio extension inside Slakh track folders")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Target sample rate")
    parser.add_argument("--chunk_size", type=float, default=10.0, help="Chunk size in seconds")
    parser.add_argument("--silence_threshold", type=float, default=0.05, help="Skip chunks below this peak amplitude")
    parser.add_argument(
        "--emb_model",
        default="music2latent",
        choices=["music2latent", "none"],
        help="Latent codec to use",
    )
    parser.add_argument("--latents_only", action="store_true", help="Save latents and metadata only, without audio")
    parser.add_argument("--save_midi", action="store_true", help="Save aligned MIDI roll chunks into LMDB")
    parser.add_argument(
        "--midi_fps",
        type=float,
        default=100.0,
        help="Fallback MIDI frames/sec when saving MIDI without a latent codec",
    )
    parser.add_argument(
        "--save_struct_aug_latent",
        action="store_true",
        help="Save structure-augmentation latent banks as z_struct_aug",
    )
    parser.add_argument(
        "--struct_aug_synth_binary",
        type=str,
        default=str(DEFAULT_SYNTH_BINARY),
        help="Path to the midi2audio synth binary",
    )
    parser.add_argument(
        "--struct_aug_soundfont",
        type=str,
        default=None,
        help="Optional soundfont path",
    )
    parser.add_argument(
        "--struct_aug_program_min",
        type=int,
        default=0,
        help="Minimum random MIDI program for structure augmentation",
    )
    parser.add_argument(
        "--struct_aug_program_max",
        type=int,
        default=127,
        help="Maximum random MIDI program for structure augmentation",
    )
    parser.add_argument(
        "--struct_aug_exclude_programs",
        type=str,
        default=DEFAULT_STRUCT_AUG_EXCLUDE_PROGRAMS,
        help="Comma-separated MIDI programs/ranges to exclude from structure augmentation",
    )
    parser.add_argument(
        "--struct_aug_seed",
        type=int,
        default=None,
        help="Optional base seed for deterministic structure-augmentation sampling",
    )
    parser.add_argument(
        "--struct_aug_strict",
        action="store_true",
        help="Fail instead of falling back to original chunks when structure augmentation fails",
    )
    parser.add_argument(
        "--struct_aug_k",
        type=int,
        default=4,
        help="Number of random MIDI programs to render per stem",
    )
    parser.add_argument(
        "--save_timbre_aug_latent",
        action="store_true",
        help="Save timbre-preserving augmentation latent banks as z_timbre_aug",
    )
    parser.add_argument(
        "--timbre_aug_k",
        type=int,
        default=4,
        help="Number of timbre-preserving variants to generate per stem",
    )
    parser.add_argument(
        "--timbre_aug_ts_min",
        type=float,
        default=0.9,
        help="Minimum tempo stretch factor for timbre augmentation",
    )
    parser.add_argument(
        "--timbre_aug_ts_max",
        type=float,
        default=1.1,
        help="Maximum tempo stretch factor for timbre augmentation",
    )
    parser.add_argument(
        "--timbre_aug_pitch_min",
        type=int,
        default=-3,
        help="Minimum pitch shift in semitones for timbre augmentation",
    )
    parser.add_argument(
        "--timbre_aug_pitch_max",
        type=int,
        default=3,
        help="Maximum pitch shift in semitones for timbre augmentation",
    )
    parser.add_argument(
        "--timbre_aug_segment_samples",
        type=int,
        default=131072,
        help="Segment size for piecewise timbre augmentation",
    )
    parser.add_argument(
        "--timbre_aug_seed",
        type=int,
        default=None,
        help="Optional base seed for deterministic timbre-augmentation sampling",
    )
    parser.add_argument(
        "--timbre_aug_strict",
        action="store_true",
        help="Fail instead of falling back to original chunks when timbre augmentation fails",
    )
    parser.add_argument(
        "--timbre_aug_silence_masks",
        type=int,
        default=2,
        help="Number of random silence masks to apply after pitch/time augmentation",
    )
    parser.add_argument(
        "--enable_timbre_aug_random_silence",
        action="store_true",
        help="Enable random silence masking in timbre-preserving augmentations",
    )
    return parser.parse_args()


def make_stem_rng(base_seed, *, split, track, stem):
    if base_seed is None:
        return random.Random()
    seed_material = f"{split}/{track}/{stem}".encode("utf-8")
    seed_offset = int(hashlib.sha1(seed_material).hexdigest()[:8], 16)
    return random.Random(int(base_seed) + seed_offset)


def load_audio_chunks(audio_path, sample_rate, chunk_size_seconds):
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.contiguous()

    chunk_samples = int(float(sample_rate) * float(chunk_size_seconds))
    total_samples = int(waveform.shape[-1])
    num_full_chunks = total_samples // chunk_samples
    if num_full_chunks == 0:
        return waveform, None, chunk_samples, 0

    truncated_samples = num_full_chunks * chunk_samples
    chunks = waveform[:truncated_samples].reshape(-1, chunk_samples).contiguous()
    return waveform, chunks, chunk_samples, truncated_samples


def prepare_structure_aug_chunk_bank(
    *,
    waveform,
    chunks,
    truncated_samples,
    chunk_samples,
    midi_path,
    original_program,
    track_name,
    stem_name,
    sample_rate,
    struct_aug_k,
    synth_binary,
    soundfont,
    program_min,
    program_max,
    excluded_programs,
    rng,
    pretty_midi_module,
    strict,
):
    programs = pick_random_programs(
        rng,
        k=int(struct_aug_k),
        min_program=int(program_min),
        max_program=int(program_max),
        avoid_program=original_program,
        excluded_programs=excluded_programs,
    )

    try:
        if midi_path is None or not os.path.exists(midi_path):
            raise FileNotFoundError(f"MIDI path not found: {midi_path}")

        bank_chunks = []
        with tempfile.TemporaryDirectory(prefix="mld_struct_aug_") as tmpdir:
            for idx, program in enumerate(programs):
                shifted_midi = os.path.join(tmpdir, f"shifted_{idx}.mid")
                rendered_wav = os.path.join(tmpdir, f"rendered_{idx}.wav")

                write_program_shifted_midi(
                    input_midi_path=midi_path,
                    output_midi_path=shifted_midi,
                    target_program=program,
                    pretty_midi_module=pretty_midi_module,
                )
                run_synth_render(
                    midi_path=shifted_midi,
                    wav_path=rendered_wav,
                    program=program,
                    synth_binary=synth_binary,
                    soundfont=soundfont,
                )
                rendered_waveform, rendered_sr = torchaudio.load(rendered_wav)
                if rendered_sr != sample_rate:
                    rendered_waveform = torchaudio.transforms.Resample(rendered_sr, sample_rate)(rendered_waveform)

                rendered_waveform = align_waveform_to_target(rendered_waveform, waveform)
                if rendered_waveform.dim() == 2:
                    rendered_waveform = rendered_waveform.mean(dim=0)
                rendered_waveform = rendered_waveform[:truncated_samples]
                bank_chunks.append(rendered_waveform.reshape(-1, chunk_samples))

        return torch.stack(bank_chunks, dim=0), programs
    except Exception as exc:
        msg = (
            f"Structure augmentation failed for track={track_name}, stem={stem_name}, "
            f"midi={midi_path}, programs={programs}: {exc}"
        )
        if strict:
            raise RuntimeError(msg) from exc
        print(f"Warning: {msg}. Falling back to original chunks.")
        fallback_program = None if original_program is None else int(original_program)
        return chunks.unsqueeze(0).repeat(int(struct_aug_k), 1, 1), [fallback_program] * int(struct_aug_k)


def prepare_timbre_aug_chunk_bank(
    *,
    waveform,
    chunks,
    sample_rate,
    chunk_samples,
    timbre_aug_k,
    ts_min,
    ts_max,
    pitch_min,
    pitch_max,
    segment_samples,
    random_silence,
    silence_masks,
    rng,
    strict,
    track_name,
    stem_name,
):
    try:
        return build_timbre_augmented_chunk_bank(
            waveform,
            sample_rate=sample_rate,
            chunk_samples=chunk_samples,
            num_augments=int(timbre_aug_k),
            ts_min=float(ts_min),
            ts_max=float(ts_max),
            pitch_min=int(pitch_min),
            pitch_max=int(pitch_max),
            segment_samples=int(segment_samples),
            random_silence=bool(random_silence),
            silence_masks=int(silence_masks),
            rng=rng,
        )
    except Exception as exc:
        msg = f"Timbre augmentation failed for track={track_name}, stem={stem_name}: {exc}"
        if strict:
            raise RuntimeError(msg) from exc
        print(f"Warning: {msg}. Falling back to original chunks.")
        bank = chunks.unsqueeze(0).repeat(int(timbre_aug_k), 1, 1)
        params = [
            [{"fallback": "original"} for _ in range(int(chunks.shape[0]))]
            for _ in range(int(timbre_aug_k))
        ]
        return bank, params


def flush_pending_examples(
    *,
    env,
    pending_examples,
    chunk_id_start,
    model,
    emb_model,
    device,
    latents_only,
    save_midi,
    midi_fps,
    chunk_size,
    pretty_midi_module,
    midi_cache,
    missing_midi_paths,
):
    if not pending_examples:
        return chunk_id_start

    chunks_tensor = torch.stack([item["chunk"] for item in pending_examples]).to(device)
    z_batch = encode_latent_batch(model, emb_model, chunks_tensor)

    z_struct_aug_batch = [None] * len(pending_examples)
    struct_aug_chunks = [item["struct_aug_chunks"] for item in pending_examples]
    if any(chunks is not None for chunks in struct_aug_chunks):
        struct_aug_bank = torch.stack(struct_aug_chunks).to(device)
        batch_size = int(struct_aug_bank.shape[0])
        bank_size = int(struct_aug_bank.shape[1])
        z_struct_flat = encode_latent_batch(
            model,
            emb_model,
            struct_aug_bank.reshape(batch_size * bank_size, *struct_aug_bank.shape[2:]),
        )
        z_struct_aug_batch = z_struct_flat.reshape(batch_size, bank_size, *z_struct_flat.shape[1:])

    z_timbre_aug_batch = [None] * len(pending_examples)
    timbre_aug_chunks = [item["timbre_aug_chunks"] for item in pending_examples]
    if any(chunks is not None for chunks in timbre_aug_chunks):
        timbre_aug_bank = torch.stack(timbre_aug_chunks).to(device)
        batch_size = int(timbre_aug_bank.shape[0])
        bank_size = int(timbre_aug_bank.shape[1])
        z_timbre_flat = encode_latent_batch(
            model,
            emb_model,
            timbre_aug_bank.reshape(batch_size * bank_size, *timbre_aug_bank.shape[2:]),
        )
        z_timbre_aug_batch = z_timbre_flat.reshape(batch_size, bank_size, *z_timbre_flat.shape[1:])

    entries = []
    chunk_id = int(chunk_id_start)
    for item, chunk_z, chunk_struct_z, chunk_timbre_z in zip(
        pending_examples,
        z_batch,
        z_struct_aug_batch,
        z_timbre_aug_batch,
    ):
        ae = AudioExample(output_type="torch")
        if not latents_only:
            ae.put_array("audio", item["chunk"].cpu())
        ae.put_metadata(item["metadata"])
        if chunk_z is not None:
            ae.put_array("z", chunk_z.cpu())
        if chunk_struct_z is not None:
            ae.put_array("z_struct_aug", chunk_struct_z.cpu())
        if chunk_timbre_z is not None:
            ae.put_array("z_timbre_aug", chunk_timbre_z.cpu())

        if save_midi:
            if chunk_z is not None:
                midi_frames = int(chunk_z.shape[-1])
            else:
                midi_frames = max(1, int(round(float(chunk_size) * float(midi_fps))))
            start_sec = float(item["metadata"]["chunk_index"]) * float(chunk_size)
            end_sec = start_sec + float(chunk_size)
            midi_obj = load_midi_file_cached(
                item["metadata"].get("midi_path"),
                midi_cache=midi_cache,
                missing_midi_paths=missing_midi_paths,
                pretty_midi_module=pretty_midi_module,
            )
            midi_roll = build_midi_roll_chunk(
                midi_obj,
                start_sec=start_sec,
                end_sec=end_sec,
                num_frames=midi_frames,
            )
            ae.put_array("midi_roll", midi_roll)

        entries.append((f"{chunk_id:08d}", ae))
        chunk_id += 1

    save_many_to_lmdb(env, entries)
    return chunk_id


def main():
    args = parse_args()

    if int(args.batch_size) <= 0:
        raise ValueError(f"--batch_size must be > 0, got {args.batch_size}")
    if float(args.chunk_size) <= 0.0:
        raise ValueError(f"--chunk_size must be > 0, got {args.chunk_size}")
    if args.latents_only and args.emb_model == "none":
        raise ValueError("--latents_only requires --emb_model music2latent.")

    env = lmdb.open(
        args.output_dir,
        map_size=int(args.db_size) * 1024**3,
        map_async=True,
        writemap=True,
        readahead=False,
    )

    model = build_latent_codec(args.emb_model, device=args.device)
    if (args.save_struct_aug_latent or args.save_timbre_aug_latent) and model is None:
        raise ValueError(
            "--save_struct_aug_latent and --save_timbre_aug_latent require --emb_model music2latent."
        )

    if args.save_struct_aug_latent:
        synth_binary = Path(args.struct_aug_synth_binary)
        if not synth_binary.exists():
            raise FileNotFoundError(
                f"Synth binary not found at '{synth_binary}'. "
                "Compile midi2audio first or set --struct_aug_synth_binary."
            )
        if args.struct_aug_soundfont is None and DEFAULT_SOUNDFONT.exists():
            args.struct_aug_soundfont = str(DEFAULT_SOUNDFONT)

        excluded_programs = parse_program_exclusion_spec(args.struct_aug_exclude_programs)
        excluded_programs = {
            program
            for program in excluded_programs
            if int(args.struct_aug_program_min) <= int(program) <= int(args.struct_aug_program_max)
        }
        allowed_count = (
            int(args.struct_aug_program_max) - int(args.struct_aug_program_min) + 1 - len(excluded_programs)
        )
        if allowed_count <= 0:
            raise ValueError(
                "No allowed structure-augmentation programs remain after exclusions. "
                f"range=[{args.struct_aug_program_min}, {args.struct_aug_program_max}], "
                f"excluded={sorted(excluded_programs)}"
            )
        print(
            "Struct-aug program policy: "
            f"range=[{args.struct_aug_program_min}, {args.struct_aug_program_max}], "
            f"excluded_count={len(excluded_programs)}, allowed_count={allowed_count}"
        )
    else:
        excluded_programs = set()

    pretty_midi_module = None
    midi_cache = {}
    missing_midi_paths = set()
    if args.save_midi or args.save_struct_aug_latent:
        try:
            import pretty_midi as pretty_midi_import
        except ImportError as exc:
            raise ImportError(
                "--save_midi and --save_struct_aug_latent require pretty_midi. "
                "Install it with: pip install pretty_midi"
            ) from exc
        pretty_midi_module = pretty_midi_import

    chunk_id = 0
    for split in ("train", "validation"):
        split_dir = os.path.join(args.input_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Warning: {split_dir} does not exist, skipping.")
            continue

        audio_files, metadata = slakh(split_dir, args.ext)
        pending_examples = []

        for audio_path, meta in zip(tqdm(audio_files, desc=f"{split}"), metadata):
            track_name = meta["track"]
            stem_name = meta["stem"]
            midi_path = meta.get("midi_path")

            waveform, chunks, chunk_samples, truncated_samples = load_audio_chunks(
                audio_path,
                sample_rate=args.sample_rate,
                chunk_size_seconds=args.chunk_size,
            )
            if chunks is None:
                continue

            struct_aug_chunks_bank = None
            struct_aug_programs = None
            if args.save_struct_aug_latent:
                struct_rng = make_stem_rng(
                    args.struct_aug_seed,
                    split=split,
                    track=track_name,
                    stem=stem_name,
                )
                struct_aug_chunks_bank, struct_aug_programs = prepare_structure_aug_chunk_bank(
                    waveform=waveform,
                    chunks=chunks,
                    truncated_samples=truncated_samples,
                    chunk_samples=chunk_samples,
                    midi_path=midi_path,
                    original_program=meta.get("program_num"),
                    track_name=track_name,
                    stem_name=stem_name,
                    sample_rate=args.sample_rate,
                    struct_aug_k=args.struct_aug_k,
                    synth_binary=args.struct_aug_synth_binary,
                    soundfont=args.struct_aug_soundfont,
                    program_min=args.struct_aug_program_min,
                    program_max=args.struct_aug_program_max,
                    excluded_programs=excluded_programs,
                    rng=struct_rng,
                    pretty_midi_module=pretty_midi_module,
                    strict=args.struct_aug_strict,
                )

            timbre_aug_chunks_bank = None
            timbre_aug_params = None
            if args.save_timbre_aug_latent:
                timbre_rng = make_stem_rng(
                    args.timbre_aug_seed,
                    split=split,
                    track=track_name,
                    stem=stem_name,
                )
                timbre_aug_chunks_bank, timbre_aug_params = prepare_timbre_aug_chunk_bank(
                    waveform=waveform,
                    chunks=chunks,
                    sample_rate=args.sample_rate,
                    chunk_samples=chunk_samples,
                    timbre_aug_k=args.timbre_aug_k,
                    ts_min=args.timbre_aug_ts_min,
                    ts_max=args.timbre_aug_ts_max,
                    pitch_min=args.timbre_aug_pitch_min,
                    pitch_max=args.timbre_aug_pitch_max,
                    segment_samples=args.timbre_aug_segment_samples,
                    random_silence=args.enable_timbre_aug_random_silence,
                    silence_masks=args.timbre_aug_silence_masks,
                    rng=timbre_rng,
                    strict=args.timbre_aug_strict,
                    track_name=track_name,
                    stem_name=stem_name,
                )

            for chunk_index, chunk in enumerate(chunks):
                if float(chunk.abs().max()) < float(args.silence_threshold):
                    continue

                chunk_meta = meta.copy()
                chunk_meta["split"] = split
                chunk_meta["chunk_index"] = int(chunk_index)
                chunk_meta["emb_model"] = str(args.emb_model)
                if args.save_struct_aug_latent:
                    chunk_meta["struct_aug_programs"] = struct_aug_programs
                if args.save_timbre_aug_latent:
                    chunk_meta["timbre_aug_params"] = [
                        timbre_aug_params[k_idx][chunk_index]
                        for k_idx in range(len(timbre_aug_params))
                    ]

                pending_examples.append(
                    {
                        "chunk": chunk,
                        "metadata": chunk_meta,
                        "struct_aug_chunks": (
                            None if struct_aug_chunks_bank is None else struct_aug_chunks_bank[:, chunk_index, :]
                        ),
                        "timbre_aug_chunks": (
                            None if timbre_aug_chunks_bank is None else timbre_aug_chunks_bank[:, chunk_index, :]
                        ),
                    }
                )

                if len(pending_examples) >= int(args.batch_size):
                    chunk_id = flush_pending_examples(
                        env=env,
                        pending_examples=pending_examples,
                        chunk_id_start=chunk_id,
                        model=model,
                        emb_model=args.emb_model,
                        device=args.device,
                        latents_only=args.latents_only,
                        save_midi=args.save_midi,
                        midi_fps=args.midi_fps,
                        chunk_size=args.chunk_size,
                        pretty_midi_module=pretty_midi_module,
                        midi_cache=midi_cache,
                        missing_midi_paths=missing_midi_paths,
                    )
                    pending_examples = []

        chunk_id = flush_pending_examples(
            env=env,
            pending_examples=pending_examples,
            chunk_id_start=chunk_id,
            model=model,
            emb_model=args.emb_model,
            device=args.device,
            latents_only=args.latents_only,
            save_midi=args.save_midi,
            midi_fps=args.midi_fps,
            chunk_size=args.chunk_size,
            pretty_midi_module=pretty_midi_module,
            midi_cache=midi_cache,
            missing_midi_paths=missing_midi_paths,
        )

    env.close()
    print(f"Dataset preparation complete. Total chunks saved: {chunk_id}")


if __name__ == "__main__":
    main()
