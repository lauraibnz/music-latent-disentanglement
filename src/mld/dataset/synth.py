import os
import subprocess
import torch


def parse_program_exclusion_spec(spec):
    """
    Parse exclusion specification like:
      "96-103,112-127,45"
    into a set of int MIDI program numbers.
    """
    if spec is None:
        return set()

    spec = str(spec).strip()
    if spec == "":
        return set()

    excluded = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid exclusion token '{token}'")
            lo = int(parts[0].strip())
            hi = int(parts[1].strip())
            if lo > hi:
                lo, hi = hi, lo
            excluded.update(range(lo, hi + 1))
        else:
            excluded.add(int(token))
    return excluded

def _candidate_programs(min_program, max_program, avoid_program=None, excluded_programs=None):
    """
    Build candidate MIDI program list from range and exclusions.
    """
    if min_program > max_program:
        raise ValueError(f"Invalid program range: min={min_program}, max={max_program}")

    excluded = set(int(p) for p in (excluded_programs or []))
    candidates = []
    for p in range(int(min_program), int(max_program) + 1):
        if p in excluded:
            continue
        if avoid_program is not None and int(p) == int(avoid_program):
            continue
        candidates.append(int(p))
    return candidates


def pick_random_programs(rng, k, min_program, max_program, avoid_program=None, excluded_programs=None):
    """
    Pick k MIDI programs from allowed candidates.
    - Avoids avoid_program when possible.
    - Excludes excluded_programs.
    - Uses sampling without replacement when enough candidates exist.
      Otherwise allows duplicates.
    """
    k = int(k)
    if k <= 0:
        return []

    candidates = _candidate_programs(
        min_program=int(min_program),
        max_program=int(max_program),
        avoid_program=avoid_program,
        excluded_programs=excluded_programs,
    )
    if not candidates:
        raise ValueError(
            "No valid MIDI programs available after applying struct-aug constraints. "
            f"range=[{min_program}, {max_program}], avoid_program={avoid_program}, "
            f"excluded_count={len(set(excluded_programs or []))}"
        )

    if len(candidates) >= k:
        return [int(p) for p in rng.sample(candidates, k=k)]

    return [int(rng.choice(candidates)) for _ in range(k)]


def run_synth_render(midi_path, wav_path, program, synth_binary, soundfont=None):
    """
    Render MIDI -> WAV using the midi2audio binary.

    Tries a couple of common argument layouts for compatibility.
    """
    binary = str(synth_binary)
    midi = str(midi_path)
    wav = str(wav_path)
    program = str(int(program))
    sf2 = str(soundfont) if soundfont else None

    candidates = []
    # This repo's rust synth style: <soundfont> <midi> <out_path>
    if sf2:
        candidates.append([binary, sf2, midi, wav])

    # Expected modern style: flags for in/out/program (+ optional soundfont)
    cmd1 = [binary, "--midi", midi, "--wav", wav, "--program", program]
    if sf2:
        cmd1 += ["--soundfont", sf2]
    candidates.append(cmd1)

    # Alternate style: positional midi/wav + flags
    cmd2 = [binary, midi, wav, "--program", program]
    if sf2:
        cmd2 += ["--soundfont", sf2]
    candidates.append(cmd2)

    # Simple positional fallback
    cmd3 = [binary, midi, wav]
    if sf2:
        cmd3.append(sf2)
    candidates.append(cmd3)

    last_err = None
    for cmd in candidates:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(
        f"Failed to run synth binary '{binary}' with supported argument layouts. "
        f"Last error: {last_err}"
    )


def write_program_shifted_midi(input_midi_path, output_midi_path, target_program, pretty_midi_module):
    """
    Write a temporary MIDI where all non-drum tracks use target_program.
    """
    midi_obj = pretty_midi_module.PrettyMIDI(input_midi_path)
    for inst in midi_obj.instruments:
        if not inst.is_drum:
            inst.program = int(target_program)
    midi_obj.write(output_midi_path)


def align_waveform_to_target(rendered_waveform, target_waveform):
    """
    Match rendered waveform channels/length to target waveform.
    """
    # Match channel count
    if rendered_waveform.shape[0] != target_waveform.shape[0]:
        if target_waveform.shape[0] == 1:
            rendered_waveform = rendered_waveform.mean(dim=0, keepdim=True)
        elif rendered_waveform.shape[0] == 1:
            rendered_waveform = rendered_waveform.repeat(target_waveform.shape[0], 1)
        else:
            rendered_waveform = rendered_waveform[:target_waveform.shape[0], :]

    # Match sample length
    tgt_len = target_waveform.shape[-1]
    cur_len = rendered_waveform.shape[-1]
    if cur_len < tgt_len:
        pad = tgt_len - cur_len
        rendered_waveform = torch.nn.functional.pad(rendered_waveform, (0, pad))
    elif cur_len > tgt_len:
        rendered_waveform = rendered_waveform[..., :tgt_len]

    return rendered_waveform
