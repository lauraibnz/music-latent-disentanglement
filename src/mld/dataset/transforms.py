import math
import random
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:
    import librosa
except ImportError:
    librosa = None

_SOX_EFFECTS_AVAILABLE = None


def _get_sox_effects():
    global _SOX_EFFECTS_AVAILABLE
    try:
        import torchaudio.sox_effects as sox_effects
    except (ImportError, OSError) as exc:
        raise ImportError(
            "torchaudio.sox_effects is required for timbre-preserving augmentations. "
            "Install a torchaudio build with SoX support."
        ) from exc
    if _SOX_EFFECTS_AVAILABLE is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sox_effects.apply_effects_tensor(
                    torch.zeros(1, 16, dtype=torch.float32),
                    16000,
                    [],
                )
            _SOX_EFFECTS_AVAILABLE = True
        except (ImportError, OSError):
            _SOX_EFFECTS_AVAILABLE = False
    if not _SOX_EFFECTS_AVAILABLE:
        raise ImportError(
            "torchaudio.sox_effects is installed but libsox is unavailable on this system."
        )
    return sox_effects


def _apply_librosa_pitch_tempo(
    waveform: torch.Tensor,
    sample_rate: int,
    stretch: float,
    pitch_semitones: int,
) -> torch.Tensor:
    if librosa is None:
        raise ImportError(
            "librosa is required as a fallback when torchaudio SoX effects are unavailable."
        )

    wav_np = _ensure_mono_waveform(waveform).detach().cpu().numpy()
    if abs(float(stretch) - 1.0) > 1e-4:
        wav_np = librosa.effects.time_stretch(wav_np, rate=float(stretch))
    if int(pitch_semitones) != 0:
        wav_np = librosa.effects.pitch_shift(
            wav_np,
            sr=int(sample_rate),
            n_steps=float(pitch_semitones),
        )
    return torch.as_tensor(wav_np, dtype=torch.float32)


def _ensure_mono_waveform(waveform: torch.Tensor) -> torch.Tensor:
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform [T] or stereo [C, T], got {tuple(waveform.shape)}")
    return waveform.contiguous()


def _align_waveform_length(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    waveform = _ensure_mono_waveform(waveform)
    cur_len = int(waveform.shape[-1])
    tgt_len = int(target_num_samples)
    if cur_len < tgt_len:
        waveform = torch.nn.functional.pad(waveform, (0, tgt_len - cur_len))
    elif cur_len > tgt_len:
        waveform = waveform[:tgt_len]
    return waveform.contiguous()


class RandomSilenceTransform:

    def __init__(
        self,
        min_width: float = 0.07,
        max_width: float = 0.15,
        min_slope: float = 0.01,
        max_slope: float = 0.05,
    ) -> None:
        self.min_width = float(min_width)
        self.max_width = float(max_width)
        self.min_slope = float(min_slope)
        self.max_slope = float(max_slope)

    def __call__(self, waveform: torch.Tensor, rng: Optional[random.Random] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        rng = rng or random.Random()
        waveform = _ensure_mono_waveform(waveform)
        length = int(waveform.shape[-1])
        if length <= 1:
            return waveform, {"start": 0.0, "width": 0.0, "fade": 0.0}

        min_width = max(1, int(round(self.min_width * length)))
        max_width = max(min_width, int(round(self.max_width * length)))
        width = rng.randint(min_width, max_width)

        min_fade = max(1, int(round(self.min_slope * length)))
        max_fade = max(min_fade, int(round(self.max_slope * length)))
        fade = min(rng.randint(min_fade, max_fade), max(1, length // 4))

        start_min = fade
        start_max = max(start_min, length - width - fade)
        start = rng.randint(start_min, start_max) if start_max >= start_min else start_min

        envelope = torch.ones_like(waveform)
        if fade > 0:
            fade_in = torch.linspace(1.0, 0.0, fade, dtype=waveform.dtype)
            fade_out = torch.linspace(0.0, 1.0, fade, dtype=waveform.dtype)
            envelope[start - fade:start] = fade_in
            envelope[start + width:start + width + fade] = fade_out[: max(0, length - (start + width))]
        envelope[start:start + width] = 0.0

        out = waveform * envelope
        params = {
            "start_frac": float(start / max(1, length)),
            "width_frac": float(width / max(1, length)),
            "fade_frac": float(fade / max(1, length)),
        }
        return out.contiguous(), params


class PitchTimeStretchTransform:

    def __init__(
        self,
        sample_rate: int,
        ts_min: float = 0.9,
        ts_max: float = 1.1,
        pitch_min: int = -3,
        pitch_max: int = 3,
        segment_samples: Optional[int] = 131072,
        random_silence: bool = False,
        silence_masks: int = 2,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.ts_min = float(ts_min)
        self.ts_max = float(ts_max)
        self.pitch_min = int(pitch_min)
        self.pitch_max = int(pitch_max)
        self.segment_samples = None if segment_samples is None or int(segment_samples) <= 0 else int(segment_samples)
        self.random_silence = bool(random_silence)
        self.silence_masks = max(0, int(silence_masks))
        self.silence_transform = RandomSilenceTransform() if self.random_silence else None

    def _sample_segment_params(self, rng: random.Random) -> Tuple[float, int]:
        stretch = rng.uniform(self.ts_min, self.ts_max)
        pitch = rng.randint(self.pitch_min, self.pitch_max) if self.pitch_min != self.pitch_max else int(self.pitch_min)
        return float(stretch), int(pitch)

    def _apply_pitch_tempo(self, waveform: torch.Tensor, stretch: float, pitch_semitones: int) -> torch.Tensor:
        waveform = _ensure_mono_waveform(waveform)
        if waveform.numel() == 0:
            return waveform

        effects: List[List[str]] = []
        if abs(float(stretch) - 1.0) > 1e-4:
            effects.append(["tempo", f"{float(stretch):.6f}"])
        if int(pitch_semitones) != 0:
            effects.append(["pitch", f"{float(pitch_semitones) * 100.0:.2f}"])
            effects.append(["rate", str(self.sample_rate)])

        if not effects:
            return waveform

        try:
            sox_effects = _get_sox_effects()
            augmented, _ = sox_effects.apply_effects_tensor(
                waveform.unsqueeze(0),
                self.sample_rate,
                effects,
            )
            return augmented.squeeze(0).to(dtype=torch.float32)
        except (ImportError, OSError):
            return _apply_librosa_pitch_tempo(
                waveform,
                sample_rate=self.sample_rate,
                stretch=stretch,
                pitch_semitones=pitch_semitones,
            )

    def __call__(
        self,
        waveform: torch.Tensor,
        rng: Optional[random.Random] = None,
        return_params: bool = False,
    ):
        rng = rng or random.Random()
        waveform = _ensure_mono_waveform(waveform)
        target_num_samples = int(waveform.shape[-1])

        if self.segment_samples is None:
            segment_starts = [0]
        else:
            segment_starts = list(range(0, target_num_samples, self.segment_samples))

        pieces: List[torch.Tensor] = []
        segment_params: List[Dict[str, Any]] = []
        for start in segment_starts:
            end = target_num_samples if self.segment_samples is None else min(target_num_samples, start + self.segment_samples)
            segment = waveform[start:end]
            stretch, pitch = self._sample_segment_params(rng)
            augmented = self._apply_pitch_tempo(segment, stretch=stretch, pitch_semitones=pitch)
            pieces.append(augmented)
            segment_params.append(
                {
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "stretch": float(stretch),
                    "pitch_semitones": int(pitch),
                }
            )

        augmented = torch.cat(pieces, dim=-1) if pieces else waveform.clone()
        silence_params: List[Dict[str, float]] = []
        if self.silence_transform is not None:
            for _ in range(self.silence_masks):
                augmented, silence_param = self.silence_transform(augmented, rng=rng)
                silence_params.append(silence_param)

        augmented = _align_waveform_length(augmented, target_num_samples)
        params = {
            "segment_samples": self.segment_samples,
            "segments": segment_params,
            "random_silence": bool(self.random_silence),
            "silence_masks": silence_params,
        }

        if return_params:
            return augmented, params
        return augmented

    def apply_global(
        self,
        waveform: torch.Tensor,
        rng: Optional[random.Random] = None,
        return_params: bool = False,
        align_output_length: bool = True,
    ):
        rng = rng or random.Random()
        waveform = _ensure_mono_waveform(waveform)
        target_num_samples = int(waveform.shape[-1])

        stretch, pitch = self._sample_segment_params(rng)
        augmented = self._apply_pitch_tempo(waveform, stretch=stretch, pitch_semitones=pitch)

        silence_params: List[Dict[str, float]] = []
        if self.silence_transform is not None:
            for _ in range(self.silence_masks):
                augmented, silence_param = self.silence_transform(augmented, rng=rng)
                silence_params.append(silence_param)

        if align_output_length:
            augmented = _align_waveform_length(augmented, target_num_samples)
        params = {
            "mode": "global_stem" if not align_output_length else "global_chunk",
            "stretch": float(stretch),
            "pitch_semitones": int(pitch),
            "random_silence": bool(self.random_silence),
            "silence_masks": silence_params,
            "output_num_samples": int(augmented.shape[-1]),
        }
        if return_params:
            return augmented, params
        return augmented


def build_timbre_augmented_chunk_bank(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    chunk_samples: int,
    num_augments: int = 4,
    ts_min: float = 0.9,
    ts_max: float = 1.1,
    pitch_min: int = -3,
    pitch_max: int = 3,
    segment_samples: Optional[int] = 131072,
    random_silence: bool = False,
    silence_masks: int = 2,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, List[List[Dict[str, Any]]]]:
    waveform = _ensure_mono_waveform(waveform)
    rng = rng or random.Random()
    num_augments = int(num_augments)
    if num_augments <= 0:
        raise ValueError(f"num_augments must be positive, got {num_augments}")

    transform = PitchTimeStretchTransform(
        sample_rate=sample_rate,
        ts_min=ts_min,
        ts_max=ts_max,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        segment_samples=segment_samples,
        random_silence=random_silence,
        silence_masks=silence_masks,
    )

    total_samples = int(waveform.shape[-1])
    num_full_chunks = total_samples // int(chunk_samples)
    if num_full_chunks <= 0:
        raise ValueError("Waveform is shorter than one full chunk.")

    bank_chunks: List[torch.Tensor] = []
    bank_params: List[List[Dict[str, Any]]] = []
    for _ in range(num_augments):
        augmented_stem, stem_params = transform.apply_global(
            waveform,
            rng=rng,
            return_params=True,
            align_output_length=False,
        )
        stretch = float(stem_params["stretch"])
        augmented_chunks: List[torch.Tensor] = []
        augment_params: List[Dict[str, Any]] = []
        for chunk_index in range(num_full_chunks):
            original_start = int(chunk_index * int(chunk_samples))
            augmented_start = int(round(float(original_start) / max(stretch, 1e-8)))
            augmented_end = augmented_start + int(chunk_samples)
            chunk_aug = augmented_stem[augmented_start:augmented_end]
            chunk_aug = _align_waveform_length(chunk_aug, int(chunk_samples))
            augmented_chunks.append(chunk_aug)
            params = dict(stem_params)
            params.update(
                {
                    "chunk_index": int(chunk_index),
                    "original_start_sample": int(original_start),
                    "augmented_start_sample": int(augmented_start),
                }
            )
            augment_params.append(params)
        bank_chunks.append(torch.stack(augmented_chunks, dim=0))
        bank_params.append(augment_params)

    return torch.stack(bank_chunks, dim=0), bank_params
