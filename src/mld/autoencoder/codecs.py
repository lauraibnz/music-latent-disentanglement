from __future__ import annotations

from typing import Optional

import torch


def normalize_latent_codec_name(codec_name: Optional[str]) -> Optional[str]:
    if codec_name is None:
        return None
    codec_name = str(codec_name).strip().lower()
    if codec_name in {"m2l", "music2latent"}:
        return "music2latent"
    if codec_name in {"none", "audio_only"}:
        return "none"
    if codec_name == "auto":
        return "auto"
    raise ValueError(
        f"Unknown latent codec '{codec_name}'. "
        "Expected one of: auto, music2latent, none."
    )


def _build_music2latent_codec(device: Optional[str] = None):
    from music2latent import EncoderDecoder as Music2LatentEncoderDecoder

    kwargs = {"device": device} if device is not None else {}
    return Music2LatentEncoderDecoder(**kwargs)


_CODEC_BUILDERS = {
    "music2latent": _build_music2latent_codec,
}


def build_latent_codec(codec_name: Optional[str], device: Optional[str] = None):
    codec_name = normalize_latent_codec_name(codec_name)
    if codec_name == "none":
        return None

    builder = _CODEC_BUILDERS.get(codec_name)
    if builder is None:
        raise ValueError(
            f"Unsupported latent codec '{codec_name}'. Expected one of: "
            f"{', '.join(sorted(_CODEC_BUILDERS))}, none."
        )
    return builder(device=device)


def _ensure_model_latent_batch(latent_like, *, latent_dim: int, device: Optional[torch.device] = None) -> torch.Tensor:
    latent = torch.as_tensor(latent_like, dtype=torch.float32, device=device)
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)
    if latent.ndim != 3:
        raise ValueError(f"Expected latent with shape [B, C, T] or [C, T], got {tuple(latent.shape)}")
    if latent.shape[1] != int(latent_dim):
        raise ValueError(
            f"Expected latent channels {latent_dim}, got {latent.shape[1]} for shape {tuple(latent.shape)}"
        )
    return latent


def encode_latent_batch(codec, codec_name: str, chunks_tensor: torch.Tensor) -> torch.Tensor | list[None]:
    codec_name = normalize_latent_codec_name(codec_name)
    if codec is None:
        return [None] * len(chunks_tensor)
    if codec_name != "music2latent":
        raise ValueError(f"Unsupported latent codec '{codec_name}' in encode_latent_batch.")
    return codec.encode(chunks_tensor)


def encode_audio_to_model_latent(
    codec,
    codec_name: str,
    waveform,
    *,
    latent_dim: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if codec is None:
        raise ValueError("encode_audio_to_model_latent requires a codec instance.")
    codec_name = normalize_latent_codec_name(codec_name)
    if codec_name != "music2latent":
        raise ValueError(f"Unsupported latent codec '{codec_name}' in encode_audio_to_model_latent.")

    raw_latent = codec.encode(waveform)
    return _ensure_model_latent_batch(raw_latent, latent_dim=latent_dim, device=device)


def encode_audio_batch_to_model_latent(
    codec,
    codec_name: str,
    waveforms,
    *,
    latent_dim: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if codec is None:
        raise ValueError("encode_audio_batch_to_model_latent requires a codec instance.")

    codec_name = normalize_latent_codec_name(codec_name)
    if codec_name != "music2latent":
        raise ValueError(f"Unsupported latent codec '{codec_name}' in encode_audio_batch_to_model_latent.")

    waveforms = torch.as_tensor(waveforms, dtype=torch.float32)
    if waveforms.ndim == 1:
        return encode_audio_to_model_latent(
            codec,
            codec_name,
            waveforms,
            latent_dim=latent_dim,
            device=device,
        )

    if waveforms.ndim == 3 and waveforms.shape[1] == 1:
        waveforms = waveforms[:, 0, :]
    elif waveforms.ndim == 3 and waveforms.shape[2] == 1:
        waveforms = waveforms[:, :, 0]

    if waveforms.ndim != 2:
        raise ValueError(
            "Expected batched waveforms with shape [B, T], [B, 1, T], or [B, T, 1], "
            f"got {tuple(waveforms.shape)}"
        )

    raw_latent = codec.encode(waveforms.to(device=device) if device is not None else waveforms)
    return _ensure_model_latent_batch(raw_latent, latent_dim=latent_dim, device=device)


def decode_model_latent_to_audio(
    codec,
    codec_name: str,
    latent_batch,
    *,
    latent_dim: int,
):
    if codec is None:
        raise ValueError("decode_model_latent_to_audio requires a codec instance.")
    codec_name = normalize_latent_codec_name(codec_name)
    if codec_name != "music2latent":
        raise ValueError(f"Unsupported latent codec '{codec_name}' in decode_model_latent_to_audio.")

    latent_batch = _ensure_model_latent_batch(latent_batch, latent_dim=latent_dim).detach().cpu().numpy()
    return codec.decode(latent_batch)
