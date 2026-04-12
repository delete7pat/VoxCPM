"""Utility functions for VoxCPM audio processing and model management."""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Default sample rate expected by VoxCPM
DEFAULT_SAMPLE_RATE = 16000


def get_model_cache_dir() -> Path:
    """Return the default model cache directory, respecting env overrides."""
    cache_dir = os.environ.get(
        "VOXCPM_CACHE_DIR",
        os.path.join(Path.home(), ".cache", "voxcpm"),
    )
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_model_dir(model_name_or_path: str) -> str:
    """
    Resolve a model name or local path to an absolute directory path.

    If `model_name_or_path` is an existing local path, it is returned as-is.
    Otherwise the name is treated as a sub-directory inside the cache dir.
    """
    local = Path(model_name_or_path)
    if local.exists():
        return str(local.resolve())

    cached = get_model_cache_dir() / model_name_or_path
    if cached.exists():
        return str(cached.resolve())

    logger.warning(
        "Model path '%s' not found locally. "
        "Make sure the model is downloaded before loading.",
        model_name_or_path,
    )
    return str(cached)


def validate_audio_file(filepath: str) -> None:
    """
    Validate that the given file path points to a supported audio file.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file extension is not supported.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(
            f"Unsupported audio format '{path.suffix}'. "
            f"Supported formats: {sorted(SUPPORTED_AUDIO_FORMATS)}"
        )


def normalize_audio(
    audio: np.ndarray,
    target_peak: float = 0.9,
) -> np.ndarray:
    """
    Normalize audio waveform so the peak absolute value equals `target_peak`.

    Args:
        audio: 1-D float32 numpy array.
        target_peak: desired peak amplitude (0.0 – 1.0).

    Returns:
        Normalized float32 numpy array.
    """
    peak = np.abs(audio).max()
    if peak < 1e-8:
        return audio.astype(np.float32)
    return (audio * (target_peak / peak)).astype(np.float32)


def pad_or_trim(
    audio: np.ndarray,
    target_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Pad or trim a 1-D audio array to exactly `target_length` samples.

    Args:
        audio: 1-D numpy array.
        target_length: desired number of samples.
        pad_value: value used for padding (default 0.0 = silence).

    Returns:
        Array of shape (target_length,).
    """
    if len(audio) >= target_length:
        return audio[:target_length]
    pad_width = target_length - len(audio)
    return np.pad(audio, (0, pad_width), constant_values=pad_value)


def seconds_to_samples(seconds: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> int:
    """Convert a duration in seconds to the equivalent number of samples."""
    return int(round(seconds * sample_rate))


def samples_to_seconds(samples: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
    """Convert a number of samples to the equivalent duration in seconds."""
    return samples / sample_rate
