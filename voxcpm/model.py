"""VoxCPM model loading and inference utilities."""

import os
from pathlib import Path
from typing import Optional, Union

import torch
import numpy as np

from voxcpm.utils import resolve_model_dir, validate_audio_file, normalize_audio, pad_or_trim


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_LEN = 30  # seconds


class VoxCPM:
    """Wrapper around the VoxCPM speech recognition model.

    Handles model loading, audio preprocessing, and transcription inference.
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            model_dir: Path to the model directory. If None, resolves from
                       environment variable or default cache location.
            device: Target device, e.g. 'cpu', 'cuda', 'cuda:0'. Defaults
                    to CUDA if available, otherwise CPU.
            dtype: Torch dtype for inference. Defaults to float16 on CUDA,
                   float32 on CPU.
        """
        self.model_dir = Path(resolve_model_dir(model_dir))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if "cuda" in self.device else torch.float32)

        self._model = None
        self._tokenizer = None
        self._processor = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "VoxCPM":
        """Load the model, tokenizer and processor into memory.

        Returns self to allow chaining: ``voxcpm = VoxCPM().load()``.
        """
        if self._model is not None:
            return self

        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required. Install it with: pip install transformers"
            ) from exc

        model_path = str(self.model_dir)
        print(f"[VoxCPM] Loading model from {model_path} on {self.device} ({self.dtype})")

        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self._model.eval()

        print("[VoxCPM] Model loaded successfully.")
        return self

    @property
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> str:
        """Transcribe speech from an audio file or raw waveform array.

        Args:
            audio: Path to an audio file or a float32 numpy array at
                   ``DEFAULT_SAMPLE_RATE`` Hz.
            language: Optional BCP-47 language code hint (e.g. 'zh', 'en').
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            The transcribed text string.
        """
        if not self.is_loaded:
            self.load()

        # Accept file paths or raw arrays
        if isinstance(audio, (str, Path)):
            validate_audio_file(audio)
            waveform = normalize_audio(audio, target_sr=DEFAULT_SAMPLE_RATE)
        elif isinstance(audio, np.ndarray):
            waveform = audio.astype(np.float32)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        # Trim / pad to the maximum supported duration
        max_samples = DEFAULT_SAMPLE_RATE * DEFAULT_MAX_AUDIO_LEN
        waveform = pad_or_trim(waveform, max_samples)

        # Build model inputs
        inputs = self._processor(
            waveform,
            sampling_rate=DEFAULT_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        generate_kwargs = {"max_new_tokens": max_new_tokens}
        if language:
            generate_kwargs["language"] = language

        with torch.no_grad():
            predicted_ids = self._model.generate(input_features, **generate_kwargs)

        transcription = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        return transcription.strip()
