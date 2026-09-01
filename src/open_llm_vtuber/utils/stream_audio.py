import base64

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import make_chunks

from ..agent.output_types import Actions
from ..agent.output_types import DisplayText


def _normalize(volumes: list) -> list:
    """Scale per-chunk volumes to 0..1 by the loudest chunk."""
    max_volume = max(volumes) if volumes else 0
    if max_volume == 0:
        raise ValueError("Audio is empty or all zero.")
    return [volume / max_volume for volume in volumes]


def _get_volume_by_chunks(audio: AudioSegment, chunk_length_ms: int) -> list:
    """
    Calculate the normalized volume (RMS) for each chunk of the audio.

    Parameters:
        audio (AudioSegment): The audio segment to process.
        chunk_length_ms (int): The length of each audio chunk in milliseconds.

    Returns:
        list: Normalized volumes for each chunk.
    """
    chunks = make_chunks(audio, chunk_length_ms)
    return _normalize([chunk.rms for chunk in chunks])


def _volumes_from_samples(samples: np.ndarray, frame_rate: int, chunk_length_ms: int):
    """Per-chunk RMS straight from int16 samples, matching AudioSegment.rms."""
    if samples.ndim > 1:  # interleave channels the way pydub does
        samples = samples.reshape(-1)
    per_chunk = max(1, int(frame_rate * chunk_length_ms / 1000))
    # Fixed-size slices with a trailing partial chunk, matching make_chunks.
    chunks = [samples[i : i + per_chunk] for i in range(0, len(samples), per_chunk)]
    volumes = [
        float(np.sqrt(np.mean(np.square(c.astype(np.float64))))) if c.size else 0.0
        for c in chunks
    ]
    return _normalize(volumes)


def _read_wav_fast(audio_path: str, chunk_length_ms: int):
    """
    Fast path for WAV: read the samples once, reuse the file bytes verbatim.

    Avoids pydub's decode + re-export round trip (and the ffmpeg subprocess it
    shells out to for compressed formats), which measured ~70ms per sentence -
    more than the local TTS synthesis itself.
    """
    samples, frame_rate = sf.read(audio_path, dtype="int16", always_2d=False)
    volumes = _volumes_from_samples(samples, frame_rate, chunk_length_ms)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    return audio_bytes, volumes


def prepare_audio_payload(
    audio_path: str | None,
    chunk_length_ms: int = 20,
    display_text: DisplayText = None,
    actions: Actions = None,
    forwarded: bool = False,
) -> dict[str, any]:
    """
    Prepares the audio payload for sending to a broadcast endpoint.
    If audio_path is None, returns a payload with audio=None for silent display.

    Parameters:
        audio_path (str | None): The path to the audio file to be processed, or None for silent display
        chunk_length_ms (int): The length of each audio chunk in milliseconds
        display_text (DisplayText, optional): Text to be displayed with the audio
        actions (Actions, optional): Actions associated with the audio

    Returns:
        dict: The audio payload to be sent
    """
    if isinstance(display_text, DisplayText):
        display_text = display_text.to_dict()

    if not audio_path:
        # Return payload for silent display
        return {
            "type": "audio",
            "audio": None,
            "volumes": [],
            "slice_length": chunk_length_ms,
            "display_text": display_text,
            "actions": actions.to_dict() if actions else None,
            "forwarded": forwarded,
        }

    audio_bytes = None
    volumes = None
    if audio_path.lower().endswith(".wav"):
        try:
            audio_bytes, volumes = _read_wav_fast(audio_path, chunk_length_ms)
        except ValueError:
            raise
        except Exception:
            audio_bytes = None  # fall through to the pydub path

    if audio_bytes is None:
        try:
            audio = AudioSegment.from_file(audio_path)
            audio_bytes = audio.export(format="wav").read()
        except Exception as e:
            raise ValueError(
                f"Error loading or converting generated audio file to wav file '{audio_path}': {e}"
            )
        volumes = _get_volume_by_chunks(audio, chunk_length_ms)

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "type": "audio",
        "audio": audio_base64,
        "volumes": volumes,
        "slice_length": chunk_length_ms,
        "display_text": display_text,
        "actions": actions.to_dict() if actions else None,
        "forwarded": forwarded,
    }

    return payload


# Example usage:
# payload, duration = prepare_audio_payload("path/to/audio.mp3", display_text="Hello", expression_list=[0,1,2])
