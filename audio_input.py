"""Audio input module for Jarvis (Phase 2).

This module handles recording audio from the microphone with automated Voice
Activity Detection (VAD) based on Root Mean Square (RMS) energy, and speech-to-text
transcription using faster-whisper.
"""

import sys
import wave

import numpy as np
import sounddevice as sd

import config

# ==============================================================================
# Tunable Audio Settings (Constants)
# ==============================================================================
SAMPLE_RATE = getattr(config, "AUDIO_SAMPLE_RATE", 16000)           # Sampling rate in Hz (16 kHz is standard for speech/Whisper)
CHANNELS = 1                  # Mono channel recording
CHUNK_DURATION = 0.05         # Duration of each buffer slice in seconds (50 ms = 800 samples)
SILENCE_THRESHOLD = getattr(config, "AUDIO_SILENCE_THRESHOLD", 0.015)     # RMS energy threshold to distinguish speech from silence
SILENCE_LIMIT_SEC = getattr(config, "AUDIO_SILENCE_LIMIT_SEC", 1.5)       # Seconds of continuous silence after speech before auto-stopping
MAX_RECORDING_SEC = getattr(config, "AUDIO_MAX_RECORDING_SEC", 30.0)      # Hard maximum recording length in seconds
INITIAL_SILENCE_TIMEOUT = getattr(config, "AUDIO_INITIAL_SILENCE_TIMEOUT", 8.0) # Seconds to wait for initial speech before giving up
DEFAULT_DEVICE = None         # None = system default; or specify an integer device index


def calculate_rms(audio_chunk: np.ndarray) -> float:
    """Calculate the Root Mean Square (RMS) energy volume of an audio chunk.

    RMS measures the signal power/volume. Values typically range from:
    - 0.000 to 0.008 for background room silence
    - 0.015 to 0.200+ for normal speaking volume

    Args:
        audio_chunk: A numpy array of float32 audio samples.

    Returns:
        The RMS value as a float.
    """
    if audio_chunk.size == 0:
        return 0.0
    # Mean of squared sample amplitudes
    mean_square = np.mean(np.square(audio_chunk))
    if np.isnan(mean_square) or mean_square < 0:
        return 0.0
    return float(np.sqrt(mean_square))


def record_until_silence(
    device: int | str | None = DEFAULT_DEVICE,
    sample_rate: int = SAMPLE_RATE,
    silence_threshold: float = SILENCE_THRESHOLD,
    silence_limit_sec: float = SILENCE_LIMIT_SEC,
    max_duration_sec: float = MAX_RECORDING_SEC,
    initial_timeout_sec: float = INITIAL_SILENCE_TIMEOUT,
) -> np.ndarray | None:
    """Record from the microphone until silence is detected after speech.

    Behavior:
    1. Records from the default (or specified) microphone at 16000 Hz, mono.
    2. Waits for speech to begin. If no speech is heard within initial_timeout_sec (8s),
       it gives up and returns None.
    3. Once speech has been detected, it continues recording until silence_limit_sec (1.5s)
       of consecutive silence occurs.
    4. Automatically stops if max_duration_sec (30s) is reached.
    5. Returns the complete audio as a 1D float32 numpy array, or None if no speech was heard.

    Args:
        device: Input device index or name (None = system default).
        sample_rate: Sampling frequency in Hz.
        silence_threshold: RMS volume above which audio is considered speech.
        silence_limit_sec: Duration of silence required to stop after speech.
        max_duration_sec: Maximum duration limit.
        initial_timeout_sec: Timeout before first speech is detected.

    Returns:
        A 1D numpy array with dtype float32 containing the audio samples,
        or None if no speech was detected.
    """
    chunk_samples = int(sample_rate * CHUNK_DURATION)
    recorded_chunks: list[np.ndarray] = []

    speech_detected = False
    silence_duration = 0.0
    total_duration = 0.0

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=CHANNELS,
            dtype="float32",
            device=device,
            blocksize=chunk_samples,
        ) as stream:
            while total_duration < max_duration_sec:
                chunk, overflowed = stream.read(chunk_samples)
                chunk_flat = chunk.flatten()
                recorded_chunks.append(chunk_flat)

                chunk_time = len(chunk_flat) / sample_rate
                total_duration += chunk_time

                rms = calculate_rms(chunk_flat)

                if not speech_detected:
                    # Waiting for user to start speaking
                    if rms >= silence_threshold:
                        speech_detected = True
                        silence_duration = 0.0
                    elif total_duration >= initial_timeout_sec:
                        # Initial timeout: no speech was detected within the limit
                        return None
                else:
                    # User has spoken; now monitor for trailing silence
                    if rms < silence_threshold:
                        silence_duration += chunk_time
                        if silence_duration >= silence_limit_sec:
                            # Silence limit reached, user has finished speaking
                            break
                    else:
                        # Still speaking, reset the silence timer
                        silence_duration = 0.0

    except Exception as err:
        print(f"[Audio Error] Failed to access microphone: {err}", file=sys.stderr)
        return None

    if not speech_detected or not recorded_chunks:
        return None

    # Concatenate all recorded chunks into a single contiguous 1D array in RAM
    audio_array = np.concatenate(recorded_chunks, axis=0).astype(np.float32)

    # Privacy Safeguard: Audio is NEVER written to disk unless explicitly opted-in via SAVE_DEBUG_AUDIO
    if getattr(config, "SAVE_DEBUG_AUDIO", False):
        try:
            from pathlib import Path
            debug_dir = getattr(config, "BASE_DIR", Path(".")).resolve() / "logs" / "debug_audio"
            debug_dir.mkdir(parents=True, exist_ok=True)
            save_wav(str(debug_dir / "last_recording.wav"), audio_array, sample_rate=sample_rate)
        except Exception:
            pass

    return audio_array


def save_wav(filename: str, audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Save a float32 numpy audio array as a standard 16-bit PCM WAV file.

    Args:
        filename: Target path/filename (e.g., 'test.wav').
        audio_data: 1D float32 numpy array with values between -1.0 and 1.0.
        sample_rate: Audio sample rate in Hz.
    """
    # Clip float values to [-1.0, 1.0] to prevent audio distortion
    clipped = np.clip(audio_data, -1.0, 1.0)
    # Convert float32 [-1.0, 1.0] to 16-bit signed integers [-32768, 32767]
    pcm_data = (clipped * 32767.0).astype(np.int16).tobytes()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes = 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


# Cached Whisper model instance for lazy loading
_whisper_model = None


def get_whisper_model():
    """Load the Whisper model lazily on first use and cache it for subsequent calls.

    Loads the model size specified in config.WHISPER_MODEL_SIZE (default: 'base')
    running on CPU with int8 quantization.
    """
    global _whisper_model
    if _whisper_model is None:
        model_size = getattr(config, "WHISPER_MODEL_SIZE", "base")
        print(f"\n[Whisper] Loading '{model_size}' speech recognition model...")
        print("[Whisper] If this is your first time, the model will download automatically (may take a minute or two)...")
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"[Whisper] '{model_size}' model loaded and ready.\n")
    return _whisper_model


def preload_whisper_model() -> None:
    """Preload the Whisper speech recognition model into memory in the background.

    Called at startup so that the first voice query does not experience initialization lag.
    """
    try:
        get_whisper_model()
    except Exception as err:
        print(f"[Whisper Warning] Background preload failed: {err}", file=sys.stderr)



def transcribe(audio: np.ndarray | None) -> str:
    """Transcribe a numpy audio array into text using faster-whisper.

    Args:
        audio: 1D float32 numpy array with 16kHz audio samples from record_until_silence().

    Returns:
        The transcribed text stripped of extra whitespace, or an empty string
        if audio is None or nothing intelligible was heard.
    """
    if audio is None or len(audio) == 0:
        return ""

    try:
        model = get_whisper_model()
        language = getattr(config, "WHISPER_LANGUAGE", "en")
        segments, _ = model.transcribe(
            audio,
            language=language,
            vad_filter=True,
        )
        # Combine all transcribed segments into a single string
        transcribed_text = " ".join(seg.text.strip() for seg in segments).strip()
        return transcribed_text
    except Exception as err:
        print(f"[Whisper Error] Transcription failed: {err}", file=sys.stderr)
        return ""


def listen(initial_timeout_sec: float = INITIAL_SILENCE_TIMEOUT) -> str:
    """Record speech from the microphone and transcribe it into text.

    Combines audio recording and speech-to-text recognition:
    1. Prints 'Listening...'
    2. Records audio using record_until_silence()
    3. Prints 'Transcribing...' while processing
    4. Returns the transcribed text, or empty string if no speech was heard.

    Args:
        initial_timeout_sec: Maximum seconds to wait for initial speech before giving up.

    Returns:
        Recognized text as a string.
    """
    print("Listening...")
    audio = record_until_silence(initial_timeout_sec=initial_timeout_sec)

    if audio is None or len(audio) == 0:
        return ""

    print("Transcribing...")
    return transcribe(audio)
