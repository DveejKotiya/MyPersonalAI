"""Wake word detection module for Jarvis (Phase 4).

This module provides real-time, low-CPU wake word detection using openwakeword.
It detects the pretrained 'hey_jarvis' model from a continuous 16 kHz mono
audio stream in a background daemon thread.

Key capabilities:
- Non-blocking background thread with start() and stop()
- pause() and resume() with complete microphone stream release so audio_input.py
  and speech synthesis can record/play without PortAudio device contention
- Configurable WAKE_THRESHOLD and 2.0s cooldown to prevent double triggers
- Automatic, friendly one-time download handler for openwakeword models
"""

import inspect
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import sounddevice as sd

import config

# Standard sample rate and frame duration for openwakeword
SAMPLE_RATE = 16000          # 16 kHz mono
FRAME_DURATION_SEC = 0.08    # 80 ms per inference frame (1280 samples)
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION_SEC)
DEFAULT_COOLDOWN_SEC = 2.0   # 2-second cooldown after a trigger


def ensure_model_downloaded(model_name: str = "hey_jarvis") -> None:
    """Ensure that the required openwakeword model weights are downloaded.

    If the requested model or required neural network feature extractors
    (melspectrogram and audio embedding) are not yet downloaded, downloads
    them with a friendly one-time notification banner.

    Args:
        model_name: The name of the wake word model to verify (default: "hey_jarvis").
    """
    import openwakeword
    import openwakeword.utils

    try:
        model_paths = openwakeword.get_pretrained_model_paths(inference_framework="onnx")
    except Exception:
        model_paths = []

    # Check if the specific model path exists on disk
    matched = [p for p in model_paths if model_name in p]
    if not matched:
        border = "=" * 70
        print(f"\n{border}")
        print(f"  [INFO] Setting up Wake Word Detection for {config.ASSISTANT_NAME}...")
        print(f"  Downloading '{model_name}' pretrained model and neural network weights.")
        print("  This is a one-time setup and may take a moment depending on your connection.")
        print(f"{border}\n")

        try:
            openwakeword.utils.download_models(model_names=[model_name])
            print("\n  [SUCCESS] Wake word model downloaded and ready!\n")
        except Exception as err:
            # Fallback to general download if targeted download raises an error
            print(f"  [Notice] Retrying download for base feature models ({err})...")
            openwakeword.utils.download_models()
            print("\n  [SUCCESS] Wake word models ready!\n")


class WakeWordListener:
    """Detects 'Hey Jarvis' wake word in a background thread."""

    def __init__(
        self,
        on_wake: Callable[..., Any] | None = None,
        threshold: float | None = None,
        model_name: str | None = None,
        cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
        device: int | str | None = None,
    ):
        """Initialize the WakeWordListener.

        Args:
            on_wake: Callback function triggered upon wake word detection.
                     Can accept (score: float) or take no arguments.
            threshold: Confidence threshold (0.0 to 1.0). If None, reads config.WAKE_THRESHOLD.
            model_name: Model identifier (default: "hey_jarvis").
            cooldown_sec: Minimum seconds between triggers (default: 2.0s).
            device: Optional sounddevice microphone device index or name.
        """
        self.on_wake = on_wake
        self.threshold = threshold if threshold is not None else getattr(config, "WAKE_THRESHOLD", 0.5)
        self.model_name = model_name if model_name is not None else getattr(config, "WAKE_WORD_MODEL", "hey_jarvis")
        self.cooldown_sec = cooldown_sec
        self.device = device

        self._running = False
        self._paused = False
        self._thread: threading.Thread | None = None
        self._stream: sd.InputStream | None = None
        self._stream_lock = threading.Lock()
        self._stream_released_event = threading.Event()
        self._stream_released_event.set()
        self._last_trigger_time = 0.0

        # Ensure model weights exist, then load model
        ensure_model_downloaded(self.model_name)
        self.model = self._load_model()

    def _load_model(self) -> Any:
        """Instantiate the openwakeword ONNX model."""
        from openwakeword.model import Model

        try:
            return Model(wakeword_models=[self.model_name], inference_framework="onnx")
        except Exception as err:
            # If loading failed because weights were missing, download and retry
            print(f"[WakeWord] Model initialization notice: {err}. Ensuring weights are present...")
            ensure_model_downloaded(self.model_name)
            return Model(wakeword_models=[self.model_name], inference_framework="onnx")

    def _release_stream(self) -> None:
        """Safely close and release the microphone input stream."""
        with self._stream_lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                except Exception:
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
                finally:
                    self._stream = None
                    self._stream_released_event.set()

    def _trigger_wake(self, score: float) -> None:
        """Invoke the on_wake callback safely."""
        if not self.on_wake:
            return

        try:
            sig = inspect.signature(self.on_wake)
            if len(sig.parameters) >= 1:
                self.on_wake(score)
            else:
                self.on_wake()
        except TypeError:
            try:
                self.on_wake()
            except Exception as cb_err:
                print(f"[WakeWord Callback Error] {cb_err}", file=sys.stderr)
        except Exception as cb_err:
            print(f"[WakeWord Callback Error] {cb_err}", file=sys.stderr)

    def _run_loop(self) -> None:
        """Main detection loop running in a background daemon thread."""
        while self._running:
            # When paused, ensure microphone stream is closed and sleep briefly
            if self._paused:
                self._release_stream()
                time.sleep(0.05)
                continue

            # Ensure the stream is open when listening
            with self._stream_lock:
                if self._stream is None and self._running and not self._paused:
                    try:
                        self._stream = sd.InputStream(
                            samplerate=SAMPLE_RATE,
                            channels=1,
                            dtype="int16",
                            blocksize=FRAME_SAMPLES,
                            device=self.device,
                        )
                        self._stream.start()
                        self._stream_released_event.clear()
                    except Exception:
                        # Microphone may be momentarily busy or disconnected
                        time.sleep(0.2)
                        continue

            if self._stream is None:
                time.sleep(0.05)
                continue

            # Read an 80 ms audio frame from the microphone
            try:
                chunk, overflowed = self._stream.read(FRAME_SAMPLES)
            except Exception:
                # Device unplugged or stream invalidated -> release stream handle and wait for reconnect
                self._release_stream()
                time.sleep(0.5)
                continue

            # Check flags again in case pause() or stop() occurred while reading
            if self._paused or not self._running:
                continue

            # Flatten to 1D int16 array for openwakeword
            chunk_flat = chunk.flatten()

            try:
                prediction = self.model.predict(chunk_flat)
            except Exception:
                continue

            # Locate the model's score in the prediction dictionary
            score = 0.0
            for key, val in prediction.items():
                if self.model_name in key or key in self.model_name:
                    score = float(val)
                    break
            else:
                if prediction:
                    score = float(max(prediction.values()))

            now = time.time()
            if score >= self.threshold and (now - self._last_trigger_time) >= self.cooldown_sec:
                self._last_trigger_time = now
                self._trigger_wake(score)

        # Loop ended; ensure audio stream is closed
        self._release_stream()

    def start(self) -> None:
        """Start wake word detection in a background daemon thread."""
        if self._running:
            return

        self._running = True
        self._paused = False
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="WakeWordListenerThread",
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop wake word detection, release the microphone, and join the thread."""
        self._running = False
        self._paused = True
        self._release_stream()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    def pause(self, timeout: float = 1.0) -> None:
        """Temporarily pause detection and fully release the microphone stream.

        Crucial for preventing audio conflicts while Jarvis is recording user
        speech or speaking output through the speaker.
        """
        self._paused = True
        self._release_stream()
        # Wait up to timeout for the stream to be completely released
        self._stream_released_event.wait(timeout=timeout)

        # Clear openwakeword feature buffer to prevent stale triggers on resume
        try:
            self.model.reset()
        except Exception:
            pass

    def resume(self) -> None:
        """Resume listening for the wake word and reopen the microphone stream."""
        try:
            self.model.reset()
        except Exception:
            pass
        self._paused = False

    def set_threshold(self, threshold: float) -> None:
        """Update detection sensitivity threshold dynamically (e.g. higher threshold while speaking)."""
        self.threshold = threshold

    def reset_cooldown(self) -> None:
        """Reset the trigger cooldown timer so the next detection triggers immediately."""
        self._last_trigger_time = 0.0

    @property
    def is_listening(self) -> bool:
        """True if the listener thread is active and not paused."""
        return self._running and not self._paused
