"""Microphone Test Script for Jarvis (Phase 2).

This script verifies your audio configuration:
1. Lists all available audio input devices.
2. Displays a 5-second live ASCII volume meter to verify mic activity.
3. Records one phrase using record_until_silence() and saves it as 'test.wav'.
4. Displays the recording duration.
"""

import sys
import time

import sounddevice as sd

from audio_input import (
    CHANNELS,
    CHUNK_DURATION,
    DEFAULT_DEVICE,
    SAMPLE_RATE,
    SILENCE_THRESHOLD,
    calculate_rms,
    record_until_silence,
    save_wav,
)


def list_input_devices() -> None:
    """List all audio input devices detected by the operating system."""
    print("=" * 60)
    print("  Detected Audio Input Devices")
    print("=" * 60)
    devices = sd.query_devices()
    default_input = sd.default.device[0]

    input_count = 0
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            is_default = " [DEFAULT]" if idx == default_input else ""
            api_name = sd.query_hostapis(dev["hostapi"])["name"]
            print(f" [{idx:2d}] {dev['name']} ({api_name}){is_default}")
            input_count += 1

    if input_count == 0:
        print(" [WARNING] No audio input devices detected! Please check your microphone connection.")
    print("=" * 60 + "\n")


def live_volume_meter(duration_sec: float = 5.0, device: int | str | None = DEFAULT_DEVICE) -> None:
    """Display a live ASCII volume meter for a given number of seconds."""
    print(f"Starting live volume meter for {int(duration_sec)} seconds...")
    print(f"Speak into your microphone now (Threshold = {SILENCE_THRESHOLD}):\n")

    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
    start_time = time.time()
    max_rms_seen = 0.0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=device,
            blocksize=chunk_samples,
        ) as stream:
            while time.time() - start_time < duration_sec:
                chunk, _ = stream.read(chunk_samples)
                rms = calculate_rms(chunk.flatten())
                max_rms_seen = max(max_rms_seen, rms)

                # Render 25-character visual volume bar
                # Full scale represents RMS of 0.10
                bar_len = int(min(rms / 0.10, 1.0) * 25)
                bar = "█" * bar_len + "░" * (25 - bar_len)

                status = "(Speaking)" if rms >= SILENCE_THRESHOLD else "(Silence) "
                elapsed = duration_sec - (time.time() - start_time)

                print(
                    f"\r[{bar}] RMS: {rms:.4f} {status} | Time left: {max(0.0, elapsed):.1f}s",
                    end="",
                    flush=True,
                )
                time.sleep(0.02)

        print("\n\nVolume meter completed.")
        print(f"Peak volume observed: {max_rms_seen:.4f} (Silence threshold: {SILENCE_THRESHOLD})")
        if max_rms_seen < SILENCE_THRESHOLD:
            print("[TIP] Peak volume was below threshold. If you spoke, try speaking closer or lowering SILENCE_THRESHOLD in audio_input.py.\n")
        else:
            print("[OK] Microphone signal detected successfully!\n")

    except Exception as err:
        print(f"\n[Error] Unable to access audio stream: {err}", file=sys.stderr)


def test_phrase_recording() -> None:
    """Test recording a phrase with record_until_silence() and save to test.wav."""
    print("=" * 60)
    print("  Testing Phrase Recording")
    print("=" * 60)
    print("Jarvis is ready to listen:")
    print("-> Speak a short phrase (e.g. 'Hello Jarvis, this is a test').")
    print("-> Once you stop speaking, it will automatically finish after 1.5 seconds.")
    print("-> (If you stay silent for 8 seconds, it will give up gracefully).\n")

    print("[Listening...]")
    audio = record_until_silence()

    if audio is None:
        print("\n[Result] No speech detected (timed out after 8 seconds of silence).")
        print("Tip: If you spoke, check your microphone volume or lower SILENCE_THRESHOLD in audio_input.py.")
        return

    duration = len(audio) / SAMPLE_RATE
    output_file = "test.wav"
    save_wav(output_file, audio, SAMPLE_RATE)

    print("\n[Result] Recording complete!")
    print(f"- Duration: {duration:.2f} seconds ({len(audio)} audio samples)")
    print(f"- Saved to: {output_file}")
    print(f"You can now play '{output_file}' in your media player to verify audio quality.\n")


def main() -> None:
    """Run all microphone diagnostic steps."""
    print("\n--- Jarvis Phase 2: Microphone Diagnostic Tool ---\n")
    list_input_devices()
    live_volume_meter(5.0)
    test_phrase_recording()


if __name__ == "__main__":
    main()
