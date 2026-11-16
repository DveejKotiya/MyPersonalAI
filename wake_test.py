"""Wake word test script for Jarvis (Phase 4).

Listens for the wake word 'Hey Jarvis' for 60 seconds, prints each detection
with its confidence score, and provides a clear guide on tuning sensitivity.
"""

import sys
import time

import config
from wake_word import WakeWordListener


def print_banner(threshold: float) -> None:
    """Print the test startup banner."""
    print("=" * 66)
    print("  Jarvis Wake Word Detection Test (60 Seconds)")
    print("=" * 66)
    print("Listening for wake word: 'Hey Jarvis'")
    print(f"Current threshold:       {threshold:.2f} (from config.py / .env)")
    print("Duration:                60 seconds (Press Ctrl+C to stop anytime)")
    print("=" * 66)
    print("-> Speak naturally: say 'Hey Jarvis' into your microphone...\n")


def print_tuning_guide(detection_count: int, threshold: float) -> None:
    """Print results summary and practical guidance on sensitivity tuning."""
    print("\n" + "=" * 66)
    print("  Test Completed - Wake Word Summary")
    print("=" * 66)
    print(f"  Total Detections: {detection_count}")
    print(f"  Tested Threshold: {threshold:.2f}")
    print("=" * 66)
    print("\n[Tuning Guide: How to Adjust Sensitivity in your '.env' file]:\n")
    print("  1. Too Many False Triggers? (Waking up to TV, chatter, or noise):")
    print("     -> Increase WAKE_THRESHOLD to make detection stricter.")
    print("        Example: WAKE_THRESHOLD=0.65 or WAKE_THRESHOLD=0.75\n")
    print("  2. Missed Detections? (You say 'Hey Jarvis' clearly, but nothing happens):")
    print("     -> Lower WAKE_THRESHOLD to make detection more sensitive.")
    print("        Example: WAKE_THRESHOLD=0.40 or WAKE_THRESHOLD=0.35\n")
    print("  3. Check Microphone Input Volume:")
    print("     -> If scores remain low (< 0.3) even when speaking closely, check")
    print("        Windows Sound Settings and ensure your mic input volume is 80-100%.\n")
    print("  4. Edit '.env' to save your preferred threshold:")
    print("        WAKE_THRESHOLD=0.50\n")
    print("=" * 66)


def main() -> None:
    """Run the 60-second wake word test."""
    threshold = getattr(config, "WAKE_THRESHOLD", 0.5)
    detections: list[float] = []

    def handle_wake(score: float = 0.0) -> None:
        detections.append(score)
        print(f"\n[TRIGGER] Wake word detected! (Confidence: {score:.3f} | Threshold: {threshold:.2f})\n")

    print_banner(threshold)

    listener = WakeWordListener(on_wake=handle_wake, threshold=threshold)
    listener.start()

    start_time = time.time()
    duration_sec = 60

    try:
        while True:
            elapsed = time.time() - start_time
            remaining = int(duration_sec - elapsed)
            if remaining <= 0:
                break

            # Print a single-line progress update every second
            sys.stdout.write(f"\r  [Listening... {remaining:02d}s remaining | Triggers: {len(detections)}]   ")
            sys.stdout.flush()
            time.sleep(1)

        print("\r  [Listening complete! 00s remaining]                    ")

    except KeyboardInterrupt:
        print("\n\n[Stopped] Test interrupted by user.")
    finally:
        listener.stop()

    print_tuning_guide(len(detections), threshold)


if __name__ == "__main__":
    main()
