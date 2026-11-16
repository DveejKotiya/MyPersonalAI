"""Jarvis - Personal AI Assistant.

Modes supported:
1. Hands-Free Mode (--listen, Phase 4): Automatic wake word detection ("Hey Jarvis")
   with explicit state machine (IDLE -> LISTENING -> THINKING -> SPEAKING -> IDLE),
   global hotkey (Ctrl+Alt+J), in-code beeps, and follow-up conversational window.
2. System Tray Integration (--tray, Phase 4): Runs assistant in background thread
   and tray icon in main thread with dynamic state colors and context menu.
3. Auto-Start Management (--enable-autostart / --disable-autostart): Configures
   Jarvis to launch automatically at login with system tray and no console window.
4. Push-to-Talk Voice Mode (--voice, Phase 2): Manual Enter-to-talk loop.
5. Text Mode (default, Phase 1): Interactive terminal prompt.
"""

import argparse
import json
import logging
import sys
import threading
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd

import autostart
import config
from audio_input import listen, record_until_silence, transcribe
from brain import Brain
from tools import set_confirmation_mode
from tray import JarvisTray, create_hotkey_listener
from tts import speak, stop_speaking
from wake_word import WakeWordListener

# ==============================================================================
# Logging & Console Redirection (Requirement 5)
# ==============================================================================

class LogStream:
    """Redirects stdout / stderr to logs/jarvis.log while preserving console output when attached."""

    def __init__(self, file_path: Path, original_stream: Any = None):
        self.file_path = file_path
        self.original_stream = original_stream
        self._lock = threading.Lock()

    def write(self, message: str) -> None:
        if not message:
            return
        with self._lock:
            # Mirror to terminal if present
            if self.original_stream and hasattr(self.original_stream, "write"):
                try:
                    self.original_stream.write(message)
                    if hasattr(self.original_stream, "flush"):
                        self.original_stream.flush()
                except Exception:
                    pass

            # Write to log file
            try:
                with open(self.file_path, "a", encoding="utf-8") as f:
                    f.write(message)
            except Exception:
                pass

    def flush(self) -> None:
        if self.original_stream and hasattr(self.original_stream, "flush"):
            try:
                self.original_stream.flush()
            except Exception:
                pass


def setup_jarvis_logging() -> None:
    """Configure persistent logging to logs/jarvis.log and install unhandled exception hooks."""
    base_dir = getattr(config, "BASE_DIR", Path(__file__).resolve().parent)
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "jarvis.log"

    # Configure root logger
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8",
    )

    # Redirect sys.stdout and sys.stderr so print() and tracebacks always append to jarvis.log
    sys.stdout = LogStream(log_file, original_stream=sys.stdout)
    sys.stderr = LogStream(log_file, original_stream=sys.stderr)

    # Global unhandled exception handlers
    def _uncaught_sys_exception(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logging.error("Uncaught exception in main thread:", exc_info=(exc_type, exc_value, exc_tb))

    def _uncaught_thread_exception(args):
        if issubclass(args.exc_type, (KeyboardInterrupt, SystemExit)):
            return
        thread_name = getattr(args.thread, "name", "unknown")
        logging.error(
            f"Uncaught exception in background thread '{thread_name}':",
            exc_info=(args.exc_type, args.exc_value, args.exc_tb),
        )

    sys.excepthook = _uncaught_sys_exception
    threading.excepthook = _uncaught_thread_exception


# ==============================================================================
# Audio & Network Readiness Probe (Requirement 4)
# ==============================================================================

def wait_for_system_readiness(
    max_retries: int = 15,
    retry_interval_sec: float = 2.0,
    initial_delay_sec: float = 3.0,
) -> bool:
    """Verify audio input devices and network services are ready before starting.

    When started at system login via autostart:
    1. Pauses briefly if launched via pythonw.exe to allow OS boot services to settle.
    2. Repeatedly probes sounddevice for at least one active microphone device.
    3. Retries in a loop if the audio driver is not ready yet.
    4. Probes local Ollama connectivity.
    5. Writes diagnostic progress to logs/jarvis.log.

    Returns:
        True if an audio input device is ready, False if timed out after retries.
    """
    is_silent_login = sys.executable.lower().endswith("pythonw.exe")
    if is_silent_login:
        logging.info(
            f"[Readiness] Detected silent login launch (pythonw.exe). "
            f"Waiting {initial_delay_sec:.1f}s for OS audio services to stabilize..."
        )
        time.sleep(initial_delay_sec)

    logging.info("[Readiness] Checking audio input devices and network services...")

    mic_ready = False
    for attempt in range(1, max_retries + 1):
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
            if input_devices:
                default_idx = sd.default.device[0]
                dev_name = "Default"
                if default_idx is not None and 0 <= default_idx < len(devices):
                    dev_name = devices[default_idx].get("name", "Default")
                logging.info(
                    f"[Readiness] Microphone ready: '{dev_name}' "
                    f"({len(input_devices)} input device(s) found)."
                )
                print(f"[Startup] Microphone device ready: '{dev_name}'.")
                mic_ready = True
                break
        except Exception as audio_err:
            logging.warning(f"[Readiness] Audio probe error (attempt {attempt}/{max_retries}): {audio_err}")

        logging.info(
            f"[Readiness] Audio devices initializing... (attempt {attempt}/{max_retries}, "
            f"retrying in {retry_interval_sec:.1f}s)"
        )
        print(f"[Startup] Waiting for audio devices to initialize... (attempt {attempt}/{max_retries})")
        time.sleep(retry_interval_sec)

    if not mic_ready:
        logging.error("[Readiness] No active microphone input device detected after all retry attempts.")
        print("[Startup Warning] Could not detect an active microphone input device.", file=sys.stderr)

    # Probe local Ollama service
    try:
        import urllib.request
        host = getattr(config, "OLLAMA_HOST", "http://localhost:11434")
        req = urllib.request.Request(f"{host}/api/tags", headers={"User-Agent": "JarvisAssistant"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                logging.info(f"[Readiness] Ollama server is reachable at {host}.")
    except Exception as net_err:
        logging.info(f"[Readiness] Ollama server notice: {net_err} (assistant will connect when request arrives).")

    return mic_ready


# ==============================================================================
# In-Memory Tone Generation (Beeps)
# ==============================================================================

def play_beep(freq: int, duration: float = 0.10, volume: float = 0.25) -> None:
    """Generate a clean audio tone in-memory with smooth envelope (no audio files needed).

    Args:
        freq: Frequency in Hertz (e.g. 880 for wake, 440 for idle).
        duration: Tone duration in seconds.
        volume: Volume scaling between 0.0 and 1.0.
    """
    try:
        sample_rate = 16000
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)
        tone = np.sin(2 * np.pi * freq * t) * volume

        # Apply gentle attack and decay envelope to prevent click/pop sounds
        fade_len = min(int(sample_rate * 0.015), n_samples // 4)
        if fade_len > 0:
            fade = np.linspace(0, 1, fade_len)
            tone[:fade_len] *= fade
            tone[-fade_len:] *= fade[::-1]

        sd.play(tone.astype(np.float32), samplerate=sample_rate)
        sd.wait()
    except Exception:
        # Fallback for Windows platform if sounddevice output encounters an issue
        try:
            import winsound
            winsound.Beep(int(freq), int(duration * 1000))
        except Exception:
            pass


def play_wake_beep() -> None:
    """High-pitch short beep (880 Hz) played immediately when wake word is detected."""
    play_beep(freq=880, duration=0.10, volume=0.25)


def play_idle_beep() -> None:
    """Lower-pitch calm beep (440 Hz) played when returning to IDLE state."""
    play_beep(freq=440, duration=0.12, volume=0.20)


# ==============================================================================
# State Machine for Hands-Free Mode (Phase 4)
# ==============================================================================

class AssistantState(Enum):
    """Explicit operational states for hands-free Jarvis."""
    IDLE = "IDLE"           # Waiting for wake word ('Hey Jarvis')
    LISTENING = "LISTENING" # Recording user speech
    THINKING = "THINKING"   # Transcribing and generating reply via Brain
    SPEAKING = "SPEAKING"   # Spoken voice output via TTS


class StateWatchdog:
    """Monitors the assistant state machine and recovers if stuck in non-IDLE state (> 60s)."""

    def __init__(self, assistant: "JarvisAssistant", timeout_sec: float = 60.0):
        self.assistant = assistant
        self.timeout_sec = timeout_sec
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the watchdog monitor in a background daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="StateWatchdogThread",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the watchdog monitor."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        """Periodic background loop verifying state machine freshness."""
        while self._running:
            time.sleep(1.0)
            if not self._running:
                break

            current_state = self.assistant.state
            # Only monitor active states: LISTENING, THINKING, SPEAKING
            if current_state != AssistantState.IDLE and not self.assistant.is_manually_paused:
                elapsed = time.time() - self.assistant.state_entered_time
                if elapsed > self.timeout_sec:
                    logging.warning(
                        f"[Watchdog Alert] State machine stuck in {current_state.value} "
                        f"for {elapsed:.1f}s (> {self.timeout_sec}s)! Forcing recovery to IDLE."
                    )
                    print(
                        f"\n[Watchdog Alert] Assistant stuck in {current_state.value} for {elapsed:.1f}s. "
                        f"Forcing reset to IDLE...\n",
                        file=sys.stderr,
                    )
                    self.assistant.force_reset_to_idle()


class JarvisAssistant:
    """Hands-free assistant controller managing the Phase 4 state machine.

    Flow:
      IDLE      -> waiting for wake word (mic owned by WakeWordListener)
      LISTENING -> wake word or hotkey heard: play high beep, record request with listen()
      THINKING  -> transcribing and calling Brain.think() (WakeWordListener paused)
      SPEAKING  -> speaking reply via TTS (WakeWordListener paused)
      Follow-up -> 6-second window without needing wake word
      then back to IDLE (play low beep, WakeWordListener resumed)
    """

    def __init__(
        self,
        brain: Brain,
        assistant_name: str,
        enable_speak: bool = True,
        debug: bool = False,
    ):
        self.state = AssistantState.IDLE
        self.state_entered_time = time.time()
        self.brain = brain
        self.assistant_name = assistant_name
        self.enable_speak = enable_speak
        self.debug = debug

        self.tray: JarvisTray | None = None
        self.is_manually_paused = False
        self._manual_wake = False
        self._running = False

        self._wake_event = threading.Event()
        self._interrupted_event = threading.Event()
        self._last_wake_score = 0.0

        # Background wake word listener
        self.listener = WakeWordListener(
            on_wake=self._on_wake_detected,
            threshold=getattr(config, "WAKE_THRESHOLD", 0.5),
        )

        # Stuck state watchdog (from config.py)
        watchdog_sec = getattr(config, "WATCHDOG_TIMEOUT_SEC", 60.0)
        self.watchdog = StateWatchdog(assistant=self, timeout_sec=watchdog_sec)

    def set_tray(self, tray: JarvisTray) -> None:
        """Attach a system tray icon manager to this assistant."""
        self.tray = tray

    def set_state(self, new_state: AssistantState) -> None:
        """Set assistant state, update timestamp, update tray icon, and print transitions."""
        old_state = self.state
        self.state = new_state
        self.state_entered_time = time.time()

        if self.tray and not self.is_manually_paused:
            self.tray.update_state(new_state.value, is_paused=False)

        if self.debug and old_state != new_state:
            print(f"[State] {old_state.value} -> {new_state.value}")

    def force_reset_to_idle(self) -> None:
        """Forcefully recover assistant state machine to IDLE when stuck."""
        logging.warning(f"[Watchdog] Forcing emergency reset of state {self.state.value} to IDLE.")

        # 1. Stop any ongoing speech or playback
        stop_speaking()

        # 2. Reset wake/interrupt events
        self._interrupted_event.set()
        self._wake_event.clear()
        self._manual_wake = False

        # 3. Transition to IDLE
        self.set_state(AssistantState.IDLE)

        # 4. Resume wake word listener if not manually paused
        if not self.is_manually_paused and hasattr(self, "listener"):
            try:
                self.listener.resume()
            except Exception as resume_err:
                logging.error(f"[Watchdog] Error resuming listener: {resume_err}")

        # 5. Play idle chime and update tray
        play_idle_beep()
        if self.tray:
            self.tray.update_state("IDLE", is_paused=self.is_manually_paused)
            self.tray.notify(
                f"{self.assistant_name} Recovered",
                "Watchdog detected a stuck state and reset to IDLE.",
            )

    def pause_listening(self) -> None:
        """Pause listening and fully release the microphone stream for other apps."""
        if self.is_manually_paused:
            return

        self.is_manually_paused = True
        if self.state == AssistantState.SPEAKING:
            self.interrupt()

        self.listener.pause()

        if self.tray:
            self.tray.update_state(self.state.value, is_paused=True)
            self.tray.notify(
                f"{self.assistant_name} Paused",
                "Microphone released for Zoom, Teams, and other apps.",
            )

        print(f"\n[{self.assistant_name}] Paused. Microphone released for other apps.")

    def resume_listening(self) -> None:
        """Resume listening and re-acquire microphone for wake word detection."""
        if not self.is_manually_paused:
            return

        self.is_manually_paused = False
        self.listener.resume()

        if self.tray:
            self.tray.update_state(self.state.value, is_paused=False)
            self.tray.notify(
                f"{self.assistant_name} Resumed",
                "Microphone active. Listening for 'Hey Jarvis'.",
            )

        print(f"\n[{self.assistant_name}] Resumed. Listening for 'Hey Jarvis'...")

    def wake_manually(self) -> None:
        """Trigger a manual wake into the LISTENING state via hotkey or tray action."""
        if self.is_manually_paused:
            self.resume_listening()

        if self.state == AssistantState.IDLE:
            self._manual_wake = True
            self._wake_event.set()

    def on_hotkey(self) -> None:
        """Global hotkey handler (Ctrl+Alt+J):

        - If idle: acts as manual wake (transitions to LISTENING).
        - If speaking: interrupts Jarvis immediately.
        """
        if self.state == AssistantState.SPEAKING:
            print(f"\n[{self.assistant_name}] Hotkey pressed during speech -> interrupting speech...")
            self.interrupt()
        elif self.state == AssistantState.IDLE or self.is_manually_paused:
            print(f"\n[{self.assistant_name}] Hotkey manual wake triggered!")
            self.wake_manually()
        else:
            if self.debug:
                print(f"[{self.assistant_name}] Hotkey pressed during {self.state.value} (ignored).")

    def _on_wake_detected(self, score: float = 0.0) -> None:
        """Callback from WakeWordListener background thread when wake word is detected."""
        if self.is_manually_paused:
            return

        if self.state == AssistantState.IDLE:
            self._last_wake_score = score
            self._wake_event.set()
        elif self.state == AssistantState.SPEAKING:
            self._last_wake_score = score
            if self.debug:
                print(f"\n[Interrupt] 'Hey Jarvis' detected during speech! (Confidence: {score:.3f})")
            self.interrupt()

    def interrupt(self) -> None:
        """Halt ongoing speech immediately and transition to LISTENING."""
        if self.state == AssistantState.SPEAKING:
            if self.debug:
                print("[Interrupt] Halting speech playback immediately...")
            self._interrupted_event.set()
            stop_speaking()

    def trigger_hotkey_interrupt(self) -> None:
        """Hotkey callback hook to interrupt speech immediately."""
        self.interrupt()

    def stop(self) -> None:
        """Stop the assistant loop and clean up all listeners, watchdogs, and threads."""
        self._running = False
        self._wake_event.set()
        self._interrupted_event.set()
        stop_speaking()

        if hasattr(self, "watchdog"):
            self.watchdog.stop()

        if hasattr(self, "listener"):
            self.listener.stop()

        if self.tray:
            self.tray.stop()

    def run(self) -> None:
        """Main loop for hands-free mode."""
        # Set security confirmation mode to voice (or text if speech is disabled)
        set_confirmation_mode("voice" if self.enable_speak else "text")

        status_label = "Spoken audio enabled" if self.enable_speak else "Quiet mode (--no-speak)"
        print("=" * 64)
        print(f"  {self.assistant_name} - Hands-Free Voice Mode (--listen)")
        print("=" * 64)
        print(f"{self.assistant_name} is online ({status_label}).")
        print("Say 'Hey Jarvis' or press the hotkey to activate.")
        print("Say 'goodbye jarvis' to exit.")
        print(f"Wake word threshold: {self.listener.threshold:.2f} (from config.py / .env)\n")

        self._running = True
        self.watchdog.start()
        self.listener.start()

        # Preload Whisper speech recognition model in background to avoid first-call lag
        from audio_input import preload_whisper_model
        threading.Thread(target=preload_whisper_model, daemon=True, name="whisper-preload").start()


        try:
            while self._running:
                # --------------------------------------------------------------
                # STATE: IDLE
                # --------------------------------------------------------------
                self.set_state(AssistantState.IDLE)
                self._wake_event.clear()
                self._interrupted_event.clear()

                if not self.is_manually_paused:
                    self.listener.set_threshold(getattr(config, "WAKE_THRESHOLD", 0.5))
                    self.listener.resume()

                # Sleep in short increments until wake word or manual wake is triggered
                while self._running and not self._wake_event.is_set():
                    if self.is_manually_paused:
                        time.sleep(0.1)
                        continue
                    time.sleep(0.05)

                if not self._running:
                    break

                self._wake_event.clear()

                # Wake event occurred: fully pause listener and release microphone stream
                self.listener.pause()

                # Process interaction cycle
                self._handle_active_cycle(is_initial_wake=True)

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{self.assistant_name}: Shutting down hands-free listener...")
        finally:
            self._running = False
            self.watchdog.stop()
            self.listener.stop()
            self.set_state(AssistantState.IDLE)

    def _handle_active_cycle(self, is_initial_wake: bool = True) -> None:
        """Execute the active conversational cycle with error trapping."""
        if self.is_manually_paused:
            return

        try:
            # ------------------------------------------------------------------
            # STATE: LISTENING
            # ------------------------------------------------------------------
            self.set_state(AssistantState.LISTENING)

            if self._manual_wake:
                self._manual_wake = False
                play_wake_beep()
                if self.debug:
                    print("\n[Wake] Manual hotkey wake activated!")
                print(f"\n{self.assistant_name}: Listening...")
                user_text = listen(initial_timeout_sec=8.0)
            elif is_initial_wake:
                play_wake_beep()
                if self.debug:
                    print(f"\n[Wake] 'Hey Jarvis' detected! (Confidence: {self._last_wake_score:.3f})")
                print(f"\n{self.assistant_name}: Listening...")
                # 8-second timeout for first request after wake word
                user_text = listen(initial_timeout_sec=8.0)
            else:
                user_text = listen(initial_timeout_sec=6.0)

            # Silence or unintelligible audio: give up and return to IDLE
            if not user_text or not user_text.strip():
                if self.state != AssistantState.IDLE:
                    if is_initial_wake:
                        print(f"{self.assistant_name}: (No speech detected within 8s. Going back to sleep.)\n")
                    else:
                        print(f"{self.assistant_name}: (No follow-up speech detected. Returning to idle.)\n")
                    play_idle_beep()
                return

            if self.state == AssistantState.IDLE:
                return

            print(f"You: {user_text}")
            command = user_text.lower().strip(" .!?,")

            # Voice command: Shutdown ("goodbye jarvis", "exit", "quit")
            if command in ("goodbye jarvis", "goodbye, jarvis", "exit", "quit", "goodbye"):
                farewell = "Goodbye! Have a great day."
                print(f"\n{self.assistant_name}: {farewell}")
                if self.enable_speak:
                    speak(farewell)
                self.stop()
                return

            # Voice command: Sleep / Stop listening ("stop listening", "go to sleep")
            if command in ("stop listening", "go to sleep", "sleep"):
                sleep_msg = "Going to sleep. Call me if you need me."
                print(f"{self.assistant_name}: {sleep_msg}\n")
                if self.enable_speak:
                    speak(sleep_msg)
                play_idle_beep()
                return

            # Voice command: Built-in memory commands ("what do you remember about me", "forget everything")
            if handle_memory_command(
                user_text,
                self.brain,
                self.assistant_name,
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                self._enter_follow_up_window()
                return

            # Voice command: Daily briefing triggers ("good morning", "give me my briefing", "what's my day look like")
            if handle_briefing_command(
                user_text,
                self.brain,
                self.assistant_name,
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                self._enter_follow_up_window()
                return

            # Voice command: Status report ("Jarvis, status", "status")
            if handle_status_command(
                user_text,
                self.assistant_name,
                current_mode="Hands-free voice mode (--listen)",
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                self._enter_follow_up_window()
                return

            # Voice command: Token usage report ("usage", "check usage")
            if handle_usage_command(
                user_text,
                self.assistant_name,
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                self._enter_follow_up_window()
                return

            # Voice command: Override token limit ("override limit")
            if handle_override_command(
                user_text,
                self.assistant_name,
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                self._enter_follow_up_window()
                return

            # Voice command: Vision & Screen ("what's on my screen", "look at this", etc.)
            if handle_vision_command(
                user_text,
                self.brain,
                self.assistant_name,
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                self._enter_follow_up_window()
                return

            # Voice command: Clipboard ("what did I copy", "summarize my clipboard", "put that in my clipboard")
            if handle_clipboard_command(
                user_text,
                self.brain,
                self.assistant_name,
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                self._enter_follow_up_window()
                return

            # Voice command: Google Calendar ("disconnect calendar")
            if handle_calendar_command(
                user_text,
                self.assistant_name,
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                self._enter_follow_up_window()
                return

            # Voice command: Conversation short-term history reset
            if command in ("reset", "clear history"):
                self.brain.reset()
                reset_msg = "Conversation history cleared."
                print(f"{self.assistant_name}: {reset_msg}\n")
                if self.enable_speak:
                    speak(reset_msg)
                self._enter_follow_up_window()
                return

            # ------------------------------------------------------------------
            # STATE: THINKING & STREAMING SPEECH
            # ------------------------------------------------------------------
            self.set_state(AssistantState.THINKING)
            print(f"{self.assistant_name} is thinking...")
            t0_llm = time.perf_counter()

            if self.enable_speak:
                from tts import SentenceSpeaker
                speaker = SentenceSpeaker()

                def on_sentence_feed(sentence: str) -> None:
                    speaker.feed_sentence(sentence)
                    if self.state != AssistantState.SPEAKING:
                        self.set_state(AssistantState.SPEAKING)
                        self._interrupted_event.clear()
                        interrupt_threshold = getattr(config, "INTERRUPT_THRESHOLD", 0.8)
                        self.listener.set_threshold(interrupt_threshold)
                        self.listener.reset_cooldown()
                        self.listener.resume()

                reply = self.brain.think(user_text, on_sentence=on_sentence_feed, mode="voice")
                llm_duration = time.perf_counter() - t0_llm
                speaker.finish()
                speaker.wait_until_done()
                self.listener.pause()
                ttfw = speaker.time_to_first_spoken_word or llm_duration
                print(f"{self.assistant_name}: {reply}\n")

                if self.debug:
                    print(f"[Latency] AI Stream: {llm_duration:.2f}s | TTFW: {ttfw:.2f}s\n")
            else:
                reply = self.brain.think(user_text, mode="voice")
                llm_duration = time.perf_counter() - t0_llm
                print(f"{self.assistant_name}: {reply}\n")
                if self.debug:
                    print(f"[Latency] AI Reply: {llm_duration:.2f}s\n")

            # Check if speaking was interrupted by wake word or hotkey
            if self._interrupted_event.is_set():
                self._interrupted_event.clear()
                if self.debug:
                    print("[Interrupt] Speech interrupted. Transitioning directly to LISTENING.\n")
                self._handle_active_cycle(is_initial_wake=True)
                return

            # Enter follow-up conversational window
            self._enter_follow_up_window()

        except Exception as err:
            # Trap any error, log it, play low beep, and safely return to IDLE
            print(f"\n[Assistant Error in {self.state.value}] {err}\n", file=sys.stderr)
            play_idle_beep()

    def _enter_follow_up_window(self) -> None:
        """Keep listening without requiring wake word for configurable follow-up duration."""
        follow_up_sec = getattr(config, "FOLLOW_UP_TIMEOUT_SEC", 6.0)
        while self._running and not self.is_manually_paused and self.state != AssistantState.IDLE:
            self.set_state(AssistantState.LISTENING)
            print(f"[Follow-up Window: {follow_up_sec:.0f}s] Speak your next request or stay quiet to sleep...")
            follow_up = listen(initial_timeout_sec=follow_up_sec)

            # Silence after 6 seconds -> end follow-up window and return to IDLE
            if not follow_up or not follow_up.strip():
                play_idle_beep()
                print(f"{self.assistant_name}: (Follow-up window closed. Returning to idle.)\n")
                break

            print(f"You: {follow_up}")
            cmd = follow_up.lower().strip(" .!?,")

            if cmd in ("goodbye jarvis", "goodbye, jarvis", "exit", "quit", "goodbye"):
                farewell = "Goodbye! Have a great day."
                print(f"\n{self.assistant_name}: {farewell}")
                if self.enable_speak:
                    speak(farewell)
                self.stop()
                break

            if cmd in ("stop listening", "go to sleep", "sleep"):
                print(f"{self.assistant_name}: Going to sleep.\n")
                play_idle_beep()
                break

            if handle_memory_command(
                follow_up,
                self.brain,
                self.assistant_name,
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                continue

            if handle_briefing_command(
                follow_up,
                self.brain,
                self.assistant_name,
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                continue

            if handle_status_command(
                follow_up,
                self.assistant_name,
                current_mode="Hands-free voice mode (--listen)",
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                continue

            if handle_usage_command(
                follow_up,
                self.assistant_name,
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                continue

            if handle_override_command(
                follow_up,
                self.assistant_name,
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                continue

            if handle_vision_command(
                follow_up,
                self.brain,
                self.assistant_name,
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                continue

            if handle_clipboard_command(
                follow_up,
                self.brain,
                self.assistant_name,
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                continue

            if handle_calendar_command(
                follow_up,
                self.assistant_name,
                enable_speak=self.enable_speak,
                confirmation_mode="voice" if self.enable_speak else "text",
            ):
                continue

            if cmd in ("reset", "clear history"):
                self.brain.reset()
                print(f"{self.assistant_name}: Conversation history cleared.\n")
                if self.enable_speak:
                    speak("Conversation history cleared.")
                continue

            # Process follow-up with Brain & streaming speech
            self.set_state(AssistantState.THINKING)
            print(f"{self.assistant_name} is thinking...")
            t0_llm = time.perf_counter()

            if self.enable_speak:
                from tts import SentenceSpeaker
                speaker = SentenceSpeaker()

                def on_sentence_feed_fu(sentence: str) -> None:
                    speaker.feed_sentence(sentence)
                    if self.state != AssistantState.SPEAKING:
                        self.set_state(AssistantState.SPEAKING)
                        self._interrupted_event.clear()
                        interrupt_threshold = getattr(config, "INTERRUPT_THRESHOLD", 0.8)
                        self.listener.set_threshold(interrupt_threshold)
                        self.listener.reset_cooldown()
                        self.listener.resume()

                reply = self.brain.think(follow_up, on_sentence=on_sentence_feed_fu, mode="voice")
                llm_duration = time.perf_counter() - t0_llm
                speaker.finish()
                speaker.wait_until_done()
                self.listener.pause()
                ttfw = speaker.time_to_first_spoken_word or llm_duration
                print(f"{self.assistant_name}: {reply}\n")

                if self.debug:
                    print(f"[Latency] AI Stream: {llm_duration:.2f}s | TTFW: {ttfw:.2f}s\n")
            else:
                reply = self.brain.think(follow_up, mode="voice")
                llm_duration = time.perf_counter() - t0_llm
                print(f"{self.assistant_name}: {reply}\n")
                if self.debug:
                    print(f"[Latency] AI Reply: {llm_duration:.2f}s\n")

            if self._interrupted_event.is_set():
                self._interrupted_event.clear()
                if self.debug:
                    print("[Interrupt] Follow-up speech interrupted. Transitioning directly to LISTENING.\n")
                self._handle_active_cycle(is_initial_wake=True)
                return


# ==============================================================================
# Long-Term Memory Built-in Commands (Phase 5)
# ==============================================================================

def handle_memory_command(
    command_text: str,
    brain: Brain,
    assistant_name: str,
    enable_speak: bool = True,
    confirmation_mode: str = "text",
) -> bool:
    """Handle built-in memory commands ('what do you remember about me', 'forget everything').

    Args:
        command_text: The user's input string.
        brain: The Brain instance for short-term history reset.
        assistant_name: Name of the assistant for messages.
        enable_speak: Whether voice output is enabled.
        confirmation_mode: 'voice' or 'text'.

    Returns:
        True if the command was recognized and handled, False otherwise.
    """
    cmd = command_text.lower().strip(" .!?,")

    # 1. "what do you remember about me"
    if cmd in ("what do you remember about me", "what do you remember", "list memories"):
        from memory import MemoryStore
        store = MemoryStore()
        memories = store.list_all()
        if not memories:
            msg = "I don't have any memories stored about you yet."
            print(f"{assistant_name}: {msg}\n")
            if enable_speak and confirmation_mode == "voice":
                try:
                    speak(msg)
                except Exception:
                    pass
            return True

        print(f"\n{assistant_name}: Here is what I remember about you ({len(memories)} item{'s' if len(memories) != 1 else ''}):")
        for i, m in enumerate(memories, 1):
            print(f"  {i}. [#{m['id']}] ({m['category']}) {m['text']}")
        print()

        if enable_speak and confirmation_mode == "voice":
            try:
                if len(memories) <= 3:
                    spoken_items = "; ".join(m["text"] for m in memories)
                    speak(f"I remember {len(memories)} things: {spoken_items}")
                else:
                    spoken_preview = "; ".join(m["text"] for m in memories[:3])
                    speak(f"I remember {len(memories)} things, including: {spoken_preview}, and {len(memories) - 3} more.")
            except Exception:
                pass
        return True

    # 2. "forget everything"
    if cmd in ("forget everything", "clear all memories", "forget all memories"):
        from memory import MemoryStore
        from tools import confirm
        store = MemoryStore()
        total_memories = store.count()

        # Confirmation question (spoken in voice mode, text prompt in text mode)
        confirm_question = (
            f"Are you sure you want to permanently delete all {total_memories} stored memories "
            f"and reset conversation history?"
        )
        if not confirm(confirm_question):
            denial_msg = "Cancelled. Your memories have been kept."
            print(f"{assistant_name}: {denial_msg}\n")
            if enable_speak and confirmation_mode == "voice":
                try:
                    speak(denial_msg)
                except Exception:
                    pass
            return True

        cleared_count = store.clear_all()
        brain.reset()
        success_msg = f"All {cleared_count} stored memories and conversation history have been cleared."
        print(f"{assistant_name}: {success_msg}\n")
        if enable_speak and confirmation_mode == "voice":
            try:
                speak(success_msg)
            except Exception:
                pass
        return True

    return False


# ==============================================================================
# Daily Briefing Handlers & Triggers (Phase 5)
# ==============================================================================

def trigger_daily_briefing(
    brain: Brain,
    assistant_name: str,
    enable_speak: bool = True,
    confirmation_mode: str = "text",
) -> str:
    """Gather the structured daily briefing and turn it into a short, natural spoken summary under 30 seconds."""
    from tools import run_tool

    print(f"\n[{assistant_name}] Gathering daily briefing...")
    raw_briefing = run_tool("daily_briefing", {})

    prompt = (
        "Here is the daily briefing reference data:\n"
        f"{raw_briefing}\n\n"
        "Please provide a short, natural, friendly spoken summary of this briefing "
        "(under 30 seconds when read aloud). "
        "Address the user directly. Highlight the weather, today's calendar events, any active timers, recent notes, and tasks. "
        "If any part is unavailable, mention it briefly."
    )

    try:
        summary = brain.think(prompt)
        # If AI model is offline or unreachable, fall back to presenting the structured briefing cleanly
        if "unable to connect to the ai service" in summary.lower():
            summary = f"Good day! Here is your daily briefing:\n\n{raw_briefing}"
    except Exception:
        summary = (
            f"Good day! Here is your daily briefing:\n\n{raw_briefing}"
        )

    print(f"\n{assistant_name}: {summary}\n")

    if enable_speak and confirmation_mode == "voice":
        try:
            from tts import speak
            speak(summary)
        except Exception as tts_err:
            print(f"[Audio Error] Speech playback failed: {tts_err}")

    return summary


def handle_briefing_command(
    command_text: str,
    brain: Brain,
    assistant_name: str,
    enable_speak: bool = True,
    confirmation_mode: str = "text",
) -> bool:
    """Voice & text trigger: saying 'good morning', 'give me my briefing', or 'what's my day look like'."""
    cmd = command_text.lower().strip(" .!?,")

    briefing_triggers = (
        "good morning",
        "good morning jarvis",
        "give me my briefing",
        "give me a briefing",
        "what's my day look like",
        "what does my day look like",
        "daily briefing",
        "briefing",
    )

    if any(cmd == trigger or cmd.startswith(trigger) for trigger in briefing_triggers):
        trigger_daily_briefing(
            brain=brain,
            assistant_name=assistant_name,
            enable_speak=enable_speak,
            confirmation_mode=confirmation_mode,
        )
        return True

    return False


# ==============================================================================
# System Status Reporter (Phase 5)
# ==============================================================================

SESSION_START_TIME = time.time()


def get_system_status(current_mode: str = "Interactive text mode") -> dict[str, Any]:
    """Compute and gather live runtime metrics for Jarvis.

    Returns:
        Dict with keys:
        - uptime: formatted string
        - current_mode: operating mode string
        - tool_calls_today: integer count
        - errors_today: integer count
        - memories_count: integer count
        - model_name: string
    """
    elapsed = max(0.0, time.time() - SESSION_START_TIME)
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    if hours > 0:
        uptime_str = f"{hours} hour{'s' if hours != 1 else ''}, {minutes} minute{'s' if minutes != 1 else ''}"
    elif minutes > 0:
        uptime_str = f"{minutes} minute{'s' if minutes != 1 else ''}, {seconds} second{'s' if seconds != 1 else ''}"
    else:
        uptime_str = f"{seconds} second{'s' if seconds != 1 else ''}"

    today_utc = datetime.now(UTC).strftime("%Y-%m-%d")
    today_local = datetime.now().strftime("%Y-%m-%d")
    tool_calls_today = 0
    errors_today = 0

    base_dir = getattr(config, "BASE_DIR", Path("."))
    log_file = base_dir / "logs" / "actions.log"
    if log_file.is_file():
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line_clean = line.strip()
                    if line_clean:
                        try:
                            entry = json.loads(line_clean)
                            ts = entry.get("timestamp", "")
                            if ts.startswith(today_utc) or ts.startswith(today_local):
                                tool_calls_today += 1
                                if entry.get("status") != "SUCCESS":
                                    errors_today += 1
                        except Exception:
                            continue
        except Exception as err:
            print(f"[Status Warning] Could not inspect actions log: {err}")

    try:
        from memory import MemoryStore
        memories_count = MemoryStore().count()
    except Exception:
        memories_count = 0

    return {
        "uptime": uptime_str,
        "current_mode": current_mode,
        "tool_calls_today": tool_calls_today,
        "errors_today": errors_today,
        "memories_count": memories_count,
        "model_name": getattr(config, "MODEL_NAME", "llama3.1"),
    }


def format_status_report(status_data: dict[str, Any]) -> tuple[str, str]:
    """Generate both text-table and concise spoken representations of system status.

    Returns:
        (text_report: str, spoken_summary: str)
    """
    text_report = (
        "=== Jarvis System Status ===\n"
        f"• Uptime:            {status_data['uptime']}\n"
        f"• Current Mode:      {status_data['current_mode']}\n"
        f"• AI Model:          {status_data['model_name']}\n"
        f"• Tool Calls Today:  {status_data['tool_calls_today']}\n"
        f"• Errors Today:      {status_data['errors_today']}\n"
        f"• Memories Stored:   {status_data['memories_count']}"
    )

    spoken_summary = (
        f"Jarvis has been online for {status_data['uptime']} in {status_data['current_mode']}. "
        f"Using AI model {status_data['model_name']}, {status_data['tool_calls_today']} tool calls "
        f"have been executed today with {status_data['errors_today']} errors. "
        f"{status_data['memories_count']} memories are stored."
    )

    return text_report, spoken_summary


def handle_status_command(
    command_text: str,
    assistant_name: str,
    current_mode: str,
    enable_speak: bool = False,
    confirmation_mode: str = "text",
) -> bool:
    """Check and handle 'status' command ('Jarvis, status', 'status', etc.)."""
    cmd = command_text.lower().strip(" .!?,")

    status_triggers = (
        "status",
        "jarvis status",
        "jarvis, status",
        "system status",
        "report status",
    )

    if cmd in status_triggers or any(cmd.startswith(t) for t in ("jarvis, status", "jarvis status")):
        data = get_system_status(current_mode=current_mode)
        text_rep, spoken_sum = format_status_report(data)

        print(f"\n{assistant_name}: {text_rep}\n")
        if enable_speak and confirmation_mode == "voice":
            try:
                from tts import speak
                speak(spoken_sum)
            except Exception:
                pass
        return True

    return False


# ==============================================================================
# Usage & Cost Management Commands (Phase 5)
# ==============================================================================

def handle_usage_command(
    command_text: str,
    assistant_name: str,
    enable_speak: bool = False,
    confirmation_mode: str = "text",
) -> bool:
    """Handle 'usage' command ('Jarvis, usage', 'usage', 'token usage', etc.)."""
    cmd = command_text.lower().strip(" .!?,")

    usage_triggers = (
        "usage",
        "jarvis usage",
        "jarvis, usage",
        "check usage",
        "token usage",
        "show usage",
        "my usage",
        "how many tokens have i used",
        "how many tokens",
        "token cost",
        "cost",
    )

    if cmd in usage_triggers or any(cmd == t for t in usage_triggers):
        from usage import format_spoken_usage_report, format_usage_report
        report = format_usage_report()
        spoken_rep = format_spoken_usage_report()

        print(f"\n{assistant_name}:\n{report}\n")
        if enable_speak and confirmation_mode == "voice":
            try:
                from tts import speak
                speak(spoken_rep)
            except Exception:
                pass
        return True

    return False


def handle_override_command(
    command_text: str,
    assistant_name: str,
    enable_speak: bool = False,
    confirmation_mode: str = "text",
) -> bool:
    """Handle 'override limit' command when daily token limit is reached."""
    cmd = command_text.lower().strip(" .!?,")

    override_triggers = (
        "override limit",
        "override token limit",
        "override daily limit",
        "jarvis override limit",
        "jarvis, override limit",
        "override the limit",
    )

    if cmd in override_triggers or any(cmd == t for t in override_triggers):
        from usage import override_daily_limit
        override_daily_limit()
        msg = "Daily token limit has been overridden for the rest of today."

        print(f"\n{assistant_name}: {msg}\n")
        if enable_speak and confirmation_mode == "voice":
            try:
                from tts import speak
                speak(msg)
            except Exception:
                pass
        return True

    return False


# ==============================================================================
# Vision & Screen Commands (Phase 6)
# ==============================================================================

def handle_vision_command(
    command_text: str,
    brain: Brain | None,
    assistant_name: str,
    enable_speak: bool = False,
    confirmation_mode: str = "text",
) -> bool:
    """Voice & text trigger: saying 'what's on my screen', 'look at this', 'explain this page', or 'what does this error mean'."""
    cmd = command_text.lower().strip(" .!?,")

    is_vision = False
    target = "screen"
    default_question = "What is on my screen?"

    # Check triggers
    if any(phrase in cmd for phrase in ("what's on my screen", "what is on my screen", "look at my screen", "look at the screen")):
        is_vision = True
        target = "screen"
        default_question = "What is on my screen?"
    elif "look at this" in cmd:
        is_vision = True
        target = "window"
        default_question = "Explain what is visible in this window."
    elif "explain this page" in cmd or "explain this window" in cmd or "explain this" in cmd:
        is_vision = True
        target = "window"
        default_question = "Explain what is displayed on this page."
    elif any(phrase in cmd for phrase in ("what does this error mean", "what does the error mean", "explain this error")):
        is_vision = True
        target = "window"
        default_question = "What does this error message mean and how can I resolve it?"

    if not is_vision:
        return False

    # Check if user asked something more specific after/with the trigger phrase
    question = default_question
    for trigger in (
        "what's on my screen",
        "what is on my screen",
        "look at my screen",
        "look at the screen",
        "look at this",
        "explain this page",
        "explain this window",
        "what does this error mean",
        "what does the error mean",
        "explain this error",
    ):
        if trigger in cmd:
            remainder = cmd.replace(trigger, "").strip(" :-,")
            if remainder and len(remainder) > 3:
                question = f"{default_question} Specifically: {remainder}"
            break

    # Execute tool via run_tool to enforce confirmation and audit logging
    from tools import run_tool, set_confirmation_mode
    set_confirmation_mode(confirmation_mode)

    print(f"\n[{assistant_name}] Checking visual context ({target})...")
    tool_result = run_tool("look_at_screen", {"question": question, "target": target})

    # Extract clean text for speech playback
    if "\n\n" in tool_result and "[Visual Analysis of" in tool_result:
        parts = tool_result.split("\n\n", 2)
        analysis_body = parts[-1].strip()
    else:
        analysis_body = tool_result.strip()

    print(f"\n{assistant_name}: {tool_result}\n")

    # Record into conversational history if brain is provided
    if brain and hasattr(brain, "history") and not tool_result.startswith("Action cancelled:"):
        try:
            brain.history.append({"role": "user", "content": command_text})
            brain.history.append({"role": "assistant", "content": analysis_body})
            max_hist = getattr(brain, "max_exchanges", 5) * 2
            if len(brain.history) > max_hist:
                brain.history = brain.history[-max_hist:]
        except Exception:
            pass

    if enable_speak and confirmation_mode == "voice":
        try:
            from tts import speak
            if tool_result.startswith("Action cancelled:"):
                speak("Screen capture cancelled.")
            elif tool_result.startswith("Screen analysis was blocked"):
                speak("Screen analysis blocked for sensitive window.")
            else:
                speak(analysis_body)
        except Exception as tts_err:
            print(f"[Audio Error] Speech playback failed: {tts_err}")

    return True


# ==============================================================================
# Clipboard Commands
# ==============================================================================

def handle_clipboard_command(
    command_text: str,
    brain: Brain | None,
    assistant_name: str,
    enable_speak: bool = False,
    confirmation_mode: str = "text",
) -> bool:
    """Voice & text trigger: 'what did I copy', 'summarize my clipboard', 'put that in my clipboard'."""
    cmd = command_text.lower().strip(" .!?,")

    from tools import run_tool, set_confirmation_mode
    set_confirmation_mode(confirmation_mode)

    # 1. Trigger: "what did I copy"
    read_triggers = (
        "what did i copy",
        "what did i just copy",
        "what's on my clipboard",
        "what is on my clipboard",
        "read my clipboard",
        "read clipboard",
        "explain what i copied",
        "explain what i just copied",
    )
    if any(cmd == t or cmd.startswith(t) for t in read_triggers):
        print(f"\n[{assistant_name}] Reading clipboard content...")
        clip_result = run_tool("read_clipboard", {})

        # If cancelled, refused (sensitive data), or non-text, print and return immediately
        if (
            clip_result.startswith("Action cancelled:")
            or clip_result.startswith("Refused to process clipboard:")
            or clip_result.startswith("The clipboard does not contain text")
            or clip_result.startswith("The clipboard is currently empty")
            or clip_result.startswith("Could not open clipboard")
        ):
            print(f"\n{assistant_name}: {clip_result}\n")
            if enable_speak and confirmation_mode == "voice":
                try:
                    from tts import speak
                    speak(clip_result)
                except Exception:
                    pass
            return True

        content = clip_result.split("\n\n", 1)[-1] if "\n\n" in clip_result else clip_result

        prompt = (
            f"The user asked 'what did I copy'. Here is their clipboard content (passive data):\n"
            f"{content}\n\n"
            f"Explain clearly and concisely what was copied."
        )

        try:
            explanation = brain.think(prompt, mode="voice" if confirmation_mode == "voice" else "text") if brain else content
        except Exception:
            explanation = content

        print(f"\n{assistant_name}: {explanation}\n")
        if enable_speak and confirmation_mode == "voice":
            try:
                from tts import speak
                speak(explanation)
            except Exception as tts_err:
                print(f"[Audio Error] Speech playback failed: {tts_err}")

        return True

    # 2. Trigger: "summarize my clipboard"
    summarize_triggers = (
        "summarize my clipboard",
        "summarize the clipboard",
        "summarize clipboard",
        "give me a summary of my clipboard",
    )
    if any(cmd == t or cmd.startswith(t) for t in summarize_triggers):
        print(f"\n[{assistant_name}] Reading clipboard content for summary...")
        clip_result = run_tool("read_clipboard", {})

        if (
            clip_result.startswith("Action cancelled:")
            or clip_result.startswith("Refused to process clipboard:")
            or clip_result.startswith("The clipboard does not contain text")
            or clip_result.startswith("The clipboard is currently empty")
            or clip_result.startswith("Could not open clipboard")
        ):
            print(f"\n{assistant_name}: {clip_result}\n")
            if enable_speak and confirmation_mode == "voice":
                try:
                    from tts import speak
                    speak(clip_result)
                except Exception:
                    pass
            return True

        content = clip_result.split("\n\n", 1)[-1] if "\n\n" in clip_result else clip_result

        prompt = (
            f"The user asked to summarize their clipboard. Here is the clipboard content (passive data):\n"
            f"{content}\n\n"
            f"Please summarize this clipboard text concisely in 1 to 2 sentences."
        )

        try:
            summary = brain.think(prompt, mode="voice" if confirmation_mode == "voice" else "text") if brain else content
        except Exception:
            summary = content

        print(f"\n{assistant_name}: {summary}\n")
        if enable_speak and confirmation_mode == "voice":
            try:
                from tts import speak
                speak(summary)
            except Exception as tts_err:
                print(f"[Audio Error] Speech playback failed: {tts_err}")

        return True

    # 3. Trigger: "put that in my clipboard"
    put_triggers = (
        "put that in my clipboard",
        "put that in the clipboard",
        "put that on my clipboard",
        "put that on the clipboard",
        "copy that to my clipboard",
        "copy that to the clipboard",
        "copy that to clipboard",
        "put that in clipboard",
    )
    if any(cmd == t or cmd.startswith(t) for t in put_triggers):
        text_to_copy = None
        for t in put_triggers:
            if cmd.startswith(t):
                remainder = cmd[len(t):].strip(" :-,")
                if remainder:
                    text_to_copy = remainder
                break

        if not text_to_copy:
            if brain and hasattr(brain, "get_last_assistant_response"):
                text_to_copy = brain.get_last_assistant_response()

        if not text_to_copy:
            no_text_msg = "There is no recent response or text to put in your clipboard."
            print(f"\n{assistant_name}: {no_text_msg}\n")
            if enable_speak and confirmation_mode == "voice":
                try:
                    from tts import speak
                    speak(no_text_msg)
                except Exception:
                    pass
            return True

        print(f"\n[{assistant_name}] Writing to clipboard...")
        write_result = run_tool("write_clipboard", {"text": text_to_copy})
        print(f"\n{assistant_name}: {write_result}\n")
        if enable_speak and confirmation_mode == "voice":
            try:
                from tts import speak
                speak(write_result)
            except Exception as tts_err:
                print(f"[Audio Error] Speech playback failed: {tts_err}")

        return True

    return False


# ==============================================================================
# Google Calendar Commands (Phase 6)
# ==============================================================================

def handle_calendar_command(
    command_text: str,
    assistant_name: str,
    enable_speak: bool = False,
    confirmation_mode: str = "text",
) -> bool:
    """Voice & text trigger: 'disconnect calendar' (with confirmation)."""
    cmd = command_text.lower().strip(" .!?,")

    disconnect_triggers = (
        "disconnect calendar",
        "disconnect my calendar",
        "disconnect google calendar",
        "unlink calendar",
        "unlink my calendar",
        "remove calendar access",
    )

    if any(cmd == t or cmd.startswith(t) for t in disconnect_triggers):
        from tools import run_tool, set_confirmation_mode

        set_confirmation_mode(confirmation_mode)
        print(f"\n[{assistant_name}] Disconnecting Google Calendar...")
        result = run_tool("disconnect_calendar", {})
        print(f"\n{assistant_name}: {result}\n")
        if enable_speak and confirmation_mode == "voice":
            try:
                from tts import speak

                speak("Google Calendar disconnected.")
            except Exception as tts_err:
                print(f"[Audio Error] Speech playback failed: {tts_err}")
        return True

    return False


# ==============================================================================
# Existing Text and Voice Modes (Preserved from Phases 1-3)
# ==============================================================================

def show_help() -> None:
    """Display the list of available commands for text mode."""
    print("\nCommands:")
    print("  help                          - Show available commands")
    print("  status / Jarvis, status       - Report system uptime and operational metrics")
    print("  good morning / briefing       - Get your daily briefing")
    print("  look at this                  - Inspect and explain the active window")
    print("  what's on my screen           - Inspect and explain the entire screen")
    print("  what did I copy               - Read and explain clipboard contents")
    print("  summarize my clipboard        - Summarize text on the clipboard")
    print("  put that in my clipboard      - Copy the last assistant reply to clipboard")
    print("  what's on my calendar today   - View today's scheduled events")
    print("  when am I free tomorrow       - Find free time slots (30+ mins)")
    print("  disconnect calendar           - Disconnect Google Calendar (requires confirmation)")
    print("  what do you remember about me - List long-term memories")
    print("  forget everything             - Clear all memories (requires confirmation)")
    print("  reset                         - Clear short-term conversation history")
    print("  exit                          - Quit the chat (or 'quit')\n")


def show_banner(assistant_name: str) -> None:
    """Display the startup welcome banner for text mode."""
    print("=" * 50)
    print(f"  {assistant_name} - Personal AI Assistant")
    print("=" * 50)
    print(f"{assistant_name} is online. Type 'help' for commands, or 'exit' to quit.\n")


def run_text_mode(brain: Brain, assistant_name: str) -> None:
    """Run the interactive terminal text chat loop (Phase 1)."""
    set_confirmation_mode("text")
    show_banner(assistant_name)

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        command = user_input.lower()

        if command in ("exit", "quit"):
            print(f"\n{assistant_name}: Goodbye! Have a great day.")
            break

        if handle_memory_command(
            user_input,
            brain,
            assistant_name,
            enable_speak=False,
            confirmation_mode="text",
        ):
            continue

        if handle_briefing_command(
            user_input,
            brain,
            assistant_name,
            enable_speak=False,
            confirmation_mode="text",
        ):
            continue

        if handle_status_command(
            user_input,
            assistant_name,
            current_mode="Interactive text chat mode",
            enable_speak=False,
            confirmation_mode="text",
        ):
            continue

        if handle_usage_command(
            user_input,
            assistant_name,
            enable_speak=False,
            confirmation_mode="text",
        ):
            continue

        if handle_override_command(
            user_input,
            assistant_name,
            enable_speak=False,
            confirmation_mode="text",
        ):
            continue

        if handle_vision_command(
            user_input,
            brain,
            assistant_name,
            enable_speak=False,
            confirmation_mode="text",
        ):
            continue

        if handle_clipboard_command(
            user_input,
            brain,
            assistant_name,
            enable_speak=False,
            confirmation_mode="text",
        ):
            continue

        if handle_calendar_command(
            user_input,
            assistant_name,
            enable_speak=False,
            confirmation_mode="text",
        ):
            continue

        if command == "reset":
            brain.reset()
            print(f"{assistant_name}: Conversation memory has been cleared.\n")
            continue

        if command == "help":
            show_help()
            continue

        print(f"{assistant_name} is thinking...")
        reply = brain.think(user_input, mode="text")
        print(f"{assistant_name}: {reply}\n")


def run_voice_mode(
    brain: Brain,
    assistant_name: str,
    enable_speak: bool = True,
    debug: bool = False,
) -> None:
    """Run the push-to-talk voice mode loop (Phase 2)."""
    set_confirmation_mode("voice" if enable_speak else "text")

    # Preload Whisper speech recognition model in background to avoid first-call lag
    from audio_input import preload_whisper_model
    threading.Thread(target=preload_whisper_model, daemon=True, name="whisper-preload").start()

    status_label = "Spoken audio enabled" if enable_speak else "Quiet mode (--no-speak, text replies only)"
    print("=" * 60)
    print(f"  {assistant_name} - Voice Mode (Push-to-Talk)")
    print("=" * 60)
    print(f"{assistant_name} is online ({status_label}).")
    print("Press Enter to speak into your mic, or type 'exit' to quit.\n")

    while True:
        try:
            prompt_input = input("Press Enter to speak, or type 'exit' to quit: ").strip()

            if prompt_input.lower() in ("exit", "quit"):
                farewell = "Goodbye! Have a great day."
                print(f"\n{assistant_name}: {farewell}")
                if enable_speak:
                    speak(farewell)
                break

            if prompt_input.lower() == "reset":
                brain.reset()
                confirm_msg = "Conversation memory has been cleared."
                print(f"{assistant_name}: {confirm_msg}\n")
                if enable_speak:
                    speak(confirm_msg)
                continue

            # Record speech and transcribe with Whisper
            stt_duration = 0.0
            try:
                print("Listening...")
                audio = record_until_silence()

                if audio is None or len(audio) == 0:
                    transcribed_text = ""
                else:
                    print("Transcribing...")
                    t0_stt = time.perf_counter()
                    transcribed_text = transcribe(audio)
                    stt_duration = time.perf_counter() - t0_stt
            except Exception as audio_err:
                print(f"[Audio Error] Microphone capture failed: {audio_err}\n")
                continue

            if not transcribed_text:
                missed_msg = "I didn't catch that."
                print(f"{assistant_name}: {missed_msg}\n")
                if enable_speak:
                    speak(missed_msg)
                continue

            print(f"You: {transcribed_text}")
            voice_command = transcribed_text.lower().strip(" .!?,")

            if voice_command in ("exit", "quit", "goodbye"):
                farewell = "Goodbye! Have a great day."
                print(f"{assistant_name}: {farewell}\n")
                if enable_speak:
                    speak(farewell)
                break

            if handle_memory_command(
                transcribed_text,
                brain,
                assistant_name,
                enable_speak=enable_speak,
                confirmation_mode="voice" if enable_speak else "text",
            ):
                continue

            if handle_briefing_command(
                transcribed_text,
                brain,
                assistant_name,
                enable_speak=enable_speak,
                confirmation_mode="voice" if enable_speak else "text",
            ):
                continue

            if handle_status_command(
                transcribed_text,
                assistant_name,
                current_mode="Push-to-talk voice mode (--voice)",
                enable_speak=enable_speak,
                confirmation_mode="voice" if enable_speak else "text",
            ):
                continue

            if handle_usage_command(
                transcribed_text,
                assistant_name,
                enable_speak=enable_speak,
                confirmation_mode="voice" if enable_speak else "text",
            ):
                continue

            if handle_override_command(
                transcribed_text,
                assistant_name,
                enable_speak=enable_speak,
                confirmation_mode="voice" if enable_speak else "text",
            ):
                continue

            if handle_vision_command(
                transcribed_text,
                brain,
                assistant_name,
                enable_speak=enable_speak,
                confirmation_mode="voice" if enable_speak else "text",
            ):
                continue

            if handle_clipboard_command(
                transcribed_text,
                brain,
                assistant_name,
                enable_speak=enable_speak,
                confirmation_mode="voice" if enable_speak else "text",
            ):
                continue

            if handle_calendar_command(
                transcribed_text,
                assistant_name,
                enable_speak=enable_speak,
                confirmation_mode="voice" if enable_speak else "text",
            ):
                continue

            if voice_command in ("reset", "clear history"):
                brain.reset()
                reset_msg = "Conversation history cleared. Starting a fresh conversation."
                print(f"{assistant_name}: {reset_msg}\n")
                if enable_speak:
                    speak(reset_msg)
                continue

            print(f"{assistant_name} is thinking...")
            t0_llm = time.perf_counter()

            if enable_speak:
                from tts import SentenceSpeaker
                speaker = SentenceSpeaker()
                reply = brain.think(transcribed_text, on_sentence=speaker.feed_sentence, mode="voice")
                llm_duration = time.perf_counter() - t0_llm
                speaker.finish()
                speaker.wait_until_done()
                ttfw = speaker.time_to_first_spoken_word or llm_duration
                print(f"{assistant_name}: {reply}\n")
            else:
                reply = brain.think(transcribed_text, mode="voice")
                llm_duration = time.perf_counter() - t0_llm
                ttfw = None
                print(f"{assistant_name}: {reply}\n")

            if debug:
                if enable_speak:
                    total_latency = stt_duration + (ttfw or llm_duration)
                    print(
                        f"[Latency] Transcribe: {stt_duration:.2f}s | "
                        f"TTFW: {ttfw:.2f}s | "
                        f"AI Total: {llm_duration:.2f}s | "
                        f"Time to speak: {total_latency:.2f}s\n"
                    )
                else:
                    print(
                        f"[Latency] Transcribe: {stt_duration:.2f}s | "
                        f"AI Reply: {llm_duration:.2f}s | "
                        f"Total: {(stt_duration + llm_duration):.2f}s\n"
                    )

        except Exception as loop_err:
            print(f"[Error] An unexpected error occurred: {loop_err}\n")


def main() -> None:
    """Initialize Jarvis and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=f"{config.ASSISTANT_NAME} - Personal AI Assistant"
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Run in hands-free wake word mode (say 'Hey Jarvis' or press hotkey to activate)",
    )
    parser.add_argument(
        "--tray",
        action="store_true",
        help="Run with system tray icon (draws icon in code, runs assistant in background thread)",
    )
    parser.add_argument(
        "--enable-autostart",
        action="store_true",
        help="Enable Jarvis to start automatically at login with system tray icon",
    )
    parser.add_argument(
        "--disable-autostart",
        action="store_true",
        help="Disable automatic startup at login and remove the startup shortcut",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Run in push-to-talk voice mode using your microphone and speaker",
    )
    parser.add_argument(
        "--no-speak",
        action="store_true",
        help="Disable spoken voice output in voice mode (prints replies only)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show latency metrics and state machine transitions",
    )

    args = parser.parse_args()
    assistant_name = config.ASSISTANT_NAME

    # Initialize persistent logging (Requirement 5)
    setup_jarvis_logging()

    # Handle Autostart CLI commands (Requirement 3)
    if args.enable_autostart:
        success, message = autostart.enable()
        if success:
            print("=" * 64)
            print(f"  {assistant_name} - Automatic Startup Enabled")
            print("=" * 64)
            print(f"[Success] {message}\n")
            print("Startup Details:")
            print(f"  Startup File: {autostart.get_autostart_path()}")
            print(f"  Executable:   {autostart.get_python_executable()}")
            print("  Arguments:    main.py --listen --tray\n")
            print("Behavior:")
            print("  - Starts silently at login in the background (no console window).")
            print("  - Displays the status icon in your system tray.")
            print("  - Automatically waits for audio devices and network to be ready.")
            print("  - Writes all runtime logs and errors to logs/jarvis.log.\n")
            print("How to inspect or remove manually if needed:")
            print(autostart.get_manual_instructions())
            print("\nTo disable via CLI: python main.py --disable-autostart\n")
            sys.exit(0)
        else:
            print(f"[Error] Failed to enable autostart: {message}", file=sys.stderr)
            print(autostart.get_manual_instructions(), file=sys.stderr)
            sys.exit(1)

    if args.disable_autostart:
        success, message = autostart.disable()
        if success:
            print("=" * 64)
            print(f"  {assistant_name} - Automatic Startup Disabled")
            print("=" * 64)
            print(f"[Success] {message}\n")
            print(f"Removed entry: {autostart.get_autostart_path()}\n")
            print("Jarvis will no longer start automatically at login.")
            print("To re-enable anytime: python main.py --enable-autostart\n")
            sys.exit(0)
        else:
            print(f"[Error] Failed to disable autostart: {message}", file=sys.stderr)
            print(autostart.get_manual_instructions(), file=sys.stderr)
            sys.exit(1)

    logging.info(f"=== {assistant_name} session started ===")

    brain = Brain()

    # Check optional startup briefing (Requirement 5)
    from state import set_last_briefing_date, should_run_startup_briefing
    if should_run_startup_briefing():
        set_last_briefing_date()
        print(f"[{assistant_name}] Preparing first-launch daily briefing for today...")
        trigger_daily_briefing(
            brain=brain,
            assistant_name=assistant_name,
            enable_speak=not args.no_speak if (args.listen or args.voice) else False,
            confirmation_mode="voice" if (args.listen or args.voice) and not args.no_speak else "text",
        )

    # --tray implies hands-free listening mode
    if args.tray and not args.listen and not args.voice:
        args.listen = True

    try:
        if args.listen:
            # Verify audio and network readiness before starting (Requirement 4)
            wait_for_system_readiness()

            assistant = JarvisAssistant(
                brain=brain,
                assistant_name=assistant_name,
                enable_speak=not args.no_speak,
                debug=args.debug,
            )

            # Global hotkey listener (Ctrl+Alt+J)
            hotkey_combo = getattr(config, "HOTKEY_COMBINATION", "<ctrl>+<alt>+j")
            hotkey_listener = create_hotkey_listener(
                hotkey_combo,
                on_activate=assistant.on_hotkey,
            )
            if hotkey_listener:
                try:
                    hotkey_listener.start()
                except Exception as hk_err:
                    print(f"[Hotkey Warning] Could not start global hotkey listener: {hk_err}")
                    hotkey_listener = None

            if args.tray:
                tray = JarvisTray(assistant=assistant)
                assistant.set_tray(tray)

                # Start assistant logic in background thread
                assistant_thread = threading.Thread(
                    target=assistant.run,
                    daemon=True,
                    name="JarvisAssistantThread",
                )
                assistant_thread.start()

                print(f"[{assistant_name}] System tray active.")
                print(f"[{assistant_name}] Global hotkey registered: {hotkey_combo}")
                print(f"[{assistant_name}] Access controls by right-clicking the system tray icon.\n")

                try:
                    # Run tray icon in the main thread (required on macOS, safe everywhere)
                    tray.run()
                except (KeyboardInterrupt, SystemExit):
                    print(f"\n[{assistant_name}] Shutting down from console interrupt...")
                finally:
                    assistant.stop()
                    if hotkey_listener and hotkey_listener.is_alive():
                        hotkey_listener.stop()
                    tray.stop()
                    if assistant_thread.is_alive():
                        assistant_thread.join(timeout=2.0)
            else:
                print(f"[{assistant_name}] Global hotkey registered: {hotkey_combo}\n")
                try:
                    assistant.run()
                finally:
                    if hotkey_listener and hotkey_listener.is_alive():
                        hotkey_listener.stop()

        elif args.voice:
            run_voice_mode(
                brain,
                assistant_name,
                enable_speak=not args.no_speak,
                debug=args.debug,
            )
        else:
            run_text_mode(brain, assistant_name)

    except (KeyboardInterrupt, EOFError):
        print(f"\n\n{assistant_name}: Goodbye! Have a great day.")
    finally:
        logging.info(f"=== {assistant_name} session ended ===")


if __name__ == "__main__":
    main()
