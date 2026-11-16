"""Text-to-Speech (TTS) module for Jarvis (Phase 2 & Phase 4).

This module handles speech synthesis for Jarvis:
1. Primary engine: edge-tts (high-quality natural neural voice, e.g. en-GB-RyanNeural).
2. Fallback engine: pyttsx3 (100% offline, uses Windows SAPI5 / system TTS).
3. Sentence-by-sentence streaming playback: splits long responses into individual
   sentences and pipelines synthesis/playback so replies start speaking quickly.
4. Immediate interruption: stop_speaking() halts playback within ~30ms and cleans
   up all temporary audio files.
5. Text preprocessing: strips markdown symbols (*, #, `, _, etc.) before speaking.
"""

import asyncio
import concurrent.futures
import os
import queue
import re
import sys
import tempfile
import threading
import time

import config

# Suppress pygame welcome banner
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# Module-level singletons for audio mixer, offline TTS engine, and speech synchronization lock
_mixer_initialized = False
_pyttsx3_engine = None
SPEECH_LOCK = threading.Lock()

# Interruption control and active temp files tracker
_stop_event = threading.Event()
_active_temp_files: set[str] = set()
_temp_lock = threading.Lock()


def strip_markdown(text: str) -> str:
    """Remove markdown syntax and formatting so text reads naturally when spoken.

    Removes:
    - Code blocks: ```code``` and inline code: `code`
    - Links: [anchor text](url) -> anchor text
    - Images: ![alt](url) -> ""
    - Bold / Italic: **bold**, *italic*, __bold__, _italic_
    - Strikethrough: ~~text~~
    - Headers: # Header -> Header
    - Blockquotes: > Quote -> Quote
    - Bullet points: - item, * item, + item -> item
    - Extra whitespace and newlines

    Args:
        text: Raw text string containing potential markdown formatting.

    Returns:
        Cleaned plain text suitable for speech synthesis.
    """
    if not text:
        return ""

    # 1. Remove multi-line code blocks ```...```
    text = re.sub(r"```[\s\S]*?```", "", text)

    # 2. Convert inline code `code` to code
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # 3. Convert markdown links [text](url) to just text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # 4. Remove image tags ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", text)

    # 5. Remove bold & italics: ***text***, **text**, *text*, ___text___, __text__, _text_
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)

    # 6. Remove strikethrough: ~~text~~
    text = re.sub(r"~~(.*?)~~", r"\1", text)

    # 7. Remove headers: #, ##, ###, etc. at start of lines
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)

    # 8. Remove blockquotes: > at start of lines
    text = re.sub(r"^\s*>\s*", "", text, flags=re.MULTILINE)

    # 9. Remove bullet list markers: -, *, + at start of lines
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)

    # 10. Replace newlines with spaces and condense multiple spaces
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences for sentence-by-sentence TTS synthesis.

    Allows long replies to begin playing quickly and enables interruptions
    to take effect within approximately one second.

    Args:
        text: Preprocessed text without markdown.

    Returns:
        List of individual sentence strings.
    """
    if not text or not text.strip():
        return []

    # Split on period, exclamation, or question mark followed by whitespace, or newlines
    raw_chunks = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    sentences = [chunk.strip() for chunk in raw_chunks if chunk.strip()]
    return sentences if sentences else [text.strip()]


def _ensure_mixer_init() -> None:
    """Initialize pygame mixer once for audio playback."""
    global _mixer_initialized
    if not _mixer_initialized:
        import pygame

        pygame.mixer.init()
        _mixer_initialized = True


def _cleanup_temp_file(path: str | None) -> None:
    """Safely remove a temporary audio file and unregister it."""
    if not path:
        return
    with _temp_lock:
        _active_temp_files.discard(path)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def stop_speaking() -> None:
    """Halt any ongoing speech playback immediately and clean up temporary audio files.

    Thread-safe method called when an interruption (wake word or hotkey) is detected.
    """
    _stop_event.set()

    # 1. Stop and unload hardware playback in pygame
    try:
        import pygame

        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
    except Exception:
        pass

    # 2. Stop offline fallback pyttsx3 engine if currently running
    global _pyttsx3_engine
    if _pyttsx3_engine is not None:
        try:
            _pyttsx3_engine.stop()
        except Exception:
            pass

    # 3. Clean up all registered temporary audio files on disk
    with _temp_lock:
        for file_path in list(_active_temp_files):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass
            _active_temp_files.discard(file_path)


def _play_audio_file(file_path: str) -> bool:
    """Play an audio file using pygame.mixer with immediate interruption checking.

    Returns:
        True if played to completion, False if interrupted or stopped.
    """
    import pygame

    _ensure_mixer_init()
    if _stop_event.is_set():
        return False

    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    except Exception:
        return False

    clock = pygame.time.Clock()
    interrupted = False
    try:
        while pygame.mixer.music.get_busy():
            if _stop_event.is_set():
                interrupted = True
                pygame.mixer.music.stop()
                break
            clock.tick(30)  # Check 30 times/sec (~33ms latency)
    finally:
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass

    return not interrupted


async def _generate_edge_tts(text: str, voice: str, output_path: str) -> None:
    """Generate speech audio file asynchronously using edge-tts."""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def _synthesize_edge_tts(sentence: str, voice: str) -> str | None:
    """Synthesize a single sentence to a temporary MP3 file using edge-tts.

    Returns:
        Path to the temporary MP3 file, or None if failed or stopped.
    """
    if _stop_event.is_set():
        return None

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    with _temp_lock:
        _active_temp_files.add(temp_path)

    try:
        if _stop_event.is_set():
            _cleanup_temp_file(temp_path)
            return None

        asyncio.run(_generate_edge_tts(sentence, voice, temp_path))

        if _stop_event.is_set():
            _cleanup_temp_file(temp_path)
            return None

        return temp_path
    except Exception:
        _cleanup_temp_file(temp_path)
        return None


def _get_pyttsx3_engine():
    """Lazily initialize and return the offline pyttsx3 engine."""
    global _pyttsx3_engine
    if _pyttsx3_engine is None:
        import pyttsx3

        _pyttsx3_engine = pyttsx3.init()
        try:
            _pyttsx3_engine.setProperty("rate", 175)
        except Exception:
            pass
    return _pyttsx3_engine


def speak_fallback(text: str) -> float:
    """Speak text using the offline pyttsx3 fallback engine with stop checks.

    Args:
        text: Text to speak.

    Returns:
        Synthesis prep time in seconds.
    """
    cleaned = strip_markdown(text)
    if not cleaned or _stop_event.is_set():
        return 0.0

    print("[TTS] Using offline fallback engine (pyttsx3)...")
    t0 = time.perf_counter()
    try:
        engine = _get_pyttsx3_engine()
        engine.say(cleaned)
        synthesis_time = time.perf_counter() - t0
        engine.runAndWait()
        return synthesis_time
    except Exception as err:
        print(f"[TTS Error] Fallback engine failed: {err}", file=sys.stderr)
        return 0.0


class SentenceSpeaker:
    """Streams sentences to speech synthesis and playback in real time as they arrive.

    Allows the assistant to begin speaking the first sentence immediately while
    subsequent sentences are still being generated by the LLM stream.
    Tracks time_to_first_spoken_word for latency benchmarking.
    """

    def __init__(self, voice: str | None = None) -> None:
        self.voice = voice or getattr(config, "TTS_VOICE", "en-GB-RyanNeural")
        self.queue: queue.Queue[str | None] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.time_to_first_spoken_word: float | None = None
        self.t0 = time.perf_counter()
        self._stopped = False
        self._done_event = threading.Event()
        self.start()

    def start(self) -> None:
        """Start the background consumer thread."""
        _stop_event.clear()
        self.worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="SentenceSpeakerWorker",
        )
        self.worker_thread.start()

    def feed_sentence(self, sentence: str) -> None:
        """Feed a completed sentence into the streaming synthesis queue."""
        if self._stopped or _stop_event.is_set():
            return
        cleaned = strip_markdown(sentence).strip()
        if cleaned:
            self.queue.put(cleaned)

    def finish(self) -> None:
        """Signal that no more sentences will arrive from the generator."""
        if not self._stopped:
            self.queue.put(None)

    def wait_until_done(self, timeout: float | None = None) -> None:
        """Block until all queued sentences have finished playing or stopped."""
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)

    def stop(self) -> None:
        """Stop streaming immediately and discard remaining sentences."""
        self._stopped = True
        stop_speaking()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break
        self.queue.put(None)

    def _worker(self) -> None:
        """Background consumer worker synthesizing and playing sentences in FIFO order."""
        with SPEECH_LOCK:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            next_future = None
            pending_sentence = None

            try:
                while not _stop_event.is_set():
                    try:
                        if pending_sentence is None:
                            sentence = self.queue.get()
                            if sentence is None:
                                break
                        else:
                            sentence = pending_sentence
                            pending_sentence = None

                        if _stop_event.is_set():
                            break

                        # Synthesize this sentence if not already scheduled
                        if next_future is None:
                            current_path = _synthesize_edge_tts(sentence, self.voice)
                        else:
                            try:
                                current_path = next_future.result(timeout=12.0)
                            except Exception:
                                current_path = None
                            next_future = None

                        if _stop_event.is_set():
                            _cleanup_temp_file(current_path)
                            break

                        # Lookahead: if another sentence is ready in the queue, synthesize it
                        if not self.queue.empty():
                            try:
                                nxt = self.queue.get_nowait()
                                if nxt is not None:
                                    pending_sentence = nxt
                                    next_future = executor.submit(
                                        _synthesize_edge_tts, nxt, self.voice
                                    )
                                else:
                                    # Put termination signal back into queue so the outer loop terminates
                                    self.queue.put(None)
                            except queue.Empty:
                                pass

                        if current_path is None:
                            # Edge-tts failed, fallback to offline pyttsx3
                            if not _stop_event.is_set():
                                if self.time_to_first_spoken_word is None:
                                    self.time_to_first_spoken_word = (
                                        time.perf_counter() - self.t0
                                    )
                                speak_fallback(sentence)
                            continue

                        # Mark time-to-first-spoken-word right before playing audio
                        if self.time_to_first_spoken_word is None:
                            self.time_to_first_spoken_word = (
                                time.perf_counter() - self.t0
                            )

                        try:
                            played = _play_audio_file(current_path)
                            if not played or _stop_event.is_set():
                                break
                        finally:
                            _cleanup_temp_file(current_path)

                    except Exception as err:
                        print(
                            f"[TTS Error] Streaming playback exception: {err}",
                            file=sys.stderr,
                        )
                        break
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
                stop_speaking()
                self._done_event.set()


def speak(text: str) -> float | None:
    """Speak the given text sentence-by-sentence with immediate interruption support.

    Behavior:
    1. Splits long replies into natural sentences.
    2. Uses SentenceSpeaker to pipeline synthesis and playback seamlessly.
    3. Checks _stop_event throughout playback so calling stop_speaking() halts
       sound immediately (< 50ms) and cleans up all temporary audio files.
    4. Automatically falls back to pyttsx3 if edge-tts is unavailable.

    Args:
        text: Text to speak.

    Returns:
        Time taken in seconds before first spoken word began, or None if text was empty.
    """
    cleaned = strip_markdown(text)
    if not cleaned:
        return None

    sentences = split_into_sentences(cleaned)
    if not sentences:
        return None

    speaker = SentenceSpeaker()
    for s in sentences:
        speaker.feed_sentence(s)
    speaker.finish()
    speaker.wait_until_done()

    return speaker.time_to_first_spoken_word or (time.perf_counter() - speaker.t0)
