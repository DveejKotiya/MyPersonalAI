"""Configuration module for Jarvis (Phase 1).

Loads environment variables from .env and exposes project settings:
- ASSISTANT_NAME: Name of the AI assistant (default: "Jarvis")
- MODEL_NAME: The model identifier (default: "llama3.1")
- API_KEY: The API key or local authentication token
- OLLAMA_HOST: URL for local Ollama instance (default: "http://localhost:11434")

Validates configuration and provides beginner-friendly guidance if
critical settings are missing.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Locate project root and .env file
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

# Load variables from .env file into os.environ
load_dotenv(dotenv_path=ENV_FILE)

# Expose required settings
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Jarvis")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "90.0"))
API_KEY = os.getenv("API_KEY")

# Whisper Speech-to-Text settings (Phase 2)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

# Text-to-Speech (TTS) settings (Phase 2)
TTS_VOICE = os.getenv("TTS_VOICE", "en-GB-RyanNeural")

# Wake Word Detection & System Tray settings (Phase 4)
WAKE_THRESHOLD = float(os.getenv("WAKE_THRESHOLD", "0.5"))
INTERRUPT_THRESHOLD = float(os.getenv("INTERRUPT_THRESHOLD", "0.8"))
WAKE_WORD_MODEL = os.getenv("WAKE_WORD_MODEL", "hey_jarvis")
# Global hotkey for manual wake (idle) and interruption (speaking).
# Formats accepted: "<ctrl>+<alt>+j", "Ctrl+Alt+J", "<ctrl>+<shift>+j", etc.
HOTKEY_COMBINATION = os.getenv("HOTKEY_COMBINATION", "<ctrl>+<alt>+j")

# Privacy & Audio Retention (Strictly False by default)
# When False: zero audio is ever written to disk. Audio stays in RAM and is discarded after STT.
# When True: saves temporary debug WAV recordings to logs/debug_audio/ for troubleshooting.
SAVE_DEBUG_AUDIO = os.getenv("SAVE_DEBUG_AUDIO", "false").lower() in ("true", "1")

# Daily Briefing settings (Phase 5)
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "New York")
BRIEFING_ON_STARTUP = os.getenv("BRIEFING_ON_STARTUP", "false").lower() in ("true", "1", "yes")

# Cost & Speed Controls (Phase 5)
DAILY_TOKEN_LIMIT = int(os.getenv("DAILY_TOKEN_LIMIT", "200000"))
PRICE_PER_M_INPUT_TOKENS = float(os.getenv("PRICE_PER_M_INPUT_TOKENS", "0.15"))
PRICE_PER_M_OUTPUT_TOKENS = float(os.getenv("PRICE_PER_M_OUTPUT_TOKENS", "0.60"))
MAX_TOKENS_VOICE = int(os.getenv("MAX_TOKENS_VOICE", "300"))
MAX_TOKENS_TEXT = int(os.getenv("MAX_TOKENS_TEXT", "800"))

# State Machine & Watchdog Settings
WATCHDOG_TIMEOUT_SEC = float(os.getenv("WATCHDOG_TIMEOUT_SEC", "60.0"))
FOLLOW_UP_TIMEOUT_SEC = float(os.getenv("FOLLOW_UP_TIMEOUT_SEC", "6.0"))

# Long-Term Memory Limits (Phase 5)
MAX_MEMORIES_CAP = int(os.getenv("MAX_MEMORIES_CAP", "500"))
MAX_MEMORY_TEXT_LENGTH = int(os.getenv("MAX_MEMORY_TEXT_LENGTH", "300"))

# Action Log Rotation Settings (Phase 5)
ACTIONS_LOG_MAX_BYTES = int(os.getenv("ACTIONS_LOG_MAX_BYTES", "1048576"))  # 1 MB
ACTIONS_LOG_BACKUP_COUNT = int(os.getenv("ACTIONS_LOG_BACKUP_COUNT", "3"))

# Audio Recording & VAD Settings (Phase 2)
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_SILENCE_THRESHOLD = float(os.getenv("AUDIO_SILENCE_THRESHOLD", "0.015"))
AUDIO_SILENCE_LIMIT_SEC = float(os.getenv("AUDIO_SILENCE_LIMIT_SEC", "1.5"))
AUDIO_MAX_RECORDING_SEC = float(os.getenv("AUDIO_MAX_RECORDING_SEC", "30.0"))
AUDIO_INITIAL_SILENCE_TIMEOUT = float(os.getenv("AUDIO_INITIAL_SILENCE_TIMEOUT", "8.0"))

# Vision & Screen Analysis Settings (Phase 6)
# When False (default), zero screenshot bytes are written to disk (in-memory only).
KEEP_SCREENSHOTS = os.getenv("KEEP_SCREENSHOTS", "false").lower() in ("true", "1", "yes")
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "llama3.2-vision")
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "1568"))

# Default keywords in window titles that trigger an automatic privacy block
_DEFAULT_SENSITIVE_KEYWORDS = (
    "password,1password,bitwarden,keepass,lastpass,bank,banking,paypal,"
    "credit,checkout,login,signin,auth,keychain,vault,private,incognito"
)
raw_keywords = os.getenv("SENSITIVE_WINDOW_KEYWORDS", _DEFAULT_SENSITIVE_KEYWORDS)
SENSITIVE_WINDOW_KEYWORDS: list[str] = [
    k.strip().lower() for k in raw_keywords.split(",") if k.strip()
]

# Semantic Memory & Embedding Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.35"))

# Whitelist of allowed applications for the open_app tool (Windows commands)
# Maps application identifier to a command list executed safely via subprocess without shell=True
APPS: dict[str, list[str]] = {
    "chrome": ["cmd.exe", "/c", "start", "", "chrome"],
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
    "spotify": ["cmd.exe", "/c", "start", "", "spotify:"],
}

# Google Calendar (Read-Only) Settings
SECRETS_DIR = (BASE_DIR / "data" / "secrets").resolve()
CALENDAR_CREDENTIALS_FILE = Path(os.getenv("CALENDAR_CREDENTIALS_FILE", str(SECRETS_DIR / "credentials.json"))).resolve()
CALENDAR_TOKEN_FILE = Path(os.getenv("CALENDAR_TOKEN_FILE", str(SECRETS_DIR / "token.json"))).resolve()
CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
CALENDAR_WORK_HOURS_START = int(os.getenv("CALENDAR_WORK_HOURS_START", "9"))
CALENDAR_WORK_HOURS_END = int(os.getenv("CALENDAR_WORK_HOURS_END", "18"))

# File operations sandbox directory (defaults to user's Documents folder)
DEFAULT_DOCUMENTS_DIR = Path.home() / "Documents"
ALLOWED_FOLDER = Path(os.getenv("ALLOWED_FOLDER", str(DEFAULT_DOCUMENTS_DIR))).resolve()


def _print_missing_key_error() -> None:
    """Print a helpful, beginner-friendly error message when API_KEY is missing."""
    border = "=" * 68
    print(f"\n{border}", file=sys.stderr)
    print("  [ERROR] Configuration Incomplete: API_KEY is missing!", file=sys.stderr)
    print(f"{border}\n", file=sys.stderr)
    print("Jarvis requires configuration from a '.env' file to get started.", file=sys.stderr)
    print("\nFollow these simple steps to fix this:\n", file=sys.stderr)
    print("  1. Create your '.env' file from the template:", file=sys.stderr)
    print("       PowerShell:     Copy-Item .env.example .env", file=sys.stderr)
    print("       Command Prompt: copy .env.example .env\n", file=sys.stderr)
    print("  2. Open '.env' in your editor and set API_KEY:", file=sys.stderr)
    print("       Since you are running Ollama locally, you can set:", file=sys.stderr)
    print("       API_KEY=local\n", file=sys.stderr)
    print("  3. Make sure Ollama is installed and running:", file=sys.stderr)
    print("       ollama run llama3.1\n", file=sys.stderr)
    print("  4. Run your application again:", file=sys.stderr)
    print("       python main.py\n", file=sys.stderr)
    print(f"{border}\n", file=sys.stderr)


# Validate settings
if not API_KEY or not API_KEY.strip():
    _print_missing_key_error()
    sys.exit(1)
