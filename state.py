"""Persistent application state manager for Jarvis (Phase 5).

Stores lightweight key-value state (such as the last daily briefing date)
in data/state.json. Automatically creates the data/ folder and handles
thread-safe file access.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import config

_STATE_LOCK = threading.Lock()


def _get_state_path() -> Path:
    """Return the absolute path to data/state.json, creating parent directory if needed."""
    base_dir = getattr(config, "BASE_DIR", Path(__file__).resolve().parent)
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "state.json"


def load_state() -> dict[str, Any]:
    """Read the full state dictionary from data/state.json.

    Returns:
        Dictionary of stored state keys, or an empty dict if the file does not exist.
    """
    with _STATE_LOCK:
        path = _get_state_path()
        if not path.is_file():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def save_state(state: dict[str, Any]) -> None:
    """Write the state dictionary to data/state.json.

    Args:
        state: State dictionary to persist.
    """
    with _STATE_LOCK:
        path = _get_state_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as err:
            print(f"[State Warning] Could not save state to {path}: {err}")


def get_last_briefing_date() -> str | None:
    """Get the ISO date string (YYYY-MM-DD) when the daily briefing was last given."""
    state = load_state()
    return state.get("last_briefing_date")


def set_last_briefing_date(date_str: str | None = None) -> None:
    """Record that the daily briefing was given on the specified date (defaults to today)."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    state = load_state()
    state["last_briefing_date"] = date_str
    save_state(state)


def should_run_startup_briefing() -> bool:
    """Determine whether a startup briefing should be presented on this run.

    Returns:
        True if BRIEFING_ON_STARTUP is enabled in config.py and no briefing has
        been recorded yet for today's date.
    """
    if not getattr(config, "BRIEFING_ON_STARTUP", False):
        return False

    today = datetime.now().strftime("%Y-%m-%d")
    last_date = get_last_briefing_date()
    return last_date != today


def get_token_limit_override_date() -> str | None:
    """Get the ISO date string (YYYY-MM-DD) for which the daily token limit was overridden."""
    state = load_state()
    return state.get("token_limit_override_date")


def set_token_limit_override_date(date_str: str | None = None) -> None:
    """Record that the daily token limit has been overridden for the specified date."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    state = load_state()
    state["token_limit_override_date"] = date_str
    save_state(state)


