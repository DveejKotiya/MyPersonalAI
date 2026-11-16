"""System Tray Icon and Desktop Integration for Jarvis (Phase 4).

Provides a system tray icon with pystray and Pillow, dynamic state colors,
interactive menus, global hotkey management, and desktop notifications.

State Colors:
- Grey:   Paused (Microphone released for Zoom/Teams)
- Blue:   Idle / Waiting for wake word ('Hey Jarvis')
- Green:  Actively listening to user speech
- Orange: Thinking (AI reasoning) or Speaking (TTS output)
"""

import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pystray
from PIL import Image, ImageDraw
from pynput import keyboard
from pynput.keyboard import HotKey
from pystray import MenuItem as item

import config

# Color palette definition for assistant states (RGB)
COLORS = {
    "grey": (140, 140, 140),     # Paused / mic released
    "blue": (30, 144, 255),      # Idle / listening for wake word
    "green": (46, 204, 113),     # Actively listening to speech
    "orange": (243, 156, 18),    # Thinking / Speaking
}


def create_tray_icon(color_name: str = "blue", size: int = 64) -> Image.Image:
    """Generate a clean, high-DPI antialiased circular tray icon in-memory.

    No external image files needed. Renders at 4x resolution and downscales
    using Lanczos filtering for smooth edges on all desktop display scales.

    Args:
        color_name: One of 'grey', 'blue', 'green', or 'orange'.
        size: Target icon width and height in pixels (default: 64).

    Returns:
        A PIL Image.Image instance.
    """
    rgb = COLORS.get(color_name, COLORS["blue"])
    scale = 4
    canvas_size = size * scale

    image = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Outer solid circular body
    padding = 4 * scale
    draw.ellipse(
        (padding, padding, canvas_size - padding, canvas_size - padding),
        fill=rgb + (255,),
        outline=(255, 255, 255, 210),
        width=3 * scale,
    )

    # Inner bright accent center
    inner_padding = 18 * scale
    draw.ellipse(
        (
            inner_padding,
            inner_padding,
            canvas_size - inner_padding,
            canvas_size - inner_padding,
        ),
        fill=(255, 255, 255, 230),
    )

    return image.resize((size, size), Image.Resampling.LANCZOS)


def open_file_in_editor(file_path: Path | str) -> None:
    """Open a file with the system default text editor across Windows, macOS, and Linux.

    Args:
        file_path: Path to the target file.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    try:
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as err:
        print(f"[Tray Error] Could not open file {path}: {err}", file=sys.stderr)


def normalize_hotkey(combo_str: str) -> str:
    """Normalize user-friendly hotkey strings into pynput format.

    Examples:
        'Ctrl+Alt+J'      -> '<ctrl>+<alt>+j'
        '<ctrl>+<alt>+j'  -> '<ctrl>+<alt>+j'
        'ctrl+shift+j'    -> '<ctrl>+<shift>+j'
        'ctrl+alt+cmd+j'  -> '<ctrl>+<alt>+<cmd>+j'

    Args:
        combo_str: Hotkey string from configuration.

    Returns:
        Normalized string compatible with pynput HotKey parser.
    """
    if not combo_str or not combo_str.strip():
        return "<ctrl>+<alt>+j"

    parts = combo_str.split("+")
    modifiers = {"ctrl", "alt", "shift", "cmd", "super", "win", "windows", "option"}
    normalized = []

    for part in parts:
        p = part.strip().lower()
        if p.startswith("<") and p.endswith(">"):
            normalized.append(p)
        elif p in modifiers:
            if p in ("win", "windows", "super"):
                normalized.append("<cmd>")
            elif p == "option":
                normalized.append("<alt>")
            else:
                normalized.append(f"<{p}>")
        else:
            normalized.append(p)

    return "+".join(normalized)


def create_hotkey_listener(
    hotkey_str: str,
    on_activate: Callable[[], None],
) -> keyboard.GlobalHotKeys | None:
    """Create a pynput GlobalHotKeys listener with validation and graceful fallbacks.

    Args:
        hotkey_str: Hotkey combination string (e.g. '<ctrl>+<alt>+j' or 'Ctrl+Alt+J').
        on_activate: Callback invoked when the hotkey is triggered.

    Returns:
        A pynput GlobalHotKeys listener instance ready to be started, or None on error.
    """
    normalized = normalize_hotkey(hotkey_str)

    try:
        HotKey.parse(normalized)
    except Exception as parse_err:
        print(
            f"[Hotkey Warning] Invalid hotkey combination '{hotkey_str}': {parse_err}. "
            f"Falling back to '<ctrl>+<alt>+j'.",
            file=sys.stderr,
        )
        normalized = "<ctrl>+<alt>+j"

    try:
        listener = keyboard.GlobalHotKeys({normalized: on_activate})
        return listener
    except Exception as err:
        print(f"[Hotkey Error] Failed to register global hotkey ({err}).", file=sys.stderr)
        return None


class JarvisTray:
    """Manages the system tray icon, context menu, and desktop notifications."""

    def __init__(self, assistant: Any, on_quit: Callable[[], None] | None = None):
        """Initialize the system tray icon.

        Args:
            assistant: Reference to the JarvisAssistant instance.
            on_quit: Optional callback executed when user clicks Quit in the menu.
        """
        self.assistant = assistant
        self.on_quit_callback = on_quit
        self._current_color = "blue"
        self._lock = threading.Lock()

        # Build context menu strictly following requirements:
        # 1. Pause/Resume listening
        # 2. Show recent activity
        # 3. Open log file
        # 4. Open notes
        # 5. Toggle speech on/off
        # 6. Quit
        menu = pystray.Menu(
            item(self._get_pause_label, self._on_toggle_pause),
            item("Show Recent Activity", self._on_show_recent_activity),
            item("Open Log File", self._on_open_log),
            item("Open Notes", self._on_open_notes),
            item(self._get_speech_label, self._on_toggle_speech),
            pystray.Menu.SEPARATOR,
            item("Quit", self._on_quit),
        )

        initial_image = create_tray_icon("blue")
        assistant_name = getattr(self.assistant, "assistant_name", "Jarvis")
        self.icon = pystray.Icon(
            name=assistant_name,
            icon=initial_image,
            title=f"{assistant_name} - Online",
            menu=menu,
        )

    # --------------------------------------------------------------------------
    # Menu Label Callbacks
    # --------------------------------------------------------------------------

    def _get_pause_label(self, item_ref: Any) -> str:
        """Dynamic text for the Pause / Resume menu item."""
        if getattr(self.assistant, "is_manually_paused", False):
            return "Resume Listening"
        return "Pause Listening (Release Mic)"

    def _get_speech_label(self, item_ref: Any) -> str:
        """Dynamic text for the Speech Toggle menu item."""
        if getattr(self.assistant, "enable_speak", True):
            return "Toggle Speech (Voice: ON)"
        return "Toggle Speech (Voice: OFF)"

    # --------------------------------------------------------------------------
    # Menu Action Callbacks
    # --------------------------------------------------------------------------

    def _on_toggle_pause(self, icon_ref: Any, item_ref: Any) -> None:
        """Toggle pause state to fully release or re-acquire the microphone."""
        if getattr(self.assistant, "is_manually_paused", False):
            self.assistant.resume_listening()
        else:
            self.assistant.pause_listening()
        self.icon.update_menu()

    def _on_toggle_speech(self, icon_ref: Any, item_ref: Any) -> None:
        """Toggle spoken audio replies on/off."""
        self.assistant.enable_speak = not self.assistant.enable_speak
        from tools import set_confirmation_mode
        set_confirmation_mode("voice" if self.assistant.enable_speak else "text")

        status = "enabled" if self.assistant.enable_speak else "disabled (quiet mode)"
        self.notify(
            f"Speech {status.capitalize()}",
            f"Voice replies are now {status}.",
        )
        self.icon.update_menu()

    def _on_show_recent_activity(self, icon_ref: Any, item_ref: Any) -> None:
        """Display a window or notification with the last 10 logged actions (Requirement 4)."""
        import json
        base_dir = getattr(config, "BASE_DIR", Path("."))
        log_path = base_dir / "logs" / "actions.log"

        entries = []
        if log_path.is_file():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line_clean = line.strip()
                        if line_clean:
                            try:
                                entries.append(json.loads(line_clean))
                            except Exception:
                                continue
            except Exception as err:
                print(f"[Tray Error] Failed reading log file: {err}", file=sys.stderr)

        recent_entries = entries[-10:] if entries else []

        if not recent_entries:
            display_text = "No recent actions recorded yet in logs/actions.log."
        else:
            lines = [f"=== Jarvis Recent Activity (Last {len(recent_entries)} Actions) ==="]
            for i, e in enumerate(recent_entries, 1):
                ts = e.get("timestamp", "")[:19].replace("T", " ")
                tool = e.get("tool", "unknown")
                args = str(e.get("arguments", ""))
                if len(args) > 50:
                    args = args[:47] + "..."
                status = e.get("status", "UNKNOWN")
                lines.append(f"{i}. [{ts}] {tool}({args}) -> [{status}]")
            display_text = "\n\n".join(lines)

        def _open_window():
            try:
                import tkinter as tk
                from tkinter import scrolledtext

                root = tk.Tk()
                root.title("Jarvis - Recent Activity")
                root.geometry("700x420")
                root.minsize(450, 250)
                root.configure(bg="#1e1e1e")

                # Bring window to front
                root.lift()
                root.attributes("-topmost", True)
                root.after_idle(root.attributes, "-topmost", False)

                lbl = tk.Label(
                    root,
                    text="Recent Actions (Audit Log)",
                    font=("Segoe UI", 12, "bold"),
                    bg="#1e1e1e",
                    fg="#ffffff",
                    padx=10,
                    pady=8,
                )
                lbl.pack(anchor="w")

                txt = scrolledtext.ScrolledText(
                    root,
                    wrap=tk.WORD,
                    font=("Consolas", 10),
                    bg="#252526",
                    fg="#d4d4d4",
                    insertbackground="white",
                    padx=10,
                    pady=10,
                )
                txt.insert(tk.END, display_text)
                txt.configure(state="disabled")
                txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

                btn_frame = tk.Frame(root, bg="#1e1e1e")
                btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

                close_btn = tk.Button(
                    btn_frame,
                    text="Close",
                    font=("Segoe UI", 9),
                    command=root.destroy,
                    width=10,
                )
                close_btn.pack(side=tk.RIGHT)

                root.mainloop()
            except Exception:
                # Fallback to desktop tray notification if GUI window fails
                preview = "\n".join(display_text.splitlines()[:4])
                self.notify("Jarvis Recent Activity", preview)

        # Launch window in background thread so tray loop is never blocked
        threading.Thread(target=_open_window, daemon=True, name="RecentActivityWindowThread").start()

    def _on_open_log(self, icon_ref: Any, item_ref: Any) -> None:
        """Open the audit actions log file in system default editor."""
        base_dir = getattr(config, "BASE_DIR", Path("."))
        log_path = base_dir / "logs" / "actions.log"
        open_file_in_editor(log_path)

    def _on_open_notes(self, icon_ref: Any, item_ref: Any) -> None:
        """Open the user's notes file in system default editor."""
        base_dir = getattr(config, "BASE_DIR", Path("."))
        notes_path = base_dir / "notes.md"
        open_file_in_editor(notes_path)

    def _on_quit(self, icon_ref: Any, item_ref: Any) -> None:
        """Exit the assistant and close the tray icon cleanly."""
        name = getattr(self.assistant, "assistant_name", "Jarvis")
        print(f"\n{name}: Exiting from system tray menu...")
        self.stop()
        if self.on_quit_callback:
            self.on_quit_callback()
        else:
            self.assistant.stop()

    # --------------------------------------------------------------------------
    # Tray State & Notification Methods
    # --------------------------------------------------------------------------

    def update_state(self, state_name: str, is_paused: bool = False) -> None:
        """Update tray icon color and tooltip based on assistant state.

        Colors:
        - grey:   Paused (mic released for Zoom/Teams)
        - blue:   IDLE (listening for wake word)
        - green:  LISTENING (recording user speech)
        - orange: THINKING or SPEAKING

        Args:
            state_name: Current state name ('IDLE', 'LISTENING', 'THINKING', 'SPEAKING').
            is_paused: True if user manually paused the assistant.
        """
        with self._lock:
            name = getattr(self.assistant, "assistant_name", "Jarvis")
            if is_paused:
                color = "grey"
                title = f"{name} - Paused (Mic Released)"
            elif state_name == "IDLE":
                color = "blue"
                title = f"{name} - Waiting for 'Hey Jarvis'"
            elif state_name == "LISTENING":
                color = "green"
                title = f"{name} - Listening to you..."
            elif state_name in ("THINKING", "SPEAKING"):
                color = "orange"
                title = f"{name} - {state_name.capitalize()}..."
            else:
                color = "blue"
                title = f"{name} - Online"

            if color != self._current_color or self.icon.title != title:
                self._current_color = color
                try:
                    self.icon.icon = create_tray_icon(color)
                    self.icon.title = title
                except Exception:
                    pass

    def notify(self, title: str, message: str) -> None:
        """Show a desktop balloon/toast notification via the system tray icon."""
        try:
            self.icon.notify(message=message, title=title)
        except Exception:
            # Notifications may be disabled by the OS notification center
            pass

    def run(self) -> None:
        """Run the system tray icon loop in the main thread (blocking)."""
        self.icon.run()

    def stop(self) -> None:
        """Stop and remove the system tray icon."""
        try:
            self.icon.stop()
        except Exception:
            pass
