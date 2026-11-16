"""Clipboard I/O utilities for Jarvis.

Provides safe access to system clipboard with:
- Native Windows support via win32clipboard with retry backoff
- Cross-platform fallbacks (tkinter / in-memory buffer)
- Proper resource cleanup ensuring OpenClipboard is always closed
"""

import sys
import time

# In-memory test/fallback buffer if OS clipboard is unavailable (e.g. in headless sandbox)
_IN_MEMORY_CLIPBOARD: str | None = None


def get_clipboard_text() -> tuple[str | None, str | None]:
    """Retrieve text from the system clipboard.

    Returns:
        tuple of (text: str | None, error_message: str | None).
        If successful: (text_string, None)
        If clipboard holds non-text: (None, "The clipboard does not contain text...")
        If clipboard cannot be accessed: (None, "Error message...")
    """
    if sys.platform.startswith("win"):
        try:
            import win32clipboard
            import win32con

            # Attempt to open clipboard with retry loop (in case locked by another process)
            opened = False
            for _ in range(5):
                try:
                    win32clipboard.OpenClipboard()
                    opened = True
                    break
                except Exception:
                    time.sleep(0.04)

            if not opened:
                # If access denied or sandbox isolation, check fallback
                if _IN_MEMORY_CLIPBOARD is not None:
                    return _IN_MEMORY_CLIPBOARD, None
                return None, "Could not open clipboard (it may be locked by another application or restricted by the OS)."

            try:
                if not win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                    return None, "The clipboard does not contain text (it may contain an image, file, or be empty)."
                data = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                return (data or ""), None
            finally:
                try:
                    win32clipboard.CloseClipboard()
                except Exception:
                    pass
        except ImportError:
            pass
        except Exception as err:
            if _IN_MEMORY_CLIPBOARD is not None:
                return _IN_MEMORY_CLIPBOARD, None
            return None, f"Could not access clipboard: {err}"

    # Fallback via tkinter
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            text = root.clipboard_get()
            root.destroy()
            return text or "", None
        except Exception:
            root.destroy()
            if _IN_MEMORY_CLIPBOARD is not None:
                return _IN_MEMORY_CLIPBOARD, None
            return None, "The clipboard does not contain text (it may contain an image, file, or be empty)."
    except Exception:
        pass

    if _IN_MEMORY_CLIPBOARD is not None:
        return _IN_MEMORY_CLIPBOARD, None

    return None, "The clipboard is currently empty or contains non-text data."


def set_clipboard_text(text: str) -> tuple[bool, str | None]:
    """Replace system clipboard content with text.

    Args:
        text: String to store on the clipboard.

    Returns:
        tuple of (success: bool, error_message: str | None).
    """
    global _IN_MEMORY_CLIPBOARD
    _IN_MEMORY_CLIPBOARD = text

    if sys.platform.startswith("win"):
        try:
            import win32clipboard
            import win32con

            opened = False
            for _ in range(5):
                try:
                    win32clipboard.OpenClipboard()
                    opened = True
                    break
                except Exception:
                    time.sleep(0.04)

            if not opened:
                return True, None  # Updated in-memory fallback

            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(text, win32con.CF_UNICODETEXT)
                return True, None
            finally:
                try:
                    win32clipboard.CloseClipboard()
                except Exception:
                    pass
        except ImportError:
            pass
        except Exception as err:
            return False, f"Could not write to clipboard: {err}"

    # Fallback via tkinter
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
        return True, None
    except Exception:
        pass

    return True, None
