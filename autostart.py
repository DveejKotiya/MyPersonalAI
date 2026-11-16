"""Autostart Configuration Module for Jarvis (Phase 4).

Manages automatic startup on user login across operating systems:
- Windows: Creates a shortcut (.lnk) in the user's Startup folder running pythonw.exe
           (no console window) with '--listen --tray'.
- macOS:   Creates a LaunchAgent property list (.plist) in ~/Library/LaunchAgents.
- Linux:   Creates a desktop entry (.desktop) in ~/.config/autostart.

Provides enable() and disable() functions, file path inspection, and
instructions for manual verification or removal.
"""

import os
import subprocess
import sys
from pathlib import Path

import config


def get_autostart_path() -> Path:
    """Return the absolute path of the startup entry for the current OS.

    Returns:
        Path object pointing to the OS-specific startup file.
    """
    if sys.platform == "win32":
        appdata = os.getenv("APPDATA")
        if not appdata:
            appdata = str(Path.home() / "AppData" / "Roaming")
        return (
            Path(appdata)
            / "Microsoft"
            / "Windows"
            / "Start Menu"
            / "Programs"
            / "Startup"
            / "Jarvis.lnk"
        )
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "LaunchAgents" / "com.jarvis.assistant.plist"
    else:
        # Linux and other UNIX systems
        return Path.home() / ".config" / "autostart" / "jarvis.desktop"


def is_enabled() -> bool:
    """Check whether automatic startup is currently enabled.

    Returns:
        True if the startup file exists, False otherwise.
    """
    return get_autostart_path().exists()


def get_python_executable() -> Path:
    """Locate the Python executable in the virtual environment.

    On Windows, prioritizes 'pythonw.exe' so that Jarvis runs silently
    in the background with no command prompt window.

    Returns:
        Path to pythonw.exe (Windows) or sys.executable.
    """
    current_exe = Path(sys.executable).resolve()
    venv_dir = current_exe.parent

    if sys.platform == "win32":
        pythonw_exe = venv_dir / "pythonw.exe"
        if pythonw_exe.exists():
            return pythonw_exe

    return current_exe


def get_manual_instructions() -> str:
    """Return user-friendly instructions for manually checking or removing the startup entry.

    Returns:
        Formatted instructions string.
    """
    target = get_autostart_path()

    if sys.platform == "win32":
        return (
            "Windows Manual Removal:\n"
            "  1. Press Win + R, type 'shell:startup' and press Enter to open the Startup folder.\n"
            f"  2. Locate 'Jarvis.lnk' and delete it.\n"
            "  Alternatively, in PowerShell run:\n"
            f"  Remove-Item \"{target}\" -ErrorAction SilentlyContinue"
        )
    elif sys.platform == "darwin":
        return (
            "macOS Manual Removal:\n"
            "  Run the following commands in your Terminal:\n"
            f"  launchctl unload \"{target}\"\n"
            f"  rm \"{target}\""
        )
    else:
        return (
            "Linux Manual Removal:\n"
            "  Run the following command in your terminal:\n"
            f"  rm \"{target}\""
        )


# ==============================================================================
# Platform-Specific Implementation Helpers
# ==============================================================================

def _enable_windows(
    target_path: Path,
    python_exe: Path,
    main_py: Path,
    base_dir: Path,
) -> tuple[bool, str]:
    """Create a Windows .lnk shortcut in the user's Startup folder."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    arguments = f'"{main_py}" --listen --tray'
    description = f"{getattr(config, 'ASSISTANT_NAME', 'Jarvis')} Personal AI Assistant"

    # 1. Try creating shortcut via win32com (WScript.Shell)
    try:
        import win32com.client

        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(str(target_path))
        shortcut.TargetPath = str(python_exe)
        shortcut.Arguments = arguments
        shortcut.WorkingDirectory = str(base_dir)
        shortcut.Description = description
        shortcut.Save()

        if target_path.exists():
            return True, f"Startup shortcut created successfully via COM at:\n  {target_path}"
    except Exception:
        pass

    # 2. Fallback to PowerShell script
    try:
        ps_script = (
            f"$ws = New-Object -ComObject WScript.Shell; "
            f"$s = $ws.CreateShortcut('{target_path!s}'); "
            f"$s.TargetPath = '{python_exe!s}'; "
            f"$s.Arguments = '{arguments}'; "
            f"$s.WorkingDirectory = '{base_dir!s}'; "
            f"$s.Description = '{description}'; "
            f"$s.Save()"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and target_path.exists():
            return True, f"Startup shortcut created successfully via PowerShell at:\n  {target_path}"
        else:
            err_msg = result.stderr.strip() or "PowerShell returned non-zero exit code"
            return False, f"PowerShell shortcut creation failed: {err_msg}"
    except Exception as ps_err:
        return False, f"Could not create Windows startup shortcut: {ps_err}"


def _enable_macos(
    target_path: Path,
    python_exe: Path,
    main_py: Path,
    base_dir: Path,
) -> tuple[bool, str]:
    """Create a LaunchAgent plist for macOS."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = base_dir / "logs" / "jarvis.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.jarvis.assistant</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_exe}</string>
        <string>{main_py}</string>
        <string>--listen</string>
        <string>--tray</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{base_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>{log_file}</string>
    <key>StandardErrorPath</key>
    <string>{log_file}</string>
</dict>
</plist>
"""
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(plist_content)

        # Attempt to load into launchctl if session is active
        try:
            subprocess.run(["launchctl", "load", str(target_path)], capture_output=True, timeout=5)
        except Exception:
            pass

        return True, f"LaunchAgent plist created at:\n  {target_path}"
    except Exception as err:
        return False, f"Failed to write LaunchAgent plist: {err}"


def _enable_linux(
    target_path: Path,
    python_exe: Path,
    main_py: Path,
    base_dir: Path,
) -> tuple[bool, str]:
    """Create a .desktop autostart entry for Linux."""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    desktop_content = f"""[Desktop Entry]
Type=Application
Name={getattr(config, 'ASSISTANT_NAME', 'Jarvis')}
Comment=Jarvis Personal AI Assistant
Exec="{python_exe}" "{main_py}" --listen --tray
Path={base_dir}
Terminal=false
X-GNOME-Autostart-enabled=true
Categories=Utility;AudioVideo;
"""
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(desktop_content)
        # Set execute permissions
        target_path.chmod(0o755)
        return True, f"Desktop autostart entry created at:\n  {target_path}"
    except Exception as err:
        return False, f"Failed to write Linux autostart desktop entry: {err}"


# ==============================================================================
# Public API
# ==============================================================================

def enable() -> tuple[bool, str]:
    """Enable automatic startup of Jarvis at user login.

    Uses absolute paths and configures the command:
        <python_executable> <main.py> --listen --tray
    with no console window (using pythonw.exe on Windows).

    Returns:
        Tuple of (success: bool, status_message: str).
    """
    target_path = get_autostart_path()
    python_exe = get_python_executable()
    base_dir = getattr(config, "BASE_DIR", Path(__file__).resolve().parent)
    main_py = (base_dir / "main.py").resolve()

    if not main_py.exists():
        return False, f"Cannot enable autostart: main.py not found at {main_py}"

    if sys.platform == "win32":
        return _enable_windows(target_path, python_exe, main_py, base_dir)
    elif sys.platform == "darwin":
        return _enable_macos(target_path, python_exe, main_py, base_dir)
    else:
        return _enable_linux(target_path, python_exe, main_py, base_dir)


def disable() -> tuple[bool, str]:
    """Disable automatic startup of Jarvis at user login.

    Removes the startup shortcut / plist / desktop entry.

    Returns:
        Tuple of (success: bool, status_message: str).
    """
    target_path = get_autostart_path()

    if not target_path.exists():
        return True, f"Autostart is not active (no startup file found at {target_path})."

    # On macOS, unload from launchctl first
    if sys.platform == "darwin":
        try:
            subprocess.run(["launchctl", "unload", str(target_path)], capture_output=True, timeout=5)
        except Exception:
            pass

    try:
        target_path.unlink()
        return True, f"Startup entry successfully removed from:\n  {target_path}"
    except Exception as err:
        return False, f"Could not delete startup entry ({err}). Please remove manually:\n{get_manual_instructions()}"

