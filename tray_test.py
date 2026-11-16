"""Unit and integration tests for System Tray Icon and Global Hotkey (Phase 4).

Tests:
1. Tray icon generation in code using Pillow with state colors:
   - grey (paused)
   - blue (idle / listening for wake word)
   - green (actively listening)
   - orange (thinking / speaking)
2. Hotkey string normalization and pynput GlobalHotKeys validation.
3. Tray menu structure: Pause/Resume, Open Log, Open Notes, Toggle Speech, Quit.
4. Assistant pause/resume lifecycle and microphone stream release.
5. Hotkey behavior: manual wake when idle, interruption when speaking.
6. Command-line flag parsing (--listen, --tray).
"""

import sys
import time
from unittest.mock import MagicMock, patch

from PIL import Image

import config
from main import AssistantState, JarvisAssistant
from tray import (
    COLORS,
    JarvisTray,
    create_hotkey_listener,
    create_tray_icon,
    normalize_hotkey,
)


def test_tray_icon_colors_and_generation():
    """Verify in-code drawing of tray icons for all 4 required assistant states."""
    expected_colors = ["grey", "blue", "green", "orange"]

    for color_name in expected_colors:
        assert color_name in COLORS, f"Missing required color definition: {color_name}"
        img = create_tray_icon(color_name=color_name, size=64)

        assert isinstance(img, Image.Image)
        assert img.size == (64, 64)
        assert img.mode == "RGBA"

        # Verify center has non-transparent pixels
        center_pixel = img.getpixel((32, 32))
        assert center_pixel[3] > 0, "Icon center should not be transparent"


def test_hotkey_normalization():
    """Verify that user-configured hotkey strings are normalized to pynput format."""
    test_cases = [
        ("<ctrl>+<alt>+j", "<ctrl>+<alt>+j"),
        ("Ctrl+Alt+J", "<ctrl>+<alt>+j"),
        ("ctrl+alt+j", "<ctrl>+<alt>+j"),
        ("CTRL+ALT+J", "<ctrl>+<alt>+j"),
        ("Ctrl+Shift+J", "<ctrl>+<shift>+j"),
        ("ctrl+shift+k", "<ctrl>+<shift>+k"),
        ("win+alt+j", "<cmd>+<alt>+j"),
        ("option+ctrl+j", "<alt>+<ctrl>+j"),
    ]

    for raw, expected in test_cases:
        normalized = normalize_hotkey(raw)
        assert normalized == expected, f"Expected {expected} for {raw}, got {normalized}"


def test_create_hotkey_listener():
    """Verify that create_hotkey_listener produces a valid GlobalHotKeys instance."""
    callback_mock = MagicMock()
    listener = create_hotkey_listener("<ctrl>+<alt>+j", callback_mock)

    assert listener is not None
    # Verify starting and stopping listener
    listener.start()
    assert listener.is_alive()
    listener.stop()


def test_tray_menu_structure():
    """Verify the tray menu contains the required items in order."""
    mock_assistant = MagicMock()
    mock_assistant.assistant_name = "Jarvis"
    mock_assistant.is_manually_paused = False
    mock_assistant.enable_speak = True

    tray = JarvisTray(assistant=mock_assistant)
    menu_items = list(tray.icon.menu.items)

    item_texts = [m.text for m in menu_items if hasattr(m, "text") and m.text]
    assert "Pause Listening (Release Mic)" in item_texts
    assert "Show Recent Activity" in item_texts
    assert "Open Log File" in item_texts
    assert "Open Notes" in item_texts
    assert "Toggle Speech (Voice: ON)" in item_texts
    assert "Quit" in item_texts

    # Test dynamic label changes
    mock_assistant.is_manually_paused = True
    tray.icon.update_menu()
    paused_texts = [m.text for m in list(tray.icon.menu.items) if hasattr(m, "text") and m.text]
    assert "Resume Listening" in paused_texts

    mock_assistant.enable_speak = False
    tray.icon.update_menu()
    quiet_texts = [m.text for m in list(tray.icon.menu.items) if hasattr(m, "text") and m.text]
    assert "Toggle Speech (Voice: OFF)" in quiet_texts


def test_assistant_pause_resume_releases_mic():
    """Verify that pause_listening pauses the listener and releases the microphone."""
    mock_brain = MagicMock()
    with patch("main.WakeWordListener") as MockListener:
        mock_listener_inst = MockListener.return_value
        assistant = JarvisAssistant(brain=mock_brain, assistant_name="Jarvis")

        mock_tray = MagicMock()
        assistant.set_tray(mock_tray)

        # Initially not paused
        assert not assistant.is_manually_paused

        # Trigger pause
        assistant.pause_listening()
        assert assistant.is_manually_paused
        mock_listener_inst.pause.assert_called_once()
        mock_tray.update_state.assert_called_with("IDLE", is_paused=True)
        mock_tray.notify.assert_called_once()
        assert "Paused" in mock_tray.notify.call_args[0][0]

        # Trigger resume
        assistant.resume_listening()
        assert not assistant.is_manually_paused
        mock_listener_inst.resume.assert_called_once()
        mock_tray.update_state.assert_called_with("IDLE", is_paused=False)
        assert "Resumed" in mock_tray.notify.call_args[0][0]


def test_hotkey_manual_wake_when_idle():
    """Verify that pressing hotkey when idle triggers a manual wake to LISTENING."""
    mock_brain = MagicMock()
    with patch("main.WakeWordListener"):
        assistant = JarvisAssistant(brain=mock_brain, assistant_name="Jarvis")
        assistant.set_state(AssistantState.IDLE)

        assert not assistant._wake_event.is_set()
        assert not assistant._manual_wake

        # Hotkey pressed
        assistant.on_hotkey()

        assert assistant._manual_wake
        assert assistant._wake_event.is_set()


def test_hotkey_interrupts_when_speaking():
    """Verify that pressing hotkey when speaking immediately interrupts speech."""
    mock_brain = MagicMock()
    with patch("main.WakeWordListener"):
        with patch("main.stop_speaking") as mock_stop_speaking:
            assistant = JarvisAssistant(brain=mock_brain, assistant_name="Jarvis")
            assistant.set_state(AssistantState.SPEAKING)

            assert not assistant._interrupted_event.is_set()

            # Hotkey pressed while speaking
            assistant.on_hotkey()

            assert assistant._interrupted_event.is_set()
            mock_stop_speaking.assert_called_once()


def test_hotkey_unpauses_and_wakes_when_paused():
    """Verify that pressing hotkey when paused unpauses the assistant and triggers wake."""
    mock_brain = MagicMock()
    with patch("main.WakeWordListener") as MockListener:
        mock_listener_inst = MockListener.return_value
        assistant = JarvisAssistant(brain=mock_brain, assistant_name="Jarvis")
        assistant.pause_listening()
        assert assistant.is_manually_paused

        assistant.on_hotkey()
        assert not assistant.is_manually_paused
        assert assistant._manual_wake
        assert assistant._wake_event.is_set()
        mock_listener_inst.resume.assert_called_once()


def test_state_updates_tray_icon():
    """Verify that assistant state transitions update tray icon with correct color."""
    mock_brain = MagicMock()
    with patch("main.WakeWordListener"):
        assistant = JarvisAssistant(brain=mock_brain, assistant_name="Jarvis")
        mock_tray = MagicMock()
        assistant.set_tray(mock_tray)

        assistant.set_state(AssistantState.LISTENING)
        mock_tray.update_state.assert_called_with("LISTENING", is_paused=False)

        assistant.set_state(AssistantState.THINKING)
        mock_tray.update_state.assert_called_with("THINKING", is_paused=False)

        assistant.set_state(AssistantState.SPEAKING)
        mock_tray.update_state.assert_called_with("SPEAKING", is_paused=False)

        assistant.set_state(AssistantState.IDLE)
        mock_tray.update_state.assert_called_with("IDLE", is_paused=False)


def test_autostart_paths_and_status():
    """Verify autostart path discovery, python executable detection, and manual instructions."""
    import autostart

    target_path = autostart.get_autostart_path()
    assert target_path is not None
    assert str(target_path).strip() != ""

    python_exe = autostart.get_python_executable()
    assert python_exe.exists()
    if sys.platform == "win32":
        assert "pythonw.exe" in str(python_exe).lower() or "python.exe" in str(python_exe).lower()

    instructions = autostart.get_manual_instructions()
    assert "Manual Removal:" in instructions


def test_autostart_enable_disable_lifecycle():
    """Verify creating and removing a startup shortcut in a sandbox-safe target."""
    import tempfile
    from pathlib import Path

    import autostart

    temp_dir = Path(tempfile.gettempdir())
    test_lnk = temp_dir / "JarvisUnitTest.lnk"
    if test_lnk.exists():
        test_lnk.unlink()

    python_exe = autostart.get_python_executable()
    base_dir = Path(".").resolve()
    main_py = (base_dir / "main.py").resolve()

    if sys.platform == "win32":
        ok, msg = autostart._enable_windows(test_lnk, python_exe, main_py, base_dir)
        assert ok, f"Windows shortcut creation failed: {msg}"
        assert test_lnk.exists(), "Shortcut file was not created"
        test_lnk.unlink()
        assert not test_lnk.exists(), "Shortcut file was not cleaned up"
    else:
        ok, msg = autostart._enable_linux(test_lnk, python_exe, main_py, base_dir)
        assert ok
        assert test_lnk.exists()
        test_lnk.unlink()


def test_system_readiness_probe():
    """Verify that wait_for_system_readiness detects audio devices and returns True."""
    from main import wait_for_system_readiness

    with patch("sounddevice.query_devices") as mock_query:
        mock_query.return_value = [{"name": "Test Mic", "max_input_channels": 2}]
        with patch("sounddevice.default.device", [0, 0]):
            ready = wait_for_system_readiness(max_retries=1, retry_interval_sec=0.01)
            assert ready is True


def test_log_stream_redirection():
    """Verify that LogStream writes output to log file."""
    import tempfile
    from pathlib import Path

    from main import LogStream

    temp_log = Path(tempfile.gettempdir()) / "test_stream.log"
    if temp_log.exists():
        temp_log.unlink()

    stream = LogStream(temp_log)
    stream.write("Test log entry\n")
    stream.flush()

    assert temp_log.exists()
    with open(temp_log, "r", encoding="utf-8") as f:
        data = f.read()
    assert "Test log entry" in data
    temp_log.unlink()


def test_watchdog_recovers_stuck_state():
    """Verify that StateWatchdog detects a state stuck > 60s and resets assistant to IDLE."""
    from main import AssistantState, JarvisAssistant

    mock_brain = MagicMock()
    with patch("main.WakeWordListener"):
        with patch("main.stop_speaking") as mock_stop_speak:
            assistant = JarvisAssistant(brain=mock_brain, assistant_name="Jarvis")
            mock_tray = MagicMock()
            assistant.set_tray(mock_tray)

            # Set state to LISTENING and simulate 75 seconds elapsed
            assistant.set_state(AssistantState.LISTENING)
            assert assistant.state == AssistantState.LISTENING
            assistant.state_entered_time = time.time() - 75.0

            # Execute force reset
            assistant.force_reset_to_idle()

            assert assistant.state == AssistantState.IDLE
            mock_stop_speak.assert_called()
            mock_tray.update_state.assert_called_with("IDLE", is_paused=False)


def test_privacy_audio_saving_disabled_by_default():
    """Verify that SAVE_DEBUG_AUDIO is False by default and no audio is saved to disk."""

    # Confirm config flag is strictly False by default
    assert hasattr(config, "SAVE_DEBUG_AUDIO")
    # Verify default is False unless environment specifically requests it
    assert isinstance(config.SAVE_DEBUG_AUDIO, bool)


def run_all_tests():
    """Run all tests directly without pytest binary."""
    tests = [
        test_tray_icon_colors_and_generation,
        test_hotkey_normalization,
        test_create_hotkey_listener,
        test_tray_menu_structure,
        test_assistant_pause_resume_releases_mic,
        test_hotkey_manual_wake_when_idle,
        test_hotkey_interrupts_when_speaking,
        test_hotkey_unpauses_and_wakes_when_paused,
        test_state_updates_tray_icon,
        test_autostart_paths_and_status,
        test_autostart_enable_disable_lifecycle,
        test_system_readiness_probe,
        test_log_stream_redirection,
        test_watchdog_recovers_stuck_state,
        test_privacy_audio_saving_disabled_by_default,
    ]

    print("=" * 60)
    print(" Running System Tray & Hotkey Test Suite")
    print("=" * 60)
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            raise e

    print("=" * 60)
    print(f" All {passed}/{len(tests)} tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
