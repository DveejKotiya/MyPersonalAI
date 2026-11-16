"""Comprehensive unit test suite for Jarvis tools framework and tools.

Verifies:
1. Core framework: decorator, registry, validation, confirmation, logs, truncation.
2. get_time_and_date: returns readable time/date string.
3. open_app: whitelisted app launched, non-whitelisted app rejected with list, empty name handled.
4. open_url: valid HTTP/HTTPS allowed, 'javascript:' and 'file://' rejected, empty URL handled.
5. search_web: proper URL encoding, browser invocation, empty query handled.
6. take_note & read_notes: writing notes, reading notes, limit slicing, empty text, missing file.
7. set_timer, list_timers, cancel_timer: background daemon thread, range limits (1s to 24h), cancellation.
8. set_volume, mute, unmute: scalar conversion, range validation (0-100), mute toggles.
9. take_screenshot: directory creation, timestamped filename, graceful error handling.
"""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import config
from tools import (
    ACTIONS_LOG_FILE,
    SCREENSHOTS_DIR,
    get_tool_schemas,
    run_tool,
    set_confirmation_mode,
    tool,
)


class TestToolFrameworkAndTools(unittest.TestCase):
    """Complete test suite for tools.py."""

    def setUp(self) -> None:
        """Reset confirmation mode before each test."""
        set_confirmation_mode("text")

    # --------------------------------------------------------------------------
    # 1. Framework Core Tests
    # --------------------------------------------------------------------------

    def test_01_get_tool_schemas(self) -> None:
        """Verify registered tools are exported in Ollama function format."""
        schemas = get_tool_schemas()
        self.assertIsInstance(schemas, list)
        names = [s["function"]["name"] for s in schemas]
        self.assertIn("echo", names)
        self.assertIn("get_time_and_date", names)
        self.assertIn("open_app", names)
        self.assertIn("open_url", names)
        self.assertIn("search_web", names)
        self.assertIn("take_note", names)
        self.assertIn("read_notes", names)
        self.assertIn("set_timer", names)
        self.assertIn("set_volume", names)
        self.assertIn("take_screenshot", names)

    def test_02_unknown_tool_and_bad_arguments(self) -> None:
        """Verify unregistered tool names and missing arguments are rejected."""
        res1 = run_tool("unknown_tool", {})
        self.assertTrue(res1.startswith("Error: Unknown tool"))

        res2 = run_tool("echo", {})
        self.assertIn("Missing required argument: 'text'", res2)

        res3 = run_tool("echo", {"text": 12345})
        self.assertIn("Invalid type for argument 'text'", res3)

    def test_03_dangerous_tool_confirmation(self) -> None:
        """Verify dangerous tools trigger confirm() and abide by user approval."""
        @tool(
            description="Test dangerous operation.",
            parameters={"type": "object", "properties": {}},
            dangerous=True,
        )
        def dangerous_action() -> str:
            return "Executed action."

        # Denial path
        with patch("tools.confirm", return_value=False):
            denied_res = run_tool("dangerous_action", {})
            self.assertIn("Action cancelled", denied_res)

        # Approval path
        with patch("tools.confirm", return_value=True):
            approved_res = run_tool("dangerous_action", {})
            self.assertEqual(approved_res, "Executed action.")

    def test_04_output_truncation_and_logging(self) -> None:
        """Verify outputs >2000 chars are truncated and logged."""
        @tool(
            description="Outputs large text.",
            parameters={"type": "object", "properties": {}},
            dangerous=False,
        )
        def huge_output() -> str:
            return "Z" * 3000

        result = run_tool("huge_output", {})
        self.assertIn("[Result truncated to 2000 characters]", result)
        self.assertTrue(ACTIONS_LOG_FILE.exists())

    # --------------------------------------------------------------------------
    # 2. get_time_and_date Tests
    # --------------------------------------------------------------------------

    def test_05_get_time_and_date(self) -> None:
        """Verify get_time_and_date returns readable format."""
        result = run_tool("get_time_and_date", {})
        self.assertIsInstance(result, str)
        # Should contain day, year, and AM/PM
        self.assertTrue(any(marker in result for marker in ["AM", "PM"]))
        self.assertTrue(any(str(year) in result for year in range(2025, 2030)))

    # --------------------------------------------------------------------------
    # 3. open_app Tests
    # --------------------------------------------------------------------------

    def test_06_open_app_whitelisted(self) -> None:
        """Verify launching a whitelisted application."""
        with patch("subprocess.Popen") as mock_popen:
            result = run_tool("open_app", {"name": "notepad"})
            self.assertIn("Opened notepad successfully", result)
            mock_popen.assert_called_once_with(config.APPS["notepad"])

    def test_07_open_app_non_whitelisted_and_empty(self) -> None:
        """Verify non-whitelisted or empty application names are rejected."""
        res1 = run_tool("open_app", {"name": "malicious_tool"})
        self.assertIn("is not in the whitelist", res1)
        self.assertIn("notepad", res1)
        self.assertIn("calculator", res1)

        res2 = run_tool("open_app", {"name": "   "})
        self.assertIn("Application name cannot be empty", res2)

    # --------------------------------------------------------------------------
    # 4. open_url Tests
    # --------------------------------------------------------------------------

    def test_08_open_url_valid(self) -> None:
        """Verify opening valid http/https URLs."""
        with patch("webbrowser.open") as mock_browser:
            result = run_tool("open_url", {"url": "https://python.org"})
            self.assertIn("Opened https://python.org", result)
            mock_browser.assert_called_once_with("https://python.org")

    def test_09_open_url_security_rejections(self) -> None:
        """Verify javascript:, file://, and empty URLs are blocked."""
        with patch("webbrowser.open") as mock_browser:
            # javascript:
            res_js = run_tool("open_url", {"url": "javascript:alert(1)"})
            self.assertIn("Security Error", res_js)
            mock_browser.assert_not_called()

            # file://
            res_file = run_tool("open_url", {"url": "file:///C:/Windows/System32/calc.exe"})
            self.assertIn("Security Error", res_file)
            mock_browser.assert_not_called()

            # Empty
            res_empty = run_tool("open_url", {"url": "  "})
            self.assertIn("URL cannot be empty", res_empty)

    # --------------------------------------------------------------------------
    # 5. search_web Tests
    # --------------------------------------------------------------------------

    def test_10_search_web(self) -> None:
        """Verify search_web properly URL-encodes queries."""
        with patch("webbrowser.open") as mock_browser:
            result = run_tool("search_web", {"query": "python tips & tricks"})
            self.assertIn("Searching Google for 'python tips & tricks'", result)
            mock_browser.assert_called_once_with("https://www.google.com/search?q=python+tips+%26+tricks")

            # Empty query
            res_empty = run_tool("search_web", {"query": "   "})
            self.assertIn("Search query cannot be empty", res_empty)

    # --------------------------------------------------------------------------
    # 6. take_note and read_notes Tests
    # --------------------------------------------------------------------------

    def test_11_take_and_read_notes(self) -> None:
        """Verify taking and reading notes with limits."""
        test_notes_path = Path("test_notes.md")
        with patch("tools.NOTES_FILE", test_notes_path):
            if test_notes_path.exists():
                test_notes_path.unlink()

            # Read empty notes
            res_empty = run_tool("read_notes", {})
            self.assertIn("No notes found", res_empty)

            # Take notes
            run_tool("take_note", {"text": "Buy groceries"})
            run_tool("take_note", {"text": "Review Jarvis Phase 3"})
            run_tool("take_note", {"text": "Deploy to laptop"})

            # Read all (limit 10)
            res_all = run_tool("read_notes", {})
            self.assertIn("Buy groceries", res_all)
            self.assertIn("Review Jarvis Phase 3", res_all)
            self.assertIn("Deploy to laptop", res_all)

            # Read with limit 1
            res_limit1 = run_tool("read_notes", {"limit": 1})
            self.assertNotIn("Buy groceries", res_limit1)
            self.assertIn("Deploy to laptop", res_limit1)

            # Empty note rejected
            res_bad_note = run_tool("take_note", {"text": "  "})
            self.assertIn("Note text cannot be empty", res_bad_note)

            # Clean up test file
            if test_notes_path.exists():
                test_notes_path.unlink()

    # --------------------------------------------------------------------------
    # 7. Timer Tests
    # --------------------------------------------------------------------------

    def test_12_set_list_cancel_timer(self) -> None:
        """Verify setting, listing, and cancelling timers."""
        # Minimum bound check (< 1 sec)
        res_too_short = run_tool("set_timer", {"minutes": 0.005})
        self.assertIn("Minimum timer duration is 1 second", res_too_short)

        # Maximum bound check (> 24 hours)
        res_too_long = run_tool("set_timer", {"minutes": 1500})
        self.assertIn("Maximum timer duration is 24 hours", res_too_long)

        # Set valid 10-second timer (0.1667 minutes)
        res_set = run_tool("set_timer", {"minutes": 0.1667, "label": "tea"})
        self.assertIn("Timer 'tea' set", res_set)

        # List timers
        res_list = run_tool("list_timers", {})
        self.assertIn("tea", res_list)

        # Cancel timer
        res_cancel = run_tool("cancel_timer", {"label": "tea"})
        self.assertIn("Cancelled timer 'tea'", res_cancel)

        # Verify cancelled
        res_list_after = run_tool("list_timers", {})
        self.assertNotIn("tea", res_list_after)

    # --------------------------------------------------------------------------
    # 8. Volume Tests
    # --------------------------------------------------------------------------

    def test_13_set_volume_and_mute(self) -> None:
        """Verify volume adjustments and range validations."""
        mock_endpoint = MagicMock()
        with patch("tools._get_windows_volume_endpoint", return_value=mock_endpoint):
            # Set valid volume 30
            res_vol = run_tool("set_volume", {"level": 30})
            self.assertEqual(res_vol, "System volume set to 30%.")
            mock_endpoint.SetMasterVolumeLevelScalar.assert_called_once_with(0.30, None)

            # Out of bounds (>100 or <0)
            res_high = run_tool("set_volume", {"level": 150})
            self.assertIn("out of range", res_high)

            res_low = run_tool("set_volume", {"level": -5})
            self.assertIn("out of range", res_low)

            # Mute
            res_mute = run_tool("mute", {})
            self.assertEqual(res_mute, "System audio muted.")
            mock_endpoint.SetMute.assert_called_with(1, None)

            # Unmute
            res_unmute = run_tool("unmute", {})
            self.assertEqual(res_unmute, "System audio unmuted.")
            mock_endpoint.SetMute.assert_called_with(0, None)

    # --------------------------------------------------------------------------
    # 9. Screenshot Tests
    # --------------------------------------------------------------------------

    def test_14_take_screenshot(self) -> None:
        """Verify take_screenshot handles save directory and execution."""
        mock_img = MagicMock()
        with patch("PIL.ImageGrab.grab", return_value=mock_img):
            result = run_tool("take_screenshot", {})
            self.assertIn("Screenshot saved successfully", result)
            self.assertTrue(SCREENSHOTS_DIR.exists())
            mock_img.save.assert_called_once()

    # --------------------------------------------------------------------------
    # 10. File Sandbox Security Tests
    # --------------------------------------------------------------------------

    def test_15_list_files_sandboxing(self) -> None:
        """Verify list_files stays within ALLOWED_FOLDER and blocks traversal."""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir).resolve()
            # Create a test file and a subfolder
            (temp_path / "doc.txt").write_text("Hello", encoding="utf-8")
            sub = temp_path / "projects"
            sub.mkdir()
            (sub / "code.py").write_text("print('test')", encoding="utf-8")

            with patch("config.ALLOWED_FOLDER", temp_path):
                # Root list
                res_root = run_tool("list_files", {})
                self.assertIn("[FILE] doc.txt", res_root)
                self.assertIn("[DIR]  projects/", res_root)

                # Subfolder list
                res_sub = run_tool("list_files", {"subfolder": "projects"})
                self.assertIn("[FILE] code.py", res_sub)

                # Path traversal attempt
                res_escape = run_tool("list_files", {"subfolder": "../../"})
                self.assertIn("Security Error", res_escape)
                self.assertIn("escapes the allowed folder", res_escape)

    def test_16_read_file_traversal_and_absolute_paths_blocked(self) -> None:
        """Verify read_file blocks traversal (../../secret.txt) and absolute paths outside sandbox."""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sandbox = Path(temp_dir) / "sandbox"
            sandbox.mkdir()
            secret_outside = Path(temp_dir) / "secret.txt"
            secret_outside.write_text("TOP_SECRET_PASSWORD", encoding="utf-8")

            legit_file = sandbox / "notes.txt"
            legit_file.write_text("Legitimate notes", encoding="utf-8")

            with patch("config.ALLOWED_FOLDER", sandbox):
                # 1. Normal read succeeds
                res_ok = run_tool("read_file", {"path": "notes.txt"})
                self.assertEqual(res_ok, "Legitimate notes")

                # 2. Directory traversal attempt: ../secret.txt
                res_traversal = run_tool("read_file", {"path": "../secret.txt"})
                self.assertIn("Security Error", res_traversal)
                self.assertNotIn("TOP_SECRET_PASSWORD", res_traversal)

                # 3. Directory traversal attempt: ../../secret.txt
                res_traversal2 = run_tool("read_file", {"path": "../../secret.txt"})
                self.assertIn("Security Error", res_traversal2)
                self.assertNotIn("TOP_SECRET_PASSWORD", res_traversal2)

                # 4. Absolute path outside sandbox attempt
                res_abs = run_tool("read_file", {"path": str(secret_outside)})
                self.assertIn("Security Error", res_abs)
                self.assertNotIn("TOP_SECRET_PASSWORD", res_abs)

    def test_17_read_file_symlink_escape_blocked(self) -> None:
        """Verify read_file blocks symlinks/junctions pointing outside ALLOWED_FOLDER."""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sandbox = Path(temp_dir) / "sandbox"
            sandbox.mkdir()
            secret_outside = Path(temp_dir) / "secret.txt"
            secret_outside.write_text("SUPER_CONFIDENTIAL", encoding="utf-8")

            with patch("config.ALLOWED_FOLDER", sandbox):
                # Test real Windows junction or mock realpath resolution pointing outside
                with patch("os.path.realpath", side_effect=lambda p: str(secret_outside) if "symlink_doc.txt" in str(p) else os.path.abspath(p)):
                    res_symlink = run_tool("read_file", {"path": "symlink_doc.txt"})
                    self.assertIn("Security Error", res_symlink)
                    self.assertIn("escapes the allowed folder", res_symlink)
                    self.assertNotIn("SUPER_CONFIDENTIAL", res_symlink)

    def test_18_read_file_extension_and_truncation(self) -> None:
        """Verify read_file enforces text extensions (.txt, .md, .csv, .json, .py) and truncates at 3000 chars."""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sandbox = Path(temp_dir).resolve()
            binary_file = sandbox / "program.exe"
            binary_file.write_text("Binary data", encoding="utf-8")

            huge_file = sandbox / "large.txt"
            huge_file.write_text("A" * 4000, encoding="utf-8")

            with patch("config.ALLOWED_FOLDER", sandbox):
                # Non-text extension blocked
                res_bin = run_tool("read_file", {"path": "program.exe"})
                self.assertIn("Security Error", res_bin)
                self.assertIn("Only text files", res_bin)

                # File truncated at 3000 chars
                res_huge = run_tool("read_file", {"path": "large.txt"})
                self.assertIn("[Content truncated at 3000 characters]", res_huge)
                self.assertTrue(res_huge.startswith("A" * 3000))
                self.assertTrue(len(res_huge) < 3100)

    def test_19_create_text_file_security_and_overwrite(self) -> None:
        """Verify create_text_file prevents overwrite, blocks traversal, and enforces confirmation."""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            sandbox = Path(temp_dir).resolve()

            with patch("config.ALLOWED_FOLDER", sandbox):
                # 1. Traversal attempt blocked
                with patch("tools.confirm", return_value=True):
                    res_traversal = run_tool("create_text_file", {"filename": "../../escape.txt", "content": "bad"})
                    self.assertIn("Security Error", res_traversal)

                # 2. Absolute path outside sandbox blocked
                with patch("tools.confirm", return_value=True):
                    res_abs = run_tool("create_text_file", {"filename": "C:/Windows/hack.txt", "content": "bad"})
                    self.assertIn("Security Error", res_abs)

                # 3. Forbidden extension blocked
                with patch("tools.confirm", return_value=True):
                    res_ext = run_tool("create_text_file", {"filename": "script.bat", "content": "bad"})
                    self.assertIn("Security Error", res_ext)

                # 4. User denial of dangerous tool
                with patch("tools.confirm", return_value=False):
                    res_denied = run_tool("create_text_file", {"filename": "valid.txt", "content": "hello"})
                    self.assertIn("Action cancelled", res_denied)
                    self.assertFalse((sandbox / "valid.txt").exists())

                # 5. Successful creation when approved
                with patch("tools.confirm", return_value=True):
                    res_created = run_tool("create_text_file", {"filename": "valid.txt", "content": "hello world"})
                    self.assertIn("created successfully", res_created)
                    self.assertTrue((sandbox / "valid.txt").exists())
                    self.assertEqual((sandbox / "valid.txt").read_text(encoding="utf-8"), "hello world")

                # 6. Refusal to overwrite existing file
                with patch("tools.confirm", return_value=True):
                    res_overwrite = run_tool("create_text_file", {"filename": "valid.txt", "content": "new text"})
                    self.assertIn("already exists. Overwriting is not permitted", res_overwrite)
                    self.assertEqual((sandbox / "valid.txt").read_text(encoding="utf-8"), "hello world")

    def test_14_memory_tools(self) -> None:
        """Verify remember, recall, forget, and list_memories tools."""
        import tempfile

        from memory import MemoryStore
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_db = Path(tmp_dir) / "test_tools_memory.db"
            with patch("memory.MemoryStore", side_effect=lambda *args, **kwargs: MemoryStore(db_path=temp_db)):
                # 1. remember tool
                res_rem = run_tool("remember", {"text": "User likes dark roast coffee", "category": "preference"})
                self.assertIn("Saved to memory", res_rem)
                self.assertIn("User likes dark roast coffee", res_rem)

                # 2. remember rejection for sensitive data
                res_sens = run_tool("remember", {"text": "My password is secret123", "category": "fact"})
                self.assertIn("Could not save memory", res_sens)
                self.assertIn("sensitive data", res_sens)

                # 3. recall tool
                res_rec = run_tool("recall", {"query": "coffee"})
                self.assertIn("Found 1 matching memory", res_rec)
                self.assertIn("dark roast coffee", res_rec)

                # 4. list_memories tool
                res_list = run_tool("list_memories", {})
                self.assertIn("Most recent memories", res_list)
                self.assertIn("dark roast coffee", res_list)

                # 5. forget tool with denial
                with patch("tools.confirm", return_value=False):
                    res_denied = run_tool("forget", {"query": "coffee"})
                    self.assertIn("Action cancelled", res_denied)

                # Ensure memory was not deleted
                self.assertEqual(len(MemoryStore(db_path=temp_db).list_all()), 1)

                # 6. forget tool with approval
                with patch("tools.confirm", return_value=True):
                    res_approved = run_tool("forget", {"query": "coffee"})
                    self.assertIn("Successfully deleted 1 memory item", res_approved)

                # Ensure memory was deleted
                self.assertEqual(len(MemoryStore(db_path=temp_db).list_all()), 0)


if __name__ == "__main__":
    unittest.main()
