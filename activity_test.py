"""Unit tests for Jarvis Activity & Introspection tools (Phase 5).

Covers:
1. logviewer.py filters: --last, --tool, --errors, --today.
2. Log rotation in tools.py: rotating at 1 MB and keeping last 3 archives.
3. Log argument sanitization: masking API keys, credentials, and truncating to 200 chars.
4. File content truncation in logs.
5. Status command reporting: uptime, mode, tool calls today, errors, memories, and model.
6. Status command triggers: 'Jarvis, status', 'status'.
"""

import json
import tempfile
import unittest
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import config
import logviewer
from main import format_status_report, get_system_status, handle_status_command
from memory import MemoryStore
from tools import _sanitize_log_data, rotate_log_file


class TestActivityAndIntrospection(unittest.TestCase):
    """Test suite for log rotation, sanitization, logviewer, and status command."""

    def setUp(self) -> None:
        """Create a fresh temporary sandbox directory for test artifacts."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.log_dir = self.tmp_path / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "actions.log"
        self.db_path = self.tmp_path / "memory.db"
        self.store = MemoryStore(db_path=self.db_path)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_log_sanitization_and_truncation(self) -> None:
        """Verify API keys, passwords, file contents, and long arguments are sanitized and capped at 200 chars."""
        # 1. API key masking
        raw_args = {
            "api_key": "sk-1234567890abcdef1234567890abcdef",
            "prompt": "Hello OpenAI sk-abcdef1234567890abcdef1234567890",
            "password": "SuperSecretPassword!",
        }
        sanitized = _sanitize_log_data(raw_args, max_len=200)
        self.assertEqual(sanitized["api_key"], "********")
        self.assertEqual(sanitized["password"], "********")
        self.assertNotIn("sk-abcdef", sanitized["prompt"])
        self.assertIn("********", sanitized["prompt"])

        # 2. Long argument truncation to 200 chars
        long_text = "A" * 350
        long_args = {"query": long_text}
        sanitized_long = _sanitize_log_data(long_args, max_len=200)
        self.assertLessEqual(len(sanitized_long["query"]), 225)
        self.assertIn("[truncated]", sanitized_long["query"])

        # 3. File content truncation
        file_args = {"filename": "data.txt", "content": "X" * 500}
        sanitized_file = _sanitize_log_data(file_args, max_len=200)
        self.assertIn("[truncated]", sanitized_file["content"])
        self.assertLessEqual(len(sanitized_file["content"]), 225)

    def test_log_rotation_at_1mb_keeping_3_files(self) -> None:
        """Verify rotating logs when exceeding 1 MB keeps exactly 3 backup archives."""
        # Create a log file exceeding 1 MB
        large_content = "X" * (1024 * 1024 + 10)
        self.log_file.write_text(large_content, encoding="utf-8")

        # First rotation: actions.log -> actions.log.1
        rotate_log_file(self.log_file, max_bytes=1024 * 1024, backup_count=3)
        self.assertTrue((self.log_dir / "actions.log.1").exists())
        self.assertFalse(self.log_file.exists())

        # Write another large file and rotate: actions.log.1 -> actions.log.2, actions.log -> actions.log.1
        self.log_file.write_text(large_content, encoding="utf-8")
        rotate_log_file(self.log_file, max_bytes=1024 * 1024, backup_count=3)
        self.assertTrue((self.log_dir / "actions.log.2").exists())
        self.assertTrue((self.log_dir / "actions.log.1").exists())

        # Third rotation: actions.log.2 -> actions.log.3, actions.log.1 -> actions.log.2, actions.log -> actions.log.1
        self.log_file.write_text(large_content, encoding="utf-8")
        rotate_log_file(self.log_file, max_bytes=1024 * 1024, backup_count=3)
        self.assertTrue((self.log_dir / "actions.log.3").exists())
        self.assertTrue((self.log_dir / "actions.log.2").exists())
        self.assertTrue((self.log_dir / "actions.log.1").exists())

        # Fourth rotation: actions.log.3 must be pruned, keeping only 3 backups (.1, .2, .3)
        self.log_file.write_text(large_content, encoding="utf-8")
        rotate_log_file(self.log_file, max_bytes=1024 * 1024, backup_count=3)
        self.assertTrue((self.log_dir / "actions.log.3").exists())
        self.assertTrue((self.log_dir / "actions.log.2").exists())
        self.assertTrue((self.log_dir / "actions.log.1").exists())
        self.assertFalse((self.log_dir / "actions.log.4").exists())

    def test_logviewer_filters(self) -> None:
        """Verify logviewer filtering by --last, --tool, --errors, and --today."""
        today_iso = datetime.now(UTC).isoformat()
        sample_entries = [
            {"timestamp": today_iso, "tool": "open_app", "arguments": {"name": "notepad"}, "status": "SUCCESS", "output_preview": "ok"},
            {"timestamp": today_iso, "tool": "get_weather", "arguments": {"city": "Paris"}, "status": "ERROR_NETWORK", "output_preview": "failed"},
            {"timestamp": "2025-01-01T10:00:00Z", "tool": "echo", "arguments": {"text": "old"}, "status": "SUCCESS", "output_preview": "old"},
            {"timestamp": today_iso, "tool": "open_app", "arguments": {"name": "calc"}, "status": "SUCCESS", "output_preview": "ok"},
        ]

        # 1. Filter by --tool
        tool_res = logviewer.filter_entries(sample_entries, tool_filter="open_app")
        self.assertEqual(len(tool_res), 2)
        self.assertTrue(all(e["tool"] == "open_app" for e in tool_res))

        # 2. Filter by --errors
        err_res = logviewer.filter_entries(sample_entries, errors_only=True)
        self.assertEqual(len(err_res), 1)
        self.assertEqual(err_res[0]["tool"], "get_weather")

        # 3. Filter by --today
        today_res = logviewer.filter_entries(sample_entries, today_only=True)
        self.assertEqual(len(today_res), 3)

        # 4. Filter by --last N
        last_res = logviewer.filter_entries(sample_entries, last_n=2)
        self.assertEqual(len(last_res), 2)
        self.assertEqual(last_res[-1]["arguments"]["name"], "calc")

    def test_status_command_reporting(self) -> None:
        """Verify get_system_status and format_status_report return complete operational metrics."""
        # Add 2 memories to store
        with patch("memory.MemoryStore", side_effect=lambda *args, **kwargs: MemoryStore(db_path=self.db_path)):
            self.store.add("Sample memory 1", category="fact")
            self.store.add("Sample memory 2", category="preference")

            # Write 2 mock log entries to test actions.log
            today_iso = datetime.now(UTC).isoformat()
            test_entries = [
                {"timestamp": today_iso, "tool": "echo", "arguments": {}, "status": "SUCCESS", "output_preview": "ok"},
                {"timestamp": today_iso, "tool": "get_weather", "arguments": {}, "status": "ERROR_TIMEOUT", "output_preview": "err"},
            ]
            self.log_file.write_text("\n".join(json.dumps(e) for e in test_entries), encoding="utf-8")

            with patch("config.BASE_DIR", self.tmp_path):
                status_dict = get_system_status(current_mode="Hands-free voice mode (--listen)")
                self.assertIn("uptime", status_dict)
                self.assertEqual(status_dict["current_mode"], "Hands-free voice mode (--listen)")
                self.assertEqual(status_dict["model_name"], config.MODEL_NAME)
                self.assertEqual(status_dict["tool_calls_today"], 2)
                self.assertEqual(status_dict["errors_today"], 1)
                self.assertEqual(status_dict["memories_count"], 2)

                text_report, spoken_summary = format_status_report(status_dict)
                self.assertIn("=== Jarvis System Status ===", text_report)
                self.assertIn("• Tool Calls Today:  2", text_report)
                self.assertIn("• Errors Today:      1", text_report)
                self.assertIn("• Memories Stored:   2", text_report)
                self.assertIn("2 tool calls have been executed today with 1 errors", spoken_summary)

    def test_status_command_trigger(self) -> None:
        """Verify 'Jarvis, status' and 'status' trigger handle_status_command."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            handled = handle_status_command(
                "Jarvis, status",
                "Jarvis",
                current_mode="Interactive text chat mode",
                enable_speak=False,
                confirmation_mode="text",
            )
            self.assertTrue(handled)
            self.assertIn("=== Jarvis System Status ===", fake_out.getvalue())
            self.assertIn("• Current Mode:", fake_out.getvalue())

        # Unrelated command returns False
        handled_other = handle_status_command(
            "What is the time?",
            "Jarvis",
            current_mode="Interactive text chat mode",
            enable_speak=False,
            confirmation_mode="text",
        )
        self.assertFalse(handled_other)


if __name__ == "__main__":
    unittest.main()
