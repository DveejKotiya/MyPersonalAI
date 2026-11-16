"""Unit tests for Jarvis built-in memory commands in main.py (Phase 5).

Covers:
1. 'what do you remember about me' when memory store is empty.
2. 'what do you remember about me' when memories exist (console & speech).
3. 'forget everything' with user denial (confirmation = False).
4. 'forget everything' with user approval (confirmation = True).
5. Non-memory inputs pass through cleanly.
"""

import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from brain import Brain
from main import handle_memory_command
from memory import MemoryStore


class TestMainMemoryCommands(unittest.TestCase):
    """Test suite for handle_memory_command() in main.py."""

    def setUp(self) -> None:
        """Create a fresh temporary SQLite database and Brain for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_main_mem.db"
        self.store = MemoryStore(db_path=self.db_path)
        self.brain = Brain()

    def tearDown(self) -> None:
        """Clean up temporary test database."""
        self.temp_dir.cleanup()

    def test_what_do_you_remember_empty(self) -> None:
        """Verify output when no memories are stored."""
        with patch("memory.MemoryStore", side_effect=lambda *args, **kwargs: MemoryStore(db_path=self.db_path)):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                handled = handle_memory_command(
                    "what do you remember about me",
                    self.brain,
                    "Jarvis",
                    enable_speak=False,
                    confirmation_mode="text",
                )
                self.assertTrue(handled)
                self.assertIn("don't have any memories stored", fake_out.getvalue())

    def test_what_do_you_remember_with_items(self) -> None:
        """Verify memories are printed and summarized when items exist."""
        self.store.add("Prefers oat milk in coffee", category="preference")
        self.store.add("Works from home on Fridays", category="fact")

        with patch("memory.MemoryStore", side_effect=lambda *args, **kwargs: MemoryStore(db_path=self.db_path)):
            with patch("main.speak") as mock_speak:
                with patch("sys.stdout", new=StringIO()) as fake_out:
                    handled = handle_memory_command(
                        "what do you remember about me",
                        self.brain,
                        "Jarvis",
                        enable_speak=True,
                        confirmation_mode="voice",
                    )
                    self.assertTrue(handled)
                    output = fake_out.getvalue()
                    self.assertIn("Prefers oat milk in coffee", output)
                    self.assertIn("Works from home on Fridays", output)
                    mock_speak.assert_called_once()
                    self.assertIn("2 things", mock_speak.call_args[0][0])

    def test_forget_everything_denied(self) -> None:
        """Verify that declining confirmation aborts deletion."""
        self.store.add("Persistent fact", category="fact")
        self.brain.history.append({"role": "user", "content": "Hello"})

        with patch("memory.MemoryStore", side_effect=lambda *args, **kwargs: MemoryStore(db_path=self.db_path)):
            with patch("tools.confirm", return_value=False):
                with patch("sys.stdout", new=StringIO()) as fake_out:
                    handled = handle_memory_command(
                        "forget everything",
                        self.brain,
                        "Jarvis",
                        enable_speak=False,
                        confirmation_mode="text",
                    )
                    self.assertTrue(handled)
                    self.assertIn("Cancelled", fake_out.getvalue())

        # Stored memory and history must still exist
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(len(self.brain.history), 1)

    def test_forget_everything_approved(self) -> None:
        """Verify that approving confirmation clears both memory store and brain history."""
        self.store.add("Memory to be cleared 1", category="fact")
        self.store.add("Memory to be cleared 2", category="preference")
        self.brain.history.append({"role": "user", "content": "Hello"})
        self.assertEqual(self.store.count(), 2)

        with patch("memory.MemoryStore", side_effect=lambda *args, **kwargs: MemoryStore(db_path=self.db_path)):
            with patch("tools.confirm", return_value=True):
                with patch("sys.stdout", new=StringIO()) as fake_out:
                    handled = handle_memory_command(
                        "forget everything",
                        self.brain,
                        "Jarvis",
                        enable_speak=False,
                        confirmation_mode="text",
                    )
                    self.assertTrue(handled)
                    self.assertIn("All 2 stored memories and conversation history have been cleared", fake_out.getvalue())

        # Stored memories and brain history must both be empty
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(len(self.brain.history), 0)

    def test_unrelated_command_returns_false(self) -> None:
        """Verify non-memory commands return False to allow normal processing."""
        handled = handle_memory_command(
            "What is the weather today?",
            self.brain,
            "Jarvis",
            enable_speak=False,
            confirmation_mode="text",
        )
        self.assertFalse(handled)


if __name__ == "__main__":
    unittest.main()

