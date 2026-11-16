"""Unit tests for Jarvis Cost & Speed Controls (Phase 5).

Verifies:
1. Logging token usage to data/usage.csv with CSV header and proper formatting.
2. Accurate cost calculation and cost string formatting.
3. Usage summary calculation for today and this month.
4. Enforcement of DAILY_TOKEN_LIMIT and refusal of model calls when exceeded.
5. Override mechanism allowing calls to proceed when limit is overridden.
6. SentenceSpeaker streaming TTS playback pipeline and time-to-first-spoken-word.
7. Brain max_tokens switching per mode (voice: 300, text: 800).
8. Network error retry behavior in Brain.
9. Built-in command handling for 'usage' and 'override limit'.
"""

import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import config
import usage
from tts import SentenceSpeaker
from usage import (
    calculate_cost,
    format_cost,
    format_spoken_usage_report,
    format_usage_report,
    get_usage_summary,
    is_daily_limit_exceeded,
    log_token_usage,
    override_daily_limit,
)


class TestCostAndSpeedControls(unittest.TestCase):
    """Test suite for token usage, daily limits, streaming speech, and speed controls."""

    def setUp(self) -> None:
        """Create a temporary sandbox directory for test CSV and state files."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)
        self.test_data = self.test_path / "data"
        self.test_data.mkdir(parents=True, exist_ok=True)
        self.test_csv = self.test_data / "usage.csv"

        self.patch_csv = patch.object(usage, "get_usage_csv_path", return_value=self.test_csv)
        self.patch_csv.start()

    def tearDown(self) -> None:
        """Clean up temporary sandbox and patches."""
        self.patch_csv.stop()
        self.test_dir.cleanup()

    def test_01_log_token_usage_creates_header_and_row(self) -> None:
        """Verify log_token_usage creates CSV with header and appends entries."""
        log_token_usage("llama3.1", input_tokens=150, output_tokens=30, image_tokens=0)
        self.assertTrue(self.test_csv.is_file())

        with open(self.test_csv, "r", newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        self.assertEqual(len(reader), 2)
        self.assertEqual(reader[0], ["timestamp", "model", "input_tokens", "output_tokens", "image_tokens", "total_tokens"])
        self.assertEqual(reader[1][1], "llama3.1")
        self.assertEqual(reader[1][2], "150")
        self.assertEqual(reader[1][3], "30")
        self.assertEqual(reader[1][4], "0")
        self.assertEqual(reader[1][5], "180")

    def test_02_cost_calculation(self) -> None:
        """Test token cost math based on config prices."""
        with patch.object(config, "PRICE_PER_M_INPUT_TOKENS", 0.15), \
             patch.object(config, "PRICE_PER_M_OUTPUT_TOKENS", 0.60):
            # 1,000,000 in ($0.15) + 1,000,000 out ($0.60) = $0.75
            cost = calculate_cost(1_000_000, 1_000_000)
            self.assertAlmostEqual(cost, 0.75, places=4)

            # 10,000 in ($0.0015) + 2,000 out ($0.0012) = $0.0027
            cost_small = calculate_cost(10_000, 2_000)
            self.assertAlmostEqual(cost_small, 0.0027, places=6)
            self.assertEqual(format_cost(cost_small), "$0.0027")
            self.assertEqual(format_cost(1.25), "$1.25")

    def test_03_usage_summary_today_and_month(self) -> None:
        """Verify get_usage_summary filters and aggregates tokens for today and this month."""
        today = datetime.now().isoformat()
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()

        log_token_usage("llama3.1", 100, 20, timestamp=today)
        log_token_usage("llama3.1", 200, 40, timestamp=today)
        log_token_usage("llama3.1", 500, 50, timestamp=yesterday)

        summary = get_usage_summary()
        self.assertEqual(summary["today"]["input_tokens"], 300)
        self.assertEqual(summary["today"]["output_tokens"], 60)
        self.assertEqual(summary["today"]["total_tokens"], 360)

        self.assertEqual(summary["this_month"]["input_tokens"], 800)
        self.assertEqual(summary["this_month"]["output_tokens"], 110)
        self.assertEqual(summary["this_month"]["total_tokens"], 910)

    def test_04_daily_limit_enforcement_and_override(self) -> None:
        """Verify DAILY_TOKEN_LIMIT marks limit exceeded and override releases it."""
        with patch.object(config, "DAILY_TOKEN_LIMIT", 500), \
             patch("state.get_token_limit_override_date", return_value=None):
            # Log 400 tokens -> not exceeded
            log_token_usage("llama3.1", 300, 100)
            self.assertFalse(is_daily_limit_exceeded())

            # Log 150 tokens -> total 550 >= 500 -> exceeded!
            log_token_usage("llama3.1", 100, 50)
            self.assertTrue(is_daily_limit_exceeded())

            # Override the limit
            today_str = datetime.now().strftime("%Y-%m-%d")
            with patch("state.set_token_limit_override_date") as mock_set, \
                 patch("state.get_token_limit_override_date", return_value=today_str):
                override_daily_limit()
                mock_set.assert_called_once_with(today_str)
                self.assertFalse(is_daily_limit_exceeded())

    def test_05_format_usage_reports(self) -> None:
        """Verify text and spoken usage reports produce expected output."""
        log_token_usage("llama3.1", 1000, 200)
        rep = format_usage_report()
        self.assertIn("Token Usage & Cost Summary:", rep)
        self.assertIn("Today:", rep)
        self.assertIn("This Month:", rep)

        spoken = format_spoken_usage_report()
        self.assertIn("Today you have used 1,200 tokens", spoken)

    def test_06_brain_respects_token_limit(self) -> None:
        """Verify brain.think refuses model calls when daily limit is exceeded."""
        from brain import Brain

        brain = Brain()
        with patch("brain.is_daily_limit_exceeded", return_value=True), \
             patch.object(config, "DAILY_TOKEN_LIMIT", 200000), \
             patch.object(brain.client, "chat") as mock_chat:
            reply = brain.think("Hello Jarvis")
            self.assertIn("Daily token limit of 200,000 tokens reached", reply)
            mock_chat.assert_not_called()

    def test_07_brain_override_limit_command(self) -> None:
        """Verify saying 'override limit' inside brain.think triggers override."""
        from brain import Brain

        brain = Brain()
        with patch("brain.override_daily_limit") as mock_override:
            reply = brain.think("override limit")
            self.assertIn("Daily token limit has been overridden", reply)
            mock_override.assert_called_once()

    def test_08_brain_max_tokens_per_mode(self) -> None:
        """Verify max_tokens differs between voice (300) and text (800) modes."""
        from brain import Brain

        brain = Brain()
        mock_resp = MagicMock()
        mock_resp.message.content = "OK"
        mock_resp.message.tool_calls = []

        with patch.object(brain.client, "chat", return_value=mock_resp) as mock_chat, \
             patch.object(config, "MAX_TOKENS_VOICE", 300), \
             patch.object(config, "MAX_TOKENS_TEXT", 800):
            # Voice mode
            brain.think("Voice query", mode="voice")
            self.assertEqual(mock_chat.call_args[1]["options"]["num_predict"], 300)

            # Text mode
            brain.think("Text query", mode="text")
            self.assertEqual(mock_chat.call_args[1]["options"]["num_predict"], 800)

    def test_09_brain_network_retry_once(self) -> None:
        """Verify network errors trigger 1 retry before raising or returning error message."""

        from brain import Brain

        brain = Brain()
        mock_success = MagicMock()
        mock_success.message.content = "Recovered after retry"
        mock_success.message.tool_calls = []

        # First call fails with ConnectionError, second succeeds
        with patch.object(brain.client, "chat", side_effect=[ConnectionError("Temporary blip"), mock_success]) as mock_chat, \
             patch("time.sleep") as mock_sleep:
            reply = brain.think("Hello")
            self.assertEqual(reply, "Recovered after retry")
            self.assertEqual(mock_chat.call_count, 2)
            mock_sleep.assert_called_once_with(1.5)

    def test_10_sentence_speaker_lifecycle(self) -> None:
        """Verify SentenceSpeaker queues sentences and handles finish."""
        with patch("tts._synthesize_edge_tts", return_value="test.mp3"), \
             patch("tts._play_audio_file", return_value=True), \
             patch("tts._cleanup_temp_file"):
            speaker = SentenceSpeaker()
            speaker.feed_sentence("First sentence.")
            speaker.feed_sentence("Second sentence.")
            speaker.finish()
            speaker.wait_until_done(timeout=2.0)
            self.assertTrue(speaker._done_event.is_set())
            self.assertIsNotNone(speaker.time_to_first_spoken_word)

    def test_11_main_command_routing(self) -> None:
        """Verify handle_usage_command and handle_override_command route commands correctly."""
        from main import handle_override_command, handle_usage_command

        with patch("usage.format_usage_report", return_value="Usage report"), \
             patch("builtins.print"):
            handled = handle_usage_command("usage", "Jarvis", enable_speak=False)
            self.assertTrue(handled)

            handled_override = handle_override_command("override limit", "Jarvis", enable_speak=False)
            self.assertTrue(handled_override)

            not_handled = handle_usage_command("regular query", "Jarvis", enable_speak=False)
            self.assertFalse(not_handled)

    def test_12_image_token_tracking_and_backwards_compatibility(self) -> None:
        """Verify image tokens are tracked separately and legacy 5-column CSV rows parse cleanly."""
        # 1. Write a legacy 5-column row directly
        today = datetime.now().isoformat()
        with open(self.test_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "model", "input_tokens", "output_tokens", "total_tokens"])
            writer.writerow([today, "llama3.1", "100", "50", "150"])

        # 2. Append a new 6-column row with image tokens
        log_token_usage("llama3.2-vision", input_tokens=40, output_tokens=80, image_tokens=1600)

        summary = get_usage_summary()
        self.assertEqual(summary["today"]["input_tokens"], 140)
        self.assertEqual(summary["today"]["output_tokens"], 130)
        self.assertEqual(summary["today"]["image_tokens"], 1600)
        self.assertEqual(summary["today"]["total_tokens"], 1870)

        # 3. Test report text includes Image Tokens line
        rep = format_usage_report(summary)
        self.assertIn("• Image Tokens:  1,600", rep)

        # 4. Test spoken report mentions image tokens
        spoken = format_spoken_usage_report(summary)
        self.assertIn("including 1,600 image tokens", spoken)


if __name__ == "__main__":
    unittest.main()
