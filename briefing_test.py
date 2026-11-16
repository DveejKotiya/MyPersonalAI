"""Unit test suite for Daily Briefing and Weather features (Phase 5).

Covers:
1. get_weather with successful Open-Meteo responses.
2. get_weather handling timeouts and network errors gracefully.
3. get_weather falling back to config.DEFAULT_CITY when city=None.
4. get_weather handling unknown city coordinates gracefully.
5. daily_briefing gathering all 5 sections (date/time, weather, timers, notes, tasks).
6. daily_briefing fault-tolerance: skipping failed components while returning the rest.
7. state.py tracking of last_briefing_date in data/state.json.
8. should_run_startup_briefing() logic.
9. Voice and text triggers ("good morning", "give me my briefing", "what's my day look like").
"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

import config
import state
from brain import Brain
from main import handle_briefing_command
from memory import MemoryStore
from tools import daily_briefing, get_weather, run_tool


class TestDailyBriefingAndWeather(unittest.TestCase):
    """Test suite for weather and daily briefing tools."""

    def setUp(self) -> None:
        """Create a fresh temporary directory for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.db_path = self.tmp_path / "test_briefing_memory.db"
        self.store = MemoryStore(db_path=self.db_path)
        self.brain = Brain()

    def tearDown(self) -> None:
        """Clean up temporary test directory."""
        self.temp_dir.cleanup()

    @patch("requests.get")
    def test_get_weather_success(self, mock_get: MagicMock) -> None:
        """Verify get_weather successfully formats temperature, condition, and high/low."""
        # Mock geocoding response
        mock_geo = MagicMock()
        mock_geo.status_code = 200
        mock_geo.json.return_value = {
            "results": [{"name": "London", "country": "United Kingdom", "latitude": 51.5, "longitude": -0.12}]
        }

        # Mock forecast response
        mock_forecast = MagicMock()
        mock_forecast.status_code = 200
        mock_forecast.json.return_value = {
            "current_units": {"temperature_2m": "°C"},
            "current": {"temperature_2m": 15.2, "weather_code": 0},
            "daily": {"temperature_2m_max": [18.0], "temperature_2m_min": [10.5]},
        }

        mock_get.side_effect = [mock_geo, mock_forecast]

        result = get_weather("London")
        self.assertIn("London, United Kingdom", result)
        self.assertIn("15.2°C", result)
        self.assertIn("Clear sky", result)
        self.assertIn("High: 18.0°C", result)
        self.assertIn("Low: 10.5°C", result)

    @patch("requests.get")
    def test_get_weather_timeout(self, mock_get: MagicMock) -> None:
        """Verify get_weather handles requests.Timeout gracefully without crashing."""
        mock_get.side_effect = requests.Timeout("Connection timed out after 5.0s")

        result = get_weather("Paris")
        self.assertIn("timed out", result.lower())
        self.assertIn("Paris", result)

    @patch("requests.get")
    def test_get_weather_network_error(self, mock_get: MagicMock) -> None:
        """Verify get_weather handles requests.RequestException gracefully."""
        mock_get.side_effect = requests.ConnectionError("Name resolution failed")

        result = get_weather("Berlin")
        self.assertIn("network error", result.lower())

    @patch("requests.get")
    def test_get_weather_unknown_city(self, mock_get: MagicMock) -> None:
        """Verify get_weather handles cities with no geocoding results."""
        mock_geo = MagicMock()
        mock_geo.status_code = 200
        mock_geo.json.return_value = {"results": []}
        mock_get.return_value = mock_geo

        result = get_weather("NonexistentCityXYZ123")
        self.assertIn("Could not find coordinates", result)

    @patch("requests.get")
    def test_get_weather_default_city_fallback(self, mock_get: MagicMock) -> None:
        """Verify city=None defaults to config.DEFAULT_CITY."""
        mock_geo = MagicMock()
        mock_geo.status_code = 200
        mock_geo.json.return_value = {
            "results": [{"name": "New York", "country": "United States", "latitude": 40.7, "longitude": -74.0}]
        }

        mock_forecast = MagicMock()
        mock_forecast.status_code = 200
        mock_forecast.json.return_value = {
            "current_units": {"temperature_2m": "°C"},
            "current": {"temperature_2m": 20.0, "weather_code": 1},
            "daily": {"temperature_2m_max": [22.0], "temperature_2m_min": [14.0]},
        }
        mock_get.side_effect = [mock_geo, mock_forecast]

        with patch.object(config, "DEFAULT_CITY", "New York"):
            result = get_weather(None)
            self.assertIn("New York", result)

    @patch("tools.get_weather")
    def test_daily_briefing_gathers_all_sections(self, mock_weather: MagicMock) -> None:
        """Verify daily_briefing returns structured text with all 5 sections."""
        mock_weather.return_value = "London, United Kingdom: 15.0°C, Clear sky (High: 18.0°C, Low: 10.0°C)"

        # Populate tasks in memory store
        with patch("memory.MemoryStore", side_effect=lambda *args, **kwargs: MemoryStore(db_path=self.db_path)):
            self.store.add("Submit quarterly report", category="task")
            self.store.add("Call plumber for sink", category="task")

            briefing_text = run_tool("daily_briefing", {})
            self.assertIn("=== Daily Briefing ===", briefing_text)
            self.assertIn("• Date & Time:", briefing_text)
            self.assertIn("• Weather: London, United Kingdom", briefing_text)
            self.assertIn("• Timers:", briefing_text)
            self.assertIn("• Recent Notes:", briefing_text)
            self.assertIn("• Pending Tasks:", briefing_text)
            self.assertIn("Submit quarterly report", briefing_text)
            self.assertIn("Call plumber for sink", briefing_text)

    @patch("tools.get_weather")
    def test_daily_briefing_fault_tolerance_weather_down(self, mock_weather: MagicMock) -> None:
        """Verify that when weather is offline, the remaining briefing sections still succeed."""
        mock_weather.return_value = "Weather service network error: Connection refused"

        with patch("memory.MemoryStore", side_effect=lambda *args, **kwargs: MemoryStore(db_path=self.db_path)):
            self.store.add("Review security pull request", category="task")

            briefing_text = daily_briefing()
            # Weather should show error/unavailable notice
            self.assertIn("Weather: Weather service network error", briefing_text)
            # Other sections must still be present and intact
            self.assertIn("• Date & Time:", briefing_text)
            self.assertIn("• Timers:", briefing_text)
            self.assertIn("• Recent Notes:", briefing_text)
            self.assertIn("Review security pull request", briefing_text)

    def test_state_tracking_and_startup_briefing_logic(self) -> None:
        """Verify state.json tracking and should_run_startup_briefing()."""
        state_file = self.tmp_path / "state.json"
        with patch("state._get_state_path", return_value=state_file):
            # Initially no state file
            self.assertIsNone(state.get_last_briefing_date())

            # When BRIEFING_ON_STARTUP is False (default) -> should not run
            with patch.object(config, "BRIEFING_ON_STARTUP", False):
                self.assertFalse(state.should_run_startup_briefing())

            # When BRIEFING_ON_STARTUP is True and no date recorded -> should run
            with patch.object(config, "BRIEFING_ON_STARTUP", True):
                self.assertTrue(state.should_run_startup_briefing())

                # Record briefing for today
                today = datetime.now().strftime("%Y-%m-%d")
                state.set_last_briefing_date(today)
                self.assertEqual(state.get_last_briefing_date(), today)

                # Now that today is recorded -> should not run again today
                self.assertFalse(state.should_run_startup_briefing())

    @patch("main.trigger_daily_briefing")
    def test_voice_and_text_triggers(self, mock_trigger: MagicMock) -> None:
        """Verify 'good morning', 'give me my briefing', and 'what's my day look like' trigger briefing."""
        triggers = [
            "good morning",
            "good morning jarvis",
            "give me my briefing",
            "what's my day look like",
            "what does my day look like",
            "daily briefing",
        ]

        for phrase in triggers:
            mock_trigger.reset_mock()
            handled = handle_briefing_command(
                phrase,
                self.brain,
                "Jarvis",
                enable_speak=False,
                confirmation_mode="text",
            )
            self.assertTrue(handled, f"Expected '{phrase}' to trigger briefing")
            mock_trigger.assert_called_once()

        # Unrelated phrase should not trigger
        mock_trigger.reset_mock()
        handled_unrelated = handle_briefing_command(
            "What is the capital of France?",
            self.brain,
            "Jarvis",
            enable_speak=False,
            confirmation_mode="text",
        )
        self.assertFalse(handled_unrelated)
        mock_trigger.assert_not_called()


if __name__ == "__main__":
    unittest.main()

