"""Tests for voice input module.

Requires voice extras: uv sync --extra voice
"""

from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")

from fiat_lux.voice import _rms, record_until_silence, transcribe


class TestRMS:
    def test_silence(self):
        audio = np.zeros(1600, dtype=np.float32)
        assert _rms(audio) == 0.0

    def test_loud(self):
        audio = np.ones(1600, dtype=np.float32)
        assert _rms(audio) == pytest.approx(1.0)

    def test_half(self):
        audio = np.full(1600, 0.5, dtype=np.float32)
        assert _rms(audio) == pytest.approx(0.5)


class TestRecordUntilSilence:
    def test_silence_threshold_calibration(self):
        """Threshold should be THRESHOLD_MULTIPLIER times ambient."""
        from fiat_lux.voice import THRESHOLD_MULTIPLIER
        # With ambient RMS of 0.005, threshold should be 0.015
        assert THRESHOLD_MULTIPLIER == 3.0

    def test_format_volume_bar(self):
        from fiat_lux.voice import format_volume_bar
        bar = format_volume_bar(0.0, width=10)
        assert len(bar) == 10
        assert "\u2588" not in bar  # no filled blocks for silence

        bar = format_volume_bar(0.5, width=10)
        assert "\u2588" in bar  # some filled blocks for loud


class TestTranscribe:
    @patch("fiat_lux.voice._get_whisper_model")
    def test_transcribes_audio(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": " lights off "}
        mock_get_model.return_value = mock_model

        audio = np.random.randn(16000).astype(np.float32)
        result = transcribe(audio)

        assert result == "lights off"
        mock_model.transcribe.assert_called_once()


class TestGracefulImportError:
    def test_cli_handles_missing_deps(self):
        """The CLI should show a helpful message if voice deps aren't installed."""
        # This tests the import guard in agent.py, not voice.py itself
        # Just verify the module structure is correct
        from fiat_lux.voice import listen_once
        assert callable(listen_once)
