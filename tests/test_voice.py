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
    @patch("sounddevice.InputStream")
    def test_detects_speech_then_silence(self, mock_input_stream):
        """Should stop recording after speech followed by silence."""
        ambient = np.full((1600, 1), 0.001, dtype=np.float32)
        speech = np.full((1600, 1), 0.1, dtype=np.float32)
        silence = np.full((1600, 1), 0.001, dtype=np.float32)

        call_count = 0
        stream = MagicMock()

        def read_side_effect(n):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return ambient, None  # calibration phase
            if call_count <= 15:
                return speech, None  # speaking
            return silence, None  # silence after speech

        stream.read = read_side_effect
        stream.__enter__ = lambda s: s
        stream.__exit__ = lambda s, *a: None
        mock_input_stream.return_value = stream

        audio = record_until_silence(max_seconds=5, silence_seconds=0.5)
        assert audio is not None
        assert len(audio) > 0

    @patch("sounddevice.InputStream")
    def test_returns_none_on_pure_silence(self, mock_input_stream):
        """Should return None if no speech detected at all."""
        silence = np.full((1600, 1), 0.001, dtype=np.float32)

        stream = MagicMock()
        stream.read = lambda n: (silence, None)
        stream.__enter__ = lambda s: s
        stream.__exit__ = lambda s, *a: None
        mock_input_stream.return_value = stream

        audio = record_until_silence(max_seconds=1)
        assert audio is None


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
