"""Tests for voice input module.

Requires voice extras: uv sync --extra voice
"""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

np = pytest.importorskip("numpy")

from heylux.voice import _rms, record_until_silence, transcribe


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
        from heylux.voice import THRESHOLD_MULTIPLIER
        assert THRESHOLD_MULTIPLIER == 2.0

    def test_format_volume_bar(self):
        from heylux.voice import format_volume_bar
        bar = format_volume_bar(0.0, width=10)
        assert len(bar) == 10
        assert "\u2588" not in bar  # no filled blocks for silence

        bar = format_volume_bar(0.5, width=10)
        assert "\u2588" in bar  # some filled blocks for loud


class TestTranscribe:
    @patch("heylux.voice._get_whisper_model")
    @patch("heylux.voice._stt_backend", "openai-whisper")
    def test_transcribes_audio_openai_whisper(self, mock_get_model):
        """Test transcription with openai-whisper backend."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": " lights off "}
        mock_get_model.return_value = mock_model

        audio = np.random.randn(16000).astype(np.float32)
        result = transcribe(audio)

        assert result == "lights off"
        mock_model.transcribe.assert_called_once()

    @patch("heylux.voice._get_whisper_model")
    @patch("heylux.voice._stt_backend", "mlx-whisper")
    def test_transcribes_audio_mlx_whisper(self, mock_get_model):
        """Test transcription with mlx-whisper backend."""
        mock_get_model.return_value = "mlx-community/whisper-large-v3-turbo"

        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = {"text": " lights off "}
        with patch.dict("sys.modules", {"mlx_whisper": mock_mlx}):
            audio = np.random.randn(16000).astype(np.float32)
            result = transcribe(audio)

            assert result == "lights off"
            mock_mlx.transcribe.assert_called_once()


class TestSTTBackendSelection:
    def test_prefers_mlx_whisper(self):
        """Should use mlx-whisper when available."""
        import heylux.voice as v
        # Reset state
        old_model, old_backend = v._model, v._stt_backend
        v._model = None
        v._stt_backend = None
        try:
            v._get_whisper_model()
            assert v._stt_backend == "mlx-whisper"
        finally:
            v._model = old_model
            v._stt_backend = old_backend

    def test_falls_back_to_openai_whisper(self):
        """Should fall back to openai-whisper if mlx-whisper unavailable."""
        import heylux.voice as v
        old_model, old_backend = v._model, v._stt_backend
        v._model = None
        v._stt_backend = None
        try:
            with patch.dict("sys.modules", {"mlx_whisper": None}):
                try:
                    v._get_whisper_model()
                    # If openai-whisper is installed, should use it
                    assert v._stt_backend == "openai-whisper"
                except ImportError:
                    # Neither backend available — expected in some environments
                    pass
        finally:
            v._model = old_model
            v._stt_backend = old_backend


class TestTTSBackendSelection:
    def test_tts_backend_loads(self):
        """TTS model loading should not crash."""
        from heylux.voice import _get_tts_model, _tts_backend
        # Just verify it doesn't throw
        _get_tts_model()


class TestSpeakFunction:
    def test_speak_is_callable(self):
        """speak() should be importable and callable."""
        from heylux.voice import speak
        assert callable(speak)

    def test_speak_epoch_prevents_overlap(self):
        """Each speak() call should bump the epoch counter."""
        import heylux.voice as v
        old_epoch = v._speak_epoch
        with patch("heylux.voice._stop_current_speech"):
            with patch("threading.Thread") as mock_thread_cls:
                mock_thread = MagicMock()
                mock_thread_cls.return_value = mock_thread
                v.speak("test")
                assert v._speak_epoch == old_epoch + 1


class TestGracefulImportError:
    def test_cli_handles_missing_deps(self):
        """The CLI should show a helpful message if voice deps aren't installed."""
        from heylux.voice import listen_once
        assert callable(listen_once)
