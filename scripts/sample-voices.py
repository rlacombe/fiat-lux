#!/usr/bin/env python3
"""Play a sample phrase in every available Kokoro voice.

Usage: uv run scripts/sample-voices.py [phrase]
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_TEXT = " ".join(sys.argv[1:]) or "Hey! Setting your lights to candle mode. Done, nightstand is flickering."

# English voices only (a=American, b=British, e=European English)
ENGLISH_VOICES = [
    ("af_heart", "American Female — Heart (default)"),
    ("af_alloy", "American Female — Alloy"),
    ("af_aoede", "American Female — Aoede"),
    ("af_bella", "American Female — Bella"),
    ("af_jessica", "American Female — Jessica"),
    ("af_kore", "American Female — Kore"),
    ("af_nicole", "American Female — Nicole"),
    ("af_nova", "American Female — Nova"),
    ("af_river", "American Female — River"),
    ("af_sarah", "American Female — Sarah"),
    ("af_sky", "American Female — Sky"),
    ("am_adam", "American Male — Adam"),
    ("am_echo", "American Male — Echo"),
    ("am_eric", "American Male — Eric"),
    ("am_fenrir", "American Male — Fenrir"),
    ("am_liam", "American Male — Liam"),
    ("am_michael", "American Male — Michael"),
    ("am_onyx", "American Male — Onyx"),
    ("am_puck", "American Male — Puck"),
    ("bf_alice", "British Female — Alice"),
    ("bf_emma", "British Female — Emma"),
    ("bf_isabella", "British Female — Isabella"),
    ("bf_lily", "British Female — Lily"),
    ("bm_daniel", "British Male — Daniel"),
    ("bm_fable", "British Male — Fable"),
    ("bm_george", "British Male — George"),
    ("bm_lewis", "British Male — Lewis"),
]


def main():
    from mlx_audio.tts.utils import load_model

    print(f"Loading Kokoro model...")
    model = load_model("mlx-community/Kokoro-82M-bf16")
    print(f"Sample: \"{SAMPLE_TEXT}\"\n")

    for voice_id, label in ENGLISH_VOICES:
        print(f"  {label} ({voice_id})  ", end="", flush=True)
        try:
            audio_chunks = []
            for result in model.generate(SAMPLE_TEXT, voice=voice_id):
                audio_chunks.append(result.audio)

            if not audio_chunks:
                print("— no audio")
                continue

            audio = np.concatenate(audio_chunks)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            sf.write(tmp, audio, 24000)
            print("▶", flush=True)
            subprocess.run(["afplay", tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            Path(tmp).unlink(missing_ok=True)
        except Exception as e:
            print(f"— error: {e}")

    print("\nTo set a voice: edit ~/.config/heylux/voice.json")
    print('  {"kokoro_voice": "am_adam"}')


if __name__ == "__main__":
    main()
