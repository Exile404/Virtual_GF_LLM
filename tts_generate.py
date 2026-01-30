#!/usr/bin/env python3
import os
import sys

# CRITICAL: Unset matplotlib backend BEFORE any imports
os.environ.pop("MPLBACKEND", None)
os.environ["MPLBACKEND"] = "Agg"

import torch

# Fix torch.load for PyTorch 2.6
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from TTS.api import TTS

# Arguments
voice_sample = sys.argv[1]
output_path = sys.argv[2]
text = sys.argv[3]

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading XTTS on {device}...")

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Generate with cloned voice
print(f"Generating audio...")
tts.tts_to_file(
    text=text,
    file_path=output_path,
    speaker_wav=voice_sample,
    language="en"
)

print(f"SUCCESS: {output_path}")
