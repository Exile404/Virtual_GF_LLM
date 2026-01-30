#!/usr/bin/env python3
import os
import sys
import socket
import json

os.environ.pop("MPLBACKEND", None)
os.environ["MPLBACKEND"] = "Agg"

import torch

# Fix torch.load
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from TTS.api import TTS

# Configuration
HOST = "127.0.0.1"
PORT = 5123
VOICE_SAMPLE = sys.argv[1] if len(sys.argv) > 1 else "./voice_processed/aiko_voice_combined.wav"

# Load model ONCE
print("Loading XTTS model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print(f"Model loaded on {device}!")
print(f"Voice sample: {VOICE_SAMPLE}")

# Start server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

print(f"TTS Server running on {HOST}:{PORT}")
print("Waiting for requests...")

while True:
    try:
        conn, addr = server.accept()
        data = conn.recv(4096).decode("utf-8")

        if not data:
            conn.close()
            continue

        request = json.loads(data)
        text = request.get("text", "")
        output_path = request.get("output", "/tmp/aiko_speech.wav")

        if text == "SHUTDOWN":
            print("Shutting down server...")
            conn.send(b"OK")
            conn.close()
            break

        # Generate speech
        tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=VOICE_SAMPLE,
            language="en"
        )

        conn.send(b"OK")
        conn.close()
        print(f"Generated: {text[:50]}...")

    except Exception as e:
        print(f"Error: {e}")
        try:
            conn.send(f"ERROR: {e}".encode())
            conn.close()
        except:
            pass

server.close()
