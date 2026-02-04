import base64
from utils.audio_utils import load_audio_from_base64
from model.detector import detect_ai_voice

# Read MP3
with open("data/sample.mp3", "rb") as f:
    audio_bytes = f.read()

# Encode to Base64
audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
print("Base64 encoding OK")

# Decode + load audio
y, sr = load_audio_from_base64(audio_base64)
print("Audio loaded:", y.shape, "SR:", sr)

# Detect
classification, confidence, explanation = detect_ai_voice(y, sr)

print("\n=== RESULT ===")
print("Classification:", classification)
print("Confidence:", confidence)
print("Explanation:", explanation)
