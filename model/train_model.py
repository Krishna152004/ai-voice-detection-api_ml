# model/train_model.py
import os
import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model.feature_extractor import extract_features

DATASET = [
    # English – AI
    ("C:/Users/91981/Desktop/Voice/ai-voice-detection-api/data/sample.mp3", "English", 1),
    ("C:/Users/91981/Desktop/Voice/ai-voice-detection-api/data/english-a.mp3", "English", 1),

    # English – Human
    ("C:/Users/91981/Desktop/Voice/ai-voice-detection-api/data/english-h.mp3", "English", 0),

    # Hindi – Human
    ("C:/Users/91981/Desktop/Voice/ai-voice-detection-api/data/10.mp3", "Hindi", 0),
    ("C:/Users/91981/Desktop/Voice/ai-voice-detection-api/data/audio1.mp3", "Hindi", 0),

    # Malayalam – Human
    ("C:/Users/91981/Desktop/Voice/ai-voice-detection-api/data/malyalam-h.mp3", "Malayalam", 0),
    ("C:/Users/91981/Desktop/Voice/ai-voice-detection-api/data/malyalam-hh.mp3", "Malayalam", 0),

    # Telugu – Human
    ("C:/Users/91981/Desktop/Voice/ai-voice-detection-api/data/telgu-h.mp3", "Telugu", 0),
    ("C:/Users/91981/Desktop/Voice/ai-voice-detection-api/data/telgu-hh.mp3", "Telugu", 0),

    # Tamil – Human
    ("C:/Users/91981/Desktop/Voice/ai-voice-detection-api/data/tamil-h.mp3", "Tamil", 0),
    
]




X, y = [], []

for audio_path, language, label in DATASET:
    audio, sr = librosa.load(audio_path, sr=None)
    features = extract_features(audio, sr, language)
    X.append(features)
    y.append(label)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

os.makedirs("model/artifacts", exist_ok=True)
joblib.dump(model, "model/artifacts/voice_classifier.pkl")

print("✅ Model trained and saved")



