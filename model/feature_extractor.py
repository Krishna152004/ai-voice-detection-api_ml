# model/feature_extractor.py
import numpy as np
import librosa

def extract_features(y, sr, language=None):
    if len(y.shape) > 1:
        y = np.mean(y, axis=0)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    pitch, _ = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitch[pitch > 0]

    pitch_mean = np.mean(pitch_vals) if len(pitch_vals) else 0
    pitch_var = np.var(pitch_vals) if len(pitch_vals) else 0

    
    LANG_PITCH_RANGE = {
        "English": (80, 300),
        "Hindi": (90, 320),
        "Tamil": (85, 310),
        "Telugu": (85, 310),
        "Malayalam": (80, 300)
    }

    min_p, max_p = LANG_PITCH_RANGE.get(language, (80, 300))
    pitch_norm = np.clip((pitch_mean - min_p) / (max_p - min_p), 0, 1)

    features = [
        np.mean(mfcc),
        np.var(mfcc),
        np.mean(spectral_contrast),
        np.var(spectral_contrast),
        pitch_norm,
        pitch_var,
        np.mean(zcr)
    ]

    return np.array(features, dtype=np.float32)
