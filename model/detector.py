import librosa
import numpy as np

def detect_ai_voice(y, sr):
    
    if len(y) < sr:
        return (
            "HUMAN",
            0.0,
            "Audio too short for reliable analysis"
        )

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_variance = np.var(mfcc)

    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    mfcc_score = max(0.0, min(1.0, 1.0 - (mfcc_variance / 15000)))
    flatness_score = min(spectral_flatness, 1.0)

    ai_likelihood = round(
        (mfcc_score * 0.7) + (flatness_score * 0.3),
        2
    )

    if ai_likelihood == float(round(
            (mfcc_score * 0.7) + (flatness_score * 0.3),
            2
        ))
    else:
        classification = "HUMAN"
        explanation = (
            "Higher spectral variance and natural frequency fluctuations "
            "indicate human speech"
        )

    return classification, ai_likelihood, explanation
