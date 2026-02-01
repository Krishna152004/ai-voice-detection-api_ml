# detector.py
import librosa
import numpy as np

def detect_ai_voice(y, sr):
    
    if len(y.shape) > 1:
        y = np.mean(y, axis=0)
    
    # Extract multiple acoustic features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Extract pitch information (where available)
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0
    
    # Calculate zero crossing rate (useful for detecting synthetic artifacts)
    zero_crossing = librosa.feature.zero_crossing_rate(y)[0]
    
    # Feature analysis
    mfcc_variance = np.var(mfcc)
    pitch_variance = np.var(pitch[pitch > 0]) if np.any(pitch > 0) else 0
    spectral_variance = np.var(spectral_contrast)
    zcr_mean = np.mean(zero_crossing)
    
    # Calculate individual feature scores (normalized)
    spectral_score = min(mfcc_variance / 15000, 1.0)
    pitch_score = min(pitch_variance / 500, 1.0)
    zcr_score = min(zcr_mean * 10, 1.0)
    
  
    feature_weights = {
        'spectral': 0.5,  # MFCC variance weight
        'pitch': 0.3,     # Pitch variance weight
        'zcr': 0.2        # Zero crossing rate weight
    }
    
    # Calculate normalized inverse score (higher = more likely AI)
    ai_likelihood = (
        (1 - spectral_score) * feature_weights['spectral'] +
        (1 - pitch_score) * feature_weights['pitch'] +
        (1 - zcr_score) * feature_weights['zcr']
    )
    
    # Decision threshold for classification
    AI_THRESHOLD = 0.6
    
    # Simple, direct confidence score (matching likelihood directly)
    confidence = float(min(max(ai_likelihood, 0.0), 1.0))
    
    # Generate detailed explanation
    explanations = []
    if spectral_score < 0.4:
        explanations.append("unnatural spectral consistency")
    if pitch_variance < 200 and pitch_mean > 0:
        explanations.append("robotic pitch patterns")
    if zcr_mean < 0.05:
        explanations.append("unusually uniform speech patterns")
    
    # Final classification decision with threshold
    if ai_likelihood > AI_THRESHOLD:
        classification = "AI_GENERATED"
        if explanations:
            explanation = "Detected " + ", ".join(explanations)
        else:
            explanation = "Unnatural spectral consistency detected"
    else:
        classification = "HUMAN"
        explanation = "Natural voice variations observed"
    
    return classification, round(confidence, 2), explanation


def validate_language(language):
 
    SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    return language in SUPPORTED_LANGUAGES
