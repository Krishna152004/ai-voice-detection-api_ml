# model/detector.py
import joblib
from model.feature_extractor import extract_features

model = joblib.load("model/artifacts/voice_classifier.pkl")

def detect_ai_voice(y, sr, language):
    features = extract_features(y, sr, language)
    prob_ai = model.predict_proba([features])[0][1]

    classification = "AI_GENERATED" if prob_ai >= 0.5 else "HUMAN"
    confidence = round(
        prob_ai if classification == "AI_GENERATED" else 1 - prob_ai,
        2
    )

    explanations = []
    if features[4] < 0.2:
        explanations.append("low pitch variability typical of synthesized speech")
    if features[1] < 100:
        explanations.append("overly smooth spectral patterns")

    explanation = (
        "Detected " + ", ".join(explanations)
        if classification == "AI_GENERATED" and explanations
        else "Natural voice variations observed"
    )

    return classification, confidence, explanation
