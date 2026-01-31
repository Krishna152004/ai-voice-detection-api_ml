from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from utils.audio_utils import load_audio_from_base64
from model.detector import detect_ai_voice

API_KEY = "sk_test_123456789"

app = FastAPI()

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.post("/api/voice-detection")
def voice_detection(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 format is supported")

    try:
        y, sr = load_audio_from_base64(request.audioBase64)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Audio processing failed: {str(e)}"
        )

    classification, confidence, explanation = detect_ai_voice(y, sr)

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidence": confidence,
        "confidenceScore": confidence,   
        "explanation": explanation
    }
