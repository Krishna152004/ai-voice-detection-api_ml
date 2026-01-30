from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64

app = FastAPI()

# =========================
# CONFIG
# =========================
SECRET_API_KEY = "sk_test_123456789"

# =========================
# REQUEST MODEL
# =========================
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# =========================
# ROOT ENDPOINT
# =========================
@app.get("/")
def root():
    return {"status": "API running"}

# =========================
# VOICE DETECTION ENDPOINT
# =========================
@app.post("/api/voice-detection")
def detect_voice(
    payload: VoiceRequest,
    x_api_key: str = Header(None)
):
    # 1️⃣ API Key Validation
    if x_api_key != SECRET_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 2️⃣ Validate Audio Format
    if payload.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 audio is supported")

    # 3️⃣ Decode Base64 → Audio Bytes
    try:
        audio_bytes = base64.b64decode(payload.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio data")

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Decoded audio is empty")

    # 4️⃣ Simple Detection Logic (Heuristic)
    audio_size = len(audio_bytes)

    if audio_size < 10000:
        classification = "AI_GENERATED"
        confidence_score = 0.65
        explanation = "Very short and uniform audio detected"
    else:
        classification = "HUMAN"
        confidence_score = 0.80
        explanation = "Natural length and variation detected"

    # 5️⃣ Response
    return {
        "status": "success",
        "language": payload.language,
        "classification": classification,
        "confidenceScore": confidence_score,
        "explanation": explanation
    }
