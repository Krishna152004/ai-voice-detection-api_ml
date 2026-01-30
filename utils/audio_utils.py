import base64
import io
from pydub import AudioSegment
import librosa

def load_audio_from_base64(audio_base64: str):
    
    audio_bytes = base64.b64decode(audio_base64)

    if not audio_bytes:
        raise ValueError("Decoded audio is empty")

    audio = AudioSegment.from_file(
        io.BytesIO(audio_bytes),
        format="mp3"
    )
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    y, sr = librosa.load(wav_io, sr=None)

    return y, sr
