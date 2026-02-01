# utils/audio_utils.py

import base64
import io
import logging
from pydub import AudioSegment
import librosa
import numpy as np

def load_audio_from_base64(audio_base64: str):
    
    try:
        # 1️⃣ Decode Base64 → bytes
        audio_bytes = base64.b64decode(audio_base64)

        if not audio_bytes:
            raise ValueError("Decoded audio is empty")

        # 2️⃣ Read MP3 bytes
        audio = AudioSegment.from_file(
            io.BytesIO(audio_bytes),
            format="mp3"
        )

        # Validate audio duration (prevent extremely short or empty clips)
        if len(audio) < 500:  # Less than 500ms
            logging.warning("Audio clip is very short (<500ms), analysis may be unreliable")

        # 3️⃣ Convert to WAV in memory
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # 4️⃣ Load waveform with consistent sample rate
        y, sr = librosa.load(wav_io, sr=22050)  # Using fixed sample rate for consistency

        # Ensure minimum length for analysis
        if len(y) < sr * 0.5:  # Less than 0.5 seconds
            # Pad short samples to ensure sufficient data for analysis
            y = np.pad(y, (0, int(sr * 0.5) - len(y)))
            logging.warning("Audio padded to minimum length for analysis")

        return y, sr
        
    except Exception as e:
        logging.error(f"Error processing audio: {str(e)}")
        raise ValueError(f"Failed to process audio: {str(e)}")
