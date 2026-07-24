from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI(title="Whisper API")

# Load the model once when the app starts
model = WhisperModel("base", device="cpu", compute_type="int8")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    temp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # Transcribe
        segments, info = model.transcribe(temp_path)

        text = " ".join(segment.text.strip() for segment in segments)

        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
