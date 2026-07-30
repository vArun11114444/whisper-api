from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel
import tempfile
import os
import time

app = FastAPI(title="Whisper API")

# Load model once
model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8",
    cpu_threads=os.cpu_count() or 2
)


@app.get("/")
def root():
    return {
        "message": "Whisper API is running",
        "status": "healthy"
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    temp_path = None

    try:
        start = time.time()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        segments, info = model.transcribe(
            temp_path,
            beam_size=1,
            best_of=1,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=False
        )

        text = "".join(segment.text for segment in segments).strip()

        return {
            "text": text,
            "language": info.language,
            "probability": info.language_probability,
            "processing_time": round(time.time() - start, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
