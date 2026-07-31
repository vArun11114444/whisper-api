from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel
import tempfile
import os
import time

app = FastAPI(title="Ultra Fast Whisper API")

# Load model only once
model = WhisperModel(
    "tiny",                      # Fastest model
    device="cpu",
    compute_type="int8",
    cpu_threads=os.cpu_count() or 2,
    num_workers=1
)


@app.get("/")
async def root():
    return {"status": "running"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    temp_path = None

    try:
        start = time.perf_counter()

        # Save upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            while True:
                chunk = await file.read(1024 * 1024)   # 1 MB chunks
                if not chunk:
                    break
                tmp.write(chunk)

            temp_path = tmp.name

        # Faster decoding
        segments, info = model.transcribe(
            temp_path,
            beam_size=1,
            best_of=1,
            temperature=0,
            patience=1,
            length_penalty=1,
            compression_ratio_threshold=None,
            log_prob_threshold=None,
            no_speech_threshold=0.6,
            vad_filter=False,
            word_timestamps=False,
            condition_on_previous_text=False,
            multilingual=False
        )

        text = "".join(segment.text for segment in segments).strip()

        return {
            "text": text,
            "language": info.language,
            "processing_time": round(time.perf_counter() - start, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
