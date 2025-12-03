import os
import io
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

app = FastAPI(title="Chatterbox Voice-Cloning API")

# -----------------------
# DEVICE SETUP
# -----------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"[INFO] Using device: {DEVICE}")

# -----------------------
# LOAD MODEL ON STARTUP
# -----------------------
@app.on_event("startup")
def load_model():
    global tts_multi
    print("[INFO] Loading multilingual model...")
    tts_multi = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    print("[INFO] Model loaded successfully.")


# -----------------------
# /tts_clone ENDPOINT
# -----------------------
@app.post("/tts_clone")
async def tts_clone(
    text: str = Form(...),
    target_language: str = Form(...),   # expects language codes: "es", "hi", "fr", etc.
    voice_sample: UploadFile = File(...)
):
    """
    Generate WAV audio using:
    - text prompt
    - target language (language ID)
    - reference voice sample for cloning
    """

    # Validate request
    if voice_sample is None:
        raise HTTPException(status_code=400, detail="voice_sample file missing.")

    # Save uploaded voice sample temporarily
    try:
        temp_path = f"/tmp/{voice_sample.filename}"
        with open(temp_path, "wb") as f:
            f.write(await voice_sample.read())
        print(f"[INFO] Saved voice sample → {temp_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save voice sample: {e}")

    try:
        # Generate cloned speech
        print(f"[INFO] Generating voice clone: lang={target_language}")

        wav = tts_multi.generate(
            text,
            language_id=target_language,
            audio_prompt_path=temp_path
        )

        # Convert to CPU tensor for saving
        wav = wav.cpu()

        # Write to memory buffer as WAV
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav.unsqueeze(0), tts_multi.sr, format="wav")
        buffer.seek(0)

        filename = f"tts_clone_{abs(hash(text))}.wav"

        print("[INFO] Returning generated WAV")

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        print("[ERROR] Generation failed:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        try:
            os.remove(temp_path)
        except:
            pass


@app.get("/")
def root():
    return {"status": "Chatterbox API running", "device": DEVICE}