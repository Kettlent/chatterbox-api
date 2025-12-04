import os
import io
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

app = FastAPI(title="Chatterbox Voice-Cloning API")

#uvicorn server:app --host 0.0.0.0 --port 8000 --reload
# curl -X POST https://YOUR_POD/tts_clone \
#   -F text="Hola, ¿cómo estás? Este es el modelo de voz multilingüe…" \
#   -F target_language="es" \
#   -F voice_sample=@/path/sample.wav \
#   --output result.wav


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
# LOAD MODEL ONCE AT STARTUP
# -----------------------
@app.on_event("startup")
def load_model():
    global tts_multi
    print("[INFO] Loading Chatterbox Multilingual model...")
    tts_multi = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    print("[INFO] Model loaded successfully.")


# -----------------------
#  /tts_clone : Text + Voice Sample → WAV
# -----------------------
@app.post("/tts_clone")
async def tts_clone(
    text: str = Form(...),
    target_language: str = Form(...),
    voice_sample: UploadFile = File(...),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5)
):
    """
    Takes:
        - text prompt
        - target language (e.g., "es", "hi", "fr")
        - reference voice audio (wav/m4a/mp3)
    Returns:
        - Fully generated WAV file (voice cloned into that language)
    """

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if voice_sample is None:
        raise HTTPException(status_code=400, detail="Missing voice_sample file.")


    # -------------------------------------
    # Save uploaded reference audio to temp
    # -------------------------------------
    try:
        temp_path = f"/tmp/{voice_sample.filename}"
        with open(temp_path, "wb") as f:
            f.write(await voice_sample.read())
        print(f"[INFO] Saved voice sample → {temp_path}")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded audio file: {e}"
        )


    # -------------------------------------
    # Generate audio
    # -------------------------------------
    try:
        print(f"[INFO] Generating TTS → lang={target_language}")
        wav = tts_multi.generate(
            text=text,
            language_id=target_language,
            audio_prompt_path=temp_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        ).cpu()

        # Ensure waveform is 2D: (channels, samples)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # Save to memory buffer (return full file)
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav, tts_multi.sr, format="wav")
        buffer.seek(0)

        filename = f"tts_clone_{abs(hash(text))}.wav"

        print("[INFO] Returning full WAV file.")

        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )

    except Exception as e:
        print("[ERROR] TTS generation failed:", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass


# -----------------------
#  Health Check
# -----------------------
@app.get("/")
def root():
    return {
        "status": "Chatterbox API running",
        "device": DEVICE,
        "model": "Multilingual-TTS"
    }