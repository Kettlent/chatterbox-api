# server.py
import os
import io
import tempfile
import logging
from pathlib import Path

import torch
import torchaudio
# import ffmpeg

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

# Chatterbox Modules
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG_DIR = Path("/workspace/chatterbox-debug")
DEBUG_DIR.mkdir(exist_ok=True, parents=True)

SUPPORTED_LANGUAGES = {
    "english": "en",
    "spanish": "es",
    "hindi": "hi",
    "french": "fr",
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("chatterbox-api")

app = FastAPI(title="Chatterbox Audio Translation API")

# Global models
tts_model = None
tts_multi = None


# -------------------------------------------------------
# LOAD MODELS AT STARTUP
# -------------------------------------------------------
@app.on_event("startup")
def load_models():
    global tts_model, tts_multi

    log.info("Loading Chatterbox single-language TTS...")
    tts_model = ChatterboxTTS.from_pretrained(device=DEVICE)
    log.info("Loaded ChatterboxTTS.")

    log.info("Loading Chatterbox multilingual TTS...")
    tts_multi = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    log.info("Loaded ChatterboxMultilingualTTS.")

    log.info("All models initialized successfully.")


# -------------------------------------------------------
# 1) BASIC TTS
# -------------------------------------------------------
@app.post("/tts")
async def tts(text: str = Form(...)):
    try:
        wav = tts_model.generate(text)
        out_path = DEBUG_DIR / "tts.wav"
        torchaudio.save(str(out_path), wav.unsqueeze(0), tts_model.sr)
        return FileResponse(out_path, media_type="audio/wav")
    except Exception as e:
        log.exception("TTS error")
        raise HTTPException(500, str(e))


# -------------------------------------------------------
# 2) MULTILINGUAL TTS
# -------------------------------------------------------
@app.post("/tts_multilingual")
async def tts_multilingual(
    text: str = Form(...),
    language: str = Form("english")
):
    try:
        lang_id = SUPPORTED_LANGUAGES.get(language.lower(), "en")
        wav = tts_multi.generate(text, language_id=lang_id)

        out_path = DEBUG_DIR / "tts_multi.wav"
        torchaudio.save(str(out_path), wav.unsqueeze(0), tts_multi.sr)

        return FileResponse(out_path, media_type="audio/wav")
    except Exception as e:
        log.exception("Multilingual TTS error")
        raise HTTPException(500, str(e))


# -------------------------------------------------------
# 3) CLONE VOICE (reference audio + text)
# -------------------------------------------------------
@app.post("/clone_voice")
async def clone_voice(
    reference_audio: UploadFile = File(...),
    text: str = Form(...)
):
    try:
        ref_path = DEBUG_DIR / f"ref_{reference_audio.filename}"
        ref_path.write_bytes(await reference_audio.read())

        wav, sr = torchaudio.load(ref_path)
        wav = wav.to(DEVICE)

        # Use audio_prompt_path instead of raw tensor
        # Chatterbox expects a file path
        cloned = tts_model.generate(text, audio_prompt_path=str(ref_path))

        out_path = DEBUG_DIR / "clone.wav"
        torchaudio.save(str(out_path), cloned.unsqueeze(0), tts_model.sr)

        return FileResponse(out_path, media_type="audio/wav")

    except Exception as e:
        log.exception("Clone voice error")
        raise HTTPException(500, str(e))


# -------------------------------------------------------
# 4) TRANSLATE AUDIO → NEW AUDIO (main feature)
# -------------------------------------------------------
@app.post("/translate_audio")
async def translate_audio(
    audio: UploadFile = File(...),
    target_language: str = Form("spanish")
):
    try:
        # Step 1: Save uploaded audio
        in_path = DEBUG_DIR / f"in_{audio.filename}"
        in_path.write_bytes(await audio.read())

        # Load audio
        wav, sr = torchaudio.load(in_path)
        wav = wav.to(DEVICE)

        # Step 2: Transcribe (via multilingual model)
        transcription = tts_multi.transcribe(wav)
        original_text = transcription["text"]
        log.info(f"Transcription: {original_text}")

        # Step 3: Translate text
        lang_id = SUPPORTED_LANGUAGES.get(target_language.lower(), "es")
        translated_text = tts_multi.translate(original_text, lang_id)
        log.info(f"Translated text: {translated_text}")

        # Step 4: Clone original speaker’s voice in target language
        cloned_audio = tts_multi.generate(
            translated_text,
            language_id=lang_id,
            audio_prompt_path=str(in_path)
        )

        # Step 5: Save output
        out_path = DEBUG_DIR / f"translated_{audio.filename}.wav"
        torchaudio.save(str(out_path), cloned_audio.unsqueeze(0), tts_multi.sr)

        return FileResponse(out_path, media_type="audio/wav")

    except Exception as e:
        log.exception("Translate audio error")
        raise HTTPException(500, str(e))


@app.get("/")
def root():
    return {"status": "Chatterbox audio translation server running."}