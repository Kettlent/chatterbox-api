# server.py
import os
import io
import tempfile
import logging
from pathlib import Path

import torch
import torchaudio
#import ffmpeg

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from chatterbox import Chatterbox

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
MODEL_NAME = "resemble-ai/chatterbox:chatterbox-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG_DIR = Path("/workspace/chatterbox-debug")
DEBUG_DIR.mkdir(exist_ok=True, parents=True)

SUPPORTED_LANGUAGES = {
    "spanish": "es",
    "hindi": "hi",
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("chatterbox-api")

app = FastAPI(title="Chatterbox API with Translation + Voice Cloning")

# global model
model = None


# -------------------------------------------------------
# LOAD MODEL AT STARTUP
# -------------------------------------------------------
@app.on_event("startup")
def load_chatterbox():
    global model
    log.info(f"Loading Chatterbox model: {MODEL_NAME}")
    model = Chatterbox(MODEL_NAME, device=DEVICE)
    log.info("Chatterbox loaded successfully")


# -------------------------------------------------------
# 1) TTS (simple)
# -------------------------------------------------------
@app.post("/tts")
async def tts(
    text: str = Form(...),
    language: str = Form("english")
):
    try:
        lang = SUPPORTED_LANGUAGES.get(language.lower(), "en")

        output_path = DEBUG_DIR / f"tts_{abs(hash(text))}.wav"

        audio = model.generate_speech(
            text=text,
            voice=None,
            language=lang,
        )

        torchaudio.save(str(output_path), audio.unsqueeze(0).cpu(), 44100)

        return FileResponse(output_path, media_type="audio/wav")

    except Exception as e:
        log.exception("TTS error")
        raise HTTPException(500, str(e))


# -------------------------------------------------------
# 2) CLONE VOICE + TTS
# -------------------------------------------------------
@app.post("/clone_voice")
async def clone_voice(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    target_language: str = Form("spanish")
):
    try:
        # save reference wav
        ref_path = DEBUG_DIR / f"ref_{reference_audio.filename}"
        ref_path.write_bytes(await reference_audio.read())

        lang = SUPPORTED_LANGUAGES.get(target_language.lower(), "es")

        output_path = DEBUG_DIR / f"clone_{abs(hash(text))}.wav"

        # load reference wav
        wav, sr = torchaudio.load(ref_path)
        wav = wav.to(DEVICE)

        # generate cloned audio
        audio = model.generate_speech(
            text=text,
            voice=wav,
            language=lang,
        )

        torchaudio.save(str(output_path), audio.unsqueeze(0).cpu(), 44100)

        return FileResponse(output_path, media_type="audio/wav")

    except Exception as e:
        log.exception("Clone voice error")
        raise HTTPException(500, str(e))


# -------------------------------------------------------
# 3) TRANSLATE FULL VIDEO → NEW LANGUAGE + CLONED VOICE
# -------------------------------------------------------
# @app.post("/translate_video")
# async def translate_video(
#     video: UploadFile = File(...),
#     target_language: str = Form("spanish")
# ):
#     try:
#         lang = SUPPORTED_LANGUAGES.get(target_language.lower(), "es")

#         # -------------------------
#         # Save uploaded video
#         # -------------------------
#         temp_video = DEBUG_DIR / f"in_{video.filename}"
#         temp_video.write_bytes(await video.read())

#         log.info(f"Uploaded video saved: {temp_video}")

#         # -------------------------
#         # Extract audio track
#         # -------------------------
#         audio_path = DEBUG_DIR / "extracted.wav"
#         (
#             ffmpeg
#             .input(str(temp_video))
#             .output(str(audio_path), ac=1, ar=44100)
#             .overwrite_output()
#             .run(quiet=True)
#         )
#         log.info(f"Audio extracted → {audio_path}")

#         # load audio
#         wav, sr = torchaudio.load(audio_path)

#         # -------------------------
#         # 1️⃣ Transcribe original audio
#         # -------------------------
#         transcript = model.transcribe(wav.to(DEVICE))
#         original_text = transcript["text"]
#         log.info(f"Transcription: {original_text}")

#         # -------------------------
#         # 2️⃣ Translate text
#         # -------------------------
#         translated_text = model.translate(original_text, lang)
#         log.info(f"Translated text: {translated_text}")

#         # -------------------------
#         # 3️⃣ Voice clone in target language
#         # -------------------------
#         cloned_audio = model.generate_speech(
#             text=translated_text,
#             voice=wav.to(DEVICE),     # cloned voice
#             language=lang,
#         )

#         out_audio = DEBUG_DIR / "translated.wav"
#         torchaudio.save(str(out_audio), cloned_audio.unsqueeze(0).cpu(), 44100)

#         # -------------------------
#         # 4️⃣ Replace audio in video
#         # -------------------------
#         output_video = DEBUG_DIR / f"translated_{video.filename}"

#         (
#             ffmpeg
#             .input(str(temp_video))
#             .video
#             .input(str(out_audio))
#             .output(str(output_video), vcodec="copy", acodec="aac")
#             .overwrite_output()
#             .run(quiet=True)
#         )

#         log.info(f"FINAL VIDEO READY → {output_video}")

#         return FileResponse(output_video, media_type="video/mp4")

#     except Exception as e:
#         log.exception("Video translation error")
#         raise HTTPException(500, str(e))
    
# -------------------------------------------------------
# 4) TRANSLATE AUDIO ONLY → RETURN NEW CLONED SPEECH
# -------------------------------------------------------
@app.post("/translate_audio")
async def translate_audio(
    audio: UploadFile = File(...),
    target_language: str = Form("spanish")
):
    try:
        lang = SUPPORTED_LANGUAGES.get(target_language.lower(), "es")

        # Save uploaded audio to temp file
        temp_audio = DEBUG_DIR / f"in_{audio.filename}"
        temp_audio.write_bytes(await audio.read())

        log.info(f"[translate_audio] Uploaded audio saved: {temp_audio}")

        # Load audio
        wav, sr = torchaudio.load(temp_audio)
        wav = wav.to(DEVICE)

        # -------------------------
        # 1️⃣ Transcribe spoken audio
        # -------------------------
        transcript = model.transcribe(wav)
        original_text = transcript["text"]
        log.info(f"[translate_audio] Transcription: {original_text}")

        # -------------------------
        # 2️⃣ Translate text
        # -------------------------
        translated_text = model.translate(original_text, lang)
        log.info(f"[translate_audio] Translated text: {translated_text}")

        # -------------------------
        # 3️⃣ Clone original speaker’s voice + Generate new speech
        # -------------------------
        cloned_audio = model.generate_speech(
            text=translated_text,
            voice=wav,       # reference for cloning
            language=lang,
        )

        # Save output audio
        out_audio_path = DEBUG_DIR / f"translated_{audio.filename}.wav"
        torchaudio.save(str(out_audio_path), cloned_audio.unsqueeze(0).cpu(), 44100)

        log.info(f"[translate_audio] Output saved: {out_audio_path}")

        return FileResponse(
            out_audio_path,
            media_type="audio/wav",
            filename=out_audio_path.name
        )

    except Exception as e:
        log.exception("translate_audio error")
        raise HTTPException(500, str(e))    


@app.get("/")
def root():
    return {"status": "Chatterbox server running", "model": MODEL_NAME}