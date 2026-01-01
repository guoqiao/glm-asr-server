#!/usr/bin/env python3
"""GLM-ASR FastAPI server for audio transcription."""

import os
from contextlib import asynccontextmanager
from typing import Optional, Annotated

from loguru import logger
import torch
import librosa
from fastapi import FastAPI, Form, File, UploadFile, HTTPException

from transcribers import get_transcriber

transcriber = get_transcriber()

@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    transcriber.load()
    yield
    torch.cuda.empty_cache()
    logger.info("Model unloaded")


app = FastAPI(lifespan=lifespan)


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: Annotated[str, Form()] = "whisper-1",  # placeholder
    language: Annotated[str, Form()] = None,
    response_format: Annotated[str, Form()] = "text",  # text, srt, vtt
):
    """Transcribe audio file to text."""
    logger.info(f"{transcriber.__class__.__name__} transcribing {file.filename} -> language {language} format {response_format}")

    # load can accept a file-like object
    # file.file: SpooledTemporaryFile
    audio_ndarray, sr = librosa.load(file.file, sr=16000)
    # path/ndarray/torch.Tensor
    return transcriber.transcribe(audio_ndarray, language=language, format=response_format)


if __name__ == "__main__":
    import uvicorn
    port = os.getenv("PORT", 8000)
    uvicorn.run(app, host="0.0.0.0", port=int(port))
