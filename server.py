#!/usr/bin/env python3
"""GLM-ASR FastAPI server for audio transcription."""

import os
from contextlib import asynccontextmanager
from typing import Optional, Annotated

from loguru import logger
import torch
import librosa
from fastapi import FastAPI, Form, File, UploadFile, HTTPException

# from transcribers import GLMASRTranscriber as Transcriber
from transcribers import WhisperTranscriber as Transcriber

transcriber = Transcriber()

@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    yield
    torch.cuda.empty_cache()
    logger.info("Model unloaded")


app = FastAPI(lifespan=lifespan)


@app.get("/v1/models")
async def list_models() -> dict:
    """List available models."""
    return {"data": [{"id": "glm-nano-2512", "object": "model"}]}


#@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: Annotated[str, Form()] = "glm-nano-2512",  # only model, ommited
    language: Annotated[str, Form()] = None,
    response_format: Annotated[str, Form()] = "text",  # {"text": "<transcript>"}
):
    """Transcribe audio file to text."""
    logger.info(f"{transcriber.__class__.__name__} transcribing {file.filename} -> language {language} format {response_format}")

    try:
        transcriber.clean_format(response_format)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # load can accept a file-like object
    # file.file: SpooledTemporaryFile
    audio_ndarray, sr = librosa.load(file.file, sr=16000)
    # path/ndarray/torch.Tensor
    text = transcriber.transcribe(audio_ndarray, language=language, format=response_format)
    logger.info(f"transcript:\n{text}\n")
    if response_format == "json":
        return {"text": text}
    if response_format in ["text", "txt", "srt", "vtt"]:
        return text


if __name__ == "__main__":
    import uvicorn
    port = os.getenv("PORT", 8000)
    uvicorn.run(app, host="0.0.0.0", port=int(port))
