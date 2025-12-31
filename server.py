#!/usr/bin/env python3
"""GLM-ASR FastAPI server for audio transcription."""

import os
from contextlib import asynccontextmanager
from typing import Optional, Annotated

from loguru import logger
import torch
import librosa
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from pydantic import BaseModel

from inference import Transcriber


transcriber = Transcriber()


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    transcriber.load()
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
    response_format: Annotated[str, Form()] = "json",  # {"text": "<transcript>"}
):
    """Transcribe audio file to text."""
    logger.info(f"transcribing file {file.filename} content_type {file.content_type} -> language {language} format {response_format}")

    formats = ["json", "text"]

    if response_format not in formats:
        raise HTTPException(400, f"response_format {response_format} not in {formats}")

    # load can accept a file-like object
    # file.file: SpooledTemporaryFile
    audio_ndarray, sr = librosa.load(file.file, sr=16000)
    # path/ndarray/torch.Tensor
    text = transcriber.transcribe(audio_ndarray, lang=language)
    if response_format == "json":
        return {"text": text}
    if response_format == "text":
        return text


if __name__ == "__main__":
    import uvicorn
    port = os.getenv("PORT", 8000)
    uvicorn.run(app, host="0.0.0.0", port=int(port))
