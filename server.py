"""GLM-ASR FastAPI server for audio transcription."""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import ffmpeg
import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from pydub import AudioSegment
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
)

logger = logging.getLogger(__name__)


class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoint."""

    text: str


class ModelState:
    """Container for model and component states."""

    def __init__(self) -> None:
        """Initialize model state with None values."""
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.feature_extractor: Optional[WhisperFeatureExtractor] = None
        self.config: Optional[AutoConfig] = None
        self.device: Optional[torch.device] = None


model_state = ModelState()

WHISPER_FEAT_CFG = {
    "chunk_length": 30,
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "WhisperProcessor",
    "return_attention_mask": False,
    "sampling_rate": 16000,
}

MODEL_ID = os.getenv("MODEL_ID", "zai-org/GLM-ASR-Nano-2512")
CHUNK_DURATION_MS = 30 * 1000  # 30 second chunks
CHUNK_OVERLAP_MS = 2 * 1000  # 2 second overlap


async def load_model_fn() -> None:
    """Load model and components on startup."""
    model_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_state.config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model_state.model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=model_state.config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(model_state.device)
    model_state.tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    model_state.feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)
    model_state.model.eval()
    logger.info(f"Model loaded on device: {model_state.device}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """Manage model lifecycle (startup/shutdown)."""
    await load_model_fn()
    yield
    if model_state.model is not None:
        del model_state.model
        torch.cuda.empty_cache()
        logger.info("Model unloaded")


app = FastAPI(lifespan=lifespan)


@app.get("/v1/models")
async def list_models() -> dict:
    """List available models."""
    return {"data": [{"id": "glm-nano-2512", "object": "model"}]}


def get_audio_token_length(seconds: float, merge_factor: int = 2) -> int:
    """Calculate number of audio tokens from duration in seconds."""

    def get_T_after_cnn(L_in: int, dilation: int = 1) -> int:
        """Apply CNN transformations to compute output length."""
        for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

    mel_len = int(seconds * 100)
    audio_len_after_cnn = get_T_after_cnn(mel_len)
    audio_token_num = (audio_len_after_cnn - merge_factor) // merge_factor + 1
    audio_token_num = min(audio_token_num, 1500 // merge_factor)
    return audio_token_num


def split_audio_into_chunks(audio_path: str) -> list[tuple[int, int]]:
    """Load audio and return list of (start_ms, end_ms) tuples for chunks."""
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        chunks = []
        step = CHUNK_DURATION_MS - CHUNK_OVERLAP_MS

        for i in range(0, duration_ms, step):
            chunk_end = min(i + CHUNK_DURATION_MS, duration_ms)
            chunks.append((i, chunk_end))
            if chunk_end >= duration_ms:
                break

        logger.info(f"[AUDIO CHUNKING] Split {duration_ms}ms into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"[AUDIO CHUNKING FAILED] {str(e)}")
        return [(0, duration_ms)]


def build_prompt(audio_path: str, merge_factor: int, chunk_seconds: int = 30) -> dict:
    """Build prompt batch from audio file."""
    logger.debug(f"[BUILD_PROMPT] Loading audio from {audio_path}")
    wav, sr = torchaudio.load(audio_path)
    logger.debug(f"  - Loaded shape: {wav.shape}, sample rate: {sr}")

    wav = wav[:1, :]
    logger.debug(f"  - After mono conversion: {wav.shape}")

    assert model_state.feature_extractor is not None
    assert model_state.tokenizer is not None

    if sr != model_state.feature_extractor.sampling_rate:
        logger.debug(
            f"  - Resampling from {sr} to {model_state.feature_extractor.sampling_rate}"
        )
        wav = torchaudio.transforms.Resample(
            sr, model_state.feature_extractor.sampling_rate
        )(wav)
        logger.debug(f"  - After resampling: {wav.shape}")

    tokens = []
    tokens += model_state.tokenizer.encode("<|user|>")
    tokens += model_state.tokenizer.encode("\n")
    logger.debug(f"  - Initial tokens: {len(tokens)}")

    audios = []
    audio_offsets = []
    audio_length = []
    chunk_size = chunk_seconds * model_state.feature_extractor.sampling_rate
    total_chunks = (wav.shape[1] + chunk_size - 1) // chunk_size
    logger.debug(
        f"  - Total chunks to process: {total_chunks} (chunk_size={chunk_size})"
    )

    for chunk_idx, start in enumerate(range(0, wav.shape[1], chunk_size)):
        chunk = wav[:, start : start + chunk_size]
        logger.debug(f"    - Chunk {chunk_idx + 1}/{total_chunks}: shape={chunk.shape}")

        mel = model_state.feature_extractor(
            chunk.numpy(),
            sampling_rate=model_state.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding="max_length",
        )["input_features"]
        logger.debug(f"      - MEL shape: {mel.shape}")

        audios.append(mel)
        seconds = chunk.shape[1] / model_state.feature_extractor.sampling_rate
        num_tokens = get_audio_token_length(seconds, merge_factor)

        tokens += model_state.tokenizer.encode("<|begin_of_audio|>")
        audio_offsets.append(len(tokens))
        tokens += [0] * num_tokens
        tokens += model_state.tokenizer.encode("<|end_of_audio|>")
        audio_length.append(num_tokens)

        logger.debug(
            f"      - Duration: {seconds:.2f}s, tokens: {num_tokens}, "
            f"total tokens now: {len(tokens)}"
        )

    if not audios:
        logger.error("[BUILD_PROMPT] No audio chunks loaded!")
        raise ValueError("Audio is empty or failed to load")

    tokens += model_state.tokenizer.encode("<|user|>")
    tokens += model_state.tokenizer.encode("\nPlease transcribe this audio into text")
    tokens += model_state.tokenizer.encode("<|assistant|>")
    tokens += model_state.tokenizer.encode("\n")

    batch = {
        "input_ids": torch.tensor([tokens], dtype=torch.long),
        "audios": torch.cat(audios, dim=0),
        "audio_offsets": [audio_offsets],
        "audio_length": [audio_length],
        "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
    }
    return batch


def run_inference(batch: dict) -> str:
    """Run inference on prepared batch and return transcript."""
    assert model_state.model is not None
    assert model_state.tokenizer is not None
    assert model_state.device is not None

    # Prepare inputs
    logger.debug("[MODEL PREP] Moving tensors to device...")
    input_ids = batch["input_ids"].to(model_state.device)
    attention_mask = batch["attention_mask"].to(model_state.device)
    audios = batch["audios"].to(model_state.device)

    model_inputs = {
        "inputs": input_ids,
        "attention_mask": attention_mask,
        "audios": audios.to(torch.bfloat16),
        "audio_offsets": batch["audio_offsets"],
        "audio_length": batch["audio_length"],
    }
    prompt_len = input_ids.size(1)

    # Inference
    logger.info("[INFERENCE START] Running model.generate()...")
    with torch.no_grad():
        generated = model_state.model.generate(
            **model_inputs,
            max_new_tokens=128,
            do_sample=False,
        )
    logger.info("[INFERENCE SUCCESS] Generation completed")

    # Decode
    logger.debug("[DECODING] Extracting and decoding transcript...")
    transcript_ids = generated[0, prompt_len:].cpu().tolist()
    transcript = model_state.tokenizer.decode(
        transcript_ids, skip_special_tokens=True
    ).strip()

    # logger.info(
    #     f"[TRANSCRIPTION RESULT] '{transcript[:100]}'"
    #     f"{'...' if len(transcript) > 100 else ''}"
    # )
    print('\n\n', transcript, '\n\n')
    return transcript



@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...), language: Optional[str] = "auto"
) -> TranscriptionResponse:
    """Transcribe audio file to text."""
    logger.debug("=== TRANSCRIPTION REQUEST RECEIVED ===")

    if (
        not model_state.model
        or not model_state.tokenizer
        or not model_state.feature_extractor
        or not model_state.config
    ):
        logger.error("Model not loaded!")
        raise HTTPException(500, "Model not loaded")

    # Validate file
    if not file.filename:
        logger.error("No filename provided in request")
        raise HTTPException(400, "No filename provided")

    logger.info(f"[FRONTEND->BACKEND] File received: {file.filename}")
    logger.debug(f"  - Content-Type: {file.content_type}")
    logger.debug(f"  - Language: {language}")

    # Read file with size check
    try:
        logger.debug("Reading file content from request...")
        content = await file.read()

        if len(content) == 0:
            logger.error("File content is empty")
            raise HTTPException(400, "Empty file received")

        # Check for valid audio file headers
        file_sig = content[:4].hex().lower()
        is_mp4 = (
            file_sig.startswith("00000020")
            or file_sig[4:8] == "6674"
            or content[4:8] == b"ftyp"
        )
        is_wav = content[:4] == b"RIFF"

        if not (is_mp4 or is_wav):
            logger.warning(f"[FILE VALIDATION] Unknown format signature: {file_sig}")

        logger.info(
            f"[FILE UPLOAD SUCCESS] Received {len(content)} bytes "
            f"({len(content) / (1024 * 1024):.2f} MB)"
        )
        logger.debug(f"  - First 20 bytes (hex): {content[:20].hex()}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[FILE UPLOAD FAILED] Error reading file: {str(e)}", exc_info=True
        )
        raise HTTPException(400, f"Failed to read uploaded file: {str(e)}")

    # Check if conversion is needed
    logger.debug("[AUDIO CHECK] Checking if conversion is needed...")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input_audio")
        audio_path = os.path.join(tmpdir, "audio.wav")

        try:
            # Write uploaded content to temp file
            logger.debug(f"  - Writing {len(content)} bytes to temporary file...")
            with open(input_path, "wb") as f:
                f.write(content)

            # Check audio properties
            needs_conversion = False
            duration_seconds = 0
            try:
                # Try to load as-is to check properties
                if is_wav:
                    logger.debug("[AUDIO CHECK] File is WAV, probing properties...")
                    wav, sr = torchaudio.load(input_path)
                    duration_seconds = wav.shape[1] / sr
                    logger.debug(
                        f"  - Sample rate: {sr}, duration: {duration_seconds:.2f}s"
                    )
                    needs_conversion = sr != 16000
                    if needs_conversion:
                        logger.info(
                            f"[AUDIO CHECK] Conversion needed: sample rate {sr} != 16000"
                        )
                else:
                    logger.debug("[AUDIO CHECK] File is not WAV, conversion needed")
                    needs_conversion = True
            except Exception as e:
                logger.warning(f"[AUDIO CHECK] Could not probe, will convert: {str(e)}")
                needs_conversion = True

            # Check if audio is longer than 1 minute
            is_long_audio = duration_seconds > 60
            if is_long_audio:
                logger.info(
                    f"[AUDIO CHECK] Audio is long ({duration_seconds:.2f}s > 60s), "
                    "will convert for chunking"
                )
                needs_conversion = True

            # Convert if needed
            if needs_conversion:
                logger.info(
                    "[AUDIO CONVERSION] Starting ffmpeg conversion to WAV format..."
                )
                try:
                    logger.debug(
                        f"  - Input file size: {os.path.getsize(input_path)} bytes"
                    )
                    logger.debug(f"  - Output path: {audio_path}")
                    logger.debug(
                        "  - Ffmpeg parameters: format=wav, acodec=pcm_s16le, ar=16000"
                    )

                    input_stream = ffmpeg.input(input_path)
                    out_stream = ffmpeg.output(
                        input_stream,
                        audio_path,
                        format="wav",
                        acodec="pcm_s16le",
                        ar=16000,
                    )
                    ffmpeg.run(out_stream, overwrite_output=True)

                    converted_size = (
                        os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
                    )
                    if converted_size < 1000:
                        logger.error(
                            f"[AUDIO CONVERSION FAILED] Output file too small: {converted_size} bytes"
                        )
                        raise HTTPException(
                            400,
                            "Audio conversion failed - output file is empty or too small. "
                            "Input file may be corrupted.",
                        )

                    logger.info(
                        f"[AUDIO CONVERSION SUCCESS] Converted to WAV: {converted_size} bytes "
                        f"({converted_size / (1024 * 1024):.2f} MB)"
                    )
                except ffmpeg.Error as e:
                    error_msg = e.stderr.decode() if e.stderr else str(e)
                    logger.error(
                        f"[FFMPEG CONVERSION FAILED] FFmpeg error: {error_msg}",
                        exc_info=True,
                    )
                    raise HTTPException(400, f"Audio conversion failed: {error_msg}")
                except HTTPException:
                    raise
            else:
                # No conversion needed, use original file
                logger.info(
                    "[AUDIO CHECK] No conversion needed, using original WAV file"
                )
                audio_path = input_path

        except Exception as e:
            logger.error(
                f"[AUDIO PROCESSING ERROR] Unexpected error: {str(e)}", exc_info=True
            )
            raise HTTPException(500, f"Audio processing error: {str(e)}")

        try:
            # Split audio into chunks
            chunks = split_audio_into_chunks(audio_path)

            if len(chunks) == 1:
                # Single chunk - process normally
                logger.info("[AUDIO CHUNKING] Processing as single chunk")
                batch = build_prompt(audio_path, model_state.config.merge_factor)
                logger.info("[AUDIO PROMPT] Built successfully")
                transcript = run_inference(batch)
            else:
                # Multiple chunks - process each and merge
                logger.info(
                    f"[AUDIO CHUNKING] Processing {len(chunks)} chunks sequentially"
                )
                audio = AudioSegment.from_file(audio_path)
                transcripts = []

                for chunk_idx, (start_ms, end_ms) in enumerate(chunks, 1):
                    logger.info(
                        f"[CHUNK {chunk_idx}/{len(chunks)}] Processing {start_ms}ms-{end_ms}ms..."
                    )

                    # Extract chunk
                    audio_chunk = audio[start_ms:end_ms]

                    # Save chunk to temp file
                    chunk_path = os.path.join(
                        os.path.dirname(audio_path), f"chunk_{chunk_idx}.wav"
                    )
                    audio_chunk.export(chunk_path, format="wav")

                    try:
                        batch = build_prompt(
                            audio_path, model_state.config.merge_factor
                        )
                        chunk_text = run_inference(batch)

                        if chunk_text:
                            transcripts.append(chunk_text)
                            logger.info(
                                f"[CHUNK {chunk_idx}] Result: '{chunk_text[:80]}'"
                            )
                    finally:
                        os.unlink(chunk_path)

                transcript = " ".join(transcripts)
                logger.info(
                    f"[TRANSCRIPTION COMPLETE] Merged {len(transcripts)} chunks "
                    f"into {len(transcript)} chars"
                )

            return TranscriptionResponse(text=transcript or "[Empty transcription]")

        except ValueError as e:
            logger.error(
                f"[AUDIO PROCESSING ERROR] ValueError: {str(e)}", exc_info=True
            )
            raise HTTPException(400, f"Audio processing error: {str(e)}")
        except RuntimeError as e:
            logger.error(
                f"[MODEL INFERENCE ERROR] RuntimeError: {str(e)}", exc_info=True
            )
            raise HTTPException(500, f"Inference failed: {str(e)}")
        except Exception as e:
            logger.error(f"[INFERENCE ERROR] Unexpected error: {str(e)}", exc_info=True)
            raise HTTPException(500, f"Transcription error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    port = os.getenv("PORT", 8000)
    uvicorn.run(app, host="0.0.0.0", port=int(port))
