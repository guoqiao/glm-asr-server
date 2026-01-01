#!/usr/bin/env python3
"""ASR: Automatic Speech Recognition."""

import json
import os
import time
from pathlib import Path

from loguru import logger


def sec2ts(t: float|str) -> str:
    # 65.234 -> 00:01:05.234
    f = float(t)
    n = int(t)
    ms = int((f - n) * 1000)  # 0.234 -> 234
    h, s = divmod(n, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class Transcriber:
    supported_formats = ["text", "txt"]

    def load(self):
        # for time consuming operations, e.g. loading model, downloading, etc.
        pass

    def clean_file(self, file: str) -> str:
        # check file data type, ext, etc.
        return file

    def clean_format(self, format: str) -> bool:
        if self.supported_formats:
            if format not in self.supported_formats:
                raise ValueError(f"requested format '{format}' not in {self.supported_formats}")
        return format

    def clean_language(self, language: str) -> str:
        # check supported langs, map lang code, etc.
        return language

    def transcribe(self, file, language: str = None, format: str = "text") -> str:
        file = self.clean_file(file)
        language = self.clean_language(language)
        format = self.clean_format(format)
        text = self._transcribe(file, language=language, format=format)
        logger.info(f"transcript:\n{text}\n")
        return text


class AssemblyAITranscriber(Transcriber):

    supported_formats = ["text", "srt", "vtt"]

    def __init__(self):
        import assemblyai as aai
        aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]

    def _transcribe(self, file: str, language: str = None, format: str = "text") -> str:
        import assemblyai as aai

        if language:
            config = aai.TranscriptionConfig(
                speech_model=aai.SpeechModel.universal,
                language_code=language,
                punctuate=True,
            )
        else:
            options = aai.LanguageDetectionOptions(expected_languages=["zh", "en"])
            config = aai.TranscriptionConfig(
                speech_model=aai.SpeechModel.universal,
                language_detection=True,
                language_detection_options=options,
                punctuate=True,
            )
        transcript_obj: aai.transcriber.Transcript = aai.Transcriber(
            config=config
        ).transcribe(str(file))
        if transcript_obj.status == "error":
            raise RuntimeError(f"AssemblyAI transcribe failed for {file}: {transcript_obj.error}")

        if format in ["text", "txt"]:
            return transcript_obj.text
        elif format in ["srt"]:
            return transcript_obj.export_subtitles_srt()
        elif format in ["vtt"]:
            return transcript_obj.export_subtitles_vtt()


class OpenAITranscriber(Transcriber):

    supported_formats = ["txt", "text", "srt"]

    def __init__(self, api_key: str = None, base_url: str = None):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.model = os.getenv("OPENAI_MODEL", "whisper-1")
        # user can request these formats for this transcriber

    def call_api(self, file: str, language: str = None, format: str = "text"):
        if format == "txt":
            format = "text"
        logger.info(f"calling {self.__class__.__name__} with model {self.model}, format {format}, language {language}")
        ret = self.client.audio.transcriptions.create(
            file=open(file, "rb"),
            model=self.model,
            language=language,
            response_format=format,
        )
        logger.info(f"openai api transcribe ret type: {type(ret)}")
        return ret

    def _transcribe(self, file: str, language: str = None, format: str = "text") -> str:

        ret = self.call_api(file, language=language, format=format)
        if format == "json":
            # for 'json', ret is a Transcript obj
            return ret.text
        elif format in ["text", "srt", "vtt"]:
            # ret is a json encoded str
            return json.loads(ret)
        else:
            raise ValueError(f"Unsupported format: {format}")


class LemonfoxAITranscriber(OpenAITranscriber):

    def __init__(self):
        super().__init__(
            api_key=os.environ["LEMONFOX_AI_API_KEY"],
            base_url="https://api.lemonfox.ai/v1",
        )


class GLMASRTranscriber(Transcriber):

    supported_formats = ["text"]
    model_id = "zai-org/GLM-ASR-Nano-2512"
    max_new_tokens = 500

    def load(self):
        from transformers import AutoModelForSeq2SeqLM, AutoProcessor
        logger.info(f"Loading model {self.model_id}...")
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, dtype="auto", device_map="auto")
        t = time.time() - t0
        logger.info(f"Model loaded in {t:.1f}s")

    def _transcribe(self, file, language: str = None, format: str = "text"):
        # audio: path, ndarray, torch.Tensor
        logger.info(f"transcribing audio ...")
        t0 = time.time()
        inputs = self.processor.apply_transcription_request(file)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
        )
        decoded_outputs = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        t = time.time() - t0
        logger.info(f"transcribing audio completed in {t:.1f}s")
        return decoded_outputs[0]


class WhisperTranscriber(Transcriber):
    supported_formats = ["json", "text", "srt"]
    model_size = os.getenv("WHISPER_MODEL", "turbo")

    def load(self):
        import whisper
        self.model = whisper.load_model(self.model_size)

    def _transcribe(self, audio, language: str = None, format: str = "text") -> str:
        # audio: Union[str, np.ndarray, torch.Tensor]
        result = self.model.transcribe(audio)
        if format in ["text", "txt"]:
            return result.get("text", "")

        segments = result.get("segments", [])
        lines = []
        if format in ["srt"]:
            for i, seg in enumerate(segments, start=1):
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                parts = [
                    str(i),
                    f"{sec2ts(start)} --> {sec2ts(end)}",
                    seg.get("text", "").strip(),
                ]
                caption = "\n".join(parts).strip()
                lines.append(caption)
            return "\n\n".join(lines).strip()


class WhisperCPUTranscriber(WhisperTranscriber):

    def load(self):
        from whisper import Whisper
        self.model = Whisper(
            model=os.getenv("WHISPER_MODEL", "small"),
            device="cpu",
            compute_type="int8",
        )


class FasterWhisperTranscriber(Transcriber):

    supported_formats = ["text", "srt"]

    device = "cuda"
    compute_type = "float16"
    batch_size = 16
    beam_size = 5
    best_of = 5
    model_size = os.getenv("FASTER_WHISPER_MODEL_SIZE", "large-v3")

    def load(self):
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        self.pipeline = BatchedInferencePipeline(model=self.model)

    def _transcribe(self, file: str, language: str = None, format: str = "text") -> str:

        logger.info(f"transcibe with faster-whisper {self.model}: {file}")
        segments, info = self.pipeline.transcribe(
            file,
            language=language or None,
            batch_size=self.batch_size,
            log_progress=False,  # a progress bar, not text
            beam_size=self.beam_size,
            best_of=self.best_of,
            chunk_length=10,
        )

        lines = []
        if format in ["text", "txt"]:
            for seg in segments:
                lines.append(seg.text.strip())
        elif format in ["srt"]:
            for i, seg in enumerate(segments, start=1):
                parts = [
                    str(i),
                    f"{sec2ts(seg.start)} --> {sec2ts(seg.end)}",
                    seg.text.strip(),
                ]
                caption = "\n".join(parts) + "\n"
                lines.append(caption)

        return "\n".join(lines).strip()


class FasterWhisperCPUTranscriber(FasterWhisperTranscriber):
    device = "cpu"
    compute_type = "int8"
    batch_size = 8
    beam_size = 1
    best_of = 1
    model_size = os.getenv("FASTER_WHISPER_MODEL_SIZE_CPU", "small")


def get_transcriber(backend: str = "") -> Transcriber:
    backend = backend or os.getenv("TRANSCRIBE_BACKEND", "lemonfoxai")
    if backend in ["lemonfoxai", "lemonfox", "lemonfox-ai"]:
        return LemonfoxAITranscriber()
    if backend in ["assembly", "assemblyai", "aai"]:
        return AssemblyAITranscriber()
    if backend in ["glm", "glmasr", "glmasr-server", "glm-asr", "glm-asr-server"]:
        return GLMASRTranscriber()
    elif backend == "openai":
        return OpenAITranscriber()
    elif backend in ["whisper", "whisper-gpu"]:
        return WhisperTranscriber()
    elif backend in ["whisper-cpu"]:
        return WhisperCPUTranscriber()
    elif backend in ["faster-whisper", "faster-whisper-gpu"]:
        return FasterWhisperTranscriber()
    elif backend in ["faster-whisper-cpu"]:
        return FasterWhisperCPUTranscriber()


def transcribe(file: str, language: str = None, format: str = "text", backend: str = "lemonfoxai") -> str:
    transcriber = get_transcriber(backend)
    transcriber.load()
    return transcriber.transcribe(file, language=language, format=format)


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe audio file to text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file", help="audio file path")
    parser.add_argument("-l", "--language", help="language code, None|zh|en|etc.")
    parser.add_argument("-f", "--format", default="text", help="output format, text|srt|vtt")
    parser.add_argument("-b", "--backend", help="transcribe backend name")
    parser.add_argument("-o", "--output", help="output file path")
    args = parser.parse_args()
    text = transcribe(args.file, language=args.language, format=args.format, backend=args.backend)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)
        logger.info(f"transcript saved to {output_path}")
    else:
        print(text)


if __name__ == "__main__":
    cli()
