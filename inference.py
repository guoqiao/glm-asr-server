#!/usr/bin/env python3
import time

from loguru import logger
from transformers import AutoModelForSeq2SeqLM, AutoProcessor


class Transcriber:

    def load(self):
        self.model_id = "zai-org/GLM-ASR-Nano-2512"
        logger.info(f"Loading model {self.model_id}...")
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, dtype="auto", device_map="auto")
        t = time.time() - t0
        logger.info(f"Model loaded in {t:.1f}s")

    def transcribe(self, audio, lang: str = None, max_new_tokens: int = 500):
        # audio: path, ndarray, torch.Tensor
        logger.info(f"transcribing audio ...")
        t0 = time.time()
        inputs = self.processor.apply_transcription_request(audio)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        decoded_outputs = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        t = time.time() - t0
        logger.info(f"transcribing audio completed in {t:.1f}s")
        return decoded_outputs[0]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=str)
    args = parser.parse_args()
    transcriber = Transcriber()
    text = transcriber.transcribe(args.audio)
    print(text)


if __name__ == "__main__":
    audio = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3"
    audio = "data/audio.mp3"
    main()