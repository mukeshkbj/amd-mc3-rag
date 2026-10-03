"""VLM wrapper for MC3: one model does image transcription (index time) and
answer generation (query time). Loaded once by daemon.py.
"""
from __future__ import annotations

import os

MODEL_DIR = os.environ.get("MC3_MODEL_DIR", "/models/qwen3-vl-8b")
ALLOW_CPU = os.environ.get("ALLOW_CPU", "0") == "1"

TRANSCRIBE_PROMPT = (
    "Transcribe every piece of text visible in this image, verbatim. Include "
    "labels, values, headers, and codes. For diagrams and tables, state the "
    "mapping explicitly (e.g. 'pin B14: THERM_ALERT#' or 'part X: $12'). "
    "Output only the transcription, top to bottom."
)


class VLMEngine:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available() and not ALLOW_CPU:
            raise RuntimeError("no GPU visible and ALLOW_CPU unset")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(MODEL_DIR)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_DIR, dtype=dtype
        ).to(self.device)
        self.model.eval()

    def generate(self, prompt: str, image=None, max_new_tokens: int = 256) -> str:
        import torch

        content = ([{"type": "image"}] if image is not None else []) + [
            {"type": "text", "text": prompt}
        ]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        kw = {"text": [text], "return_tensors": "pt"}
        if image is not None:
            kw["images"] = [image]
        inputs = self.processor(**kw)
        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True
        )[0].strip()

    def transcribe_image(self, path: str) -> str:
        from PIL import Image

        img = Image.open(path)
        img.load()
        if getattr(img, "n_frames", 1) > 1:
            img.seek(0)
        return self.generate(TRANSCRIBE_PROMPT, image=img.convert("RGB"),
                             max_new_tokens=512)

    def warmup(self) -> None:
        self.generate("Reply with the single word: ok", max_new_tokens=8)
