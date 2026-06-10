"""
Baseline adapter for the current Qwen3-VL-4B benchmark stack.
This adapter mirrors the preprocessing choices used by the DFlash spec adapter
so baseline/spec comparisons are on the same input distribution.
"""

from __future__ import annotations

import time
import torch
from typing import Any, Dict

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from models.base import BaseModelAdapter, get_device


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def _normalize_video_frames(frames, sid: str):
    if not isinstance(frames, list):
        return frames
    if len(frames) == 0:
        return frames
    if len(frames) == 1:
        print(f"[Qwen3-VL-4B-Instruct] sample {sid}: only 1 frame after prune -> duplicating to 2 frames")
        return [frames[0], frames[0].copy() if hasattr(frames[0], "copy") else frames[0]]
    return frames


class Qwen3VLAdapter(BaseModelAdapter):
    MODEL_NAME = "Qwen3-VL-4B-Instruct"
    MODEL_PARAMS = "4B"
    MODALITY = "image+video"

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 128,
        num_frames: int = 4,
        **_: Any,
    ):
        self.device = get_device(device)
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.max_new_tokens = int(max_new_tokens)
        self.num_frames = int(num_frames)

        print(f"[{self.MODEL_NAME}] Loading {MODEL_ID} on {self.device} ...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=self.dtype,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()

    def _build_inputs(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = sample.get("prompt", "Describe this image.")
        sid = sample.get("id", "unknown")
        frames = sample.get("frames")
        video_path = sample.get("video_path")
        image = sample.get("image")

        if isinstance(frames, list) and len(frames) > 0:
            frames = _normalize_video_frames(frames, sid)
            messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = self.processor(text=[text], videos=[frames], return_tensors="pt", padding=True)
            mode = "frames"
        elif video_path:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(video_path), "num_frames": self.num_frames},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            encoded = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            mode = "video"
        elif image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            encoded = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            mode = "image"
        else:
            raise ValueError(f"[{self.MODEL_NAME}] sample {sid}: no valid input")

        print(f"[{self.MODEL_NAME}] input id={sid} | mode={mode}")
        return encoded

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        encoded = self._build_inputs(sample)
        model_device = next(self.model.parameters()).device
        encoded = {
            k: v.to(model_device) if isinstance(v, torch.Tensor) else v
            for k, v in encoded.items()
        }

        ttft = None
        t0 = time.perf_counter()
        first_token = [True]
        original_forward = self.model.__class__.forward

        def _patched_forward(self_inner, *args, **kwargs):
            result = original_forward(self_inner, *args, **kwargs)
            if first_token[0]:
                first_token[0] = False
                nonlocal ttft
                ttft = time.perf_counter() - t0
            return result

        self.model.__class__.forward = _patched_forward
        try:
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
            )
        finally:
            self.model.__class__.forward = original_forward

        prompt_len = int(encoded["attention_mask"][0].sum().item())
        answer_ids = generated[0, prompt_len:]
        decoded = self.processor.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        return {
            "text": decoded,
            "num_tokens": int(answer_ids.shape[0]),
            "time_to_first_token_s": round(ttft, 4) if ttft is not None else None,
            "acceptance_length": None,
            "draft_rounds": None,
        }
