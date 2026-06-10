"""
Spec benchmark adapter for the current DFlash + Qwen3-VL-4B stack.
Loads:
  - target: Qwen/Qwen3-VL-4B-Instruct
  - draft : z-lab/Qwen3-4B-DFlash-b16
  - optional fine-tuned checkpoint: *.pt from Phase A / Phase B
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch
from models.base import BaseModelAdapter, get_device


def _find_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import DFlashDraftModel, dflash_generate  # noqa: E402

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402


TARGET_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DRAFT_MODEL_ID = "z-lab/Qwen3-4B-DFlash-b16"


def _normalize_video_frames(frames, sid: str):
    if not isinstance(frames, list):
        return frames
    if len(frames) == 0:
        return frames
    if len(frames) == 1:
        print(f"[SpecVLM] sample {sid}: only 1 frame after prune -> duplicating to 2 frames")
        return [frames[0], frames[0].copy() if hasattr(frames[0], "copy") else frames[0]]
    return frames


class SpecVLMAdapter(BaseModelAdapter):
    MODEL_NAME = "DFlash-SpecVLM (Qwen3-VL-4B)"
    MODEL_PARAMS = "4B+draft"
    MODALITY = "image+video"

    def __init__(
        self,
        draft_model_path: str = DRAFT_MODEL_ID,
        target_model_path: str = TARGET_MODEL_ID,
        draft_checkpoint: str | None = None,
        device: str = "cuda",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        dtype: str = "bfloat16",
        num_frames: int = 4,
        **_: Any,
    ):
        self.device = get_device(device)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.num_frames = int(num_frames)
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        print(f"[SpecVLM] Loading processor from {target_model_path} ...")
        self.processor = AutoProcessor.from_pretrained(target_model_path)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"

        print(f"[SpecVLM] Loading target model {target_model_path} ...")
        self.target = Qwen3VLForConditionalGeneration.from_pretrained(
            target_model_path,
            dtype=self.dtype,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        for p in self.target.parameters():
            p.requires_grad = False

        print(f"[SpecVLM] Loading draft model {draft_model_path} ...")
        self.draft = DFlashDraftModel.from_pretrained(
            draft_model_path,
            dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

        if draft_checkpoint:
            state = torch.load(draft_checkpoint, map_location="cpu")
            state_dict = state.get("model_state_dict", state)
            incompat = self.draft.load_state_dict(state_dict, strict=False)
            print(
                f"[SpecVLM] Loaded checkpoint: {draft_checkpoint} | "
                f"missing={len(getattr(incompat, 'missing_keys', []))}, "
                f"unexpected={len(getattr(incompat, 'unexpected_keys', []))}"
            )

        tok = self.processor.tokenizer
        self.stop_token_ids = [tok.eos_token_id] if tok.eos_token_id is not None else None

        print(
            f"[SpecVLM] Ready | target={type(self.target).__name__} "
            f"| draft={type(self.draft).__name__} | num_frames={self.num_frames}"
        )

    def _build_inputs(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = sample.get("prompt", "Describe this image.")
        sid = sample.get("id", "unknown")
        frames = sample.get("frames")
        video_path = sample.get("video_path")
        image = sample.get("image")
        task = sample.get("task", "short_caption")

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
            raise ValueError(f"[SpecVLM] sample {sid}: no valid input for task={task}")

        print(f"[SpecVLM] input id={sid} | mode={mode}")
        return encoded

    def generate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        encoded = self._build_inputs(sample)
        input_ids = encoded["input_ids"].to(self.target.device)

        pixel_values = encoded.get("pixel_values")
        image_grid_thw = encoded.get("image_grid_thw")
        pixel_values_videos = encoded.get("pixel_values_videos")
        video_grid_thw = encoded.get("video_grid_thw")

        if pixel_values is not None:
            pixel_values = pixel_values.to(self.target.device)
            image_grid_thw = image_grid_thw.to(self.target.device)
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(self.target.device)
            video_grid_thw = video_grid_thw.to(self.target.device)

        stats: SimpleNamespace = dflash_generate(
            self.draft,
            target=self.target,
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            stop_token_ids=self.stop_token_ids,
            temperature=self.temperature,
            return_stats=True,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

        output_ids = stats.output_ids
        decoded = self.processor.tokenizer.decode(
            output_ids[0, stats.num_input_tokens:],
            skip_special_tokens=True,
        ).strip()
        acceptance = (
            float(sum(stats.acceptance_lengths) / len(stats.acceptance_lengths))
            if stats.acceptance_lengths
            else None
        )

        return {
            "text": decoded,
            "num_tokens": int(stats.num_output_tokens),
            "time_to_first_token_s": float(stats.time_to_first_token) if stats.time_to_first_token is not None else None,
            "acceptance_length": acceptance,
            "draft_rounds": int(len(stats.acceptance_lengths)) if stats.acceptance_lengths else None,
        }


class _MockSpecModel:
    def generate(self, sample: dict) -> dict:
        prompt = sample.get("prompt", "Describe this image.")
        n = max(16, len(prompt.split()) * 4)
        return {
            "text": f"[MOCK SPEC OUTPUT] {prompt[:80]}",
            "num_tokens": n,
            "time_to_first_token_s": 0.12,
            "acceptance_length": 2.0,
            "draft_rounds": max(1, n // 4),
        }
