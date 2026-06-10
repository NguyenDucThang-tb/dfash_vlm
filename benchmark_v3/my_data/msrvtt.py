"""
MSR-VTT local dataset adapter for benchmark_v3.

Expected local layout:
  /content/msrvtt/msrvtt_test_1k.json
  /content/msrvtt/videos/video/video0.mp4

Each video produces 2 samples:
  - short_caption
  - exhaustive_caption
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from my_data.base import BaseDataset, BUCKET_LONG, BUCKET_SHORT, MODE_FRAMES, MODE_VIDEO
from my_data.frame_pruner import (
    get_qwen3_base_frames,
    prune_by_motion,
    prune_by_ratio,
    prune_by_scene_change,
    prune_uniform,
)


PROMPT_SHORT = "Describe the main events in this video in one or two concise sentences."
PROMPT_EXHAUSTIVE = (
    "Describe the main events in this video exhaustively. Cover actions, objects, scene changes, "
    "temporal order, and important details from start to finish."
)


class MSRVTTDataset(BaseDataset):
    DATASET_NAME = "MSR-VTT"
    MODALITY = "video"
    _QWEN3_RATIO_STRATEGIES = {"qwen3_base_ratio"}

    def __init__(
        self,
        num_samples: int = 90,
        buckets: Optional[List[str]] = None,
        input_mode: str = MODE_FRAMES,
        annotation_path: str = "/content/msrvtt/msrvtt_test_1k.json",
        video_root: str = "/content/msrvtt/videos/video",
        num_frames: int = 4,
        pruning_strategy: str = "motion",
        keep_ratio: float = 0.5,
        ratio_sub_strategy: str = "motion",
        qwen3_fps: float = 1.0,
        dense_pool_multiplier: int = 8,
        seed: int = 42,
    ):
        super().__init__(num_samples, buckets, input_mode)
        self.annotation_path = Path(annotation_path)
        self.video_root = Path(video_root)
        self.num_frames = int(num_frames)
        self.pruning_strategy = pruning_strategy
        self.keep_ratio = float(keep_ratio)
        self.ratio_sub_strategy = ratio_sub_strategy
        self.qwen3_fps = float(qwen3_fps)
        self.dense_pool_multiplier = max(2, int(dense_pool_multiplier))
        self.seed = int(seed)
        self._load()

    def _load(self):
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"MSR-VTT annotation not found: {self.annotation_path}")
        if not self.video_root.exists():
            raise FileNotFoundError(f"MSR-VTT video root not found: {self.video_root}")

        rows = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        random.Random(self.seed).shuffle(rows)

        target_videos = max(1, self.num_samples // 2)
        used = 0
        for row in rows:
            if used >= target_videos:
                break

            vid = row.get("video_id")
            if not vid:
                continue
            video_file = row.get("video", f"{vid}.mp4")
            video_path = self.video_root / video_file
            if not video_path.exists():
                continue

            captions = row.get("caption") or []
            if isinstance(captions, str):
                captions = [captions]
            reference = captions[0].strip() if captions else ""

            duration = None
            try:
                start_t = row.get("start time")
                end_t = row.get("end time")
                if start_t is not None and end_t is not None:
                    duration = round(float(end_t) - float(start_t), 2)
            except Exception:
                duration = None

            frames = None
            if self.input_mode == MODE_FRAMES:
                frames = self._extract_frames(video_path)
                if not frames:
                    continue

            base = {
                "image": None,
                "frames": frames,
                "video_path": str(video_path),
                "input_mode": self.input_mode,
                "reference": reference,
                "dataset": self.DATASET_NAME,
                "duration_s": duration,
                "category": row.get("category"),
                "source_url": row.get("url"),
            }

            self.samples.append(
                {
                    **base,
                    "id": f"{vid}_short",
                    "prompt": PROMPT_SHORT,
                    "token_bucket": BUCKET_SHORT,
                    "task": "video_caption",
                }
            )
            self.samples.append(
                {
                    **base,
                    "id": f"{vid}_exhaustive",
                    "prompt": PROMPT_EXHAUSTIVE,
                    "token_bucket": BUCKET_LONG,
                    "task": "video_caption",
                }
            )
            used += 1

        if not self.samples:
            raise RuntimeError("MSR-VTT produced no valid samples.")

    def _extract_frames(self, video_path: Path) -> Optional[List[Image.Image]]:
        if self.pruning_strategy in self._QWEN3_RATIO_STRATEGIES:
            base_frames, _ = get_qwen3_base_frames(str(video_path), fps=self.qwen3_fps)
            if not base_frames:
                return None
            keep_ratio = max(self.num_frames / max(1, len(base_frames)), self.keep_ratio)
            frames = prune_by_ratio(
                base_frames,
                keep_ratio=keep_ratio,
                strategy=self.ratio_sub_strategy,
            )
            if len(frames) > self.num_frames:
                frames = prune_uniform(frames, self.num_frames)
            return frames or None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None

        pool_size = min(total, max(self.num_frames, self.num_frames * self.dense_pool_multiplier))
        idxs = np.linspace(0, max(0, total - 1), pool_size, dtype=int)
        candidate_frames: List[Image.Image] = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            candidate_frames.append(Image.fromarray(rgb))
        cap.release()
        if not candidate_frames:
            return None

        if self.pruning_strategy == "motion":
            frames = prune_by_motion(candidate_frames, self.num_frames)
        elif self.pruning_strategy == "scene_change":
            frames = prune_by_scene_change(candidate_frames, self.num_frames)
        else:
            frames = prune_uniform(candidate_frames, self.num_frames)
        if len(frames) > self.num_frames:
            frames = prune_uniform(frames, self.num_frames)
        return frames or None
