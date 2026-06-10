"""
Base class cho tất cả datasets.
Hỗ trợ phân bucket theo task/prompt type.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

# MSCOCO dùng 2 bucket theo loại prompt
BUCKET_SHORT  = "short_caption"
BUCKET_LONG   = "exhaustive_caption"

# Alias để không break các dataset khác
BUCKET_MEDIUM = BUCKET_SHORT
ALL_BUCKETS   = [BUCKET_SHORT, BUCKET_LONG]

# Input mode constants
MODE_FRAMES = "frames"   # Hướng 1: List[PIL.Image] đã decode + prune
MODE_VIDEO  = "video"    # Hướng 2: video_path str, model tự decode


class BaseDataset(ABC):
    """
    Interface cho mọi dataset.
    Mỗi sample phải có:
        id, prompt, task, token_bucket
    Và một trong hai tùy input_mode:
        frames    (list[PIL.Image])  — khi input_mode="frames"
        video_path (str)             — khi input_mode="video"
    image (PIL.Image) vẫn dùng cho image-only datasets (MSCOCO).
    """

    DATASET_NAME: str = "unknown"
    MODALITY: str = "image"

    def __init__(
        self,
        num_samples: int = 90,
        buckets: Optional[List[str]] = None,
        input_mode: str = MODE_FRAMES,   # "frames" hoặc "video"
    ):
        self.num_samples = num_samples
        self.buckets     = buckets or ALL_BUCKETS
        self.input_mode  = input_mode
        self.samples: List[dict] = []

    @abstractmethod
    def _load(self):
        """Load và populate self.samples."""
        ...

    def filter_by_bucket(self, bucket: str):
        return [s for s in self.samples if s.get("token_bucket") == bucket]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[dict]:
        return iter(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

    def summary(self) -> dict:
        counts = {b: 0 for b in ALL_BUCKETS}
        for s in self.samples:
            b = s.get("token_bucket", BUCKET_SHORT)
            counts[b] = counts.get(b, 0) + 1
        return {
            "dataset":    self.DATASET_NAME,
            "total":      len(self.samples),
            "buckets":    counts,
            "input_mode": self.input_mode,
        }