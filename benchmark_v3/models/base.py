"""
Base interface cho tất cả model adapters.
Mọi model đều kế thừa BaseModelAdapter.
"""

from abc import ABC, abstractmethod
import torch


def get_device(requested: str = "cuda") -> str:
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return requested
    print(f"[adapter] CUDA not available, falling back to CPU.")
    return "cpu"


def get_dtype(device: str, dtype: str = "bfloat16"):
    if device == "cpu" and dtype == "bfloat16":
        return torch.float32
    return getattr(torch, dtype)


class BaseModelAdapter(ABC):
    """
    Interface contract cho mọi model adapter.

    generate(sample) -> {
        "text": str,
        "num_tokens": int,
        "time_to_first_token_s": float | None,   # TTFT
        "acceptance_length": float | None,        # speculative decoding
        "draft_rounds": int | None,
    }
    """

    # Mỗi subclass khai báo thông tin model
    MODEL_NAME: str = "unknown"
    MODEL_PARAMS: str = "unknown"   # vd: "3B", "7B"
    MODALITY: str = "image+video"   # "image", "video", "image+video"

    @abstractmethod
    def generate(self, sample: dict) -> dict:
        """
        Args:
            sample: {
                "image": PIL.Image | None,
                "frames": list[PIL.Image] | None,
                "prompt": str,
                "task": "short_caption" | "long_caption" | "video_caption",
                "token_bucket": "<100" | "<300" | ">500",
            }
        Returns:
            {
                "text": str,
                "num_tokens": int,
                "time_to_first_token_s": float | None,
                "acceptance_length": float | None,
                "draft_rounds": int | None,
            }
        """
        ...

    def estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))

    def info(self) -> dict:
        return {
            "model_name": self.MODEL_NAME,
            "model_params": self.MODEL_PARAMS,
            "modality": self.MODALITY,
        }
