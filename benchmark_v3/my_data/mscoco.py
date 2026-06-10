"""
MSCOCO 2014 5K Test — image captioning benchmark.

Source: nlphuji/mscoco_2014_5k_test_image_text_retrieval

Schema:
  - image   : PIL Image
  - caption : list[str], 5-6 captions ngắn
  - cocoid  : int

Hai bucket theo loại prompt:
  - short_caption      : "Describe in 1-2 sentences"   → output ngắn ~20-40 token
  - exhaustive_caption : "Describe exhaustively..."     → output dài ~150-300 token

Mỗi ảnh tạo ra 2 samples (1 per bucket), xen kẽ nhau.
"""

from typing import Optional, List
from my_data.base import BaseDataset, BUCKET_SHORT, BUCKET_LONG

# ── Prompts ──────────────────────────────────────────────────────
PROMPT_SHORT = (
    "Describe this image in one or two concise sentences."
)

PROMPT_EXHAUSTIVE = (
    "Describe this image exhaustively. Cover all visible objects, "
    "their positions, colors, actions, background details, lighting, "
    "and any text or symbols present. Be thorough and specific."
)

PARQUET_URL = (
    "https://huggingface.co/datasets/nlphuji/"
    "mscoco_2014_5k_test_image_text_retrieval/"
    "resolve/refs/convert/parquet/TEST/test/0000.parquet"
)


class MSCOCODataset(BaseDataset):

    DATASET_NAME = "MSCOCO-2014-5k"
    MODALITY     = "image"

    def __init__(self, num_samples: int = 90, buckets: Optional[List[str]] = None):
        # num_samples = tổng samples; mỗi ảnh → 2 samples
        # nên số ảnh thực tế load = num_samples // 2
        super().__init__(num_samples, buckets)
        self._load()

    def _load(self):
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "parquet",
                data_files={"test": PARQUET_URL},
                split="test",
                streaming=True,
            )
            self._parse(ds)
            if len(self.samples) > 0:
                n_imgs = len(self.samples) // 2
                print(f"[MSCOCO] Loaded {len(self.samples)} samples "
                      f"({n_imgs} ảnh × 2 bucket: short_caption + exhaustive_caption)")
                return
        except Exception as e:
            print(f"[MSCOCO] Load failed: {e} -> using synthetic data")

        self._make_synthetic()

    def _parse(self, ds):
        n_images = self.num_samples // 2  # mỗi ảnh → 2 samples

        for i, item in enumerate(ds):
            if len(self.samples) >= self.num_samples:
                break

            image = self._ensure_pil(item.get("image"))
            if image is None:
                continue

            captions = item.get("caption") or []
            if not captions:
                continue

            # Reference: caption ngắn nhất trong bộ 5-6 captions
            reference = min(captions, key=lambda x: len(x.split()))
            item_id   = item.get("cocoid") or str(i)

            # Sample 1 — short caption
            self.samples.append({
                "id":           f"coco2014_{item_id}_short",
                "image":        image,
                "frames":       None,
                "prompt":       PROMPT_SHORT,
                "reference":    reference,
                "token_bucket": BUCKET_SHORT,
                "task":         "short_caption",
                "dataset":      self.DATASET_NAME,
            })

            # Sample 2 — exhaustive caption
            self.samples.append({
                "id":           f"coco2014_{item_id}_exhaustive",
                "image":        image,
                "frames":       None,
                "prompt":       PROMPT_EXHAUSTIVE,
                "reference":    reference,
                "token_bucket": BUCKET_LONG,
                "task":         "exhaustive_caption",
                "dataset":      self.DATASET_NAME,
            })

    def _ensure_pil(self, image):
        from PIL import Image as PILImage
        if image is None:
            return None
        if isinstance(image, PILImage.Image):
            return image.convert("RGB")
        try:
            return PILImage.fromarray(image).convert("RGB")
        except Exception:
            return None

    def _make_synthetic(self):
        import numpy as np
        from PIL import Image as PILImage
        n_images = self.num_samples // 2
        for i in range(n_images):
            arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = PILImage.fromarray(arr)
            ref = "A dog sitting on a grassy lawn near a white fence."

            self.samples.append({
                "id":           f"coco_synthetic_{i}_short",
                "image":        img,
                "frames":       None,
                "prompt":       PROMPT_SHORT,
                "reference":    ref,
                "token_bucket": BUCKET_SHORT,
                "task":         "short_caption",
                "dataset":      self.DATASET_NAME,
            })
            self.samples.append({
                "id":           f"coco_synthetic_{i}_exhaustive",
                "image":        img,
                "frames":       None,
                "prompt":       PROMPT_EXHAUSTIVE,
                "reference":    ref,
                "token_bucket": BUCKET_LONG,
                "task":         "exhaustive_caption",
                "dataset":      self.DATASET_NAME,
            })
