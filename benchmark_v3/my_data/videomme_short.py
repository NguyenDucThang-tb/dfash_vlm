"""
Video-MME Short — subset video ngắn (<2 phút) của benchmark Video-MME.
HuggingFace: mteb/Video-MME_short

Cấu trúc dataset thật (900 samples, split=test):
  question_id, video_id,
  video      : torchcodec.VideoDecoder  (HF decode sẵn)
  question   : str
  candidates : List[str]   e.g. ["A. A bat.", "B. A dragon.", ...]
  answer     : str         e.g. "B. A dragon."

Bucket strategy (2 bucket):
  - short_caption      : "Reply with only the letter" → BUCKET_SHORT
  - exhaustive_caption : "Explain reasoning step by step" → BUCKET_LONG

Mỗi video tạo ra 2 samples (1 per bucket), xen kẽ nhau.
Không có synthetic fallback — nếu HuggingFace load thất bại thì raise.
"""

import os
import tempfile
import numpy as np
from typing import Optional, List, Tuple

from PIL import Image as PILImage

from my_data.base import (
    BaseDataset,
    BUCKET_SHORT, BUCKET_LONG,
    MODE_FRAMES, MODE_VIDEO,
)
from my_data.frame_pruner import (
    get_qwen3_base_frames,
    prune_by_ratio,
)

NUM_FRAMES = 64  # chỉ dùng khi KHÔNG dùng qwen3_base_ratio mode

_QWEN3_RATIO_STRATEGIES = {"qwen3_base_ratio"}
THUMB_SIZE = (224, 224)

PROMPT_SHORT = (
    "Watch the video and answer the question.\n"
    "Reply with the correct answer in one short sentence.\n\n"
    "Question: {question}\n{candidates}\n\nAnswer:"
)

PROMPT_EXHAUSTIVE = (
    "Watch the video carefully and consider all options.\n"
    "Explain your reasoning step by step before selecting the final answer.\n"
    "Provide your answer in the format: Letter (A, B, C, or D) at the end.\n\n"
    "Question: {question}\n{candidates}\n\nAnswer:"
)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _format_candidates(candidates: list) -> str:
    return "\n".join(c for c in candidates if c)


def _extract_letter(answer: str) -> str:
    """'B. A dragon.' → 'B'"""
    if not answer:
        return ""
    letter = answer.strip()[0].upper()
    return letter if letter in "ABCD" else answer.strip()


def _frames_from_videodecoder(
    vd,
    n_frames: int = NUM_FRAMES,
) -> Optional[List[PILImage.Image]]:
    """
    Extract n_frames đều từ torchcodec.VideoDecoder.

    Thử 3 API theo thứ tự:
      1. get_frames_at(indices=[...])
      2. get_frames_played_at(seconds=[...])
      3. __getitem__ fallback
    """
    try:
        import torch

        meta  = vd.metadata
        total = getattr(meta, "num_frames", None) \
             or getattr(meta, "num_frames_from_header", None)

        # ── API 1: get_frames_at — dùng index ───────────────────────────────
        if total and hasattr(vd, "get_frames_at"):
            try:
                idxs  = [min(int(i * total / n_frames), total - 1)
                         for i in range(n_frames)]
                batch = vd.get_frames_at(indices=idxs)
                frames = _tensor_to_pil_list(batch.data)
                if frames:
                    return frames
            except Exception:
                pass

        # ── API 2: get_frames_played_at — dùng timestamp ────────────────────
        if hasattr(vd, "get_frames_played_at"):
            try:
                duration = getattr(meta, "duration_seconds", None) \
                        or getattr(meta, "duration_seconds_from_header", None)
                if duration and duration > 0:
                    times = [(i + 0.5) * duration / n_frames
                             for i in range(n_frames)]
                    batch = vd.get_frames_played_at(seconds=times)
                    frames = _tensor_to_pil_list(batch.data)
                    if frames:
                        return frames
            except Exception:
                pass

        # ── API 3: __getitem__ fallback ──────────────────────────────────────
        if total and hasattr(vd, "__getitem__"):
            try:
                idxs    = [min(int(i * total / n_frames), total - 1)
                           for i in range(n_frames)]
                parts   = [vd[j] for j in idxs]
                tensors = [p.data if hasattr(p, "data") else p for p in parts]
                stacked = []
                for t in tensors:
                    stacked.append(t if t.ndim == 3 else t.squeeze(0))
                frames = _tensor_to_pil_list(torch.stack(stacked))
                if frames:
                    return frames
            except Exception:
                pass

    except Exception:
        pass

    return None


def _frames_from_videodecoder_by_fps(
    vd,
    fps: float = 1.0,
) -> Tuple[Optional[List[PILImage.Image]], int]:
    """
    Decode frames từ VideoDecoder theo fps, KHÔNG cố định số frame.
    Số frame = round(duration_giây × fps), tùy độ dài video.

    Đây là hàm thay thế get_qwen3_base_frames() khi VideoDecoder đã có
    sẵn trong memory, tránh vòng lặp:
        vd → save NUM_FRAMES→mp4 → đọc lại → luôn ra NUM_FRAMES frames.

    Returns: (frames, n_frames) hoặc (None, 0) nếu thất bại.
    """
    try:
        meta     = vd.metadata
        duration = (
            getattr(meta, "duration_seconds", None)
            or getattr(meta, "duration_seconds_from_header", None)
        )
        total = (
            getattr(meta, "num_frames", None)
            or getattr(meta, "num_frames_from_header", None)
        )

        if not duration and total:
            video_fps = getattr(meta, "average_fps", None) or 25.0
            duration  = total / video_fps

        if not duration or duration <= 0:
            return None, 0

        n_frames = max(1, round(duration * fps))
        frames   = _frames_from_videodecoder(vd, n_frames)
        if frames:
            return frames, len(frames)
        return None, 0

    except Exception as e:
        print(f"[VideoMME-Short] _frames_from_videodecoder_by_fps error: {e}")
        return None, 0


def _tensor_to_pil_list(frames_t) -> Optional[List[PILImage.Image]]:
    """(N, C, H, W) uint8 → list[PIL.Image]"""
    try:
        import torch
        if not isinstance(frames_t, torch.Tensor):
            return None

        arr = frames_t.detach().cpu()

        if arr.ndim == 3:
            arr = arr.unsqueeze(0)

        n, c, h, w = arr.shape
        if c <= 4 and h > 4 and w > 4:
            arr = arr.permute(0, 2, 3, 1)

        arr = arr.numpy()

        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)

        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        return [PILImage.fromarray(f[..., :3]).resize(THUMB_SIZE) for f in arr]
    except Exception:
        return None


# ─── Dataset class ─────────────────────────────────────────────────────────────

class VideoMMEShortDataset(BaseDataset):
    """
    Video-MME Short — multiple-choice video QA benchmark.

    Mỗi video sinh 2 samples:
      - short_caption      → BUCKET_SHORT, trả lời 1 chữ cái
      - exhaustive_caption → BUCKET_LONG,  giải thích step-by-step

    sample["task"]       = "short_qa" | "exhaustive_qa"
    sample["answer"]     = ground-truth letter ("A"/"B"/"C"/"D")
    sample["real_video"] = True nếu frames từ video thật
    """

    DATASET_NAME = "VideoMME-Short"
    MODALITY     = "video"

    def __init__(
        self,
        num_samples: int              = 90,
        num_frames: int               = NUM_FRAMES,
        buckets: Optional[List[str]]  = None,
        download_video: bool          = True,
        input_mode: str               = MODE_FRAMES,
        pruning_strategy: str         = "motion",
        keep_ratio: float             = 0.5,
        qwen3_fps: float              = 1.0,
        ratio_sub_strategy: str       = "motion",
    ):
        super().__init__(num_samples, buckets, input_mode)
        self.num_frames          = num_frames
        self.download_video      = download_video
        self._tmp_dir            = tempfile.mkdtemp(prefix="videomme_")
        self.pruning_strategy    = pruning_strategy
        self.keep_ratio          = keep_ratio
        self.qwen3_fps           = qwen3_fps
        self.ratio_sub_strategy  = ratio_sub_strategy
        self._load()

    def _load(self):
        self._load_from_hf()

        real   = sum(1 for s in self.samples if s.get("real_video"))
        synth  = len(self.samples) - real
        n_vids = len(self.samples) // 2
        if self.samples and self.input_mode == MODE_FRAMES:
            n_frames = len(self.samples[0].get("frames") or [])
        else:
            n_frames = 0

        prune_info = ""
        if self.pruning_strategy in _QWEN3_RATIO_STRATEGIES:
            prune_info = (
                f" | prune=qwen3_base_ratio"
                f"(keep={self.keep_ratio*100:.0f}%,"
                f" sub={self.ratio_sub_strategy},"
                f" fps={self.qwen3_fps})"
            )
        print(
            f"[VideoMME-Short] Loaded {len(self.samples)} samples "
            f"({n_vids} video × 2 buckets: short_caption + exhaustive_caption) "
            f"| real_video={real}, frame_failed={synth} "
            f"| num_frames={n_frames} "
            f"| {self.summary()['buckets']}"
            f"{prune_info}"
        )

    def _load_from_hf(self):
        from datasets import load_dataset

        print(
            f"[VideoMME-Short] Loading mteb/Video-MME_short "
            f"(streaming, input_mode={self.input_mode})..."
        )
        ds = load_dataset("mteb/Video-MME_short", split="test", streaming=True)

        for i, row in enumerate(ds):
            if len(self.samples) >= self.num_samples:
                break

            vd          = row.get("video")
            frames      = None
            video_path  = None
            real_video  = False
            base_frames = None
            n_base      = 0

            if self.download_video and vd is not None:

                if self.input_mode == MODE_VIDEO:
                    video_path = self._save_video_to_file(
                        vd, row.get("video_id", f"video_{i}"), i
                    )
                    if video_path:
                        real_video = True
                    else:
                        print(
                            f"[VideoMME-Short] sample {i} "
                            f"({row.get('video_id', '?')}): "
                            f"video save failed — skipping"
                        )
                        continue

                elif self.pruning_strategy in _QWEN3_RATIO_STRATEGIES:
                    # ── Fix root cause ──────────────────────────────────────
                    # Decode TRỰC TIẾP từ VideoDecoder theo fps.
                    # KHÔNG đi qua _save_video_to_file vì hàm đó fallback
                    # encode NUM_FRAMES frames → video giả → luôn ra NUM_FRAMES.
                    base_frames, n_base = _frames_from_videodecoder_by_fps(
                        vd, fps=self.qwen3_fps,
                    )
                    if not base_frames:
                        print(
                            f"[VideoMME-Short] sample {i} "
                            f"({row.get('video_id', '?')}): "
                            f"decode by fps failed — skipping"
                        )
                        continue

                    print(
                        f"[frame_pruner] get_qwen3_base_frames: "
                        f"{n_base} frames từ VideoDecoder (fps={self.qwen3_fps})"
                    )

                    frames = prune_by_ratio(
                        base_frames,
                        keep_ratio=self.keep_ratio,
                        strategy=self.ratio_sub_strategy,
                    )
                    real_video = True
                    # Lưu video_path chỉ để dùng cho caption_sim Pass A
                    video_path = self._save_video_to_file(
                        vd, row.get("video_id", f"video_{i}"), i
                    )

                else:
                    frames = _frames_from_videodecoder(vd, self.num_frames)
                    if frames:
                        real_video = True
                    else:
                        print(
                            f"[VideoMME-Short] sample {i} "
                            f"({row.get('video_id', '?')}): "
                            f"frame extraction failed — skipping"
                        )
                        continue

            if frames is None and video_path is None:
                print(
                    f"[VideoMME-Short] sample {i} "
                    f"({row.get('video_id', '?')}): no video data — skipping"
                )
                continue

            candidates = row.get("candidates") or [
                "A. Option A", "B. Option B", "C. Option C", "D. Option D"
            ]
            question   = row.get("question", "What is happening in this video?")
            raw_answer = row.get("answer", "")
            answer     = _extract_letter(raw_answer)
            qid        = row.get("question_id", f"vmme_{i}")
            vid        = row.get("video_id", f"video_{i}")
            cands_str  = _format_candidates(candidates)

            base_sample = {
                "video_id":      vid,
                "frames":        frames,
                "video_path":    video_path,
                "input_mode":    self.input_mode,
                "question":      question,
                "answer":        answer,
                "answer_raw":    raw_answer,
                "candidates":    candidates,
                "options":       candidates,
                "real_video":    real_video,
                "base_frames":   base_frames if self.pruning_strategy in _QWEN3_RATIO_STRATEGIES else None,
                "n_base_frames": n_base      if self.pruning_strategy in _QWEN3_RATIO_STRATEGIES else None,
                "keep_ratio":    self.keep_ratio if self.pruning_strategy in _QWEN3_RATIO_STRATEGIES else None,
            }

            self.samples.append({
                **base_sample,
                "id":           f"{qid}_short",
                "prompt":       PROMPT_SHORT.format(question=question, candidates=cands_str),
                "task":         "short_qa",
                "token_bucket": BUCKET_SHORT,
            })

            self.samples.append({
                **base_sample,
                "id":           f"{qid}_exhaustive",
                "prompt":       PROMPT_EXHAUSTIVE.format(question=question, candidates=cands_str),
                "task":         "exhaustive_qa",
                "token_bucket": BUCKET_LONG,
            })

    def _save_video_to_file(self, vd, video_id: str, idx: int) -> Optional[str]:
        """
        Lưu video ra file .mp4 tạm — CHỈ dùng cho caption_sim Pass A.
        KHÔNG dùng để đếm frame cho qwen3_base_ratio (vì sẽ subsample NUM_FRAMES).
        """
        try:
            raw_bytes = getattr(vd, "_raw_video_bytes", None) \
                     or getattr(vd, "video_bytes", None)

            if raw_bytes is not None:
                safe_id = video_id.replace("/", "_").replace(" ", "_")
                path = os.path.join(self._tmp_dir, f"{safe_id}_{idx}.mp4")
                with open(path, "wb") as f:
                    f.write(raw_bytes)
                return path

            frames = _frames_from_videodecoder(vd, NUM_FRAMES)
            if not frames:
                return None

            safe_id = video_id.replace("/", "_").replace(" ", "_")
            path    = os.path.join(self._tmp_dir, f"{safe_id}_{idx}.mp4")

            try:
                import imageio.v3 as iio
                iio.imwrite(path, [np.array(f) for f in frames], fps=1, codec="libx264")
                return path
            except Exception:
                pass

            try:
                import imageio
                writer = imageio.get_writer(
                    path, fps=1, format="ffmpeg",
                    output_params=["-vcodec", "libx264", "-crf", "28"],
                )
                for frame in frames:
                    writer.append_data(np.array(frame))
                writer.close()
                return path
            except Exception:
                pass

            return None

        except Exception as e:
            print(f"[VideoMME-Short] _save_video_to_file error: {e}")
            return None
