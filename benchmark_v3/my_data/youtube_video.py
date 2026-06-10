"""
YouTube Video Dataset — 100 video thật cho benchmark.

Hỗ trợ 2 input mode:
  - input_mode="frames" (mặc định): decode + prune → sample["frames"]
  - input_mode="video":             chỉ lưu path   → sample["video_path"]

Pruning strategy mới (Câu 1 + Câu 2):
  - pruning_strategy="qwen3_base_ratio":
      Bước 1 — Dùng get_qwen3_base_frames() để lấy đúng bộ frame Qwen3 decode
               (thay vì OpenCV dense pool × 5). Đây là base frame nhất quán với model.
      Bước 2 — Prune trên bộ frame đó theo tỷ lệ keep_ratio (mặc định 50%).
  - pruning_strategy="motion" / "scene_change" / "uniform":
      Giữ nguyên flow cũ (OpenCV dense pool → prune đến target_num tuyệt đối).

Khi pruning_strategy="qwen3_base_ratio", mỗi sample lưu thêm:
  - "base_frames"    : List[PIL.Image] base frames của Qwen3 (TRƯỚC prune)
  - "keep_ratio"     : tỷ lệ frame giữ lại
  - "n_base_frames"  : số base frame của Qwen3
  - "n_pruned_frames": số frame sau prune
Dùng để tính caption similarity (Câu 2) trong run_benchmark.py.
"""

import cv2
import os
import numpy as np
import subprocess
from pathlib import Path
from typing import Optional, List
from PIL import Image
from my_data.base import BaseDataset, BUCKET_SHORT, BUCKET_LONG, MODE_FRAMES, MODE_VIDEO
from my_data.frame_pruner import (
    prune_uniform, prune_by_motion, prune_by_scene_change,
    prune_by_ratio, get_qwen3_base_frames,
)

YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=AGCHk5_2skY",
    "https://www.youtube.com/watch?v=N1cdUjctpG8",
    "https://www.youtube.com/watch?v=WAWyfikAndA",
    "https://www.youtube.com/watch?v=HwnB8aCn8yE",
    "https://www.youtube.com/watch?v=24i4ncHuf6A",
    "https://www.youtube.com/watch?v=40BlVzjxu-I",
    "https://www.youtube.com/watch?v=0ay2Qy3wBe8",
    "https://www.youtube.com/watch?v=95bnzxzSaso",
    "https://www.youtube.com/watch?v=sUDY-SMREtA",
    "https://www.youtube.com/watch?v=uigL9Zn18bc",
    "https://www.youtube.com/watch?v=OE5S-NbNsro",
    "https://www.youtube.com/watch?v=qvwfWyU9Gfc",
    "https://www.youtube.com/watch?v=5wLv3pCqZ9o",
    "https://www.youtube.com/watch?v=uqILuTcux_o",
    "https://www.youtube.com/watch?v=eXKE0nAMmg4",
    "https://www.youtube.com/watch?v=5_fXicEnKKk",
    "https://www.youtube.com/watch?v=O0qVPW1fUn4",
    "https://www.youtube.com/watch?v=tX8a00l_Dfs",
    "https://www.youtube.com/watch?v=mVIXU0x9ocI",
    "https://www.youtube.com/watch?v=YPaLGP_xr-w",
    "https://www.youtube.com/watch?v=TbaCxIJ_VP4",
    "https://www.youtube.com/watch?v=n3IYmdy6d4Y",
    "https://www.youtube.com/watch?v=iZYLeIJwe4w",
    "https://www.youtube.com/watch?v=jkNxmTrrZSk",
    "https://www.youtube.com/watch?v=ebzbKa32kuk",
    "https://www.youtube.com/watch?v=XWfqBKeC0g8",
    "https://www.youtube.com/watch?v=tS4a6I4-Yjo",
    "https://www.youtube.com/watch?v=6YhlYu70uNA",
    "https://www.youtube.com/watch?v=JbDRs0ja3PE",
    "https://www.youtube.com/watch?v=39HTpUG1MwQ",
    "https://www.youtube.com/watch?v=TPS22HRRY1k",
    "https://www.youtube.com/watch?v=X30VlGh3HwQ",
    "https://www.youtube.com/watch?v=zKyWRRJQbkM",
    "https://www.youtube.com/watch?v=blSnLEZe-sI",
    "https://www.youtube.com/watch?v=hysxIH_GZks",
    "https://www.youtube.com/watch?v=MXKDvuEkSF0",
    "https://www.youtube.com/watch?v=EQ-67udZEeg",
    "https://www.youtube.com/watch?v=GqeRnxSuLFI",
    "https://www.youtube.com/watch?v=XOJKiOw8Xqo",
    "https://www.youtube.com/watch?v=5ksVshqVuiM",
    "https://www.youtube.com/watch?v=yl5ZXQmrtP0",
    "https://www.youtube.com/watch?v=kRlhlCWplqk",
    "https://www.youtube.com/watch?v=9jjTGpWmc5U",
    "https://www.youtube.com/watch?v=fo0Hmch2YS0",
    "https://www.youtube.com/watch?v=990ci2H9BBc",
    "https://www.youtube.com/watch?v=gD0MvkDGAMg",
    "https://www.youtube.com/watch?v=WTQ7_e8bnfM",
    "https://www.youtube.com/watch?v=M7WOXFvwbSY",
    "https://www.youtube.com/watch?v=RvnC--JjDBw",
    "https://www.youtube.com/watch?v=08km9Yqbt-A",
    "https://www.youtube.com/watch?v=Qyg_91gNHCc",
    "https://www.youtube.com/watch?v=nYLMNQ77FjM",
    "https://www.youtube.com/watch?v=DF_J3vCcbBA",
    "https://www.youtube.com/watch?v=wVlfyhs64IY",
    "https://www.youtube.com/watch?v=K0mjUgHKfJo",
    "https://www.youtube.com/watch?v=WmVLcj-XKnM",
    "https://www.youtube.com/watch?v=zBnKgwnn7i4",
    "https://www.youtube.com/watch?v=D52rTzibFRc",
    "https://www.youtube.com/watch?v=NEFG7YcIDcI",
    "https://www.youtube.com/watch?v=8TNPeimqOO0",
    "https://www.youtube.com/watch?v=LCtOpCi5r2s",
    "https://www.youtube.com/watch?v=jTzKgI68VLc",
    "https://www.youtube.com/watch?v=eTLMWnsStuk",
    "https://www.youtube.com/watch?v=9_M4bNOxsYs",
    "https://www.youtube.com/watch?v=fo-mVfOsC-E",
    "https://www.youtube.com/watch?v=x6jPuXwtxCM",
    "https://www.youtube.com/watch?v=yXNShWnon4g",
    "https://www.youtube.com/watch?v=UZvydHZKyww",
    "https://www.youtube.com/watch?v=-O6mJ0VBTc4",
    "https://www.youtube.com/watch?v=CezlmUwMXNo",
    "https://www.youtube.com/watch?v=IzQ2siryQrM",
    "https://www.youtube.com/watch?v=VsuShNWghXk",
    "https://www.youtube.com/watch?v=7iXM5aq53Ts",
    "https://www.youtube.com/watch?v=HvjgQqNOq9A",
    "https://www.youtube.com/watch?v=uz6rjbw0ZA0",
    "https://www.youtube.com/watch?v=KOwR0Ln46Ks",
    "https://www.youtube.com/watch?v=540LkURTR7g",
    "https://www.youtube.com/watch?v=SnOc5W0PgVE",
    "https://www.youtube.com/watch?v=WViSvPFUVd8",
    "https://www.youtube.com/watch?v=21q-lDikdBg",
    "https://www.youtube.com/watch?v=FjS2LzrHEO8",
    "https://www.youtube.com/watch?v=m4qhFFdHTCc",
    "https://www.youtube.com/watch?v=lKoG2_zdoSA",
    "https://www.youtube.com/watch?v=aqTIB_q40bo",
    "https://www.youtube.com/watch?v=hUrjmA0fhsc",
    "https://www.youtube.com/watch?v=aBdQQxgxDrY",
    "https://www.youtube.com/watch?v=tF4DML7FIWk",
    "https://www.youtube.com/watch?v=jRS9fVh7MUw",
    "https://www.youtube.com/watch?v=JxkJ-FwFeVI",
    "https://www.youtube.com/watch?v=c9arR8T0Qts",
    "https://www.youtube.com/watch?v=_aVHf_jmWk8",
    "https://www.youtube.com/watch?v=3rTsETO3s9U",
    "https://www.youtube.com/watch?v=drbi6HK1gSc",
    "https://www.youtube.com/watch?v=M69Sn3OERZo",
    "https://www.youtube.com/watch?v=djr6T2C8Gfs",
    "https://www.youtube.com/watch?v=Q0B5dLHDQ2w",
    "https://www.youtube.com/watch?v=uoJDGnaVuTg",
    "https://www.youtube.com/watch?v=PU-XOFIJMlg",
    "https://www.youtube.com/watch?v=Ij-FYOrklFE",
    "https://www.youtube.com/watch?v=56yT3H_DjVE",
]

PROMPT_SHORT = "Describe this video in one or two concise sentences."
PROMPT_EXHAUSTIVE = (
    "Describe this video exhaustively. Cover all visible objects, "
    "their positions, actions, background details, movements, "
    "and any notable events from start to finish. Be thorough and specific."
)

# Strategies dùng Qwen3 base frames
_QWEN3_RATIO_STRATEGIES = {"qwen3_base_ratio"}


class YouTubeVideoDataset(BaseDataset):

    DATASET_NAME = "YouTube-100"
    MODALITY     = "video"

    def __init__(
        self,
        num_samples: int = 90,
        num_frames: int = 32,
        frame_size: int = 224,
        video_dir: str = "/kaggle/working/ytb_videos",
        buckets: Optional[List[str]] = None,
        force_download: bool = False,
        # ── Pruning strategy ────────────────────────────────────
        # "motion" / "scene_change" / "uniform": flow cũ (OpenCV dense pool)
        # "qwen3_base_ratio": CÂU 1+2 — dùng Qwen3 base frames, prune theo %
        pruning_strategy: str = "motion",
        dense_pool_multiplier: int = 5,
        # Tham số riêng cho qwen3_base_ratio
        keep_ratio: float = 0.5,        # giữ 50% frame base của Qwen3
        qwen3_fps: float = 1.0,         # fps Qwen3 dùng khi decode video
        ratio_sub_strategy: str = "motion",  # strategy nội bộ sau khi có base frames
        # ── Input mode ──────────────────────────────────────────
        input_mode: str = MODE_FRAMES,
    ):
        super().__init__(num_samples, buckets, input_mode)
        self.num_frames            = num_frames
        self.frame_size            = frame_size
        self.video_dir             = Path(video_dir)
        self.force_download        = force_download
        self.pruning_strategy      = pruning_strategy
        self.dense_pool_multiplier = dense_pool_multiplier
        self.keep_ratio            = keep_ratio
        self.qwen3_fps             = qwen3_fps
        self.ratio_sub_strategy    = ratio_sub_strategy
        self._load()

    def _load(self):
        self.video_dir.mkdir(parents=True, exist_ok=True)
        video_paths = self._download_all()
        if not video_paths:
            print("[YouTube] No videos → synthetic")
            self._make_synthetic()
            return

        n_videos_target  = self.num_samples // 2
        videos_processed = 0

        for vid_path in video_paths:
            if videos_processed >= n_videos_target:
                break

            duration = self._get_duration(vid_path)
            vid_id   = vid_path.stem

            if self.input_mode == MODE_VIDEO:
                # Hướng 2: chỉ lưu path, không decode
                frames      = None
                base_frames = None
                video_path  = str(vid_path)
                n_base      = 0
                n_pruned    = 0
            else:
                # Hướng 1: decode + prune
                if self.pruning_strategy in _QWEN3_RATIO_STRATEGIES:
                    # ── CÂU 1: lấy base frames từ Qwen3 pipeline ──────────
                    base_frames, n_base = get_qwen3_base_frames(
                        str(vid_path),
                        fps=self.qwen3_fps,
                    )
                    if not base_frames:
                        print(f"[YouTube] {vid_id}: get_qwen3_base_frames failed → skip")
                        continue

                    # ── CÂU 2: prune theo tỷ lệ keep_ratio ───────────────
                    frames = prune_by_ratio(
                        base_frames,
                        keep_ratio=self.keep_ratio,
                        strategy=self.ratio_sub_strategy,
                    )
                    n_pruned = len(frames)
                else:
                    # Flow cũ: OpenCV dense pool → prune theo target_num
                    frames = self._extract_frames(vid_path, self.num_frames)
                    if not frames:
                        continue
                    base_frames = None
                    n_base      = len(frames)
                    n_pruned    = len(frames)

                video_path = str(vid_path)

            videos_processed += 1

            base = {
                "image":           None,
                "frames":          frames,
                "video_path":      video_path,
                "input_mode":      self.input_mode,
                "reference":       "",
                "dataset":         self.DATASET_NAME,
                "duration_s":      round(duration, 1),
                "num_frames":      n_pruned if frames else self.num_frames,
                "has_real_frames": frames is not None or video_path is not None,
                # ── Thông tin cho caption similarity (Câu 2) ──────────────
                "base_frames":     base_frames,   # None nếu không phải qwen3_base_ratio
                "n_base_frames":   n_base,
                "n_pruned_frames": n_pruned,
                "keep_ratio":      self.keep_ratio if self.pruning_strategy in _QWEN3_RATIO_STRATEGIES else None,
                "pruning_strategy": self.pruning_strategy,
            }

            self.samples.append({**base,
                "id":           f"ytb_{vid_id}_short",
                "prompt":       PROMPT_SHORT,
                "token_bucket": BUCKET_SHORT,
                "task":         "short_caption",
            })
            self.samples.append({**base,
                "id":           f"ytb_{vid_id}_exhaustive",
                "prompt":       PROMPT_EXHAUSTIVE,
                "token_bucket": BUCKET_LONG,
                "task":         "exhaustive_caption",
            })

        shortage = n_videos_target - videos_processed
        if shortage > 0:
            print(f"[YouTube] Short {shortage * 2} samples → synthetic")
            for i in range(shortage):
                self._append_synthetic_pair(i)

        real = sum(1 for s in self.samples if s.get("has_real_frames"))
        strategy_label = self.pruning_strategy
        if self.pruning_strategy in _QWEN3_RATIO_STRATEGIES:
            strategy_label = (
                f"qwen3_base_ratio(keep={self.keep_ratio*100:.0f}%,"
                f"sub={self.ratio_sub_strategy})"
            )
        print(
            f"[YouTube] Loaded {len(self.samples)} samples "
            f"({videos_processed} videos × 2 bucket) "
            f"| mode={self.input_mode} | strategy={strategy_label} | real={real} "
            f"| {self.summary()['buckets']}"
        )

    # ── Download ──────────────────────────────────────────────────

    def _download_all(self) -> List[Path]:
        try:
            subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(
                    ["pip", "install", "yt-dlp", "-q", "--break-system-packages"],
                    check=True,
                )
            except subprocess.CalledProcessError:
                print("[YouTube] ⚠️  yt-dlp not found and install failed → no videos")
                return []

        video_paths = []
        for url in YOUTUBE_URLS:
            vid_id   = url.split("v=")[-1]
            existing = list(self.video_dir.glob(f"{vid_id}.*"))
            if existing and not self.force_download:
                video_paths.append(existing[0])
                continue
            try:
                subprocess.run([
                    "yt-dlp",
                    "-f", "worst[ext=mp4]/worst",
                    "--max-filesize", "150m",
                    "--no-playlist",
                    "-o", str(self.video_dir / f"{vid_id}.%(ext)s"),
                    "--quiet", "--no-warnings",
                    url,
                ], capture_output=True, text=True, timeout=180)
                downloaded = list(self.video_dir.glob(f"{vid_id}.*"))
                if downloaded:
                    video_paths.append(downloaded[0])
            except Exception:
                pass

        print(f"[YouTube] Downloaded/Found {len(video_paths)}/{len(YOUTUBE_URLS)} videos")
        return video_paths

    # ── Video utils ───────────────────────────────────────────────

    def _get_duration(self, path: Path) -> float:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return 0.0
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frames / fps if fps > 0 else 0.0

    def _extract_frames(self, video_path: Path, target_num_frames: int) -> List[Image.Image]:
        """Flow cũ: OpenCV dense pool → prune theo target_num tuyệt đối."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        num_dense = min(target_num_frames * self.dense_pool_multiplier, total)
        indices   = np.linspace(0, total - 1, max(num_dense, target_num_frames), dtype=int)

        candidate_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil = pil.resize((self.frame_size, self.frame_size), Image.LANCZOS)
                candidate_frames.append(pil)
        cap.release()

        if not candidate_frames:
            return []

        if len(candidate_frames) > target_num_frames:
            if self.pruning_strategy == "motion":
                return prune_by_motion(candidate_frames, target_num_frames)
            elif self.pruning_strategy == "scene_change":
                return prune_by_scene_change(candidate_frames, target_num_frames)
            else:
                return prune_uniform(candidate_frames, target_num_frames)

        return candidate_frames

    # ── Synthetic fallback ────────────────────────────────────────

    def _make_synthetic_frames(self, n: int) -> List[Image.Image]:
        return [
            Image.fromarray(np.random.randint(0, 255, (self.frame_size, self.frame_size, 3), dtype=np.uint8))
            for _ in range(n)
        ]

    def _append_synthetic_pair(self, i: int):
        frames = self._make_synthetic_frames(self.num_frames)
        for bucket, task, suffix in [
            (BUCKET_SHORT, "short_caption",      "short"),
            (BUCKET_LONG,  "exhaustive_caption",  "exhaustive"),
        ]:
            self.samples.append({
                "id": f"ytb_synthetic_{i}_{suffix}", "image": None,
                "frames": frames, "video_path": None, "input_mode": self.input_mode,
                "prompt": PROMPT_SHORT if task == "short_caption" else PROMPT_EXHAUSTIVE,
                "reference": "", "token_bucket": bucket, "task": task,
                "dataset": "synthetic", "duration_s": 0.0,
                "num_frames": self.num_frames, "has_real_frames": False,
                "base_frames": None, "n_base_frames": 0,
                "n_pruned_frames": self.num_frames, "keep_ratio": None,
                "pruning_strategy": "synthetic",
            })

    def _make_synthetic(self):
        for i in range(self.num_samples // 2):
            self._append_synthetic_pair(i)
