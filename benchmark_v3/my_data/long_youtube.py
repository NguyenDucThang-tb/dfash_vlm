"""
Long YouTube Video Dataset — video dài >10 phút cho benchmark.

Hỗ trợ 2 input mode:
  - input_mode="frames" (mặc định): decode + prune → sample["frames"]
  - input_mode="video":             chỉ lưu path   → sample["video_path"]

Pruning strategy mới (Câu 1 + Câu 2) — giống youtube_video.py nhưng có 2 điểm riêng:
  - Frame budget ĐỘNG theo thời lượng video (FRAME_BUDGET_TABLE: 32–128 frames).
    Khi dùng qwen3_base_ratio, keep_ratio áp dụng lên bộ base frames Qwen3,
    không phải lên budget tuyệt đối → số frame cuối cùng có thể khác budget.
  - Có tham số scene_change_threshold và remove_redundancy riêng cho video dài.

pruning_strategy="qwen3_base_ratio":
    Bước 1 — get_qwen3_base_frames(video_path, fps=qwen3_fps)
              Lấy đúng bộ frame Qwen3 decode. Video dài → Qwen3 sẽ decode 200–500+ frames.
    Bước 2 — prune_by_ratio(base_frames, keep_ratio, ratio_sub_strategy)
              Prune theo % trên bộ base frames đó.
    Kết quả: nhất quán với model, không phụ thuộc vào FRAME_BUDGET_TABLE.

pruning_strategy="scene_change" / "motion" / "uniform":
    Flow cũ: OpenCV dense pool × dense_pool_multiplier → prune đến frame budget.
    Frame budget = _get_frame_budget(duration_s).

Mỗi sample lưu thêm:
  - "base_frames"     : List[PIL.Image] base frames Qwen3 (TRƯỚC prune), None nếu strategy cũ
  - "keep_ratio"      : tỷ lệ frame giữ lại (chỉ qwen3_base_ratio)
  - "n_base_frames"   : số base frame Qwen3
  - "n_pruned_frames" : số frame sau prune
  - "frame_budget"    : frame budget theo FRAME_BUDGET_TABLE (dù strategy nào)
  - "video_bucket"    : nhãn thời lượng ("10-15min", "15-30min", ...)
"""

import cv2
import os
import re
import numpy as np
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image
from my_data.base import BaseDataset, BUCKET_SHORT, BUCKET_LONG, MODE_FRAMES, MODE_VIDEO
from my_data.frame_pruner import (
    prune_uniform, prune_by_motion, prune_by_scene_change,
    prune_by_ratio, get_qwen3_base_frames,
    _remove_global_redundancy,
)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

'''
"https://youtu.be/IV3dnLzthDA?si=AFx9ccZtq3XCUMC-",
    "https://youtu.be/ILgSesWMUEI?si=FD1BfVceiXmBrITV",
    "https://youtu.be/Xzv84ZdtlE0?si=idXUG_fb-wsUX3ch",
    "https://youtu.be/K-Fc08X56R0?si=1Fwxtiln_VIROaHs",
    "https://youtu.be/RUNDkDJ3zI8?si=YGpzKzsHP_gG38ma",
    "https://youtu.be/QP9IwVr3BOs?si=2khCxzHeC06y6eq0",
'''
LONG_YOUTUBE_URLS: List[str] = [
    "https://youtu.be/QLJSc6Jgzzs?si=6FdE394WxsSeeEYV"
]

# Frame budget động theo thời lượng — dùng cho strategy cũ (OpenCV)
# Khi dùng qwen3_base_ratio, budget này chỉ dùng để log tham khảo
FRAME_BUDGET_TABLE = [
    (0,    600,          32),
    (600,  900,          48),
    (900,  1800,         64),
    (1800, 3600,         96),
    (3600, float("inf"), 128),
]

PROMPT_SHORT = "Describe this video in one or two concise sentences."
PROMPT_EXHAUSTIVE = (
    "You are analyzing the video frames provided above. "
    "Identify and describe the entire cooking process as a detailed sequence of steps. "
    "For each step, clearly state: (1) the action being performed, "
    "(2) the ingredients or tools involved, and "
    "(3) any observable details such as quantities, textures, colors, or timing if visible. "
    "Present the output as a numbered list of steps from start to finish. "
    "Focus on actions such as cutting, mixing, frying, boiling, baking, or plating. "
    "Do not skip any visible step, even minor ones like preheating, seasoning, or garnishing. "
    "Be thorough and descriptive."
)

# Strategies dùng Qwen3 base frames
_QWEN3_RATIO_STRATEGIES = {"qwen3_base_ratio"}


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _get_frame_budget(duration_s: float) -> int:
    for lo, hi, budget in FRAME_BUDGET_TABLE:
        if lo <= duration_s < hi:
            return budget
    return 64


def _extract_vid_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else "unknown_id"


def _duration_label(duration_s: float) -> str:
    minutes = duration_s / 60
    if minutes < 15:   return "10-15min"
    elif minutes < 30: return "15-30min"
    elif minutes < 60: return "30-60min"
    else:              return "60min+"


# ─────────────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────────────

class LongYouTubeVideoDataset(BaseDataset):

    DATASET_NAME = "LongYouTube"
    MODALITY     = "video"

    def __init__(
        self,
        num_samples: int = 90,
        frame_size: int = 224,
        video_dir: str = "/kaggle/working/long_ytb_videos",
        buckets: Optional[List[str]] = None,
        force_download: bool = False,
        # ── Pruning strategy ────────────────────────────────────
        # "scene_change" / "motion" / "uniform": flow cũ (OpenCV dense pool)
        # "qwen3_base_ratio": CÂU 1+2 — Qwen3 base frames + prune theo %
        pruning_strategy: str = "scene_change",
        scene_change_threshold: float = 0.25,   # chỉ dùng cho strategy cũ
        dense_pool_multiplier: int = 4,          # chỉ dùng cho strategy cũ
        remove_redundancy: bool = True,          # áp dụng cả 2 strategy
        # Tham số riêng cho qwen3_base_ratio
        keep_ratio: float = 0.5,                 # giữ 50% base frames
        qwen3_fps: float = 1.0,                  # fps Qwen3 dùng khi decode
        ratio_sub_strategy: str = "scene_change", # strategy nội bộ (video dài → scene_change phù hợp hơn)
        # ── Input mode ──────────────────────────────────────────
        input_mode: str = MODE_FRAMES,
        min_duration_s: float = 0.0,
        
    ):
        super().__init__(num_samples, buckets, input_mode)
        self.frame_size             = frame_size
        self.video_dir              = Path(video_dir)
        self.force_download         = force_download
        self.pruning_strategy       = pruning_strategy
        self.scene_change_threshold = scene_change_threshold
        self.dense_pool_multiplier  = dense_pool_multiplier
        self.remove_redundancy      = remove_redundancy
        self.keep_ratio             = keep_ratio
        self.qwen3_fps              = qwen3_fps
        self.ratio_sub_strategy     = ratio_sub_strategy
        
        self._load()
        

    # ── Load ──────────────────────────────────────────────────────

    def _load(self):
        self.video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[LongYouTube] _load: video_dir={self.video_dir}, mode={self.input_mode}, strategy={self.pruning_strategy}")

        video_paths = self._download_all()
        if not video_paths:
            print("[LongYouTube] No videos found → synthetic fallback")
            self._make_synthetic()
            return

        n_videos_target  = self.num_samples // 2
        videos_processed = 0

        for vid_path in video_paths:
            if videos_processed >= n_videos_target:
                break

            duration = self._get_duration(vid_path)
            

            vid_id        = vid_path.stem
            frame_budget  = _get_frame_budget(duration)   # tham chiếu, luôn tính

            if self.input_mode == MODE_VIDEO:
                # Hướng 2: chỉ lưu path, không decode
                frames      = None
                base_frames = None
                video_path  = str(vid_path)
                n_base      = 0
                n_pruned    = frame_budget
            else:
                # Hướng 1: decode + prune
                frames, base_frames, n_base, n_pruned = self._extract_and_prune(
                    vid_path, duration, frame_budget
                )
                if not frames:
                    continue
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
                "num_frames":      n_pruned if frames else frame_budget,
                "has_real_frames": True,
                "video_bucket":    _duration_label(duration),
                "frame_budget":    frame_budget,
                # ── Thông tin cho caption similarity (Câu 2) ──────────
                "base_frames":      base_frames,
                "n_base_frames":    n_base,
                "n_pruned_frames":  n_pruned,
                "keep_ratio":       self.keep_ratio if self.pruning_strategy in _QWEN3_RATIO_STRATEGIES else None,
                "pruning_strategy": self.pruning_strategy,
            }

            self.samples.append({**base,
                "id":           f"long_ytb_{vid_id}_short",
                "prompt":       PROMPT_SHORT,
                "token_bucket": BUCKET_SHORT,
                "task":         "short_caption",
            })
            self.samples.append({**base,
                "id":           f"long_ytb_{vid_id}_exhaustive",
                "prompt":       PROMPT_EXHAUSTIVE,
                "token_bucket": BUCKET_LONG,
                "task":         "exhaustive_caption",
            })

        shortage = n_videos_target - videos_processed
        if shortage > 0:
            print(f"[LongYouTube] Short {shortage * 2} samples → synthetic")
            for i in range(shortage):
                self._append_synthetic_pair(i)

        real = sum(1 for s in self.samples if s.get("has_real_frames"))

        strategy_label = self.pruning_strategy
        if self.pruning_strategy in _QWEN3_RATIO_STRATEGIES:
            strategy_label = (
                f"qwen3_base_ratio(keep={self.keep_ratio*100:.0f}%,"
                f"sub={self.ratio_sub_strategy},fps={self.qwen3_fps})"
            )

        print(
            f"[LongYouTube] Loaded {len(self.samples)} samples "
            f"({videos_processed} videos × 2 bucket) "
            f"| mode={self.input_mode} | strategy={strategy_label} "
            f"| real={real}, synthetic={len(self.samples) - real} "
            f"| {self.summary()['buckets']}"
        )

    # ── Core: extract + prune ─────────────────────────────────────

    def _extract_and_prune(
        self,
        vid_path: Path,
        duration: float,
        frame_budget: int,
    ) -> Tuple[Optional[List[Image.Image]], Optional[List[Image.Image]], int, int]:
        """
        Dispatch extract+prune theo pruning_strategy.

        Returns:
            (frames, base_frames, n_base, n_pruned)
            - frames      : bộ frame cuối (sau prune), đưa vào model
            - base_frames : bộ base frames Qwen3 TRƯỚC prune (chỉ qwen3_base_ratio)
            - n_base      : len(base_frames)
            - n_pruned    : len(frames)
        """
        if self.pruning_strategy in _QWEN3_RATIO_STRATEGIES:
            return self._extract_qwen3_ratio(vid_path, duration, frame_budget)
        else:
            frames = self._extract_frames_opencv(vid_path, frame_budget)
            n = len(frames)
            return frames, None, n, n

    def _extract_qwen3_ratio(
        self,
        vid_path: Path,
        duration: float,
        frame_budget: int,
    ) -> Tuple[Optional[List[Image.Image]], Optional[List[Image.Image]], int, int]:
        """
        CÂU 1: Lấy base frames từ Qwen3 pipeline.
        CÂU 2: Prune theo keep_ratio.

        Lưu ý cho video dài: Qwen3 với fps=1.0 sẽ decode ~600–3600+ frames
        cho video 10–60 phút. keep_ratio=0.5 giảm xuống ~300–1800 frames —
        vẫn nhiều hơn frame_budget. Nếu muốn kết quả gần với budget, dùng
        keep_ratio thấp hơn (0.1–0.2) hoặc điều chỉnh qwen3_fps.
        """
        base_frames, n_base = get_qwen3_base_frames(
            str(vid_path),
            fps=self.qwen3_fps,
        )

        if not base_frames:
            print(
                f"[LongYouTube] {vid_path.name}: get_qwen3_base_frames failed "
                f"→ fallback về OpenCV (budget={frame_budget})"
            )
            frames = self._extract_frames_opencv(vid_path, frame_budget)
            n = len(frames)
            return frames, None, n, n

        # Prune theo keep_ratio trên base frames
        frames = prune_by_ratio(
            base_frames,
            keep_ratio=self.keep_ratio,
            strategy=self.ratio_sub_strategy,
        )

        # Lọc redundancy nếu bật (đặc biệt hữu ích cho video dài có nhiều cảnh tĩnh)
        if self.remove_redundancy and len(frames) > 1:
            before = len(frames)
            frames = _remove_global_redundancy(frames, similarity_threshold=0.90)
            if len(frames) < before:
                print(
                    f"[LongYouTube] remove_redundancy: {before} → {len(frames)} frames "
                    f"({vid_path.name})"
                )

        n_pruned = len(frames)

        print(
            f"[LongYouTube] {vid_path.name}: "
            f"base={n_base} → pruned={n_pruned} "
            f"(keep={self.keep_ratio*100:.0f}%, budget_ref={frame_budget}, "
            f"duration={duration/60:.1f}min)"
        )

        return frames, base_frames, n_base, n_pruned

    # ── OpenCV extract (strategy cũ) ─────────────────────────────

    def _extract_frames_opencv(
        self,
        video_path: Path,
        target_num_frames: int,
    ) -> List[Image.Image]:
        """Flow cũ: OpenCV dense pool → prune theo frame budget."""
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
        if len(candidate_frames) <= target_num_frames:
            return candidate_frames

        # Prune
        if self.pruning_strategy == "scene_change":
            selected = prune_by_scene_change(
                candidate_frames,
                target_num_frames,
                threshold=self.scene_change_threshold * 100,
            )
        elif self.pruning_strategy == "motion":
            selected = prune_by_motion(candidate_frames, target_num_frames)
        else:
            selected = prune_uniform(candidate_frames, target_num_frames)

        if self.remove_redundancy and len(selected) > 1:
            selected = _remove_global_redundancy(selected, similarity_threshold=0.90)

        return selected

    # ── Download ──────────────────────────────────────────────────

    def _download_all(self) -> List[Path]:
        if not LONG_YOUTUBE_URLS:
            print("[LongYouTube] LONG_YOUTUBE_URLS is empty")
            return []
        try:
            subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(
                    ["pip", "install", "yt-dlp", "-q", "--break-system-packages"],
                    check=True,
                )
            except subprocess.CalledProcessError:
                print("[LongYouTube] ⚠️  yt-dlp not found and install failed → no videos")
                return []

        video_paths = []
        for url in LONG_YOUTUBE_URLS:
            vid_id   = _extract_vid_id(url)
            existing = list(self.video_dir.glob(f"{vid_id}.*"))
            if existing and not self.force_download:
                valid_mp4 = [f for f in existing if f.suffix.lower() == ".mp4"]
                if valid_mp4:
                    video_paths.append(valid_mp4[0])
                    continue
                else:
                    print(f"[LongYouTube] Old format {existing[0].name} → redownload")
                    try:
                        os.remove(existing[0])
                    except Exception as e:
                        print(f"[LongYouTube] Delete failed: {e}")

            print(f"[LongYouTube] Downloading {vid_id}...")
            try:
                subprocess.run([
                    "yt-dlp",
                    "-f", "worst[ext=mp4]/worst",
                    "--merge-output-format", "mp4",
                    "--max-filesize", "500m",
                    "--no-playlist",
                    "-o", str(self.video_dir / f"{vid_id}.mp4"),
                    "--quiet", "--no-warnings",
                    url,
                ], capture_output=True, text=True, timeout=600)

                downloaded = list(self.video_dir.glob(f"{vid_id}.mp4"))
                if downloaded:
                    video_paths.append(downloaded[0])
                    print(f"[LongYouTube] ✅ {downloaded[0].name}")
                else:
                    print(f"[LongYouTube] ❌ Failed {vid_id}")
            except Exception as e:
                print(f"[LongYouTube] ❌ Error {vid_id}: {e}")

        print(f"[LongYouTube] Downloaded/Found {len(video_paths)}/{len(LONG_YOUTUBE_URLS)} mp4")
        return video_paths

    # ── Duration ──────────────────────────────────────────────────

    def _get_duration(self, path: Path) -> float:
        """Dùng ffprobe nếu có (chính xác hơn), fallback về OpenCV."""
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                out = result.stdout.strip()
                if out:
                    duration = float(out)
                    print(f"[LongYouTube] {path.name} → {duration:.1f}s ({duration/60:.1f}min)")
                    return duration
        except Exception as e:
            print(f"[LongYouTube] ffprobe exception: {e}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return 0.0
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frames / fps if fps > 0 else 0.0

    # ── Synthetic fallback ────────────────────────────────────────

    def _make_synthetic_frames(self, n: int) -> List[Image.Image]:
        return [
            Image.fromarray(np.random.randint(0, 255, (self.frame_size, self.frame_size, 3), dtype=np.uint8))
            for _ in range(n)
        ]

    def _append_synthetic_pair(self, i: int):
        n_frames = 64
        frames   = self._make_synthetic_frames(n_frames)
        base = {
            "image": None, "frames": frames, "video_path": None,
            "input_mode": self.input_mode, "reference": "",
            "dataset": "synthetic", "duration_s": 0.0,
            "num_frames": n_frames, "has_real_frames": False,
            "video_bucket": "synthetic", "frame_budget": n_frames,
            "base_frames": None, "n_base_frames": 0,
            "n_pruned_frames": n_frames, "keep_ratio": None,
            "pruning_strategy": "synthetic",
        }
        self.samples.append({**base,
            "id": f"long_ytb_synthetic_{i}_short",
            "prompt": PROMPT_SHORT,
            "token_bucket": BUCKET_SHORT, "task": "short_caption",
        })
        self.samples.append({**base,
            "id": f"long_ytb_synthetic_{i}_exhaustive",
            "prompt": PROMPT_EXHAUSTIVE,
            "token_bucket": BUCKET_LONG, "task": "exhaustive_caption",
        })

    def _make_synthetic(self):
        for i in range(self.num_samples // 2):
            self._append_synthetic_pair(i)
