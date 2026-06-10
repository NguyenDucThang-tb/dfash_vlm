"""
Frame Pruning Strategies for Long Videos.
Hỗ trợ 4 phương pháp: uniform, motion, scene_change, ratio.

Thêm mới (Câu 1 + Câu 2):
  - get_qwen3_base_frames(): lấy đúng bộ frame mà Qwen3 tự decode từ video.
  - prune_by_ratio(): prune theo % số frame giữ lại (keep_ratio 0.0–1.0).

Fix:
  - get_qwen3_base_frames KHÔNG dùng process_vision_info nữa vì torchcodec bug
    (fps bị ignore → fallback fps=24 → OOM hoặc sai số frame).
    Thay bằng: torchvision.io.read_video → extract thủ công → PIL list.
    Kết quả nhất quán với TARGET_FPS, không phụ thuộc torchcodec.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from PIL import Image


# ─────────────────────────────────────────────────────────────────
# PHẦN 1: HELPER NỘI BỘ
# ─────────────────────────────────────────────────────────────────

def _calculate_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def _remove_global_redundancy(
    pil_frames: List[Image.Image],
    similarity_threshold: float = 0.92,
) -> List[Image.Image]:
    if len(pil_frames) <= 1:
        return pil_frames
    filtered = [pil_frames[0]]
    for i in range(1, len(pil_frames)):
        img_prev = np.array(filtered[-1].resize((64, 64)))
        img_curr = np.array(pil_frames[i].resize((64, 64)))
        if _calculate_histogram_similarity(img_prev, img_curr) < similarity_threshold:
            filtered.append(pil_frames[i])
    return filtered


# ─────────────────────────────────────────────────────────────────
# PHẦN 1.5: HELPER TÍNH MAX_FRAMES (giữ nguyên cho tương thích)
# ─────────────────────────────────────────────────────────────────

def _get_video_max_frames(video_path: str, fps: float) -> int:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 512
        video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames <= 0:
            return 512
        duration_s = total_frames / video_fps
        needed = int(np.ceil(duration_s * fps)) * 2
        return max(32, min(needed, 2048))
    except Exception:
        return 512


# ─────────────────────────────────────────────────────────────────
# PHẦN 2: LẤY BASE FRAMES (CÂU 1)
# ─────────────────────────────────────────────────────────────────

def get_qwen3_base_frames(
    video_path: str,
    fps: float = 1.0,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
) -> Tuple[Optional[List[Image.Image]], int]:
    """
    Lấy bộ frame tương đương Qwen3-VL decode từ video_path.

    FIX: Không dùng process_vision_info/torchcodec nữa vì bug fps bị ignore
    (Qwen3-VL issue #1329) → torchcodec fallback fps=24 → decode hàng nghìn
    frames → OOM hoặc sai số frame hoàn toàn.

    Thay bằng torchvision.io.read_video (extract thủ công, đúng fps, nhẹ hơn).
    Fallback về OpenCV nếu torchvision fail.

    Args:
        video_path      : đường dẫn tới file video
        fps             : số frame/giây muốn sample (mặc định 1.0)
        min_pixels      : không dùng trực tiếp (giữ signature tương thích)
        max_pixels      : không dùng trực tiếp (giữ signature tương thích)

    Returns:
        (frames, n) — List[PIL.Image] và số frame, hoặc (None, 0) nếu thất bại
    """
    # --- Thử torchvision ---
    try:
        import torchvision.io as tvio

        video, _, info = tvio.read_video(video_path, pts_unit="sec", output_format="TCHW")
        native_fps   = info.get("video_fps", 25.0)
        total_frames = video.shape[0]

        if total_frames == 0:
            raise ValueError("read_video returned 0 frames")

        # Sample đều theo fps
        step    = max(1, round(native_fps / fps))
        indices = list(range(0, total_frames, step))

        frames = []
        for idx in indices:
            arr = video[idx].permute(1, 2, 0).numpy()  # C,H,W → H,W,C
            frames.append(Image.fromarray(arr))

        print(
            f"[frame_pruner] get_qwen3_base_frames (torchvision): "
            f"native_fps={native_fps:.1f} total={total_frames} "
            f"step={step} → {len(frames)} frames (fps={fps})"
        )
        return frames, len(frames)

    except Exception as e:
        print(f"[frame_pruner] torchvision failed ({e}) → OpenCV fallback")

    # --- Fallback OpenCV ---
    return _opencv_base_frames(video_path, fps)


def _opencv_base_frames(
    video_path: str,
    fps: float = 1.0,
) -> Tuple[Optional[List[Image.Image]], int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step         = max(1, int(video_fps / fps))
    indices      = list(range(0, total_frames, step))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    duration_s = total_frames / video_fps if video_fps > 0 else 0
    print(
        f"[frame_pruner] get_qwen3_base_frames (OpenCV): "
        f"{len(frames)} frames (duration={duration_s:.1f}s, fps={fps})"
    )
    return (frames if frames else None), len(frames)


# ─────────────────────────────────────────────────────────────────
# PHẦN 3: CÁC HÀM PRUNE GỐC
# ─────────────────────────────────────────────────────────────────

def prune_uniform(pil_frames: List[Image.Image], target_num: int) -> List[Image.Image]:
    n = len(pil_frames)
    if n <= target_num:
        return pil_frames
    indices = np.linspace(0, n - 1, target_num, dtype=int)
    return [pil_frames[i] for i in indices]


def prune_by_motion(
    pil_frames: List[Image.Image],
    target_num: int,
    remove_redundancy: bool = True,
) -> List[Image.Image]:
    n = len(pil_frames)
    if n <= target_num:
        return pil_frames

    resize_to   = (64, 64)
    frames_np   = [np.array(f.resize(resize_to)) for f in pil_frames]
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_np]

    segment_size     = n / target_num
    selected_indices = []

    for i in range(target_num):
        start_idx = int(i * segment_size)
        end_idx   = int((i + 1) * segment_size)

        if end_idx - start_idx <= 1:
            selected_indices.append(start_idx)
            continue

        max_motion = -1
        best_idx   = start_idx

        for j in range(start_idx, end_idx - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[j], gray_frames[j + 1], None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.2, flags=0,
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
            if magnitude > max_motion:
                max_motion = magnitude
                best_idx   = j + 1

        selected_indices.append(best_idx)

    selected_indices = sorted(set(selected_indices))
    selected_frames  = [pil_frames[i] for i in selected_indices]

    if remove_redundancy:
        selected_frames = _remove_global_redundancy(selected_frames, similarity_threshold=0.92)

    return selected_frames


def prune_by_scene_change(
    pil_frames: List[Image.Image],
    target_num: int,
    threshold: float = 30.0,
) -> List[Image.Image]:
    n = len(pil_frames)
    if n <= target_num:
        return pil_frames

    keyframe_indices = [0]
    prev_hist        = None

    for i in range(1, n):
        curr_img  = cv2.cvtColor(np.array(pil_frames[i]), cv2.COLOR_RGB2HSV)
        curr_hist = cv2.calcHist([curr_img], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > (threshold / 100.0):
                keyframe_indices.append(i)

        prev_hist = curr_hist

    if len(keyframe_indices) > target_num:
        return prune_uniform([pil_frames[i] for i in keyframe_indices], target_num)
    elif len(keyframe_indices) < target_num:
        missing           = target_num - len(keyframe_indices)
        remaining_indices = [i for i in range(n) if i not in keyframe_indices]
        extra_indices     = prune_uniform(remaining_indices, missing) if remaining_indices else []
        return [pil_frames[i] for i in keyframe_indices] + [pil_frames[i] for i in extra_indices]

    return [pil_frames[i] for i in keyframe_indices]


# ─────────────────────────────────────────────────────────────────
# PHẦN 4: PRUNE THEO % FRAME (CÂU 2)
# ─────────────────────────────────────────────────────────────────

def prune_by_ratio(
    pil_frames: List[Image.Image],
    keep_ratio: float,
    strategy: str = "motion",
) -> List[Image.Image]:
    """
    Prune theo tỷ lệ % frame giữ lại.

    Args:
        pil_frames : list frame gốc (base frames từ get_qwen3_base_frames)
        keep_ratio : tỷ lệ giữ lại, 0.0 < keep_ratio <= 1.0
        strategy   : "motion" | "scene_change" | "uniform"
    """
    if not pil_frames:
        return pil_frames

    keep_ratio = max(0.05, min(1.0, keep_ratio))
    target_num = max(1, int(len(pil_frames) * keep_ratio))

    print(
        f"[frame_pruner] prune_by_ratio: {len(pil_frames)} → {target_num} frames "
        f"(keep={keep_ratio*100:.0f}%, strategy={strategy})"
    )

    if strategy == "motion":
        return prune_by_motion(pil_frames, target_num)
    elif strategy == "scene_change":
        return prune_by_scene_change(pil_frames, target_num)
    else:
        return prune_uniform(pil_frames, target_num)
