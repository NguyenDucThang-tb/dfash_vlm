"""
Quick test: chạy model trên vài video.
Hỗ trợ 3 nguồn:
  - videomme    : VideoMME-Short từ HuggingFace (mặc định)
  - youtube     : YouTube-100 (download bằng yt-dlp, cần internet)
  - long_youtube: Long YouTube >10 phút (download bằng yt-dlp, cần internet)

Hỗ trợ 2 input mode:
  - frames (mặc định): decode + prune frames trước, truyền List[PIL.Image] vào model
  - video            : truyền video_path thẳng vào model (chỉ Qwen hỗ trợ native;
                       Llama tự fallback về frames)

Chạy:
    python preview.py
    python preview.py --source youtube --num-videos 3
    python preview.py --source youtube --num-videos 3 --num-frames 8
    python preview.py --source long_youtube --num-videos 2
    python preview.py --source videomme --num-videos 3 --num-frames 4
    python preview.py --model llama --source videomme --num-videos 2
    python preview.py --model qwen3vl --source youtube --input-model video --num-videos 3
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image as PILImage


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source", choices=["videomme", "youtube", "long_youtube"], default="videomme",
        help="Nguồn video: 'videomme', 'youtube', hoặc 'long_youtube'",
    )
    p.add_argument(
        "--num-videos", type=int, default=3,
        help="Số video test (mỗi video = 2 samples: short + exhaustive)",
    )
    p.add_argument(
        "--num-frames", type=int, default=32,
        help="Frames extract mỗi video. long_youtube dùng dynamic budget tự động",
    )
    p.add_argument(
        "--video-dir", default="/kaggle/working/ytb_videos",
        help="Thư mục cache video YouTube",
    )
    p.add_argument(
        "--long-video-dir", default="/kaggle/working/long_ytb_videos",
        help="Thư mục cache video dài",
    )
    p.add_argument(
        "--pruning", choices=["scene_change", "motion", "uniform"], default="scene_change",
        help="Pruning strategy. Chỉ áp dụng khi --input-model frames",
    )
    p.add_argument(
        "--model", choices=["qwen3vl", "llama"], default="qwen3vl",
        help="Model inference: 'qwen3vl' (mặc định) hoặc 'llama'",
    )
    p.add_argument(
        "--input-model", choices=["frames", "video"], default="frames",
        dest="input_mode",
        help=(
            "'frames' (mặc định): decode + prune → List[PIL.Image]. "
            "'video': truyền video_path thẳng vào model (Qwen native; Llama tự fallback). "
            "Không áp dụng cho --source videomme."
        ),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output-dir", default="results/test_videomme",
        help="Thư mục lưu ảnh output",
    )
    return p.parse_args()


# ── Load dataset ───────────────────────────────────────────────────────────────

def load_videomme_pairs(num_videos: int, num_frames: int):
    from my_data.videomme_short import VideoMMEShortDataset
    from my_data.base import BUCKET_SHORT, BUCKET_LONG

    ds = VideoMMEShortDataset(num_samples=num_videos * 2, num_frames=num_frames)

    by_video: dict = {}
    for s in ds:
        vid = s.get("video_id") or s["id"].rsplit("_", 1)[0]
        by_video.setdefault(vid, {})[s["token_bucket"]] = s

    pairs = []
    for vid, buckets in list(by_video.items())[:num_videos]:
        s_short = buckets.get(BUCKET_SHORT)
        s_long  = buckets.get(BUCKET_LONG)
        if s_short and s_long:
            pairs.append((s_short, s_long))
    return pairs


def load_youtube_pairs(num_videos: int, num_frames: int, video_dir: str, input_mode: str):
    from my_data.youtube_video import YouTubeVideoDataset
    from my_data.base import BUCKET_SHORT, BUCKET_LONG

    ds = YouTubeVideoDataset(
        num_samples=num_videos * 2,
        num_frames=num_frames,
        video_dir=video_dir,
        input_mode=input_mode,
    )

    by_video: dict = {}
    for s in ds:
        vid = s.get("video_id") or s["id"].replace("ytb_", "").rsplit("_", 1)[0]
        by_video.setdefault(vid, {})[s["token_bucket"]] = s

    pairs = []
    for vid, buckets in list(by_video.items())[:num_videos]:
        s_short = buckets.get(BUCKET_SHORT)
        s_long  = buckets.get(BUCKET_LONG)
        if s_short and s_long:
            pairs.append((s_short, s_long))
    return pairs


def load_long_youtube_pairs(num_videos: int, video_dir: str, pruning: str, input_mode: str):
    from my_data.long_youtube import LongYouTubeVideoDataset
    from my_data.base import BUCKET_SHORT, BUCKET_LONG

    ds = LongYouTubeVideoDataset(
        num_samples=max(num_videos * 2, 2),
        video_dir=video_dir,
        pruning_strategy=pruning,
        input_mode=input_mode,
    )

    by_video: dict = {}
    for s in ds:
        vid = s["id"].replace("long_ytb_", "").rsplit("_", 1)[0]
        by_video.setdefault(vid, {})[s["token_bucket"]] = s

    pairs = []
    for vid, buckets in list(by_video.items())[:num_videos]:
        s_short = buckets.get(BUCKET_SHORT)
        s_long  = buckets.get(BUCKET_LONG)
        if s_short and s_long:
            pairs.append((s_short, s_long))
    return pairs


def load_video_pairs(source, num_videos, num_frames, video_dir, long_video_dir, pruning, input_mode):
    if source == "youtube":
        print(f"[preview] Source: YouTube-100  (video_dir={video_dir}, input_mode={input_mode})")
        return load_youtube_pairs(num_videos, num_frames, video_dir, input_mode)
    elif source == "long_youtube":
        print(f"[preview] Source: LongYouTube >10min  (video_dir={long_video_dir}, pruning={pruning}, input_mode={input_mode})")
        return load_long_youtube_pairs(num_videos, long_video_dir, pruning, input_mode)
    else:
        if input_mode == "video":
            print("[preview] ⚠️  VideoMME không hỗ trợ input_mode='video' → fallback về frames")
        print("[preview] Source: VideoMME-Short (HuggingFace)")
        return load_videomme_pairs(num_videos, num_frames)


# ── Load model ─────────────────────────────────────────────────────────────────

def load_model(model_name: str, device: str):
    if model_name == "llama":
        from models.baselines.llama import LlamaVisionAdapter
        return LlamaVisionAdapter(device=device)
    else:
        from models.baselines.qwen3vl import Qwen3VLAdapter
        return Qwen3VLAdapter(device=device)


# ── Llama fallback ─────────────────────────────────────────────────────────────

def _maybe_fallback_to_frames(model, sample: dict) -> dict:
    """Llama không hỗ trợ native video_path → decode frames tạm nếu cần."""
    try:
        from models.baselines.llama import LlamaVisionAdapter
        if not isinstance(model, LlamaVisionAdapter):
            return sample
    except ImportError:
        return sample

    if sample.get("input_mode") != "video" or sample.get("frames"):
        return sample

    video_path = sample.get("video_path")
    if not video_path:
        return sample

    print(f"  [fallback] Llama: decode frames từ {os.path.basename(video_path)}")
    import cv2
    import numpy as np

    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs   = np.linspace(0, total - 1, min(8, total), dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil.resize((224, 224)))
    cap.release()

    patched = dict(sample)
    patched["frames"]     = frames
    patched["input_mode"] = "frames"
    return patched


# ── Generate ───────────────────────────────────────────────────────────────────

def run_pair(model, short_sample: dict, exh_sample: dict) -> dict:
    label_short = short_sample.get("task", "short")
    label_exh   = exh_sample.get("task", "long")

    short_sample = _maybe_fallback_to_frames(model, short_sample)
    exh_sample   = _maybe_fallback_to_frames(model, exh_sample)

    print(f"  → {label_short:<20} ...", end=" ", flush=True)
    t0        = time.perf_counter()
    out_short = model.generate(short_sample)
    print(f"{time.perf_counter() - t0:.1f}s  |  '{out_short['text'].strip()[:80]}'")

    print(f"  → {label_exh:<20} ...", end=" ", flush=True)
    t0      = time.perf_counter()
    out_exh = model.generate(exh_sample)
    print(f"{time.perf_counter() - t0:.1f}s  |  '{out_exh['text'].strip()[:80]}...'")

    vis_frames = (
        out_short.get("used_frames")
        or short_sample.get("frames")
        or []
    )

    video_id   = short_sample.get("video_id") or short_sample.get("id", "unknown")
    model_name = getattr(model, "MODEL_NAME", "VLM")
    input_mode = short_sample.get("input_mode", "frames")

    return {
        "model_name":   model_name,
        "input_mode":   input_mode,
        "video_id":     video_id,
        "question":     _extract_question(short_sample["prompt"]),
        "candidates":   short_sample.get("candidates", []),
        "answer_gt":    short_sample.get("answer", "—"),
        "frames":       vis_frames,
        "duration_s":   short_sample.get("duration_s"),
        "num_frames":   short_sample.get("num_frames"),
        "task":         short_sample.get("task"),
        "dataset":      short_sample.get("dataset"),
        "video_bucket": short_sample.get("video_bucket"),
        "short_prompt": short_sample["prompt"],
        "short_text":   out_short["text"],
        "short_tokens": out_short.get("num_tokens", 0),
        "short_ttft":   out_short.get("time_to_first_token_s"),
        "short_label":  label_short,
        "exh_prompt":   exh_sample["prompt"],
        "exh_text":     out_exh["text"],
        "exh_tokens":   out_exh.get("num_tokens", 0),
        "exh_ttft":     out_exh.get("time_to_first_token_s"),
        "exh_label":    label_exh,
    }


def _extract_question(prompt: str) -> str:
    for line in prompt.split("\n"):
        if line.startswith("Question:"):
            return line.replace("Question:", "").strip()
    return ""


# ── Visualize ──────────────────────────────────────────────────────────────────

def _wrap_text(text: str, width: int = 80) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width))


def save_result_figure(result: dict, out_path: str):
    frames   = result.get("frames") or []
    n_frames = len(frames)

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#1a1a2e")

    if n_frames > 0:
        n_cols = min(n_frames, 32)
        gs = gridspec.GridSpec(
            4, n_cols, figure=fig,
            hspace=0.55, wspace=0.05,
            height_ratios=[2.5, 1, 1.2, 2.5],
        )
        for j, frame in enumerate(frames[:n_cols]):
            ax = fig.add_subplot(gs[0, j])
            ax.imshow(frame)
            ax.set_title(f"f{j}", fontsize=6, color="#aaaacc", pad=2)
            ax.axis("off")
    else:
        n_cols = 4
        gs = gridspec.GridSpec(
            4, n_cols, figure=fig,
            hspace=0.55, wspace=0.05,
            height_ratios=[2.5, 1, 1.2, 2.5],
        )
        ax_ph = fig.add_subplot(gs[0, :])
        ax_ph.set_facecolor("#0a0a1a")
        ax_ph.text(
            0.5, 0.5,
            f"[native video mode — frames not decoded]\nvideo: {result.get('video_id', '?')}",
            transform=ax_ph.transAxes, ha="center", va="center",
            fontsize=10, color="#aaaacc", fontfamily="monospace",
        )
        ax_ph.axis("off")

    # Row 1: metadata
    ax_meta = fig.add_subplot(gs[1, :])
    ax_meta.axis("off")
    input_mode_str = f"  |  input_mode: {result.get('input_mode', 'frames')}"
    if result.get("candidates"):
        cands_str = "  |  ".join(result["candidates"])
        meta_text = (
            f"video_id: {result['video_id']}{input_mode_str}\n"
            f"Q: {result['question']}\n"
            f"{cands_str}\n"
            f"Ground truth: {result['answer_gt']}"
        )
    else:
        duration_str   = f"  |  duration: {result.get('duration_s')}s" if result.get("duration_s") else ""
        num_frames_str = f"  |  frames: {result.get('num_frames')}" if result.get("num_frames") else ""
        bucket_str     = f"  |  bucket: {result['video_bucket']}" if result.get("video_bucket") else ""
        meta_text = (
            f"video_id: {result['video_id']}{duration_str}{num_frames_str}{bucket_str}{input_mode_str}\n"
            f"task: {result.get('task', 'video_caption')}\n"
            f"dataset: {result.get('dataset', 'YouTube')}"
        )
    ax_meta.text(
        0.01, 0.95, meta_text,
        transform=ax_meta.transAxes, fontsize=9, color="#e0e0ff",
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#16213e", alpha=0.9),
    )

    # Row 2: short output
    ax_short = fig.add_subplot(gs[2, :])
    ax_short.axis("off")
    ttft_str    = f"  TTFT={result['short_ttft']:.2f}s" if result["short_ttft"] else ""
    short_label = result.get("short_label", "short")
    short_body  = (
        f"[PROMPT: {short_label}]\n"
        f"MODEL OUTPUT ({result['short_tokens']} tokens{ttft_str}):\n"
        f"{result['short_text'].strip()}"
    )
    ax_short.text(
        0.01, 0.95, _wrap_text(short_body, 120),
        transform=ax_short.transAxes, fontsize=9, color="#90ee90",
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f3460", alpha=0.9),
    )
    ax_short.set_title(short_label, fontsize=9, color="#90ee90", loc="left", pad=3)

    # Row 3: exhaustive output
    ax_exh = fig.add_subplot(gs[3, :])
    ax_exh.axis("off")
    ttft_str  = f"  TTFT={result['exh_ttft']:.2f}s" if result["exh_ttft"] else ""
    exh_label = result.get("exh_label", "long")
    exh_body  = (
        f"[PROMPT: {exh_label}]\n\n"
        f"MODEL OUTPUT ({result['exh_tokens']} tokens{ttft_str}):\n"
        f"{result['exh_text'].strip()}"
    )
    ax_exh.text(
        0.01, 0.95, _wrap_text(exh_body, 120),
        transform=ax_exh.transAxes, fontsize=9, color="#ffd700",
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f3460", alpha=0.9),
    )
    ax_exh.set_title(exh_label, fontsize=9, color="#ffd700", loc="left", pad=3)

    dataset_label = result.get("dataset") or "YouTube"
    bucket_label  = f"  ·  {result['video_bucket']}" if result.get("video_bucket") else ""
    mode_label    = f"  ·  [{result.get('input_mode', 'frames')}]"
    model_label   = result.get("model_name", "VLM")
    fig.suptitle(
        f"{model_label}  ·  {dataset_label}{bucket_label}{mode_label}  ·  {result['video_id']}",
        fontsize=12, color="white", y=0.995,
    )

    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ saved → {out_path}")


# ── Print console summary ──────────────────────────────────────────────────────

def print_summary(results: list):
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"{'VIDEO SUMMARY':^90}")
    print(sep)

    for i, r in enumerate(results):
        duration_str = f"  ({r['duration_s']}s / {r['duration_s']/60:.1f}min)" if r.get("duration_s") else ""
        bucket_str   = f"  [{r['video_bucket']}]" if r.get("video_bucket") else ""
        mode_str     = f"  [input_mode={r.get('input_mode', 'frames')}]"
        print(f"\n[{i+1}] video_id : {r['video_id']}{duration_str}{bucket_str}{mode_str}")

        if r.get("question"):
            print(f"    question : {r['question']}")
            print(f"    gt answer: {r['answer_gt']}")
        print()

        ttft_s = f"TTFT={r['short_ttft']:.2f}s" if r["short_ttft"] else "TTFT=n/a"
        print(f"    [{r['short_label']:<16}] {r['short_tokens']:>4} tokens  {ttft_s}")
        print(f"    output → '{r['short_text'].strip()}'")
        print()

        ttft_e      = f"TTFT={r['exh_ttft']:.2f}s" if r["exh_ttft"] else "TTFT=n/a"
        exh_preview = r["exh_text"].strip().replace("\n", " ")[:200]
        print(f"    [{r['exh_label']:<16}] {r['exh_tokens']:>4} tokens  {ttft_e}")
        print(f"    output → '{exh_preview}...'")

    print(f"\n{sep}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_mode == "video" and args.model == "llama":
        print(
            "[preview] ⚠️  Llama không hỗ trợ input_mode='video' native. "
            "Sẽ tự fallback decode frames khi chạy inference."
        )

    print(f"\n{'='*60}")
    print(f"Loading {args.num_videos} video(s) from '{args.source}' [{args.input_mode} mode]...")
    print(f"{'='*60}")
    pairs = load_video_pairs(
        source         = args.source,
        num_videos     = args.num_videos,
        num_frames     = args.num_frames,
        video_dir      = args.video_dir,
        long_video_dir = args.long_video_dir,
        pruning        = args.pruning,
        input_mode     = args.input_mode,
    )
    print(f"→ Got {len(pairs)} video pair(s)\n")

    if not pairs:
        print("ERROR: no samples loaded. Kiểm tra kết nối HuggingFace hoặc thư mục video.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Loading model '{args.model}' on {args.device}...")
    print(f"{'='*60}")
    model = load_model(args.model, args.device)

    print(f"\n{'='*60}")
    print(f"Running inference...")
    print(f"{'='*60}")

    results = []
    for idx, (s_short, s_exh) in enumerate(pairs):
        vid_label = s_short.get("video_id") or s_short.get("id", "unknown")
        print(f"\n[video {idx+1}/{len(pairs)}] {vid_label}")
        result = run_pair(model, s_short, s_exh)
        results.append(result)

        fig_path = os.path.join(
            args.output_dir,
            f"video_{idx+1:02d}_{result['video_id']}_{args.source}_{args.model}_{args.input_mode}.png",
        )
        save_result_figure(result, fig_path)

    print_summary(results)
    print(f"\n✅ Done [{args.model} / {args.input_mode}]. Figures saved → {args.output_dir}/")


if __name__ == "__main__":
    main()
