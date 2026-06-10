"""
Benchmark Runner v3

Thêm mới so với v2:
  CÂU 1: --pruning-strategy qwen3_base_ratio
          Dùng get_qwen3_base_frames() để lấy đúng bộ frame Qwen3 decode,
          thay vì OpenCV dense pool. Prune trên bộ frame đó.

  CÂU 2: --keep-ratio 0.5  (chỉ áp dụng khi strategy=qwen3_base_ratio)
          Prune theo % frame giữ lại.
          Tự động chạy 2 pass để tính caption similarity:
            Pass A: model chạy với base frames (Qwen3 full)
            Pass B: model chạy với pruned frames
          Tính ROUGE-L + BERTScore(nếu cài) giữa 2 caption.

  CÂU 3: VideoMME accuracy tự động được tính nếu sample có "answer" field.
          Không cần thêm tham số — tracker.record() nhận predicted_text + ground_truth_answer.

Chạy:

    # Test nhanh — mock mode
    python run_benchmark.py --datasets videomme_short --models qwen3vl --num-videos 3

    # Hướng 1 cũ — frames pipeline (OpenCV dense pool + motion prune)
    python run_benchmark.py --real --datasets youtube --models qwen3vl --input-model frames --num-videos 3

    # Hướng 2 — native video pipeline (Qwen tự decode)
    python run_benchmark.py --real --datasets youtube --models qwen3vl --input-model video --num-videos 3

    # CÂU 1+2: Qwen3 base frames + ratio prune + caption similarity
    python run_benchmark.py --real --datasets youtube --models qwen3vl \\
        --pruning-strategy qwen3_base_ratio --keep-ratio 0.5 --num-videos 5

    # CÂU 3: VideoMME accuracy tự động có khi dùng videomme_short
    python run_benchmark.py --real --datasets videomme_short --models qwen3vl --num-videos 10

    # Full benchmark
    python run_benchmark.py --real --datasets mscoco videomme_short youtube --models qwen3vl spec
"""

import argparse
import sys
import os
import time
from typing import Optional

from my_data import DATASET_REGISTRY
from models.spec_vlm import SpecVLMAdapter
from models.baselines import BASELINE_REGISTRY
from metrics.tracker import (
    MetricsTracker, compute_speedups, compute_caption_similarity,
    print_result, print_table, save_results,
)

_FRAMES_ONLY_DATASETS = {"mscoco"}
_VIDEO_DATASETS       = {"msrvtt", "youtube", "long_youtube", "videomme_short"}


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VLM Speculative Decoding Benchmark v3")

    p.add_argument(
        "--datasets", nargs="+",
        default=["mscoco", "videomme_short"],
        choices=list(DATASET_REGISTRY.keys()),
    )
    p.add_argument(
        "--models", nargs="+",
        default=["qwen3vl"],
        choices=["spec"] + list(BASELINE_REGISTRY.keys()),
    )
    p.add_argument(
        "--num-videos", type=int, default=None,
        help="Giới hạn số video test (mỗi video = 2 samples).",
    )
    p.add_argument("--num-samples", type=int, default=90)
    p.add_argument("--real", action="store_true", help="Load real models (requires GPU).")
    p.add_argument(
        "--input-model", choices=["frames", "video"], default="frames",
        dest="input_mode",
        help=(
            "'frames': decode + prune → List[PIL.Image]. "
            "'video': truyền video_path thẳng vào model (Qwen native)."
        ),
    )
    # ── CÂU 1+2: Pruning strategy mới ───────────────────────────
    p.add_argument(
        "--pruning-strategy",
        default="motion",
        choices=["motion", "scene_change", "uniform", "qwen3_base_ratio"],
        help=(
            "Chiến lược prune frame cho video datasets. "
            "'qwen3_base_ratio': lấy base frames từ Qwen3 pipeline rồi prune theo --keep-ratio."
        ),
    )
    p.add_argument(
        "--keep-ratio", type=float, default=0.5,
        help=(
            "Tỷ lệ frame giữ lại khi --pruning-strategy=qwen3_base_ratio. "
            "0.5 = giữ 50%%. Mặc định: 0.5."
        ),
    )
    p.add_argument(
        "--ratio-sub-strategy", default="motion",
        choices=["motion", "scene_change", "uniform"],
        help="Strategy nội bộ dùng trong prune_by_ratio (sau khi có Qwen3 base frames).",
    )
    p.add_argument(
        "--qwen3-fps", type=float, default=1.0,
        help="FPS Qwen3 dùng khi decode video trong get_qwen3_base_frames. Mặc định: 1.0.",
    )
    # ── CÂU 2: Caption similarity ────────────────────────────────
    p.add_argument(
        "--caption-sim", action="store_true",
        help=(
            "Bật chế độ đo caption similarity (2-pass). "
            "Pass A: full base frames. Pass B: pruned frames. "
            "Chỉ áp dụng khi --pruning-strategy=qwen3_base_ratio và --input-model=frames."
        ),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--target-model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--draft-model", default="z-lab/Qwen3-4B-DFlash-b16")
    p.add_argument("--draft-checkpoint", default=None)
    p.add_argument("--num-frames", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--output-dir", default="results/")
    p.add_argument(
        "--baseline-for-speedup", default="Qwen3-VL-4B-Instruct",
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────

def _resolve_num_samples(args, ds_name: str) -> int:
    if args.num_videos is not None:
        return args.num_videos * 2
    return args.num_samples


def load_datasets(args) -> dict:
    datasets = {}

    for name in args.datasets:
        cls            = DATASET_REGISTRY[name]
        num_samples    = _resolve_num_samples(args, name)
        effective_mode = args.input_mode

        if name in _FRAMES_ONLY_DATASETS and args.input_mode == "video":
            print(f"[runner] ⚠️  '{name}' không hỗ trợ input_mode='video' → fallback về frames")
            effective_mode = "frames"

        print(f"\n[runner] Loading dataset: {name}  (samples={num_samples}, input_mode={effective_mode})")

        kwargs = {"num_samples": num_samples, "input_mode": effective_mode}

        # Truyền pruning params vào video datasets (fix 1: bao gồm videomme_short)
        if name in _VIDEO_DATASETS:
            kwargs["pruning_strategy"] = args.pruning_strategy
            if args.pruning_strategy == "qwen3_base_ratio":
                kwargs["keep_ratio"]         = args.keep_ratio
                kwargs["ratio_sub_strategy"] = args.ratio_sub_strategy
                kwargs["qwen3_fps"]          = args.qwen3_fps
                print(
                    f"  → Pruning: qwen3_base_ratio "
                    f"(keep={args.keep_ratio*100:.0f}%, sub={args.ratio_sub_strategy}, fps={args.qwen3_fps})"
                )

        try:
            datasets[name] = cls(**kwargs)
        except TypeError:
            # Dataset không nhận một số kwargs (vd: pruning_strategy) — thử lại với params cơ bản
            # Giữ lại input_mode để không fallback về MODE_FRAMES ngoài ý muốn
            safe_kwargs = {k: v for k, v in kwargs.items()
                          if k in ("num_samples", "input_mode", "num_frames", "buckets", "download_video")}
            try:
                datasets[name] = cls(**safe_kwargs)
            except TypeError:
                datasets[name] = cls(num_samples=num_samples)

        s = datasets[name].summary()
        print(f"  → {s['total']} samples | buckets: {s['buckets']} | mode: {s.get('input_mode', 'frames')}")

    return datasets


def load_models(args) -> dict:
    models = {}
    names = args.models
    real = args.real
    device = args.device

    if "spec" in names:
        print(f"\n[runner] Loading SpecVLM (mock={not real})")
        models["SpecVLM"] = SpecVLMAdapter(
            device=device,
            target_model_path=args.target_model,
            draft_model_path=args.draft_model,
            draft_checkpoint=args.draft_checkpoint,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            num_frames=args.num_frames,
        )

    if real:
        for name in names:
            if name == "spec":
                continue
            cls = BASELINE_REGISTRY.get(name)
            if cls is None:
                print(f"[runner] Unknown model: {name}, skipping.")
                continue
            print(f"\n[runner] Loading baseline: {name} ({cls.MODEL_NAME}, {cls.MODEL_PARAMS})")
            adapter = cls(
                device=device,
                dtype=args.dtype,
                max_new_tokens=args.max_new_tokens,
                num_frames=args.num_frames,
            )
            models[adapter.MODEL_NAME] = adapter
    else:
        from models.spec_vlm import _MockSpecModel
        from models.base import BaseModelAdapter

        class _MockBaseline(BaseModelAdapter):
            def __init__(self, name, params, modality):
                self.MODEL_NAME   = name
                self.MODEL_PARAMS = params
                self.MODALITY     = modality
                self._mock        = _MockSpecModel()

            def generate(self, sample):
                r = self._mock.generate(sample)
                r["num_tokens"]        = max(1, r["num_tokens"] - 20)
                r["acceptance_length"] = None
                return r

        for name in names:
            if name == "spec":
                continue
            cls = BASELINE_REGISTRY.get(name)
            if cls is None:
                continue
            models[cls.MODEL_NAME] = _MockBaseline(cls.MODEL_NAME, cls.MODEL_PARAMS, cls.MODALITY)

    return models


# ─────────────────────────────────────────────────────────────────
# LLAMA FALLBACK
# ─────────────────────────────────────────────────────────────────

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

    print(f"    [fallback] Llama: decode frames từ {os.path.basename(video_path)}")
    import cv2
    import numpy as np
    from PIL import Image as PILImage

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


# ─────────────────────────────────────────────────────────────────
# CÂU 2: 2-PASS CAPTION SIMILARITY
# ─────────────────────────────────────────────────────────────────

def _run_caption_sim_pass(model, sample: dict) -> Optional[str]:
    """Chạy model trên 1 sample và trả về text output. None nếu lỗi."""
    try:
        out = model.generate(sample)
        return out.get("text", "")
    except Exception as e:
        print(f"    [caption_sim] generate lỗi: {e}")
        return None


def _build_native_video_sample(sample: dict) -> Optional[dict]:
    """Tạo sample cho pass A: model nhận video trực tiếp."""
    video_path = sample.get("video_path")
    if not video_path:
        return None
    # Sentinel path (.CACHED) là file placeholder, không phải video thật —
    # dùng base_frames thay thế nếu có
    if video_path.endswith(".CACHED"):
        base_frames = sample.get("base_frames")
        if not base_frames:
            return None
        patched = dict(sample)
        patched["input_mode"] = "frames"
        patched["frames"]     = base_frames   # Pass A dùng full base frames
        patched["video_path"] = None
        return patched
    patched = dict(sample)
    patched["input_mode"] = "video"
    patched["frames"] = None
    patched["video_path"] = video_path
    return patched


def _build_pruned_frames_sample(sample: dict) -> Optional[dict]:
    """Tạo sample cho pass B: model nhận frames đã prune."""
    frames = sample.get("frames")
    if not frames:
        return None
    patched = dict(sample)
    patched["input_mode"] = "frames"
    patched["frames"] = frames
    return patched


# ─────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────

def run(args):
    use_caption_sim = (
        args.caption_sim
        and args.pruning_strategy == "qwen3_base_ratio"
    )
    # fix 3: videomme_short giờ nằm trong _VIDEO_DATASETS nên sẽ được tính caption_sim

    if args.caption_sim and not use_caption_sim:
        print(
            "[runner] ⚠️  --caption-sim chỉ áp dụng khi "
            "--pruning-strategy=qwen3_base_ratio. Bỏ qua."
        )

    datasets = load_datasets(args)
    models   = load_models(args)

    print(f"\n{'='*60}")
    print(f"Models          : {list(models.keys())}")
    print(f"Datasets        : {list(datasets.keys())}")
    print(f"Input mode      : {args.input_mode}")
    print(f"Pruning strategy: {args.pruning_strategy}")
    if args.pruning_strategy == "qwen3_base_ratio":
        print(f"Keep ratio      : {args.keep_ratio*100:.0f}%")
    if use_caption_sim:
        print(f"Caption sim     : ENABLED (native video vs pruned frames)")
    print(f"Real model      : {args.real}")
    print(f"{'='*60}")

    all_results = {}

    for ds_name, dataset in datasets.items():
        num_samples = _resolve_num_samples(args, ds_name)
        print(f"\n{'='*60}\nDataset: {ds_name.upper()}  ({len(dataset)} samples)\n{'='*60}")
        all_results[ds_name] = {}

        for model_name, model in models.items():
            info = model.info()
            print(f"\n  [{info['model_name']}] ({info['model_params']}, {info['modality']})")

            tracker = MetricsTracker(
                model_name   = info["model_name"],
                model_params = info["model_params"],
                dataset_name = ds_name,
            )

            skipped = 0
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break

                sample = _maybe_fallback_to_frames(model, sample)

                caption_sim_result = None
                caption_base = None
                if use_caption_sim and ds_name in _VIDEO_DATASETS:
                    print(f"    [caption_sim] sample={sample['id']} video_path={sample.get('video_path') is not None} frames={sample.get('frames') is not None}")
                    video_sample = _build_native_video_sample(sample)
                    print(f"    [caption_sim] passA build_native_video_sample -> {'OK' if video_sample is not None else 'NONE'}")
                    if video_sample is not None:
                        video_sample = _maybe_fallback_to_frames(model, video_sample)
                        caption_base = _run_caption_sim_pass(model, video_sample)
                        print(f"    [caption_sim] passA caption_base_none={caption_base is None} len={len(caption_base) if caption_base else 0}")

                try:
                    pruned_sample = _build_pruned_frames_sample(sample)
                    if pruned_sample is None:
                        print(
                            f"    ⚠️  [warn] sample {sample.get('id')}: frames=None "
                            f"sau khi load — qwen3_base_ratio có thể đã fail. "
                            f"Dùng sample gốc (có thể fallback 32 frames!)"
                        )
                        pruned_sample = dict(sample)
                    pruned_sample["input_mode"] = "frames"

                    t0      = time.perf_counter()
                    out     = model.generate(pruned_sample)
                    elapsed = time.perf_counter() - t0

                    if use_caption_sim and ds_name in _VIDEO_DATASETS and caption_base is not None:
                        caption_pruned = out.get("text", "")
                        print(f"    [caption_sim] passB len={len(caption_pruned)}")
                        caption_sim_result = compute_caption_similarity(caption_base, caption_pruned)
                        print(f"    [caption_sim] result={caption_sim_result}")

                    # ── CÂU 3: accuracy cho VideoMME ──────────────────────
                    predicted_text = out.get("text", "")
                    ground_truth   = sample.get("answer")   # có trong videomme_short
                    answer_options = sample.get("options") or sample.get("choices") or sample.get("candidates")

                    if sample.get("token_bucket") == "exhaustive_caption":
                        print("\n" + "=" * 100)
                        print(f"[RAW-EXHAUSTIVE] sample_id   : {sample.get('id')}")
                        print(f"[RAW-EXHAUSTIVE] bucket      : {sample.get('token_bucket')}")
                        print(f"[RAW-EXHAUSTIVE] question    : {sample.get('question')}")
                        print(f"[RAW-EXHAUSTIVE] options     : {answer_options}")
                        print(f"[RAW-EXHAUSTIVE] gt_answer   : {ground_truth}")
                        print(f"[RAW-EXHAUSTIVE] pred_text   : {predicted_text}")
                        print("=" * 100 + "\n")

                    tracker.record(
                        elapsed_s             = elapsed,
                        num_tokens            = out.get("num_tokens") or model.estimate_tokens(predicted_text),
                        sample_id             = sample["id"],
                        token_bucket          = sample.get("token_bucket", "short_caption"),
                        time_to_first_token_s = out.get("time_to_first_token_s"),
                        acceptance_length     = out.get("acceptance_length"),
                        draft_rounds          = out.get("draft_rounds"),
                        video_bucket          = sample.get("video_bucket"),
                        predicted_text        = predicted_text,
                        ground_truth_answer   = ground_truth,
                        answer_options        = answer_options,
                        caption_sim           = caption_sim_result,
                    )

                except Exception as e:
                    import traceback
                    skipped += 1
                    print(f"    [skip] sample {i} ({sample.get('id', '')}): {e}")
                    traceback.print_exc()
                    continue

                if (i + 1) % 20 == 0:
                    s         = tracker.running_summary()
                    alpha_str = f"  α={s['alpha']}" if s.get("alpha") else ""
                    ttft_str  = f"  TTFT={s['ttft']}s" if s.get("ttft") else ""
                    print(f"    {i+1}/{num_samples}  tps={s['tps']}{ttft_str}{alpha_str}")

            if skipped > 0:
                print(f"    ⚠️  {skipped} samples skipped")

            result = tracker.finalize()
            all_results[ds_name][info["model_name"]] = result
            print_result(result)

    compute_speedups(all_results, baseline_model=args.baseline_for_speedup)
    save_results(all_results, args.output_dir)
    print_table(all_results)
    return all_results


if __name__ == "__main__":
    run(parse_args())
