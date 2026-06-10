import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from model import DFlashDraftModel, dflash_generate
except ImportError:
    from dflash.model import DFlashDraftModel, dflash_generate


def load_video_paths(
    *,
    manifest: Optional[str],
    video_root: Optional[str],
    limit: int,
    seed: int,
    from_bottom: bool,
    shuffle: bool,
) -> list[Path]:
    out: list[Path] = []

    if manifest:
        manifest_path = Path(manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        paths: list[Path] = []
        for row in rows:
            p = Path(str(row.get("video_path", "")))
            if p.exists():
                paths.append(p)
        if shuffle:
            random.Random(seed).shuffle(paths)
        elif from_bottom:
            paths = list(reversed(paths))
        out = paths[:limit]
    elif video_root:
        root = Path(video_root)
        if not root.exists():
            raise FileNotFoundError(f"video_root not found: {root}")
        video_paths = sorted(
            [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}]
        )
        if shuffle:
            random.Random(seed).shuffle(video_paths)
        elif from_bottom:
            video_paths = list(reversed(video_paths))
        out = video_paths[:limit]
    else:
        raise ValueError("either --manifest or --video-root must be provided")

    if not out:
        raise RuntimeError("No valid videos found for evaluation.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Test draft->target acceptance on video-only prompts.")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--draft-model", type=str, default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--draft-checkpoint", type=str, default=None)
    parser.add_argument("--manifest", type=str, default="/content/phaseB_target_answers.jsonl")
    parser.add_argument("--video-root", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-prompt", type=str, default="Describe the main events in this video.")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument(
        "--from-bottom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Take evaluation videos from the end of the sorted manifest/file list to avoid overlap with train rows.",
    )
    parser.add_argument(
        "--shuffle-videos",
        action="store_true",
        help="Shuffle video candidates by seed instead of taking from top/bottom.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("[Init] Loading target...")
    target = Qwen3VLForConditionalGeneration.from_pretrained(
        args.target_model,
        dtype=dtype,
        device_map="cuda",
    ).eval()
    for p in target.parameters():
        p.requires_grad = False

    print("[Init] Loading processor...")
    processor = AutoProcessor.from_pretrained(args.target_model)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"

    print("[Init] Loading draft...")
    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        dtype=dtype,
        trust_remote_code=True,
    ).to("cuda").eval()
    if args.draft_checkpoint:
        state = torch.load(args.draft_checkpoint, map_location="cpu")
        state_dict = state.get("model_state_dict", state)
        incompat = draft.load_state_dict(state_dict, strict=False)
        print(
            f"[Init] Loaded checkpoint: {args.draft_checkpoint} | "
            f"missing={len(getattr(incompat, 'missing_keys', []))}, "
            f"unexpected={len(getattr(incompat, 'unexpected_keys', []))}"
        )

    print("[Init] Preparing video samples...")
    video_paths = load_video_paths(
        manifest=args.manifest,
        video_root=args.video_root,
        limit=args.num_samples,
        seed=args.seed,
        from_bottom=args.from_bottom,
        shuffle=args.shuffle_videos,
    )
    if args.shuffle_videos:
        print(f"[Init] Video selection: shuffled | seed={args.seed}")
    else:
        direction = "bottom" if args.from_bottom else "top"
        print(f"[Init] Video selection: sorted-{direction}")

    acceptance_lengths: list[int] = []
    tpot_list: list[float] = []
    output_tokens_list: list[int] = []
    failures: list[dict[str, Any]] = []

    stop_ids = [processor.tokenizer.eos_token_id] if processor.tokenizer.eos_token_id is not None else None

    print(f"[Run] Evaluating {len(video_paths)} videos...")
    tic = time.perf_counter()
    for i, video_path in enumerate(video_paths, start=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path), "num_frames": args.num_frames},
                    {"type": "text", "text": args.video_prompt},
                ],
            }
        ]
        try:
            encoded = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(target.device)
            pixel_values_videos = encoded.get("pixel_values_videos")
            video_grid_thw = encoded.get("video_grid_thw")
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.to(target.device)
                video_grid_thw = video_grid_thw.to(target.device)

            stats = dflash_generate(
                draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop_ids,
                temperature=args.temperature,
                return_stats=True,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
        except Exception as exc:
            failures.append({"video_path": str(video_path), "error": str(exc)})
            print(f"[Warn] failed {video_path.name}: {exc}")
            torch.cuda.empty_cache()
            continue

        acceptance_lengths.extend(stats.acceptance_lengths)
        tpot_list.append(float(stats.time_per_output_token))
        output_tokens_list.append(int(stats.num_output_tokens))

        if i % 10 == 0 or i == len(video_paths):
            last_accept = statistics.mean(stats.acceptance_lengths) if stats.acceptance_lengths else float("nan")
            print(
                f"  processed {i:>3}/{len(video_paths)} | "
                f"last_out_tokens={stats.num_output_tokens} | "
                f"last_mean_accept={last_accept:.3f}"
            )

    elapsed = time.perf_counter() - tic
    total_generated = sum(output_tokens_list)
    mean_accept = statistics.mean(acceptance_lengths) if acceptance_lengths else float("nan")
    std_accept = statistics.pstdev(acceptance_lengths) if len(acceptance_lengths) > 1 else 0.0
    mean_tpot = statistics.mean(tpot_list) if tpot_list else float("nan")
    mean_tps = (1.0 / mean_tpot) if mean_tpot > 0 else float("nan")

    print("\n=== Video Acceptance Report ===")
    print(f"target_model: {args.target_model}")
    print(f"draft_model:  {args.draft_model}")
    print(f"checkpoint:   {args.draft_checkpoint if args.draft_checkpoint else '(pretrained)'}")
    print(f"num_videos:   {len(video_paths)}")
    print(f"successful_videos: {len(output_tokens_list)}")
    print(f"failed_videos:     {len(failures)}")
    print(f"total_gen_tokens: {total_generated}")
    print(f"acceptance_length_mean: {mean_accept:.4f}")
    print(f"acceptance_length_std:  {std_accept:.4f}")
    print(f"mean_time_per_token:    {mean_tpot:.6f} s")
    print(f"mean_tokens_per_sec:    {mean_tps:.2f}")
    print(f"wall_time:              {elapsed:.2f} s")

    block_size = int(getattr(draft, "block_size", 16))
    hist_pct: list[str] = []
    if acceptance_lengths:
        hist = [acceptance_lengths.count(k) for k in range(1, block_size + 1)]
        total_hist = sum(hist)
        if total_hist > 0:
            hist_pct = [f"{100.0 * c / total_hist:.1f}%" for c in hist]
            print(f"acceptance_histogram_1..{block_size}: {hist_pct}")

    if failures:
        print(f"[Warn] Example failure: {failures[0]['video_path']} | {failures[0]['error']}")

    if args.save_json:
        payload = {
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "checkpoint": args.draft_checkpoint if args.draft_checkpoint else "(pretrained)",
            "num_videos": len(video_paths),
            "successful_videos": len(output_tokens_list),
            "failed_videos": len(failures),
            "total_gen_tokens": total_generated,
            "acceptance_length_mean": mean_accept,
            "acceptance_length_std": std_accept,
            "mean_time_per_token": mean_tpot,
            "mean_tokens_per_sec": mean_tps,
            "wall_time": elapsed,
            "acceptance_histogram_1_to_block_size": hist_pct,
            "failures": failures,
            "video_paths": [str(p) for p in video_paths],
            "video_prompt": args.video_prompt,
            "num_frames": args.num_frames,
            "max_new_tokens": args.max_new_tokens,
        }
        out_path = Path(args.save_json)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
