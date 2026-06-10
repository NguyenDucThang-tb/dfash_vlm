import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from model import DFlashDraftModel, dflash_generate
except ImportError:
    from dflash.model import DFlashDraftModel, dflash_generate


def load_coco_image_paths(
    coco_root: str,
    coco_ann: Optional[str],
    limit: int,
    seed: int,
    from_bottom: bool,
    shuffle: bool,
) -> list[Path]:
    root = Path(coco_root)
    if not root.exists():
        raise FileNotFoundError(f"COCO_ROOT not found: {root}")

    out: list[Path] = []
    ann_path = Path(coco_ann) if coco_ann else None
    if ann_path is not None and ann_path.exists():
        with ann_path.open("r", encoding="utf-8") as f:
            ann = json.load(f)

        files: list[str] = sorted(img["file_name"] for img in ann.get("images", []) if "file_name" in img)
        if shuffle:
            random.Random(seed).shuffle(files)
        elif from_bottom:
            files = list(reversed(files))
        for name in files:
            p = root / name
            if p.exists():
                out.append(p)
            if len(out) >= limit:
                break
    else:
        image_paths = sorted(
            [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        )
        if shuffle:
            random.Random(seed).shuffle(image_paths)
        elif from_bottom:
            image_paths = list(reversed(image_paths))
        out = image_paths[:limit]

    if not out:
        raise RuntimeError("No valid COCO images found for evaluation.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Test draft->target acceptance on image-only prompts.")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--draft-model", type=str, default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--draft-checkpoint", type=str, default=None)
    parser.add_argument("--coco-root", type=str, default="/content/coco/images/train2017")
    parser.add_argument("--coco-ann", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-prompt", type=str, default="Describe this image.")
    parser.add_argument(
        "--from-bottom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Take evaluation images from the end of the sorted COCO file list to avoid train-set overlap.",
    )
    parser.add_argument(
        "--shuffle-images",
        action="store_true",
        help="Shuffle image candidates by seed instead of taking from top/bottom.",
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

    print("[Init] Preparing image samples...")
    image_paths = load_coco_image_paths(
        coco_root=args.coco_root,
        coco_ann=args.coco_ann,
        limit=args.num_samples,
        seed=args.seed,
        from_bottom=args.from_bottom,
        shuffle=args.shuffle_images,
    )
    if args.shuffle_images:
        print(f"[Init] Image selection: shuffled | seed={args.seed}")
    else:
        direction = "bottom" if args.from_bottom else "top"
        print(f"[Init] Image selection: sorted-{direction}")

    acceptance_lengths: list[int] = []
    tpot_list: list[float] = []
    output_tokens_list: list[int] = []

    stop_ids = [processor.tokenizer.eos_token_id] if processor.tokenizer.eos_token_id is not None else None

    print(f"[Run] Evaluating {len(image_paths)} images...")
    tic = time.perf_counter()
    for i, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.image_prompt},
                ],
            }
        ]
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(target.device)
        pixel_values = encoded.get("pixel_values")
        image_grid_thw = encoded.get("image_grid_thw")
        if pixel_values is not None:
            pixel_values = pixel_values.to(target.device)
            image_grid_thw = image_grid_thw.to(target.device)

        stats = dflash_generate(
            draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=stop_ids,
            temperature=args.temperature,
            return_stats=True,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        acceptance_lengths.extend(stats.acceptance_lengths)
        tpot_list.append(float(stats.time_per_output_token))
        output_tokens_list.append(int(stats.num_output_tokens))

        if i % 10 == 0 or i == len(image_paths):
            last_accept = statistics.mean(stats.acceptance_lengths) if stats.acceptance_lengths else float("nan")
            print(
                f"  processed {i:>3}/{len(image_paths)} | "
                f"last_out_tokens={stats.num_output_tokens} | "
                f"last_mean_accept={last_accept:.3f}"
            )

    elapsed = time.perf_counter() - tic
    total_generated = sum(output_tokens_list)
    mean_accept = statistics.mean(acceptance_lengths) if acceptance_lengths else float("nan")
    std_accept = statistics.pstdev(acceptance_lengths) if len(acceptance_lengths) > 1 else 0.0
    mean_tpot = statistics.mean(tpot_list) if tpot_list else float("nan")
    mean_tps = (1.0 / mean_tpot) if mean_tpot > 0 else float("nan")

    print("\n=== Image Acceptance Report ===")
    print(f"target_model: {args.target_model}")
    print(f"draft_model:  {args.draft_model}")
    print(f"checkpoint:   {args.draft_checkpoint if args.draft_checkpoint else '(pretrained)'}")
    print(f"num_images:   {len(image_paths)}")
    print(f"total_gen_tokens: {total_generated}")
    print(f"acceptance_length_mean: {mean_accept:.4f}")
    print(f"acceptance_length_std:  {std_accept:.4f}")
    print(f"mean_time_per_token:    {mean_tpot:.6f} s")
    print(f"mean_tokens_per_sec:    {mean_tps:.2f}")
    print(f"wall_time:              {elapsed:.2f} s")

    block_size = int(getattr(draft, "block_size", 16))
    if acceptance_lengths:
        hist = [acceptance_lengths.count(k) for k in range(1, block_size + 1)]
        total_hist = sum(hist)
        if total_hist > 0:
            pct = [f"{100.0 * c / total_hist:.1f}%" for c in hist]
            print(f"acceptance_histogram_1..{block_size}: {pct}")


if __name__ == "__main__":
    main()
