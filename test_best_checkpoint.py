import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from datasets import get_dataset_config_names, load_dataset
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from model import DFlashDraftModel, dflash_generate
except ImportError:
    from dflash.model import DFlashDraftModel, dflash_generate


def extract_turn_pair(sample: dict[str, Any]) -> Optional[Tuple[str, str]]:
    def _to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    chunks.append(item.strip())
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
            return "\n".join(chunks).strip()
        return ""

    messages = sample.get("messages")
    if isinstance(messages, list):
        user_text: Optional[str] = None
        assistant_text: Optional[str] = None
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower()
            text = _to_text(msg.get("content"))
            if not text:
                continue
            if role == "user" and user_text is None:
                user_text = text
            elif role == "assistant" and user_text is not None:
                assistant_text = text
                break
        if user_text and assistant_text:
            return user_text, assistant_text

    prompt = sample.get("prompt") or sample.get("instruction") or sample.get("question")
    response = sample.get("response") or sample.get("answer") or sample.get("output")
    if isinstance(prompt, str) and isinstance(response, str):
        prompt = prompt.strip()
        response = response.strip()
        if prompt and response:
            return prompt, response
    return None


def load_text_dataset(dataset_name: str, split: str):
    try:
        return load_dataset(dataset_name, split=split)
    except Exception:
        cfgs = get_dataset_config_names(dataset_name)
        for cfg in cfgs:
            try:
                return load_dataset(dataset_name, cfg, split=split)
            except Exception:
                continue
        raise RuntimeError(f"Unable to load dataset: {dataset_name} split={split}")


def load_text_prompts(dataset_name: str, split: str, limit: int, seed: int) -> list[str]:
    ds = load_text_dataset(dataset_name, split)
    if hasattr(ds, "shuffle"):
        ds = ds.shuffle(seed=seed)
    prompts: list[str] = []
    for sample in ds:
        turns = extract_turn_pair(sample)
        if turns is None:
            continue
        user_turn, _ = turns
        user_turn = user_turn.strip()
        if user_turn:
            prompts.append(user_turn)
        if len(prompts) >= limit:
            break
    if not prompts:
        raise RuntimeError("No valid text prompts found.")
    return prompts


def load_coco_image_paths(coco_root: str, coco_ann: str, limit: int, seed: int) -> list[Path]:
    root = Path(coco_root)
    ann_path = Path(coco_ann)
    if not root.exists():
        raise FileNotFoundError(f"COCO_ROOT not found: {root}")
    if not ann_path.exists():
        raise FileNotFoundError(f"COCO_ANN not found: {ann_path}")

    with ann_path.open("r", encoding="utf-8") as f:
        ann = json.load(f)

    names = [img["file_name"] for img in ann.get("images", []) if "file_name" in img]
    random.Random(seed).shuffle(names)

    out: list[Path] = []
    for name in names:
        p = root / name
        if p.exists():
            out.append(p)
        if len(out) >= limit:
            break
    if not out:
        raise RuntimeError("No valid COCO image paths found.")
    return out


@torch.no_grad()
def eval_text(
    draft: DFlashDraftModel,
    target: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    prompts: list[str],
    max_input_len: int,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, float]:
    stop_ids = [processor.tokenizer.eos_token_id] if processor.tokenizer.eos_token_id is not None else None
    acceptance_lengths: list[int] = []
    tpot_list: list[float] = []
    out_tokens_total = 0

    tic = time.perf_counter()
    for i, prompt in enumerate(prompts, start=1):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"][:, :max_input_len].to(target.device)
        stats = dflash_generate(
            draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_ids,
            temperature=temperature,
            return_stats=True,
        )
        acceptance_lengths.extend(stats.acceptance_lengths)
        tpot_list.append(float(stats.time_per_output_token))
        out_tokens_total += int(stats.num_output_tokens)
        if i % 10 == 0 or i == len(prompts):
            m = statistics.mean(stats.acceptance_lengths) if stats.acceptance_lengths else float("nan")
            print(f"  [text] {i:>3}/{len(prompts)} | last_mean_accept={m:.3f}")
    elapsed = time.perf_counter() - tic

    mean_acc = statistics.mean(acceptance_lengths) if acceptance_lengths else float("nan")
    std_acc = statistics.pstdev(acceptance_lengths) if len(acceptance_lengths) > 1 else 0.0
    mean_tpot = statistics.mean(tpot_list) if tpot_list else float("nan")
    tps = (1.0 / mean_tpot) if mean_tpot > 0 else float("nan")
    return {
        "num_samples": float(len(prompts)),
        "total_gen_tokens": float(out_tokens_total),
        "acceptance_length_mean": float(mean_acc),
        "acceptance_length_std": float(std_acc),
        "mean_time_per_token": float(mean_tpot),
        "mean_tokens_per_sec": float(tps),
        "wall_time_sec": float(elapsed),
    }


@torch.no_grad()
def eval_image(
    draft: DFlashDraftModel,
    target: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image_paths: list[Path],
    image_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, float]:
    stop_ids = [processor.tokenizer.eos_token_id] if processor.tokenizer.eos_token_id is not None else None
    acceptance_lengths: list[int] = []
    tpot_list: list[float] = []
    out_tokens_total = 0

    tic = time.perf_counter()
    for i, path in enumerate(image_paths, start=1):
        with Image.open(path) as im:
            image = im.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": image_prompt},
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
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_ids,
            temperature=temperature,
            return_stats=True,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        acceptance_lengths.extend(stats.acceptance_lengths)
        tpot_list.append(float(stats.time_per_output_token))
        out_tokens_total += int(stats.num_output_tokens)
        if i % 10 == 0 or i == len(image_paths):
            m = statistics.mean(stats.acceptance_lengths) if stats.acceptance_lengths else float("nan")
            print(f"  [image] {i:>3}/{len(image_paths)} | last_mean_accept={m:.3f}")
    elapsed = time.perf_counter() - tic

    mean_acc = statistics.mean(acceptance_lengths) if acceptance_lengths else float("nan")
    std_acc = statistics.pstdev(acceptance_lengths) if len(acceptance_lengths) > 1 else 0.0
    mean_tpot = statistics.mean(tpot_list) if tpot_list else float("nan")
    tps = (1.0 / mean_tpot) if mean_tpot > 0 else float("nan")
    return {
        "num_samples": float(len(image_paths)),
        "total_gen_tokens": float(out_tokens_total),
        "acceptance_length_mean": float(mean_acc),
        "acceptance_length_std": float(std_acc),
        "mean_time_per_token": float(mean_tpot),
        "mean_tokens_per_sec": float(tps),
        "wall_time_sec": float(elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test current best checkpoint on text+image acceptance.")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--draft-model", type=str, default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--draft-checkpoint", type=str, default="/content/drive/MyDrive/dflash_phaseA/best_checkpoint.pt")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-dataset", type=str, default="HuggingFaceTB/smoltalk")
    parser.add_argument("--text-split", type=str, default="test")
    parser.add_argument("--num-text", type=int, default=15)
    parser.add_argument("--num-image", type=int, default=15)
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--coco-root", type=str, default="/content/coco/images/train2017")
    parser.add_argument("--coco-ann", type=str, default="/content/coco/annotations/captions_train2017.json")
    parser.add_argument("--image-prompt", type=str, default="Describe this image.")
    parser.add_argument("--save-json", type=str, default=None)
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
    state = torch.load(args.draft_checkpoint, map_location="cpu")
    state_dict = state.get("model_state_dict", state)
    incompat = draft.load_state_dict(state_dict, strict=False)
    print(
        f"[Init] Loaded checkpoint: {args.draft_checkpoint} | "
        f"missing={len(getattr(incompat, 'missing_keys', []))}, "
        f"unexpected={len(getattr(incompat, 'unexpected_keys', []))}"
    )

    print("[Init] Loading eval text prompts...")
    text_prompts = load_text_prompts(args.text_dataset, args.text_split, args.num_text, args.seed)
    print("[Init] Loading eval image paths...")
    image_paths = load_coco_image_paths(args.coco_root, args.coco_ann, args.num_image, args.seed)

    print(f"[Run] text={len(text_prompts)} | image={len(image_paths)}")
    text_report = eval_text(
        draft=draft,
        target=target,
        processor=processor,
        prompts=text_prompts,
        max_input_len=args.max_input_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    image_report = eval_image(
        draft=draft,
        target=target,
        processor=processor,
        image_paths=image_paths,
        image_prompt=args.image_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print("\n=== Best Checkpoint Report ===")
    print(f"target_model: {args.target_model}")
    print(f"draft_model:  {args.draft_model}")
    print(f"checkpoint:   {args.draft_checkpoint}")
    print(
        f"text_acc_mean={text_report['acceptance_length_mean']:.4f} | "
        f"text_tps={text_report['mean_tokens_per_sec']:.2f}"
    )
    print(
        f"image_acc_mean={image_report['acceptance_length_mean']:.4f} | "
        f"image_tps={image_report['mean_tokens_per_sec']:.2f}"
    )

    payload = {
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "checkpoint": args.draft_checkpoint,
        "num_text": len(text_prompts),
        "num_image": len(image_paths),
        "text_report": text_report,
        "image_report": image_report,
    }
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"[Saved] {out}")


if __name__ == "__main__":
    main()
