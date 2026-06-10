import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


TARGET_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_COCO_ROOT = "/content/mscoco/images/train2017"
DEFAULT_OUTPUT = "/content/phaseA_target_answers.jsonl"
DEFAULT_DRIVE_OUTPUT = "/content/drive/MyDrive/dflash_phaseA_20k_fullctx_answer64_mrope/phaseA_target_answers.jsonl"
DEFAULT_PROMPT = "Describe this image."


def iter_image_paths(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def load_done_paths(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_path = row.get("image_path")
            if isinstance(image_path, str):
                done.add(image_path)
    return done


def build_messages(image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


@torch.no_grad()
def generate_answer_batch(
    *,
    target: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image_paths: List[Path],
    prompt: str,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    images: List[Image.Image] = []
    for image_path in image_paths:
        with Image.open(image_path) as im:
            images.append(im.convert("RGB").resize((448, 448)))

    texts = [
        processor.apply_chat_template(
            build_messages(image, prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
        for image in images
    ]
    encoded = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(target.device)

    prompt_lens = encoded["attention_mask"].sum(dim=1).to(torch.long)
    generated = target.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
    )
    rows: List[Dict[str, Any]] = []
    for i, image_path in enumerate(image_paths):
        answer_ids = generated[i, int(prompt_lens[i].item()):]
        answer = processor.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        rows.append(
            {
                "image_path": str(image_path),
                "file_name": image_path.name,
                "prompt": prompt,
                "answer": answer,
                "target_model": TARGET_MODEL_ID,
                "max_new_tokens": int(max_new_tokens),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate target answers for Phase A image-only training.")
    parser.add_argument("--coco-root", type=str, default=DEFAULT_COCO_ROOT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--drive-output", type=str, default=DEFAULT_DRIVE_OUTPUT)
    parser.add_argument("--no-drive-copy", action="store_true")
    parser.add_argument("--target-model", type=str, default=TARGET_MODEL_ID)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--num-images", type=int, default=10000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    root = Path(args.coco_root)
    if not root.exists():
        raise FileNotFoundError(f"COCO_ROOT not found: {root}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = iter_image_paths(root)
    rnd = random.Random(args.seed)
    rnd.shuffle(image_paths)
    if args.num_images > 0:
        image_paths = image_paths[: args.num_images]

    done = load_done_paths(output_path) if args.resume else set()
    mode = "a" if args.resume else "w"

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print("[Init] Loading target model...")
    target = Qwen3VLForConditionalGeneration.from_pretrained(
        args.target_model,
        dtype=dtype,
        device_map="cuda",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.target_model)

    total = 0
    failed: list[str] = []
    start = time.perf_counter()
    pending_paths = [p for p in image_paths if str(p) not in done]
    batch_size = max(1, int(args.batch_size))
    print(f"[Init] images={len(image_paths)} pending={len(pending_paths)} done={len(done)} batch_size={batch_size}")
    with output_path.open(mode, encoding="utf-8") as f:
        for batch_start in tqdm(range(0, len(pending_paths), batch_size), desc="Generating target answers"):
            batch_paths = pending_paths[batch_start : batch_start + batch_size]
            try:
                rows = generate_answer_batch(
                    target=target,
                    processor=processor,
                    image_paths=batch_paths,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                print(f"[Warn] batch {batch_start} failed: {exc}. Falling back to single-image generation.")
                torch.cuda.empty_cache()
                rows = []
                for image_path in batch_paths:
                    try:
                        rows.extend(
                            generate_answer_batch(
                                target=target,
                                processor=processor,
                                image_paths=[image_path],
                                prompt=args.prompt,
                                max_new_tokens=args.max_new_tokens,
                            )
                        )
                    except Exception as single_exc:
                        print(f"[Warn] failed {image_path}: {single_exc}")
                        failed.append(str(image_path))
                torch.cuda.empty_cache()

            for row in rows:
                if not row["answer"]:
                    failed.append(str(row["image_path"]))
                    continue
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                total += 1
            f.flush()
    elapsed = time.perf_counter() - start
    print(f"[Done] wrote={total} skipped_existing={len(done)} failed={len(failed)} output={output_path}")
    print(f"[Done] elapsed_sec={elapsed:.1f}")
    if not args.no_drive_copy and args.drive_output:
        drive_output = Path(args.drive_output)
        drive_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, drive_output)
        print(f"[Done] copied_to_drive={drive_output}")
    if failed:
        print("[Done] failed examples:")
        for path in failed[:20]:
            print(path)


if __name__ == "__main__":
    main()
