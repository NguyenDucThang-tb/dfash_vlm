import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


TARGET_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_RAW_MANIFEST = "/content/phaseB_raw_videos.jsonl"
DEFAULT_OUTPUT = "/content/phaseB_target_answers.jsonl"
DEFAULT_PROMPT = "Describe the main events in this video."
DEFAULT_BAD_OUTPUT = "/content/phaseB_bad_videos.jsonl"


def load_raw_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"raw manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            video_path = Path(str(row.get("video_path", "")))
            if video_path.exists():
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No usable videos found in raw manifest: {path}")
    return rows


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            video_id = row.get("video_id")
            answer = str(row.get("answer", "")).strip()
            if video_id and answer:
                done.add(str(video_id))
    return done


def build_messages(video_path: str, prompt: str, num_frames: int) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "num_frames": int(num_frames)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def encode_batch(
    *,
    processor: AutoProcessor,
    rows: List[Dict[str, Any]],
    prompt: str,
    num_frames: int,
) -> Dict[str, torch.Tensor]:
    messages_list = [build_messages(str(row["video_path"]), prompt, num_frames) for row in rows]
    texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]
    video_paths = [str(row["video_path"]) for row in rows]
    try:
        return processor(
            text=texts,
            videos=video_paths,
            return_tensors="pt",
            padding=True,
        )
    except Exception:
        if len(rows) != 1:
            raise
        return processor.apply_chat_template(
            messages_list[0],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )


@torch.no_grad()
def generate_answer_batch(
    *,
    target: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    rows: List[Dict[str, Any]],
    prompt: str,
    num_frames: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    encoded = encode_batch(processor=processor, rows=rows, prompt=prompt, num_frames=num_frames)
    encoded = encoded.to(target.device)
    prompt_lens = encoded["attention_mask"].sum(dim=1).to(torch.long)
    generated = target.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
    )

    out_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        answer_ids = generated[i, int(prompt_lens[i].item()):]
        answer = processor.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        payload = {
            "video_id": str(row.get("video_id", "")),
            "video_path": str(row["video_path"]),
            "prompt": prompt,
            "answer": answer,
            "target_model": TARGET_MODEL_ID,
            "max_new_tokens": int(max_new_tokens),
            "num_frames": int(num_frames),
            "duration": row.get("duration"),
            "source_url": row.get("source_url"),
            "source_caption": row.get("source_caption"),
        }
        out_rows.append(payload)
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen3-VL target answers for Phase B video training.")
    parser.add_argument("--raw-manifest", type=str, default=DEFAULT_RAW_MANIFEST)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-model", type=str, default=TARGET_MODEL_ID)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--bad-output", type=str, default=DEFAULT_BAD_OUTPUT)
    parser.add_argument("--num-videos", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    raw_rows = load_raw_rows(Path(args.raw_manifest))
    if args.num_videos > 0:
        raw_rows = raw_rows[: args.num_videos]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bad_output_path = Path(args.bad_output)
    bad_output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output_path) if args.resume else set()
    pending_rows = [row for row in raw_rows if str(row.get("video_id", "")) not in done_ids]

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print("[Init] Loading target model...")
    target = Qwen3VLForConditionalGeneration.from_pretrained(
        args.target_model,
        dtype=dtype,
        device_map="cuda",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.target_model)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"

    mode = "a" if args.resume else "w"
    batch_size = max(1, int(args.batch_size))
    total = 0
    failed: List[str] = []
    start = time.perf_counter()
    print(
        f"[Init] raw={len(raw_rows)} pending={len(pending_rows)} done={len(done_ids)} "
        f"num_frames={args.num_frames} max_new_tokens={args.max_new_tokens} output={output_path}"
    )
    with output_path.open(mode, encoding="utf-8") as f, bad_output_path.open(mode, encoding="utf-8") as bad_f:
        for batch_start in tqdm(range(0, len(pending_rows), batch_size), desc="Generating video answers"):
            batch_rows = pending_rows[batch_start : batch_start + batch_size]
            try:
                rows = generate_answer_batch(
                    target=target,
                    processor=processor,
                    rows=batch_rows,
                    prompt=args.prompt,
                    num_frames=args.num_frames,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                print(f"[Warn] batch {batch_start} failed: {exc}. Falling back to single-video generation.")
                torch.cuda.empty_cache()
                rows = []
                for row in batch_rows:
                    try:
                        rows.extend(
                            generate_answer_batch(
                                target=target,
                                processor=processor,
                                rows=[row],
                                prompt=args.prompt,
                                num_frames=args.num_frames,
                                max_new_tokens=args.max_new_tokens,
                            )
                        )
                    except Exception as single_exc:
                        video_id = str(row.get("video_id", row.get("video_path", "")))
                        print(f"[Warn] failed {video_id}: {single_exc}")
                        failed.append(video_id)
                        bad_f.write(json.dumps({
                            "video_id": video_id,
                            "video_path": str(row.get("video_path", "")),
                            "error": str(single_exc),
                        }, ensure_ascii=True) + "\n")
                torch.cuda.empty_cache()

            for row in rows:
                if not str(row.get("answer", "")).strip():
                    video_id = str(row.get("video_id", row.get("video_path", "")))
                    failed.append(video_id)
                    bad_f.write(json.dumps({
                        "video_id": video_id,
                        "video_path": str(row.get("video_path", "")),
                        "error": "empty_answer",
                    }, ensure_ascii=True) + "\n")
                    continue
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                total += 1
            f.flush()
            bad_f.flush()

    elapsed = time.perf_counter() - start
    print(
        f"[Done] wrote={total} failed={len(failed)} output={output_path} "
        f"bad_output={bad_output_path} elapsed_sec={elapsed:.1f}"
    )
    if failed:
        print("[Done] failed examples:")
        for item in failed[:20]:
            print(item)


if __name__ == "__main__":
    main()
