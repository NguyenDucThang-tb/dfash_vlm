import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from datasets import get_dataset_config_names, load_dataset
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
        config_names = get_dataset_config_names(dataset_name)
        for cfg in config_names:
            try:
                return load_dataset(dataset_name, cfg, split=split)
            except Exception:
                continue
        raise RuntimeError(f"Unable to load dataset: {dataset_name}, split={split}")


def evaluate_variant(
    *,
    name: str,
    draft: DFlashDraftModel,
    target: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    prompts: list[str],
    max_input_len: int,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    print(f"[Run] Evaluating variant={name} on {len(prompts)} prompts...")
    acceptance_lengths: list[int] = []
    tpot_list: list[float] = []
    output_tokens_list: list[int] = []

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
            stop_token_ids=[processor.tokenizer.eos_token_id] if processor.tokenizer.eos_token_id is not None else None,
            temperature=temperature,
            return_stats=True,
        )
        acceptance_lengths.extend(stats.acceptance_lengths)
        tpot_list.append(float(stats.time_per_output_token))
        output_tokens_list.append(int(stats.num_output_tokens))

        if i % 10 == 0 or i == len(prompts):
            print(
                f"  [{name}] processed {i:>3}/{len(prompts)} | "
                f"last_out_tokens={stats.num_output_tokens} | "
                f"last_mean_accept={statistics.mean(stats.acceptance_lengths):.3f}"
            )

    elapsed = time.perf_counter() - tic
    total_generated = sum(output_tokens_list)
    mean_accept = statistics.mean(acceptance_lengths) if acceptance_lengths else float("nan")
    std_accept = statistics.pstdev(acceptance_lengths) if len(acceptance_lengths) > 1 else 0.0
    mean_tpot = statistics.mean(tpot_list) if tpot_list else float("nan")
    mean_tps = (1.0 / mean_tpot) if mean_tpot > 0 else float("nan")

    block_size = int(getattr(draft, "block_size", 16))
    hist_pct: Optional[list[str]] = None
    if acceptance_lengths:
        hist = [acceptance_lengths.count(k) for k in range(1, block_size + 1)]
        total_hist = sum(hist)
        if total_hist > 0:
            hist_pct = [f"{100.0 * c / total_hist:.1f}%" for c in hist]

    report = {
        "variant": name,
        "num_prompts": len(prompts),
        "total_gen_tokens": total_generated,
        "acceptance_length_mean": float(mean_accept),
        "acceptance_length_std": float(std_accept),
        "mean_time_per_token": float(mean_tpot),
        "mean_tokens_per_sec": float(mean_tps),
        "wall_time_sec": float(elapsed),
        "block_size": block_size,
        "acceptance_histogram_1_to_block_size": hist_pct,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Test draft->target acceptance (text-only, no training).")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--draft-model", type=str, default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument(
        "--draft-checkpoint",
        action="append",
        default=[],
        help="Path to training checkpoint .pt (can be passed multiple times for comparison).",
    )
    parser.add_argument(
        "--draft-label",
        action="append",
        default=[],
        help="Label for each --draft-checkpoint (same order).",
    )
    parser.add_argument("--dataset", type=str, default="HuggingFaceTB/smoltalk")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", type=str, default=None, help="Optional path to save compare report as JSON.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

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

    print("[Init] Loading dataset...")
    dataset = load_text_dataset(args.dataset, args.split)
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=args.seed)
    if len(dataset) == 0:
        raise RuntimeError("Dataset split is empty.")

    prompts: list[str] = []
    for sample in dataset:
        turns = extract_turn_pair(sample)
        if turns is None:
            continue
        user_turn, _ = turns
        if user_turn.strip():
            prompts.append(user_turn.strip())
        if len(prompts) >= args.num_samples:
            break
    if not prompts:
        raise RuntimeError("No valid text-only prompts extracted from dataset.")

    ckpts = args.draft_checkpoint or []
    labels = args.draft_label or []
    if labels and len(labels) != len(ckpts):
        raise ValueError("If --draft-label is provided, number of labels must match --draft-checkpoint count.")

    variants: list[tuple[str, Optional[str]]] = []
    if not ckpts:
        variants.append(("pretrained", None))
    else:
        for i, ckpt in enumerate(ckpts):
            label = labels[i] if i < len(labels) else Path(ckpt).stem
            variants.append((label, ckpt))

    all_reports: list[dict[str, Any]] = []
    for label, ckpt_path in variants:
        print("[Init] Loading draft...")
        draft = DFlashDraftModel.from_pretrained(
            args.draft_model,
            dtype=dtype,
            trust_remote_code=True,
        ).to("cuda").eval()

        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu")
            state_dict = state.get("model_state_dict", state)
            incompat = draft.load_state_dict(state_dict, strict=False)
            missing = len(getattr(incompat, "missing_keys", []))
            unexpected = len(getattr(incompat, "unexpected_keys", []))
            print(
                f"[Init] Loaded checkpoint for variant={label}: {ckpt_path} | "
                f"missing_keys={missing}, unexpected_keys={unexpected}"
            )

        report = evaluate_variant(
            name=label,
            draft=draft,
            target=target,
            processor=processor,
            prompts=prompts,
            max_input_len=args.max_input_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        all_reports.append(report)

        print("\n=== Acceptance Report ===")
        print(f"variant:      {label}")
        print(f"target_model: {args.target_model}")
        print(f"draft_model:  {args.draft_model}")
        if ckpt_path is not None:
            print(f"checkpoint:   {ckpt_path}")
        print(f"num_prompts:  {report['num_prompts']}")
        print(f"total_gen_tokens: {report['total_gen_tokens']}")
        print(f"acceptance_length_mean: {report['acceptance_length_mean']:.4f}")
        print(f"acceptance_length_std:  {report['acceptance_length_std']:.4f}")
        print(f"mean_time_per_token:    {report['mean_time_per_token']:.6f} s")
        print(f"mean_tokens_per_sec:    {report['mean_tokens_per_sec']:.2f}")
        print(f"wall_time:              {report['wall_time_sec']:.2f} s")
        hist_pct = report.get("acceptance_histogram_1_to_block_size")
        if hist_pct:
            print(f"acceptance_histogram_1..{report['block_size']}: {hist_pct}")

        del draft
        torch.cuda.empty_cache()

    if len(all_reports) > 1:
        print("\n=== Compare Summary ===")
        ranked = sorted(all_reports, key=lambda r: r["acceptance_length_mean"], reverse=True)
        for i, rep in enumerate(ranked, start=1):
            print(
                f"{i}. {rep['variant']} | "
                f"acc_len={rep['acceptance_length_mean']:.4f} | "
                f"tps={rep['mean_tokens_per_sec']:.2f}"
            )
        best = ranked[0]
        print(f"[Best] {best['variant']} | acc_len={best['acceptance_length_mean']:.4f}")

    if args.save_json:
        payload = {
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "dataset": args.dataset,
            "split": args.split,
            "seed": args.seed,
            "num_samples": len(prompts),
            "max_input_len": args.max_input_len,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "variants": all_reports,
        }
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
