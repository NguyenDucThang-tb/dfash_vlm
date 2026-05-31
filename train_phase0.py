import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import get_dataset_config_names, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration, get_scheduler

try:
    from model import DFlashDraftModel, _get_embed_tokens, _get_lm_head, dflash_generate, extract_context_feature
except ImportError:
    from dflash import DFlashDraftModel, _get_embed_tokens, _get_lm_head, extract_context_feature
    from dflash.model import dflash_generate


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 3072
BATCH_SIZE = 2
ACCUMULATION_STEPS = 4
MAX_STEPS = 5000
WARMUP_STEPS = 200  # ~0.04 of MAX_STEPS(5000)
LR = 2e-4
LR_MIN = 1e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LOG_EVERY = 50
SAVE_EVERY = 500
EARLY_STOP_PATIENCE = 3
EARLY_STOP_ACC_DELTA = 1e-4
CHECKPOINT_ROOT = "/content/drive/MyDrive/dflash_phase0_paper_v2"
DATASET_NAME = "HuggingFaceTB/smoltalk"
DRAFT_CONFIG_PATH = "./config.json"
TARGET_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
WARM_START_MODEL_ID = "z-lab/Qwen3-4B-DFlash-b16"
USE_WARM_START = True
# "fc_only" | "attention" | "full" | "draft_layers_only"
UNFREEZE_POLICY = "draft_layers_only"

SEED = 42
IGNORE_INDEX = -100
DEVICE = "cuda"
HIDDEN_ALIGN_EVERY = 100
GPU_LOG_EVERY = 100
EVAL_NUM_SAMPLES = 50
EVAL_MAX_NEW_TOKENS = 64
EVAL_LOSS_NUM_BATCHES = 50
BLOCK_CONTEXT_LEN = 64
LOSS_DECAY_GAMMA = None  # None => auto by block size
NUM_BLOCKS_PER_SAMPLE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 training for DFlash draft model.")
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Checkpoint subdirectory name under CHECKPOINT_ROOT.",
    )
    parser.add_argument(
        "--fresh_start",
        action="store_true",
        help="Do not resume from existing checkpoints, start at step 0.",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=CHECKPOINT_ROOT,
        help="Root folder for all training runs.",
    )
    parser.add_argument(
        "--resume_source",
        type=str,
        choices=["latest", "best_acc", "best_loss", "path", "none"],
        default="latest",
        help="Where to resume from inside run checkpoint directory.",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Checkpoint path when --resume_source=path.",
    )
    parser.add_argument(
        "--lr_override",
        type=float,
        default=None,
        help="Override optimizer LR after resume (or from fresh start).",
    )
    parser.add_argument(
        "--unfreeze_policy",
        type=str,
        choices=["fc_only", "attention", "full", "draft_layers_only"],
        default=UNFREEZE_POLICY,
        help="Trainable parameter policy for draft model.",
    )
    return parser.parse_args()


def make_default_run_name(unfreeze_policy: str) -> str:
    warm = "warm" if USE_WARM_START else "cold"
    return f"phase0_block_{unfreeze_policy}_{warm}_lr{LR:.0e}_bs{BATCH_SIZE}x{ACCUMULATION_STEPS}"


def resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name.strip()
    base = make_default_run_name(args.unfreeze_policy)
    if args.fresh_start:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{suffix}"
    return base


def resolve_resume_checkpoint(
    *,
    args: argparse.Namespace,
    ckpt_dir: Path,
    best_ckpt_path: Path,
    best_acc_ckpt_path: Path,
) -> Optional[Path]:
    if args.fresh_start or args.resume_source == "none":
        return None
    if args.resume_source == "best_acc":
        return best_acc_ckpt_path if best_acc_ckpt_path.exists() else None
    if args.resume_source == "best_loss":
        return best_ckpt_path if best_ckpt_path.exists() else None
    if args.resume_source == "path":
        if args.resume_ckpt is None:
            raise ValueError("--resume_ckpt is required when --resume_source=path")
        p = Path(args.resume_ckpt)
        if not p.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {p}")
        return p
    return latest_checkpoint(ckpt_dir)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def print_gpu_memory(prefix: str) -> None:
    allocated_gb = torch.cuda.memory_allocated() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"{prefix} GPU memory: {allocated_gb:.1f}GB / {total_gb:.1f}GB")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        f.flush()


def _auto_loss_gamma(block_size: int) -> float:
    if LOSS_DECAY_GAMMA is not None:
        return float(LOSS_DECAY_GAMMA)
    if block_size >= 16:
        return 7.0
    if block_size >= 10:
        return 5.0
    if block_size >= 8:
        return 4.0
    return max(2.0, block_size / 2.0)


def build_block_training_batch(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    target_hidden: torch.Tensor,
    mask_token_id: int,
    block_size: int,
    context_len: int,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs, _ = input_ids.shape
    ctx_dim = target_hidden.shape[-1]
    device = input_ids.device

    block_input_ids = torch.full((bs, block_size), mask_token_id, dtype=torch.long, device=device)
    block_hidden_ctx = torch.zeros((bs, context_len, ctx_dim), dtype=target_hidden.dtype, device=device)
    block_labels = torch.full((bs, block_size - 1), IGNORE_INDEX, dtype=torch.long, device=device)
    block_weights = torch.zeros((bs, block_size - 1), dtype=torch.float32, device=device)
    block_valid = torch.zeros((bs,), dtype=torch.bool, device=device)

    for i in range(bs):
        valid_pos = torch.nonzero(attention_mask[i] > 0, as_tuple=True)[0]
        if valid_pos.numel() < 2:
            continue
        first_valid = int(valid_pos[0].item())
        last_valid = int(valid_pos[-1].item())
        if last_valid <= first_valid:
            continue

        # Prefer anchors whose next token is inside assistant/answer span.
        ans_positions = torch.nonzero(answer_mask[i] > 0, as_tuple=True)[0]
        candidates: List[int] = []
        if ans_positions.numel() > 0:
            first_ans = int(ans_positions[0].item())
            last_ans = int(ans_positions[-1].item())
            lo = max(first_valid, first_ans - 1)
            hi = min(last_valid - 1, last_ans - 1)
            if lo <= hi:
                candidates = list(range(lo, hi + 1))
        anchor = random.choice(candidates) if candidates else random.randint(first_valid, last_valid - 1)
        block_valid[i] = True

        block_input_ids[i, 0] = input_ids[i, anchor]
        if block_size > 1:
            block_input_ids[i, 1:] = mask_token_id

        # Use local prefix context ending at anchor to approximate runtime verifier context.
        ctx_start = max(first_valid, anchor - context_len + 1)
        ctx_slice = target_hidden[i, ctx_start : anchor + 1]
        ctx_n = int(ctx_slice.shape[0])
        if ctx_n > 0:
            block_hidden_ctx[i, context_len - ctx_n : context_len] = ctx_slice

        # Labels correspond to masked positions (k=1..block_size-1).
        max_pred = min(block_size - 1, last_valid - anchor)
        for k in range(max_pred):
            t = target_tokens[i, anchor + 1 + k]
            if t != IGNORE_INDEX:
                block_labels[i, k] = t
                block_weights[i, k] = float(math.exp(-k / gamma))

    return block_input_ids, block_hidden_ctx, block_labels, block_weights, block_valid


def extract_turn_pair(sample: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    def _to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: List[str] = []
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


class SmolTalkDataset(Dataset):
    def __init__(self, hf_dataset, processor: AutoProcessor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        sample = self.dataset[idx]
        turns = extract_turn_pair(sample)
        if turns is None:
            return None
        user_turn, assistant_turn = turns
        user_messages = [{"role": "user", "content": [{"type": "text", "text": user_turn}]}]
        messages = [
            {"role": "user", "content": [{"type": "text", "text": user_turn}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_turn}]},
        ]
        user_encoded = self.processor.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        answer_start = int(user_encoded["input_ids"].shape[1])
        input_ids = encoded["input_ids"].squeeze(0)[:MAX_SEQ_LEN]
        if input_ids.numel() < 4:
            return None
        if answer_start >= int(input_ids.shape[0] - 1):
            return None
        attention_mask = encoded["attention_mask"].squeeze(0)[: input_ids.shape[0]]
        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "answer_start": torch.tensor(answer_start, dtype=torch.long),
        }


def build_collate_fn(pad_token_id: int):
    def collate_fn(batch: List[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
        samples = [x for x in batch if x is not None]
        if not samples:
            return None

        max_len = max(s["input_ids"].shape[0] for s in samples)
        bs = len(samples)

        input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        labels = torch.full((bs, max_len), IGNORE_INDEX, dtype=torch.long)

        for i, sample in enumerate(samples):
            ids = sample["input_ids"]
            mask = sample["attention_mask"]
            answer_start = int(sample["answer_start"].item())
            n = ids.shape[0]
            row_start = max_len - n
            input_ids[i, -n:] = ids
            attention_mask[i, -n:] = mask
            if 0 <= answer_start < n:
                labels[i, row_start + answer_start : row_start + n] = ids[answer_start:n]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn


def load_text_dataset():
    try:
        train_ds = load_dataset(DATASET_NAME, split="train").shuffle(seed=SEED)
        test_ds = load_dataset(DATASET_NAME, split="test")
        return train_ds, test_ds
    except Exception:
        config_names = get_dataset_config_names(DATASET_NAME)
        for cfg in config_names:
            try:
                train_ds = load_dataset(DATASET_NAME, cfg, split="train").shuffle(seed=SEED)
                try:
                    test_ds = load_dataset(DATASET_NAME, cfg, split="test")
                except Exception:
                    test_ds = train_ds.select(range(min(5000, len(train_ds))))
                return train_ds, test_ds
            except Exception:
                continue
        raise RuntimeError(f"Unable to load dataset: {DATASET_NAME}")


def validate_draft_config(config: AutoConfig) -> None:
    if not hasattr(config, "dflash_config"):
        raise ValueError("config.json missing `dflash_config`.")
    if not hasattr(config, "num_target_layers"):
        raise ValueError("config.json missing `num_target_layers`.")
    if not hasattr(config, "block_size"):
        raise ValueError("config.json missing `block_size`.")
    if "target_layer_ids" not in config.dflash_config:
        raise ValueError("config.json dflash_config missing `target_layer_ids`.")
    if "mask_token_id" not in config.dflash_config:
        raise ValueError("config.json dflash_config missing `mask_token_id`.")


def load_draft_config(config_path: str) -> AutoConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Draft config not found: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    model_type = raw_cfg.get("model_type")
    if model_type is None:
        raise ValueError("config.json missing `model_type`.")
    cfg_kwargs = dict(raw_cfg)
    cfg_kwargs.pop("model_type", None)
    config = AutoConfig.for_model(model_type, **cfg_kwargs)
    validate_draft_config(config)
    return config


def init_draft_model(config: AutoConfig) -> DFlashDraftModel:
    if USE_WARM_START:
        try:
            print(f"[Init] Warm-start draft from: {WARM_START_MODEL_ID}")
            model = DFlashDraftModel.from_pretrained(
                WARM_START_MODEL_ID,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
            )
            return model.to(device=DEVICE, dtype=torch.bfloat16)
        except Exception as e:
            print(f"[Warn] Warm-start failed, fallback to random init. Error: {e}")
    return DFlashDraftModel(config).to(device=DEVICE, dtype=torch.bfloat16)


def freeze_for_phase0(
    draft: DFlashDraftModel,
    unfreeze_policy: str,
) -> Tuple[List[torch.nn.Parameter], Dict[str, int]]:
    for p in draft.parameters():
        p.requires_grad = False

    group_counts: Dict[str, int] = {
        "fc": 0,
        "hidden_norm": 0,
        "final_norm": 0,
        "attn": 0,
        "mlp_norm": 0,
    }

    if unfreeze_policy == "draft_layers_only":
        # Paper-like option: only update draft Transformer layers.
        for p in draft.norm.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                group_counts["final_norm"] += p.numel()
        for layer in draft.layers:
            # attention-related modules
            for mod in [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                layer.self_attn.q_norm,
                layer.self_attn.k_norm,
            ]:
                for p in mod.parameters():
                    if not p.requires_grad:
                        p.requires_grad = True
                        group_counts["attn"] += p.numel()
            # mlp + norms inside each decoder layer
            for mod in [layer.mlp, layer.input_layernorm, layer.post_attention_layernorm]:
                for p in mod.parameters():
                    if not p.requires_grad:
                        p.requires_grad = True
                        group_counts["mlp_norm"] += p.numel()
        trainable = [p for p in draft.parameters() if p.requires_grad]
        return trainable, group_counts

    for p in draft.fc.parameters():
        p.requires_grad = True
        group_counts["fc"] += p.numel()
    for p in draft.hidden_norm.parameters():
        p.requires_grad = True
        group_counts["hidden_norm"] += p.numel()

    if unfreeze_policy in {"attention", "full"}:
        for p in draft.norm.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                group_counts["final_norm"] += p.numel()
        for layer in draft.layers:
            attn_modules = [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                layer.self_attn.q_norm,
                layer.self_attn.k_norm,
            ]
            for mod in attn_modules:
                for p in mod.parameters():
                    if not p.requires_grad:
                        p.requires_grad = True
                        group_counts["attn"] += p.numel()

    if unfreeze_policy == "full":
        for layer in draft.layers:
            for mod in [layer.mlp, layer.input_layernorm, layer.post_attention_layernorm]:
                for p in mod.parameters():
                    if not p.requires_grad:
                        p.requires_grad = True
                        group_counts["mlp_norm"] += p.numel()

    trainable = [p for p in draft.parameters() if p.requires_grad]
    return trainable, group_counts


def build_eval_samples(hf_dataset, limit: int) -> List[str]:
    prompts: List[str] = []
    for sample in hf_dataset:
        turns = extract_turn_pair(sample)
        if turns is None:
            continue
        user_turn, _ = turns
        if user_turn.strip():
            prompts.append(user_turn.strip())
        if len(prompts) >= limit:
            break
    return prompts


@torch.no_grad()
def evaluate_acceptance(
    draft: DFlashDraftModel,
    target: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    prompts: List[str],
) -> Dict[str, float]:
    if not prompts:
        return {
            "eval_acceptance_length_mean": float("nan"),
            "eval_acceptance_length_std": float("nan"),
            "eval_tokens_per_sec": float("nan"),
            "eval_time_per_token": float("nan"),
        }

    draft.eval()
    acceptance_all: List[int] = []
    total_tokens = 0
    total_decode_time = 0.0

    for prompt in prompts:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        stats = dflash_generate(
            draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            stop_token_ids=[processor.tokenizer.eos_token_id] if processor.tokenizer.eos_token_id is not None else None,
            temperature=0.0,
            return_stats=True,
        )
        acceptance_all.extend(stats.acceptance_lengths)
        total_tokens += int(stats.num_output_tokens)
        total_decode_time += float(stats.time_per_output_token) * int(stats.num_output_tokens)

    draft.train()
    if not acceptance_all:
        return {
            "eval_acceptance_length_mean": float("nan"),
            "eval_acceptance_length_std": float("nan"),
            "eval_tokens_per_sec": float("nan"),
            "eval_time_per_token": float("nan"),
        }

    acc_tensor = torch.tensor(acceptance_all, dtype=torch.float32)
    eval_tpot = total_decode_time / max(total_tokens, 1)
    eval_tps = total_tokens / max(total_decode_time, 1e-9)
    return {
        "eval_acceptance_length_mean": float(acc_tensor.mean().item()),
        "eval_acceptance_length_std": float(acc_tensor.std(unbiased=False).item()),
        "eval_tokens_per_sec": float(eval_tps),
        "eval_time_per_token": float(eval_tpot),
    }


@torch.no_grad()
def evaluate_loss(
    draft: DFlashDraftModel,
    target: Qwen3VLForConditionalGeneration,
    embed_tokens,
    lm_head,
    eval_loader: DataLoader,
    num_batches: int,
) -> float:
    draft.eval()
    total = 0.0
    count = 0
    block_size = int(getattr(draft, "block_size", 16))
    context_len = int(BLOCK_CONTEXT_LEN)
    gamma = _auto_loss_gamma(block_size)
    mask_token_id = int(getattr(draft, "mask_token_id", 0))

    for batch in eval_loader:
        if batch is None:
            continue
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        answer_mask = labels != IGNORE_INDEX
        bs, seq_len = input_ids.shape
        if seq_len < 4:
            continue

        target_output = target(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        target_tokens = torch.argmax(target_output.logits, dim=-1)
        target_tokens = target_tokens.masked_fill(labels == IGNORE_INDEX, IGNORE_INDEX)
        target_hidden = extract_context_feature(target_output.hidden_states, draft.target_layer_ids)
        batch_num = torch.zeros((), device=DEVICE, dtype=torch.float32)
        batch_den = torch.zeros((), device=DEVICE, dtype=torch.float32)
        for _ in range(NUM_BLOCKS_PER_SAMPLE):
            (
                block_input_ids,
                block_hidden_ctx,
                block_labels,
                block_weights,
                block_valid,
            ) = build_block_training_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                answer_mask=answer_mask,
                target_tokens=target_tokens,
                target_hidden=target_hidden,
                mask_token_id=mask_token_id,
                block_size=block_size,
                context_len=context_len,
                gamma=gamma,
            )
            if not bool(block_valid.any()):
                continue

            noise_embedding = embed_tokens(block_input_ids)
            position_ids = torch.arange(context_len + block_size, device=DEVICE).unsqueeze(0).expand(bs, -1)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                draft_hidden = draft(
                    noise_embedding=noise_embedding,
                    target_hidden=block_hidden_ctx,
                    position_ids=position_ids,
                )
                draft_logits = lm_head(draft_hidden)
                vocab_size = draft_logits.shape[-1]
                per_tok_loss = F.cross_entropy(
                    draft_logits[:, 1:, :].reshape(-1, vocab_size),
                    block_labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="none",
                )
                per_tok_loss = per_tok_loss.view(bs, block_size - 1)
                weighted = per_tok_loss * block_weights
                denom = block_weights.sum()
                if float(denom.item()) > 0.0:
                    batch_num = batch_num + weighted.sum().float()
                    batch_den = batch_den + denom.float()
        if float(batch_den.item()) <= 0.0:
            continue
        loss = batch_num / batch_den.clamp(min=1e-6)
        total += float(loss.item())
        count += 1
        if count >= num_batches:
            break

    draft.train()
    if count == 0:
        return float("nan")
    return total / count


def latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    files = []
    for p in ckpt_dir.glob("step_*.pt"):
        try:
            step = int(p.stem.split("_")[1])
            files.append((step, p))
        except Exception:
            continue
    if not files:
        return None
    files.sort(key=lambda x: x[0])
    return files[-1][1]


def save_checkpoint(
    ckpt_path: Path,
    step: int,
    draft: DFlashDraftModel,
    optimizer: AdamW,
    scheduler,
    loss_value: float,
    trainable_param_count: Optional[int] = None,
    eval_acceptance_length: Optional[float] = None,
    eval_loss: Optional[float] = None,
    unfreeze_policy: Optional[str] = None,
) -> None:
    checkpoint = {
        "step": step,
        "model_state_dict": draft.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss_value,
        "eval_acceptance_length": eval_acceptance_length,
        "eval_loss": eval_loss,
        "trainable_param_count": trainable_param_count,
        "unfreeze_policy": unfreeze_policy if unfreeze_policy is not None else UNFREEZE_POLICY,
        "use_warm_start": USE_WARM_START,
        "warm_start_model_id": WARM_START_MODEL_ID if USE_WARM_START else None,
    }
    torch.save(checkpoint, ckpt_path)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    set_seed(SEED)
    run_name = resolve_run_name(args)
    unfreeze_policy = args.unfreeze_policy
    ckpt_dir = Path(args.checkpoint_root) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_checkpoint.pt"
    best_acc_ckpt_path = ckpt_dir / "best_acceptance_checkpoint.pt"
    log_path = ckpt_dir / "logs.jsonl"
    summary_path = ckpt_dir / "summary.json"
    print(f"[Run] run_name={run_name}")
    print(f"[Run] checkpoint_dir={ckpt_dir}")
    if args.fresh_start:
        print("[Run] fresh_start=True (skip checkpoint resume)")
    print(f"[Run] resume_source={args.resume_source}")
    print(f"[Run] unfreeze_policy={unfreeze_policy}")
    if args.lr_override is not None:
        print(f"[Run] lr_override={args.lr_override:.2e}")

    print("[Init] Loading target model...")
    target = Qwen3VLForConditionalGeneration.from_pretrained(
        TARGET_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    target.eval()
    for p in target.parameters():
        p.requires_grad = False
    processor = AutoProcessor.from_pretrained(TARGET_MODEL_ID)
    embed_tokens = _get_embed_tokens(target)
    lm_head = _get_lm_head(target)
    for p in lm_head.parameters():
        p.requires_grad = False

    print("[Init] Loading draft config and instantiating draft model...")
    draft_config = load_draft_config(DRAFT_CONFIG_PATH)
    draft = init_draft_model(draft_config)
    trainable_params, group_counts = freeze_for_phase0(draft, unfreeze_policy=unfreeze_policy)
    n_total = sum(p.numel() for p in draft.parameters())
    n_trainable = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    print(f"[Phase 0] Trainable params: {n_trainable:,} / {n_total:,}")
    print(
        "[Phase 0] Trainable breakdown | "
        f"fc={group_counts['fc']:,}, hidden_norm={group_counts['hidden_norm']:,}, "
        f"final_norm={group_counts['final_norm']:,}, "
        f"attn={group_counts['attn']:,}, mlp_norm={group_counts['mlp_norm']:,}, "
        f"policy={unfreeze_policy}, warm_start={USE_WARM_START}"
    )
    draft.train()

    effective_lr = float(args.lr_override) if args.lr_override is not None else LR
    optimizer = AdamW(trainable_params, lr=effective_lr, weight_decay=WEIGHT_DECAY)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=MAX_STEPS,
    )

    train_dataset, test_dataset = load_text_dataset()
    eval_prompts = build_eval_samples(test_dataset, EVAL_NUM_SAMPLES)
    print(f"[Init] Eval samples: {len(eval_prompts)}")
    wrapped_dataset = SmolTalkDataset(train_dataset, processor)
    eval_wrapped_dataset = SmolTalkDataset(test_dataset, processor)
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=build_collate_fn(pad_id),
        drop_last=False,
        num_workers=2,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_wrapped_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=build_collate_fn(pad_id),
        drop_last=False,
        num_workers=2,
        pin_memory=True,
    )

    step = 0
    best_loss = float("inf")
    best_acc_len = float("-inf")
    no_improve_acc_streak = 0
    grad_norm_value = 0.0
    best_step = 0
    best_acc_step = 0
    last_eval_acceptance_mean = float("nan")
    last_eval_loss = float("nan")

    ckpt_path = resolve_resume_checkpoint(
        args=args,
        ckpt_dir=ckpt_dir,
        best_ckpt_path=best_ckpt_path,
        best_acc_ckpt_path=best_acc_ckpt_path,
    )
    if ckpt_path is not None:
        print(f"[Resume] Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        draft.load_state_dict(state["model_state_dict"])
        saved_trainable = state.get("trainable_param_count")
        can_resume_optim = (saved_trainable is None) or (int(saved_trainable) == int(n_trainable))
        if can_resume_optim:
            try:
                optimizer.load_state_dict(state["optimizer_state_dict"])
                scheduler.load_state_dict(state["scheduler_state_dict"])
                step = int(state.get("step", 0))
            except Exception as e:
                print(f"[Warn] Optimizer/scheduler resume failed, restart optimizer state. Error: {e}")
                step = 0
        else:
            print(
                "[Warn] Checkpoint trainable_param_count mismatch "
                f"(ckpt={saved_trainable}, current={n_trainable}). Restarting optimizer/scheduler."
            )
            step = 0
        last_loss = float(state.get("loss", float("inf")))
        best_loss = min(best_loss, last_loss)
        if step > 0:
            print(f"[Resume] Resuming from step {step}, loss={last_loss:.4f}")
        else:
            print(f"[Resume] Loaded model weights from {ckpt_path}, optimizer reset.")

        if args.lr_override is not None:
            for group in optimizer.param_groups:
                group["lr"] = float(args.lr_override)
            if hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = [float(args.lr_override)] * len(scheduler.base_lrs)
            if hasattr(scheduler, "_last_lr"):
                scheduler._last_lr = [float(args.lr_override)] * len(optimizer.param_groups)
            print(f"[Resume] Applied LR override: {args.lr_override:.2e}")

        if best_acc_ckpt_path.exists():
            try:
                acc_state = torch.load(best_acc_ckpt_path, map_location="cpu")
                best_acc_len = float(acc_state.get("eval_acceptance_length", float("-inf")))
                best_acc_step = int(acc_state.get("step", 0))
                if best_acc_len != float("-inf"):
                    print(f"[Resume] Best acc checkpoint: step={best_acc_step}, acc_len={best_acc_len:.4f}")
            except Exception as e:
                print(f"[Warn] Failed to load best acceptance checkpoint metadata: {e}")
    else:
        print("[Fresh] Starting from step 0")

    if step == 0 or step % 1000 == 0:
        print_gpu_memory("[Info]")

    start_time = time.perf_counter()
    window_start = time.perf_counter()
    window_tokens = 0
    window_loss_sum = 0.0
    window_micro_steps = 0
    micro_step = 0
    update_tokens = 0
    update_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    data_iter = iter(dataloader)
    block_size = int(getattr(draft, "block_size", 16))
    context_len = int(BLOCK_CONTEXT_LEN)
    gamma = _auto_loss_gamma(block_size)
    mask_token_id = int(getattr(draft, "mask_token_id", 0))
    print(
        f"[Phase 0] Block training enabled | block_size={block_size}, "
        f"context_len={context_len}, blocks/sample={NUM_BLOCKS_PER_SAMPLE}, "
        f"loss_decay_gamma={gamma:.2f}, mask_token_id={mask_token_id}"
    )
    while step < MAX_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if batch is None:
            continue

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        answer_mask = labels != IGNORE_INDEX
        bs, seq_len = input_ids.shape
        if seq_len < 4:
            continue

        with torch.no_grad():
            target_output = target(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            target_tokens = torch.argmax(target_output.logits, dim=-1)
            target_tokens = target_tokens.masked_fill(labels == IGNORE_INDEX, IGNORE_INDEX)
            target_hidden = extract_context_feature(
                target_output.hidden_states,
                draft.target_layer_ids,
            )

        total_num = torch.zeros((), device=DEVICE, dtype=torch.float32)
        total_den = torch.zeros((), device=DEVICE, dtype=torch.float32)
        valid_tokens = 0
        last_block_hidden_ctx: Optional[torch.Tensor] = None
        last_noise_embedding: Optional[torch.Tensor] = None

        for _ in range(NUM_BLOCKS_PER_SAMPLE):
            (
                block_input_ids,
                block_hidden_ctx,
                block_labels,
                block_weights,
                block_valid,
            ) = build_block_training_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                answer_mask=answer_mask,
                target_tokens=target_tokens,
                target_hidden=target_hidden,
                mask_token_id=mask_token_id,
                block_size=block_size,
                context_len=context_len,
                gamma=gamma,
            )

            if not bool(block_valid.any()):
                continue

            noise_embedding = embed_tokens(block_input_ids)
            position_ids = torch.arange(context_len + block_size, device=DEVICE).unsqueeze(0).expand(bs, -1)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                draft_hidden = draft(
                    noise_embedding=noise_embedding,
                    target_hidden=block_hidden_ctx,
                    position_ids=position_ids,
                )
                # lm_head is frozen, but this op must remain in grad mode so gradients flow to draft_hidden.
                draft_logits = lm_head(draft_hidden)
                vocab_size = draft_logits.shape[-1]
                per_tok_loss = F.cross_entropy(
                    draft_logits[:, 1:, :].reshape(-1, vocab_size),
                    block_labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="none",
                )
                per_tok_loss = per_tok_loss.view(bs, block_size - 1)
                weighted = per_tok_loss * block_weights
                denom = block_weights.sum()
                if float(denom.item()) > 0.0:
                    total_num = total_num + weighted.sum().float()
                    total_den = total_den + denom.float()
                    valid_tokens += int((block_labels != IGNORE_INDEX).sum().item())
                    last_block_hidden_ctx = block_hidden_ctx
                    last_noise_embedding = noise_embedding

        if float(total_den.item()) <= 0.0:
            continue
        loss = total_num / total_den.clamp(min=1e-6)

        scaled_loss = loss / ACCUMULATION_STEPS
        scaled_loss.backward()
        micro_step += 1

        window_tokens += int(valid_tokens)
        window_loss_sum += float(loss.item())
        window_micro_steps += 1
        update_tokens += int(valid_tokens)

        if micro_step % ACCUMULATION_STEPS == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(draft.parameters(), GRAD_CLIP)
            grad_norm_value = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            step_now = time.perf_counter()
            step_elapsed = max(step_now - update_start, 1e-6)
            step_tps = update_tokens / step_elapsed
            update_start = step_now
            update_tokens = 0

            hidden_cosine_sim = None
            if step % HIDDEN_ALIGN_EVERY == 0 and last_block_hidden_ctx is not None and last_noise_embedding is not None:
                with torch.no_grad():
                    projected = draft.hidden_norm(draft.fc(last_block_hidden_ctx))
                    m = min(projected.shape[1], last_noise_embedding.shape[1])
                    projected = projected[:, -m:, :]
                    noise_cmp = last_noise_embedding[:, -m:, :]
                    hidden_cosine_sim = float(
                        F.cosine_similarity(projected.float(), noise_cmp.float(), dim=-1).mean().item()
                    )

            gpu_memory_gb = None
            if step % GPU_LOG_EVERY == 0:
                gpu_memory_gb = float(torch.cuda.memory_allocated() / 1e9)

            log_entry: Dict[str, Any] = {
                "step": step,
                "loss": float(loss.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm_value),
                "tokens_per_sec": float(step_tps),
                "gpu_memory_gb": gpu_memory_gb,
                "elapsed_sec": float(time.perf_counter() - start_time),
                "hidden_cosine_sim": hidden_cosine_sim,
                "eval_acceptance_length_mean": None,
                "eval_acceptance_length_std": None,
                "eval_tokens_per_sec": None,
                "eval_time_per_token": None,
                "eval_loss": None,
            }

            if step % LOG_EVERY == 0:
                elapsed_window = max(time.perf_counter() - window_start, 1e-6)
                tps = window_tokens / elapsed_window
                avg_loss = window_loss_sum / max(1, window_micro_steps)
                lr = optimizer.param_groups[0]["lr"]
                total_elapsed = format_elapsed(time.perf_counter() - start_time)
                print(
                    f"Step {step:5d}/{MAX_STEPS} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                    f"Grad norm: {grad_norm_value:.3f} | Tokens/sec: {tps:.0f} | Elapsed: {total_elapsed}"
                )
                window_tokens = 0
                window_loss_sum = 0.0
                window_micro_steps = 0
                window_start = time.perf_counter()

            if step % SAVE_EVERY == 0:
                loss_value = float(loss.item())
                step_ckpt = ckpt_dir / f"step_{step}.pt"
                save_checkpoint(
                    step_ckpt,
                    step,
                    draft,
                    optimizer,
                    scheduler,
                    loss_value,
                    trainable_param_count=n_trainable,
                    unfreeze_policy=unfreeze_policy,
                )

                eval_metrics = evaluate_acceptance(
                    draft=draft,
                    target=target,
                    processor=processor,
                    prompts=eval_prompts,
                )
                eval_loss = evaluate_loss(
                    draft=draft,
                    target=target,
                    embed_tokens=embed_tokens,
                    lm_head=lm_head,
                    eval_loader=eval_dataloader,
                    num_batches=EVAL_LOSS_NUM_BATCHES,
                )
                log_entry.update(eval_metrics)
                log_entry["eval_loss"] = float(eval_loss)
                last_eval_acceptance_mean = float(eval_metrics["eval_acceptance_length_mean"])
                last_eval_loss = float(eval_loss)
                print(
                    f"[Eval] step={step} | eval_loss={eval_loss:.4f} | "
                    f"acc_len={eval_metrics['eval_acceptance_length_mean']:.3f} | "
                    f"eval_tps={eval_metrics['eval_tokens_per_sec']:.1f}"
                )

                if loss_value < best_loss:
                    best_loss = loss_value
                    best_step = step
                    save_checkpoint(
                        best_ckpt_path,
                        step,
                        draft,
                        optimizer,
                        scheduler,
                        loss_value,
                        trainable_param_count=n_trainable,
                        eval_acceptance_length=last_eval_acceptance_mean,
                        eval_loss=last_eval_loss,
                        unfreeze_policy=unfreeze_policy,
                    )

                improved_acc = (
                    not math.isnan(last_eval_acceptance_mean)
                    and last_eval_acceptance_mean > (best_acc_len + EARLY_STOP_ACC_DELTA)
                )
                if improved_acc:
                    best_acc_len = last_eval_acceptance_mean
                    best_acc_step = step
                    save_checkpoint(
                        best_acc_ckpt_path,
                        step,
                        draft,
                        optimizer,
                        scheduler,
                        loss_value,
                        trainable_param_count=n_trainable,
                        eval_acceptance_length=last_eval_acceptance_mean,
                        eval_loss=last_eval_loss,
                        unfreeze_policy=unfreeze_policy,
                    )
                    print(f"[BestAcc] step={best_acc_step} | acc_len={best_acc_len:.4f}")
                    no_improve_acc_streak = 0
                else:
                    no_improve_acc_streak += 1

                if no_improve_acc_streak >= EARLY_STOP_PATIENCE:
                    print(
                        f"[Early Stop] acc_len not improved by >= {EARLY_STOP_ACC_DELTA:.1e} "
                        f"for {EARLY_STOP_PATIENCE} evals. Done."
                    )
                    final_ckpt = ckpt_dir / f"step_{step}_final.pt"
                    save_checkpoint(
                        final_ckpt,
                        step,
                        draft,
                        optimizer,
                        scheduler,
                        loss_value,
                        trainable_param_count=n_trainable,
                        unfreeze_policy=unfreeze_policy,
                    )
                    append_jsonl(log_path, log_entry)
                    break

            append_jsonl(log_path, log_entry)

            if step % 1000 == 0:
                torch.cuda.empty_cache()
                print_gpu_memory("[Info]")

    elapsed_total = time.perf_counter() - start_time
    summary = {
        "phase": 0,
        "total_steps": int(step),
        "final_loss": float(loss.item()) if "loss" in locals() else None,
        "best_loss": float(best_loss),
        "best_step": int(best_step),
        "best_acceptance_length": float(best_acc_len) if best_acc_len != float("-inf") else None,
        "best_acceptance_step": int(best_acc_step),
        "final_acceptance_length": float(last_eval_acceptance_mean),
        "final_eval_loss": float(last_eval_loss),
        "training_time_hours": float(elapsed_total / 3600.0),
        "gpu": torch.cuda.get_device_name(0),
        "target_model": TARGET_MODEL_ID,
        "trainable_params": int(n_trainable),
        "total_params": int(n_total),
        "dataset": DATASET_NAME,
        "max_seq_len": MAX_SEQ_LEN,
        "effective_batch_size": BATCH_SIZE * ACCUMULATION_STEPS,
        "training_mode": "block_verify_aligned",
        "block_size": int(block_size),
        "block_context_len": int(context_len),
        "loss_decay_gamma": float(gamma),
        "unfreeze_policy": unfreeze_policy,
        "early_stop_acc_delta": float(EARLY_STOP_ACC_DELTA),
        "use_warm_start": USE_WARM_START,
        "warm_start_model_id": WARM_START_MODEL_ID if USE_WARM_START else None,
        "run_name": run_name,
        "checkpoint_dir": str(ckpt_dir),
        "fresh_start": bool(args.fresh_start),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
        f.flush()

    print(f"[Done] Training finished. Logs: {log_path} | Summary: {summary_path}")


if __name__ == "__main__":
    main()
