import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils import clip_grad_norm_
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
MAX_STEPS = 10000
WARMUP_STEPS = 400  # paper uses warmup ratio 0.04 over 10k steps
LORA_LR = 1e-5
FC_HIDDEN_LR = 5e-7
LR_MIN = 1e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
# Kept for compatibility with helper functions; unused in phase0-style run.
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
TRAIN_DRAFT_LAYERS_ONLY = True
TRAIN_ALL_DRAFT_LAYERS = False
FULL_SEQ_LABELS = False
IMAGE_RATIO = 1.0
IMAGE_TRAIN_SAMPLES = 10000
IMAGE_EPOCHS = 2
LAMBDA_KL_STAGE1 = 0.0
LAMBDA_L2SP_STAGE1 = 0.0
NUM_BLOCKS_PER_SAMPLE = 32
EVAL_IMAGE_SAMPLES = 10
EVAL_MAX_NEW_TOKENS = 64
BLOCK_CONTEXT_LEN = 64
USE_FULL_CONTEXT = False
LOSS_DECAY_GAMMA = None  # None => auto by block size
LOG_EVERY = 200
SAVE_EVERY = 1000
POSITION_DEBUG = False
POSITION_DEBUG_EVERY = 200
EARLY_STOP_PATIENCE = 5
QUICK_STOP_ENABLED = False
IMAGE_ACC_MIN_GAIN = 0.05
ANCHOR_STRATIFIED_SAMPLING = False  # paper uses random anchor sampling
ANCHOR_STRATIFIED_BINS = 5
CHECKPOINT_DIR = "/content/drive/MyDrive/dflash_phaseA_answer_only_v2_mrope"
PHASE0_CKPT = "/content/drive/MyDrive/dflash_phase0_paper_v2/phase0_paper_v2_run1/best_acceptance_checkpoint.pt"
COCO_ROOT = "/content/mscoco/images/train2017"
COCO_ANN = "/content/mscoco/annotations/annotations/captions_train2017.json"
TARGET_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DRAFT_CONFIG_PATH = "./config.json"

SEED = 42
IGNORE_INDEX = -100
DEVICE = "cuda"
IMAGE_PROMPT = "Describe this image."
IMAGE_ASSISTANT_FALLBACK = "The image shows a scene."


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


def next_batch(loader: DataLoader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


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


def _build_full_rope_position_ids(
    *,
    target: Qwen3VLForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
) -> torch.Tensor:
    # Use Qwen3-VL native RoPE index builder when available.
    if hasattr(target, "model") and hasattr(target.model, "get_rope_index"):
        pos_ids, _ = target.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        return pos_ids.to(input_ids.device)

    # Fallback: plain 1D expanded to 3 axes.
    pos_1d = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(1, input_ids.shape[0], -1)
    return pos_1d.expand(3, -1, -1)


def _build_block_position_ids(
    *,
    full_pos_ids: torch.Tensor,
    block_valid: torch.Tensor,
    block_ctx_start: torch.Tensor,
    block_anchor_pos: torch.Tensor,
    block_last_valid: torch.Tensor,
    context_len: int,
    block_size: int,
) -> torch.Tensor:
    # Shape expected by Qwen3VLTextRotaryEmbedding: [3, bs, context_len + block_size]
    _, bs, _ = full_pos_ids.shape
    out = torch.zeros((3, bs, context_len + block_size), dtype=full_pos_ids.dtype, device=full_pos_ids.device)

    for i in range(bs):
        if not bool(block_valid[i].item()):
            continue
        ctx_start = int(block_ctx_start[i].item())
        anchor = int(block_anchor_pos[i].item())
        last_valid = int(block_last_valid[i].item())
        if ctx_start < 0 or anchor < 0 or last_valid < 0:
            continue

        ctx_slice = full_pos_ids[:, i, ctx_start : anchor + 1]
        ctx_n = int(ctx_slice.shape[1])
        if ctx_n > 0:
            # Left pad (if any) with the first available context position.
            out[:, i, : context_len - ctx_n] = ctx_slice[:, :1]
            out[:, i, context_len - ctx_n : context_len] = ctx_slice

        last_pos = full_pos_ids[:, i, last_valid]
        for k in range(block_size):
            tok_idx = anchor + k
            if tok_idx <= last_valid:
                out[:, i, context_len + k] = full_pos_ids[:, i, tok_idx]
            else:
                delta = tok_idx - last_valid
                out[:, i, context_len + k] = last_pos + delta

    return out


def _check_block_position_alignment(
    *,
    full_pos_ids: torch.Tensor,
    block_pos_ids: torch.Tensor,
    block_valid: torch.Tensor,
    block_ctx_start: torch.Tensor,
    block_anchor_pos: torch.Tensor,
    block_last_valid: torch.Tensor,
    context_len: int,
    block_size: int,
) -> Dict[str, int]:
    ctx_token_mismatch = 0
    noise_token_mismatch = 0
    checked_ctx_tokens = 0
    checked_noise_tokens = 0

    _, bs, _ = full_pos_ids.shape
    for i in range(bs):
        if not bool(block_valid[i].item()):
            continue
        ctx_start = int(block_ctx_start[i].item())
        anchor = int(block_anchor_pos[i].item())
        last_valid = int(block_last_valid[i].item())
        if ctx_start < 0 or anchor < 0 or last_valid < 0:
            continue

        # Context part expected from full_pos_ids with left padding by first ctx token.
        ctx_slice = full_pos_ids[:, i, ctx_start : anchor + 1]
        ctx_n = int(ctx_slice.shape[1])
        expected_ctx = torch.zeros((3, context_len), dtype=full_pos_ids.dtype, device=full_pos_ids.device)
        if ctx_n > 0:
            expected_ctx[:, : context_len - ctx_n] = ctx_slice[:, :1]
            expected_ctx[:, context_len - ctx_n : context_len] = ctx_slice
        actual_ctx = block_pos_ids[:, i, :context_len]
        ctx_diff = (actual_ctx != expected_ctx).any(dim=0)
        ctx_token_mismatch += int(ctx_diff.sum().item())
        checked_ctx_tokens += int(context_len)

        # Noise part expected from [anchor .. anchor+block_size-1], extending past last_valid linearly.
        expected_noise = torch.zeros((3, block_size), dtype=full_pos_ids.dtype, device=full_pos_ids.device)
        last_pos = full_pos_ids[:, i, last_valid]
        for k in range(block_size):
            tok_idx = anchor + k
            if tok_idx <= last_valid:
                expected_noise[:, k] = full_pos_ids[:, i, tok_idx]
            else:
                expected_noise[:, k] = last_pos + (tok_idx - last_valid)
        actual_noise = block_pos_ids[:, i, context_len : context_len + block_size]
        noise_diff = (actual_noise != expected_noise).any(dim=0)
        noise_token_mismatch += int(noise_diff.sum().item())
        checked_noise_tokens += int(block_size)

    return {
        "checked_ctx_tokens": checked_ctx_tokens,
        "ctx_token_mismatch": ctx_token_mismatch,
        "checked_noise_tokens": checked_noise_tokens,
        "noise_token_mismatch": noise_token_mismatch,
    }


def _inspect_visual_rope_positions(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    full_pos_ids: torch.Tensor,
    image_token_id: Optional[int],
    video_token_id: Optional[int],
) -> Dict[str, Any]:
    # Return compact stats to verify image/video token positions in MRoPE space.
    bs = int(input_ids.shape[0])
    seq_len = int(input_ids.shape[1])
    stats: Dict[str, Any] = {
        "bs": bs,
        "seq_len": seq_len,
        "image_token_id": int(image_token_id) if image_token_id is not None else None,
        "video_token_id": int(video_token_id) if video_token_id is not None else None,
        "visual_tokens_total": 0,
        "samples_with_visual": 0,
        "visual_axes_equal_all": None,
        "t_unique": 0,
        "h_unique": 0,
        "w_unique": 0,
        "t_range": None,
        "h_range": None,
        "w_range": None,
    }

    if image_token_id is None and video_token_id is None:
        return stats

    all_t: List[torch.Tensor] = []
    all_h: List[torch.Tensor] = []
    all_w: List[torch.Tensor] = []
    eq_all = True

    for i in range(bs):
        valid = attention_mask[i] > 0
        ids = input_ids[i]
        vis_mask = torch.zeros_like(valid, dtype=torch.bool)
        if image_token_id is not None:
            vis_mask |= (ids == int(image_token_id))
        if video_token_id is not None:
            vis_mask |= (ids == int(video_token_id))
        vis_mask &= valid
        n_vis = int(vis_mask.sum().item())
        if n_vis <= 0:
            continue
        stats["samples_with_visual"] += 1
        stats["visual_tokens_total"] += n_vis

        pos = full_pos_ids[:, i, vis_mask]  # [3, n_vis]
        t_axis = pos[0]
        h_axis = pos[1]
        w_axis = pos[2]
        all_t.append(t_axis)
        all_h.append(h_axis)
        all_w.append(w_axis)
        eq_all = eq_all and bool(torch.equal(t_axis, h_axis) and torch.equal(h_axis, w_axis))

    if stats["visual_tokens_total"] > 0:
        t_cat = torch.cat(all_t, dim=0)
        h_cat = torch.cat(all_h, dim=0)
        w_cat = torch.cat(all_w, dim=0)
        stats["visual_axes_equal_all"] = bool(eq_all)
        stats["t_unique"] = int(torch.unique(t_cat).numel())
        stats["h_unique"] = int(torch.unique(h_cat).numel())
        stats["w_unique"] = int(torch.unique(w_cat).numel())
        stats["t_range"] = [int(t_cat.min().item()), int(t_cat.max().item())]
        stats["h_range"] = [int(h_cat.min().item()), int(h_cat.max().item())]
        stats["w_range"] = [int(w_cat.min().item()), int(w_cat.max().item())]

    return stats


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
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    bs, _ = input_ids.shape
    ctx_dim = target_hidden.shape[-1]
    device = input_ids.device

    block_input_ids = torch.full((bs, block_size), mask_token_id, dtype=torch.long, device=device)
    block_hidden_ctx = torch.zeros((bs, context_len, ctx_dim), dtype=target_hidden.dtype, device=device)
    block_labels = torch.full((bs, block_size - 1), IGNORE_INDEX, dtype=torch.long, device=device)
    block_weights = torch.zeros((bs, block_size - 1), dtype=torch.float32, device=device)
    block_valid = torch.zeros((bs,), dtype=torch.bool, device=device)
    block_anchor_pos = torch.full((bs,), -1, dtype=torch.long, device=device)
    block_anchor_rel = torch.full((bs,), -1.0, dtype=torch.float32, device=device)
    block_ctx_start = torch.full((bs,), -1, dtype=torch.long, device=device)
    block_last_valid = torch.full((bs,), -1, dtype=torch.long, device=device)

    for i in range(bs):
        valid_pos = torch.nonzero(attention_mask[i] > 0, as_tuple=True)[0]
        if valid_pos.numel() < 2:
            continue
        first_valid = int(valid_pos[0].item())
        last_valid = int(valid_pos[-1].item())
        if last_valid <= first_valid:
            continue

        # Prefer anchors that predict assistant answer tokens.
        # We need answer at (anchor + 1), so candidate anchors are where answer_mask[:, 1:] is True.
        ans_next_idx = torch.nonzero(answer_mask[i, 1:] > 0, as_tuple=True)[0]
        if ans_next_idx.numel() > 0:
            candidates = (ans_next_idx + 0).tolist()  # anchor positions in original indexing
            # Keep candidates inside valid range.
            candidates = [c for c in candidates if first_valid <= c <= (last_valid - 1)]
        else:
            candidates = []

        if candidates:
            if ANCHOR_STRATIFIED_SAMPLING and ANCHOR_STRATIFIED_BINS > 1:
                denom = max(last_valid - first_valid, 1)
                bins: List[List[int]] = [[] for _ in range(ANCHOR_STRATIFIED_BINS)]
                for c in candidates:
                    rel = float(c - first_valid) / float(denom)
                    rel = min(max(rel, 0.0), 1.0)
                    bi = min(int(rel * ANCHOR_STRATIFIED_BINS), ANCHOR_STRATIFIED_BINS - 1)
                    bins[bi].append(c)
                non_empty = [b for b in bins if b]
                chosen_bin = random.choice(non_empty) if non_empty else candidates
                anchor = int(random.choice(chosen_bin))
            else:
                anchor = int(random.choice(candidates))
        else:
            anchor = random.randint(first_valid, last_valid - 1)
        block_valid[i] = True
        block_anchor_pos[i] = anchor
        block_last_valid[i] = last_valid
        denom = max(last_valid - first_valid, 1)
        block_anchor_rel[i] = float(anchor - first_valid) / float(denom)

        block_input_ids[i, 0] = input_ids[i, anchor]
        if block_size > 1:
            block_input_ids[i, 1:] = mask_token_id

        ctx_start = max(first_valid, anchor - context_len + 1)
        block_ctx_start[i] = ctx_start
        ctx_slice = target_hidden[i, ctx_start : anchor + 1]
        ctx_n = int(ctx_slice.shape[0])
        if ctx_n > 0:
            block_hidden_ctx[i, context_len - ctx_n : context_len] = ctx_slice

        max_pred = min(block_size - 1, last_valid - anchor)
        for k in range(max_pred):
            t = target_tokens[i, anchor + 1 + k]
            if t != IGNORE_INDEX:
                block_labels[i, k] = t
                block_weights[i, k] = float(math.exp(-k / gamma))

    return (
        block_input_ids,
        block_hidden_ctx,
        block_labels,
        block_weights,
        block_valid,
        block_anchor_pos,
        block_anchor_rel,
        block_ctx_start,
        block_last_valid,
    )


class CocoCaptionDataset(Dataset):
    def __init__(self, coco_root: str, coco_ann: str, processor: AutoProcessor):
        self.processor = processor
        self.coco_root = Path(coco_root)
        ann_path = Path(coco_ann)
        if not self.coco_root.exists():
            raise FileNotFoundError(f"COCO_ROOT not found: {self.coco_root}")
        self.items: List[Tuple[Path, List[str]]] = []

        if ann_path.exists():
            with ann_path.open("r", encoding="utf-8") as f:
                ann = json.load(f)

            id_to_file = {img["id"]: img["file_name"] for img in ann.get("images", [])}
            id_to_caps: Dict[int, List[str]] = {}
            for obj in ann.get("annotations", []):
                image_id = int(obj["image_id"])
                cap = str(obj.get("caption", "")).strip()
                if not cap:
                    continue
                id_to_caps.setdefault(image_id, []).append(cap)

            for image_id, captions in id_to_caps.items():
                file_name = id_to_file.get(image_id)
                if file_name is None:
                    continue
                image_path = self.coco_root / file_name
                if image_path.exists():
                    self.items.append((image_path, captions))
        else:
            # Fallback mode: no annotation file required, read images directly.
            print(f"[Warn] COCO_ANN not found: {ann_path}. Falling back to image-folder-only mode.")
            for image_path in sorted(self.coco_root.glob("*.jpg")):
                self.items.append((image_path, [IMAGE_ASSISTANT_FALLBACK]))

        if not self.items:
            raise RuntimeError("No valid image samples found in COCO_ROOT.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        image_path, captions = self.items[idx]
        caption = random.choice(captions).strip()
        if not caption:
            return None

        with Image.open(image_path) as im:
            image = im.convert("RGB").resize((448, 448))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": IMAGE_PROMPT},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": caption}]},
        ]
        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": IMAGE_PROMPT},
                ],
            }
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
        full_input_ids = encoded["input_ids"]
        full_attention_mask = encoded["attention_mask"]
        answer_start = int(user_encoded["input_ids"].shape[1])
        input_ids = full_input_ids[:, :MAX_SEQ_LEN]
        attention_mask = full_attention_mask[:, :MAX_SEQ_LEN]
        if input_ids.shape[1] < 4:
            return None
        if (not FULL_SEQ_LABELS) and answer_start >= int(input_ids.shape[1] - 1):
            return None
        if "pixel_values" not in encoded or "image_grid_thw" not in encoded:
            return None
        return {
            "input_ids": input_ids.squeeze(0).long(),
            "attention_mask": attention_mask.squeeze(0).long(),
            "answer_start": int(answer_start),
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": encoded["image_grid_thw"],
        }


def build_image_collate_fn(pad_token_id: int):
    def collate_fn(samples: List[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
        max_len = max(s["input_ids"].shape[0] for s in samples)
        bs = len(samples)
        input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        labels = torch.full((bs, max_len), IGNORE_INDEX, dtype=torch.long)
        pixel_values_list = []
        image_grid_list = []
        for i, sample in enumerate(samples):
            ids = sample["input_ids"]
            mask = sample["attention_mask"]
            n = ids.shape[0]
            answer_start = int(sample["answer_start"])
            input_ids[i, -n:] = ids
            attention_mask[i, -n:] = mask
            base = max_len - n
            if FULL_SEQ_LABELS:
                labels[i, base : base + n] = ids
            else:
                start = max(0, answer_start)
                if start < n:
                    labels[i, base + start : base + n] = ids[start:n]
            pixel_values_list.append(sample["pixel_values"])
            image_grid_list.append(sample["image_grid_thw"])
        try:
            pixel_values = torch.cat(pixel_values_list, dim=0)
            image_grid_thw = torch.cat(image_grid_list, dim=0)
        except RuntimeError:
            return None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    return collate_fn


def build_eval_image_paths(image_ds: "CocoCaptionDataset", limit: int) -> List[Path]:
    paths: List[Path] = []
    for image_path, _caps in image_ds.items:
        paths.append(image_path)
        if len(paths) >= limit:
            break
    return paths


@torch.no_grad()
def evaluate_acceptance_10_10(
    *,
    draft: DFlashDraftModel,
    target: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image_paths: List[Path],
    step: Optional[int] = None,
    detail_log_path: Optional[Path] = None,
) -> Dict[str, float]:
    draft.eval()
    stop_ids = [processor.tokenizer.eos_token_id] if processor.tokenizer.eos_token_id is not None else None

    img_accept: List[int] = []
    img_tokens = 0
    img_decode_time = 0.0
    per_answer_rows: List[Dict[str, Any]] = []

    for idx, image_path in enumerate(image_paths):
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": IMAGE_PROMPT},
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
        input_ids = encoded["input_ids"].to(DEVICE)
        pixel_values = encoded.get("pixel_values")
        image_grid_thw = encoded.get("image_grid_thw")
        if pixel_values is not None:
            pixel_values = pixel_values.to(DEVICE)
            image_grid_thw = image_grid_thw.to(DEVICE)
        stats = dflash_generate(
            draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            stop_token_ids=stop_ids,
            temperature=0.0,
            return_stats=True,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        img_accept.extend(stats.acceptance_lengths)
        img_tokens += int(stats.num_output_tokens)
        img_decode_time += float(stats.time_per_output_token) * int(stats.num_output_tokens)
        acc_seq = [int(x) for x in stats.acceptance_lengths]
        per_answer_rows.append(
            {
                "step": int(step) if step is not None else None,
                "sample_type": "image",
                "sample_index": int(idx),
                "image_path": str(image_path),
                "num_decode_steps": int(len(acc_seq)),
                "acceptance_lengths": acc_seq,
                "mean_acceptance_length": float(sum(acc_seq) / len(acc_seq)) if acc_seq else None,
                "num_output_tokens": int(stats.num_output_tokens),
            }
        )

    draft.train()

    if detail_log_path is not None:
        for row in per_answer_rows:
            append_jsonl(detail_log_path, row)

    img_mean = float("nan") if not img_accept else float(torch.tensor(img_accept, dtype=torch.float32).mean().item())
    img_tps = float("nan") if img_decode_time <= 0 else float(img_tokens / img_decode_time)
    return {
        "eval_image_acc_len": img_mean,
        "eval_image_tps": img_tps,
    }


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
    raw_cfg = json.loads(path.read_text(encoding="utf-8"))
    model_type = raw_cfg.get("model_type")
    if model_type is None:
        raise ValueError("config.json missing `model_type`.")
    cfg_kwargs = dict(raw_cfg)
    cfg_kwargs.pop("model_type", None)
    config = AutoConfig.for_model(model_type, **cfg_kwargs)
    validate_draft_config(config)
    return config


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, linear.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scale * self.lora_B(self.lora_A(self.dropout(x)))


def inject_lora(draft: DFlashDraftModel) -> None:
    for layer in draft.layers:
        attn = layer.self_attn
        for module_name in LORA_TARGET_MODULES:
            original = getattr(attn, module_name)
            if isinstance(original, LoRALinear):
                continue
            if not isinstance(original, nn.Linear):
                raise TypeError(f"{module_name} is not nn.Linear, got {type(original)}")
            lora_mod = LoRALinear(
                original,
                rank=LORA_RANK,
                alpha=LORA_ALPHA,
                dropout=LORA_DROPOUT,
            )
            # Keep LoRA params on same device/dtype as the original projection.
            lora_mod = lora_mod.to(device=original.weight.device, dtype=original.weight.dtype)
            setattr(attn, module_name, lora_mod)


def freeze_and_select_trainable(
    draft: DFlashDraftModel,
    train_fc_norm: bool,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], int, int]:
    for p in draft.parameters():
        p.requires_grad = False

    n_lora = 0
    n_fc_norm = 0
    lora_params: List[nn.Parameter] = []
    fc_norm_params: List[nn.Parameter] = []

    for module in draft.modules():
        if isinstance(module, LoRALinear):
            for p in module.lora_A.parameters():
                p.requires_grad = True
                n_lora += p.numel()
                lora_params.append(p)
            for p in module.lora_B.parameters():
                p.requires_grad = True
                n_lora += p.numel()
                lora_params.append(p)

    for p in draft.fc.parameters():
        p.requires_grad = train_fc_norm
        n_fc_norm += p.numel()
        fc_norm_params.append(p)
    for p in draft.hidden_norm.parameters():
        p.requires_grad = train_fc_norm
        n_fc_norm += p.numel()
        fc_norm_params.append(p)

    return lora_params, fc_norm_params, n_lora, n_fc_norm


def set_fc_norm_trainable(draft: DFlashDraftModel, enabled: bool) -> None:
    for p in draft.fc.parameters():
        p.requires_grad = enabled
    for p in draft.hidden_norm.parameters():
        p.requires_grad = enabled


def select_trainable_full_draft(
    draft: DFlashDraftModel,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], int, int]:
    non_fc_params: List[nn.Parameter] = []
    fc_norm_params: List[nn.Parameter] = []
    n_lora = 0
    n_fc_norm = 0
    for name, p in draft.named_parameters():
        p.requires_grad = True
        if name.startswith("fc.") or name.startswith("hidden_norm."):
            fc_norm_params.append(p)
            n_fc_norm += p.numel()
        else:
            non_fc_params.append(p)
        if ".lora_A." in name or ".lora_B." in name:
            n_lora += p.numel()
    return non_fc_params, fc_norm_params, n_lora, n_fc_norm


def select_trainable_draft_layers_only(
    draft: DFlashDraftModel,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], int, int]:
    for p in draft.parameters():
        p.requires_grad = False
    for p in draft.layers.parameters():
        p.requires_grad = True
    non_fc_params = [p for p in draft.layers.parameters() if p.requires_grad]
    fc_norm_params: List[nn.Parameter] = []
    n_lora = 0
    n_fc_norm = 0
    return non_fc_params, fc_norm_params, n_lora, n_fc_norm


def freeze_fc_hidden_only(draft: DFlashDraftModel) -> Tuple[List[nn.Parameter], int]:
    for p in draft.parameters():
        p.requires_grad = False
    n_fc_norm = 0
    for p in draft.fc.parameters():
        p.requires_grad = True
        n_fc_norm += p.numel()
    for p in draft.hidden_norm.parameters():
        p.requires_grad = True
        n_fc_norm += p.numel()
    trainable = [p for p in draft.parameters() if p.requires_grad]
    return trainable, n_fc_norm


def build_ref_params(
    draft: DFlashDraftModel,
    phase0_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    ref_params: Dict[str, torch.Tensor] = {}
    for name, param in draft.named_parameters():
        keep = param.requires_grad or name.startswith("fc.") or name.startswith("hidden_norm.") or ".lora_" in name
        if not keep:
            continue
        if name in phase0_state_dict:
            ref = phase0_state_dict[name].detach().clone().to(device=param.device, dtype=param.dtype)
        else:
            ref = param.detach().clone()
        ref_params[name] = ref
    return ref_params


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
    loss_ce: float,
    loss_kl: float,
    loss_l2sp: float,
    train_state: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "step": step,
        "model_state_dict": draft.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss_ce": loss_ce,
        "loss_kl": loss_kl,
        "loss_l2sp": loss_l2sp,
        "train_state": train_state or {},
    }
    torch.save(payload, ckpt_path)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    set_seed(SEED)
    ckpt_dir = Path(CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_checkpoint.pt"
    log_path = ckpt_dir / "logs.jsonl"
    eval_detail_path = ckpt_dir / "eval_acceptance_steps.jsonl"
    summary_path = ckpt_dir / "summary.json"

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
    image_token_id = getattr(target.config, "image_token_id", None)
    video_token_id = getattr(target.config, "video_token_id", None)
    for p in lm_head.parameters():
        p.requires_grad = False

    print("[Init] Loading draft config and Phase 0 checkpoint...")
    draft_config = load_draft_config(DRAFT_CONFIG_PATH)
    print(
        f"[Init][DraftConfig] use_mrope={draft_config.dflash_config.get('use_mrope', True)} | "
        f"rope_scaling={getattr(draft_config, 'rope_scaling', None)} | "
        f"rope_parameters={getattr(draft_config, 'rope_parameters', None)}"
    )
    draft = DFlashDraftModel(draft_config).to(device=DEVICE, dtype=torch.bfloat16)
    phase0_path = Path(PHASE0_CKPT)
    if not phase0_path.exists():
        raise FileNotFoundError(f"PHASE0_CKPT not found: {phase0_path}")
    phase0_state = torch.load(phase0_path, map_location="cpu")
    phase0_state_dict = phase0_state.get("model_state_dict", phase0_state)
    draft.load_state_dict(phase0_state_dict, strict=True)
    old_draft = deepcopy(draft).eval()
    for p in old_draft.parameters():
        p.requires_grad = False

    if TRAIN_DRAFT_LAYERS_ONLY:
        print("[Init] Train mode: draft-layers-only (no LoRA, no fc/hidden_norm)")
        non_fc_params, fc_norm_params, n_lora, n_fc_norm = select_trainable_draft_layers_only(draft)
        trainable_params = [*non_fc_params, *fc_norm_params]
    else:
        print("[Init] Injecting LoRA into draft attention...")
        inject_lora(draft)
    if TRAIN_ALL_DRAFT_LAYERS:
        non_fc_params, fc_norm_params, n_lora, n_fc_norm = select_trainable_full_draft(draft)
        trainable_params = [*non_fc_params, *fc_norm_params]
    elif not TRAIN_DRAFT_LAYERS_ONLY:
        non_fc_params, fc_norm_params, n_lora, n_fc_norm = freeze_and_select_trainable(
            draft,
            train_fc_norm=True,
        )
        trainable_params = [*non_fc_params, *fc_norm_params]
    n_total = sum(p.numel() for p in draft.parameters())
    n_trainable = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    print(f"[Phase A] Trainable: {n_trainable:,} / {n_total:,} params")
    print(f"LoRA params:       {n_lora:,}")
    print(f"fc + hidden_norm:  {n_fc_norm:,}")
    print(f"train_draft_layers_only: {TRAIN_DRAFT_LAYERS_ONLY}")
    print(f"train_all_draft:   {TRAIN_ALL_DRAFT_LAYERS}")
    draft.train()

    ref_params = build_ref_params(draft, phase0_state_dict)

    optimizer = AdamW(
        [
            {"params": non_fc_params, "lr": LORA_LR, "weight_decay": WEIGHT_DECAY, "name": "non_fc"},
            {"params": fc_norm_params, "lr": FC_HIDDEN_LR, "weight_decay": WEIGHT_DECAY, "name": "fc_norm"},
        ]
    )
    print("[Init] Loading datasets...")
    image_ds = CocoCaptionDataset(COCO_ROOT, COCO_ANN, processor)
    if IMAGE_TRAIN_SAMPLES > 0 and len(image_ds.items) > IMAGE_TRAIN_SAMPLES:
        rnd = random.Random(SEED)
        rnd.shuffle(image_ds.items)
        image_ds.items = image_ds.items[:IMAGE_TRAIN_SAMPLES]
    eval_image_paths = build_eval_image_paths(image_ds, EVAL_IMAGE_SAMPLES)
    effective_images_per_step = max(1, BATCH_SIZE * ACCUMULATION_STEPS)
    train_max_steps = max(1, math.ceil((len(image_ds) * IMAGE_EPOCHS) / effective_images_per_step))
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=min(WARMUP_STEPS, train_max_steps),
        num_training_steps=train_max_steps,
    )
    print(f"[Init] Eval set: image={len(eval_image_paths)}")
    print(f"[Init] Eval detail log: {eval_detail_path}")
    print(f"[Init] Train image-only=True | image_samples={len(image_ds)} | image_epochs={IMAGE_EPOCHS} | train_steps={train_max_steps}")
    print(f"[Init] Mix ratio image={IMAGE_RATIO:.1f} | blocks/sample={NUM_BLOCKS_PER_SAMPLE}")
    baseline_image_acc = float("nan")

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    image_loader = DataLoader(
        image_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=build_image_collate_fn(pad_id),
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    image_iter = iter(image_loader)

    step = 0
    best_ce = float("inf")
    best_image_acc = float("-inf")
    best_image_gain = float("-inf")
    best_image_step = 0
    no_improve_saves = 0
    grad_norm_value = 0.0
    last_loss_ce = float("nan")
    last_loss_kl = float("nan")
    last_loss_l2sp = float("nan")
    current_image_ratio = IMAGE_RATIO
    current_lambda_kl = LAMBDA_KL_STAGE1
    current_lambda_l2sp = LAMBDA_L2SP_STAGE1

    ckpt_path = latest_checkpoint(ckpt_dir)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")
        draft.load_state_dict(state["model_state_dict"], strict=False)
        train_state = state.get("train_state", {})
        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            step = int(state.get("step", 0))
        except Exception as e:
            print(f"[Warn] Failed to resume optimizer/scheduler, restart states. Error: {e}")
            step = 0
        last_loss_ce = float(state.get("loss_ce", float("nan")))
        last_loss_kl = float(state.get("loss_kl", float("nan")))
        last_loss_l2sp = float(state.get("loss_l2sp", float("nan")))
        baseline_image_acc = float(train_state.get("baseline_image_acc", float("nan")))
        current_image_ratio = IMAGE_RATIO
        current_lambda_kl = LAMBDA_KL_STAGE1
        current_lambda_l2sp = LAMBDA_L2SP_STAGE1
        best_ce = float(train_state.get("best_ce", best_ce))
        best_image_acc = float(train_state.get("best_image_acc", best_image_acc))
        best_image_gain = float(train_state.get("best_image_gain", best_image_gain))
        best_image_step = int(train_state.get("best_image_step", best_image_step))
        no_improve_saves = int(train_state.get("no_improve_saves", no_improve_saves))
        print(
            f"[Resume] step={step}, "
            f"loss_ce={last_loss_ce:.4f}, loss_kl={last_loss_kl:.4f}, loss_l2sp={last_loss_l2sp:.4f}"
        )
        print(
            f"[Resume] single_stage=True | KL={current_lambda_kl:.2f} "
            f"| L2SP={current_lambda_l2sp:.1e} | image_ratio={current_image_ratio:.1f}"
        )
    else:
        print("[Fresh] Starting from step 0")

    if math.isnan(baseline_image_acc):
        print("[Init] Running baseline acceptance eval...")
        baseline_eval = evaluate_acceptance_10_10(
            draft=draft,
            target=target,
            processor=processor,
            image_paths=eval_image_paths,
            step=0,
            detail_log_path=eval_detail_path,
        )
        baseline_image_acc = float(baseline_eval["eval_image_acc_len"])
    print(f"[Baseline] image_acc={baseline_image_acc:.3f}")

    print_gpu_memory("[Info]")
    optimizer.zero_grad(set_to_none=True)

    start_time = time.perf_counter()
    window_start = time.perf_counter()
    window_tokens = 0
    window_loss = 0.0
    window_loss_ce = 0.0
    window_loss_kl = 0.0
    window_loss_l2 = 0.0
    window_micro_steps = 0
    window_anchor_total = 0
    window_anchor_sum_rel = 0.0
    window_anchor_bins = [0, 0, 0, 0, 0]  # 0-20,20-40,40-60,60-80,80-100%
    micro_step = 0
    update_tokens = 0
    update_start = time.perf_counter()
    block_size = int(getattr(draft, "block_size", 16))
    base_context_len = int(BLOCK_CONTEXT_LEN)
    last_context_len = int(BLOCK_CONTEXT_LEN)
    pos_mismatch_warned = False
    last_pos_debug: Dict[str, Any] = {}
    gamma = _auto_loss_gamma(block_size)
    mask_token_id = int(getattr(draft, "mask_token_id", 0))
    context_mode = "full_seq" if USE_FULL_CONTEXT else f"fixed_{base_context_len}"
    print(
        f"[Phase A] Block training enabled | block_size={block_size}, "
        f"context_mode={context_mode}, loss_decay_gamma={gamma:.2f}, mask_token_id={mask_token_id}"
    )
    stage_tag = "Stage"
    stage_span = f"0..{train_max_steps}"
    print(
        f"[{stage_tag}] steps={stage_span} | KL={current_lambda_kl:.2f} | "
        f"L2SP={current_lambda_l2sp:.1e} | image_ratio={current_image_ratio:.1f}"
    )
    if FULL_SEQ_LABELS:
        print("[TrainMode] full-seq labels from target logits (no answer-only bias)")
    else:
        print("[TrainMode] answer-only labels from target logits (assistant span only)")

    def _train_state() -> Dict[str, Any]:
        return {
            "baseline_image_acc": float(baseline_image_acc),
            "current_image_ratio": float(current_image_ratio),
            "current_lambda_kl": float(current_lambda_kl),
            "current_lambda_l2sp": float(current_lambda_l2sp),
            "best_ce": float(best_ce),
            "best_image_acc": float(best_image_acc),
            "best_image_gain": float(best_image_gain),
            "best_image_step": int(best_image_step),
            "no_improve_saves": int(no_improve_saves),
        }

    while step < train_max_steps:
        cur_image_ratio = current_image_ratio
        is_image_batch = True
        batch, image_iter = next_batch(image_loader, image_iter)
        if batch is None:
            continue

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        answer_mask = (labels != IGNORE_INDEX)
        bs, seq_len = input_ids.shape
        if seq_len < 4:
            continue
        context_len = int(seq_len) if USE_FULL_CONTEXT else int(base_context_len)
        last_context_len = context_len

        pixel_values = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")
        if pixel_values is not None:
            pixel_values = pixel_values.to(DEVICE)
            image_grid_thw = image_grid_thw.to(DEVICE)

        target_kwargs = {}
        if pixel_values is not None:
            target_kwargs["pixel_values"] = pixel_values
            target_kwargs["image_grid_thw"] = image_grid_thw

        with torch.no_grad():
            target_out = target(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **target_kwargs,
            )
            target_tokens = torch.argmax(target_out.logits, dim=-1)
            target_tokens = target_tokens.masked_fill(labels == IGNORE_INDEX, IGNORE_INDEX)
            target_hidden = extract_context_feature(target_out.hidden_states, draft.target_layer_ids)
            full_pos_ids = _build_full_rope_position_ids(
                target=target,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
            )
            visual_pos_stats = _inspect_visual_rope_positions(
                input_ids=input_ids,
                attention_mask=attention_mask,
                full_pos_ids=full_pos_ids,
                image_token_id=image_token_id,
                video_token_id=video_token_id,
            )
        loss_ce_num = torch.zeros((), device=DEVICE, dtype=torch.float32)
        loss_ce_den = torch.zeros((), device=DEVICE, dtype=torch.float32)
        loss_kl_num = torch.zeros((), device=DEVICE, dtype=torch.float32)
        loss_kl_den = torch.zeros((), device=DEVICE, dtype=torch.float32)
        valid_tokens_total = 0
        used_blocks = 0
        first_block_debug: Optional[Dict[str, Any]] = None

        for _ in range(NUM_BLOCKS_PER_SAMPLE):
            (
                block_input_ids,
                block_hidden_ctx,
                block_labels,
                block_weights,
                block_valid,
                block_anchor_pos,
                block_anchor_rel,
                block_ctx_start,
                block_last_valid,
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

            valid_anchor_rel = block_anchor_rel[block_anchor_rel >= 0.0]
            if valid_anchor_rel.numel() > 0:
                anchor_vals = valid_anchor_rel.detach().cpu().tolist()
                window_anchor_total += len(anchor_vals)
                window_anchor_sum_rel += float(sum(anchor_vals))
                for r in anchor_vals:
                    rr = min(max(float(r), 0.0), 1.0)
                    bi = min(int(rr * 5.0), 4)
                    window_anchor_bins[bi] += 1

            valid_tokens_total += int((block_labels != IGNORE_INDEX).sum().item())
            used_blocks += 1
            pos_ids = _build_block_position_ids(
                full_pos_ids=full_pos_ids,
                block_valid=block_valid,
                block_ctx_start=block_ctx_start,
                block_anchor_pos=block_anchor_pos,
                block_last_valid=block_last_valid,
                context_len=context_len,
                block_size=block_size,
            )
            pos_align_stats = None
            if POSITION_DEBUG and used_blocks == 1:
                pos_align_stats = _check_block_position_alignment(
                    full_pos_ids=full_pos_ids,
                    block_pos_ids=pos_ids,
                    block_valid=block_valid,
                    block_ctx_start=block_ctx_start,
                    block_anchor_pos=block_anchor_pos,
                    block_last_valid=block_last_valid,
                    context_len=context_len,
                    block_size=block_size,
                )
            draft_use_mrope = bool(getattr(draft, "use_mrope", False))
            pos_ids_for_draft = pos_ids if draft_use_mrope else pos_ids[0]
            noise_len = int(block_input_ids.shape[1])
            target_ctx_len = int(block_hidden_ctx.shape[1])
            pos_len = int(pos_ids_for_draft.shape[-1])
            expected_kv_len = target_ctx_len + noise_len
            pos_match_kv = pos_len == expected_kv_len
            cur_pos_debug = {
                "context_len": int(context_len),
                "target_ctx_len": target_ctx_len,
                "noise_len": noise_len,
                "block_size": int(block_size),
                "pos_len": pos_len,
                "expected_kv_len": int(expected_kv_len),
                "pos_match_kv": bool(pos_match_kv),
                "use_mrope": bool(draft_use_mrope),
            }
            if pos_align_stats is not None:
                cur_pos_debug.update(pos_align_stats)
            if POSITION_DEBUG and used_blocks == 1:
                cur_pos_debug.update(
                    {
                        "visual_tokens_total": int(visual_pos_stats.get("visual_tokens_total", 0)),
                        "samples_with_visual": int(visual_pos_stats.get("samples_with_visual", 0)),
                        "visual_axes_equal_all": visual_pos_stats.get("visual_axes_equal_all", None),
                        "visual_t_unique": int(visual_pos_stats.get("t_unique", 0)),
                        "visual_h_unique": int(visual_pos_stats.get("h_unique", 0)),
                        "visual_w_unique": int(visual_pos_stats.get("w_unique", 0)),
                        "visual_t_range": visual_pos_stats.get("t_range", None),
                        "visual_h_range": visual_pos_stats.get("h_range", None),
                        "visual_w_range": visual_pos_stats.get("w_range", None),
                    }
                )
                first_block_debug = dict(cur_pos_debug)
            last_pos_debug = cur_pos_debug
            if (not pos_match_kv) and (not pos_mismatch_warned):
                print(
                    f"[Warn][PosIDs] pos_len({pos_len}) != expected_kv_len({expected_kv_len}) | "
                    f"context_len={context_len}, block_size={block_size}, target_ctx_len={target_ctx_len}, noise_len={noise_len}"
                )
                pos_mismatch_warned = True

            old_log_probs = None
            if current_lambda_kl > 0.0:
                with torch.no_grad():
                    old_noise = embed_tokens(block_input_ids)
                    old_hidden = old_draft(
                        noise_embedding=old_noise,
                        target_hidden=block_hidden_ctx,
                        position_ids=pos_ids_for_draft,
                    )
                    old_logits = lm_head(old_hidden)
                    old_log_probs = F.log_softmax(old_logits[:, 1:, :], dim=-1)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                noise_emb = embed_tokens(block_input_ids).detach()
                draft_hidden = draft(
                    noise_embedding=noise_emb,
                    target_hidden=block_hidden_ctx,
                    position_ids=pos_ids_for_draft,
                )
                draft_logits = lm_head(draft_hidden)
                vocab_size = draft_logits.shape[-1]
                per_tok_ce = F.cross_entropy(
                    draft_logits[:, 1:, :].reshape(-1, vocab_size),
                    block_labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="none",
                )
                per_tok_ce = per_tok_ce.view(bs, block_size - 1)
                weighted_ce = per_tok_ce * block_weights
                denom = block_weights.sum().float()
                loss_ce_num = loss_ce_num + weighted_ce.sum().float()
                loss_ce_den = loss_ce_den + denom

                if old_log_probs is not None:
                    new_log_probs = F.log_softmax(draft_logits[:, 1:, :], dim=-1)
                    per_tok_kl = F.kl_div(
                        new_log_probs,
                        old_log_probs.exp(),
                        reduction="none",
                    ).sum(dim=-1)
                    weighted_kl = per_tok_kl * block_weights
                    loss_kl_num = loss_kl_num + weighted_kl.sum().float()
                    loss_kl_den = loss_kl_den + denom

        if used_blocks == 0:
            continue
        if first_block_debug is not None:
            last_pos_debug = first_block_debug

        loss_ce = loss_ce_num / loss_ce_den.clamp(min=1e-6)
        if current_lambda_kl > 0.0:
            loss_kl = loss_kl_num / loss_kl_den.clamp(min=1e-6)
        else:
            loss_kl = torch.zeros((), device=DEVICE, dtype=torch.float32)

        loss_l2sp = torch.zeros((), device=DEVICE, dtype=torch.float32)
        if current_lambda_l2sp > 0.0:
            for name, p in draft.named_parameters():
                if p.requires_grad and name in ref_params:
                    diff = p.float() - ref_params[name].float()
                    loss_l2sp = loss_l2sp + torch.sum(diff * diff)

        loss_total = (
            loss_ce
            + current_lambda_kl * loss_kl
            + current_lambda_l2sp * loss_l2sp.to(loss_ce.dtype)
        )
        scaled_loss = loss_total / ACCUMULATION_STEPS
        scaled_loss.backward()
        micro_step += 1

        valid_tokens = valid_tokens_total
        window_tokens += valid_tokens
        window_loss += float(loss_total.item())
        window_loss_ce += float(loss_ce.item())
        window_loss_kl += float(loss_kl.item())
        window_loss_l2 += float(loss_l2sp.item())
        window_micro_steps += 1
        update_tokens += valid_tokens

        if micro_step % ACCUMULATION_STEPS != 0:
            continue

        grad_norm = clip_grad_norm_(draft.parameters(), GRAD_CLIP)
        grad_norm_value = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        optimizer.step()
        scheduler.step()
        for group in optimizer.param_groups:
            group["lr"] = max(group["lr"], LR_MIN)
        optimizer.zero_grad(set_to_none=True)
        step += 1

        step_now = time.perf_counter()
        step_elapsed = max(step_now - update_start, 1e-6)
        step_tps = update_tokens / step_elapsed
        update_start = step_now
        update_tokens = 0
        ce_value = float(loss_ce.item())
        kl_value = float(loss_kl.item())
        l2_value = float(loss_l2sp.item())
        total_value = float(loss_total.item())

        log_entry = {
            "step": int(step),
            "loss": total_value,
            "loss_ce": ce_value,
            "loss_kl": kl_value,
            "loss_l2sp": l2_value,
            "lr": float(optimizer.param_groups[0]["lr"]),
            # Backward-compatible keys:
            "lr_lora": float(optimizer.param_groups[0]["lr"]),
            "lr_fc_hidden": float(optimizer.param_groups[1]["lr"]),
            # Preferred explicit keys:
            "lr_non_fc": float(optimizer.param_groups[0]["lr"]),
            "lr_fc_norm": float(optimizer.param_groups[1]["lr"]),
            "grad_norm": float(grad_norm_value),
            "tokens_per_sec": float(step_tps),
            "is_image_batch": True,
            "image_ratio_used": float(cur_image_ratio),
            "lambda_kl": float(current_lambda_kl),
            "lambda_l2sp": float(current_lambda_l2sp),
            "elapsed_sec": float(time.perf_counter() - start_time),
            "eval_image_acc_len": None,
            "eval_image_tps": None,
        }

        if step % LOG_EVERY == 0:
            elapsed_window = max(time.perf_counter() - window_start, 1e-6)
            tps = window_tokens / elapsed_window
            avg_loss = window_loss / max(1, window_micro_steps)
            avg_ce = window_loss_ce / max(1, window_micro_steps)
            avg_kl = window_loss_kl / max(1, window_micro_steps)
            avg_l2 = window_loss_l2 / max(1, window_micro_steps)
            lora_lr = optimizer.param_groups[0]["lr"]
            fc_lr = optimizer.param_groups[1]["lr"]
            total_elapsed = format_elapsed(time.perf_counter() - start_time)
            print(
                f"Step {step:5d}/{train_max_steps} | "
                f"Loss: {avg_loss:.4f} (CE:{avg_ce:.4f} KL:{avg_kl:.4f} L2:{avg_l2:.4f}) | "
                f"Batch: IMG | LR(non_fc/fc_norm): {lora_lr:.2e}/{fc_lr:.2e} | "
                f"KL/L2: {current_lambda_kl:.2f}/{current_lambda_l2sp:.1e} | "
                f"Grad: {grad_norm_value:.3f} | "
                f"Tokens/sec: {tps:.0f} | Elapsed: {total_elapsed}"
            )
            if POSITION_DEBUG and last_pos_debug and (step % POSITION_DEBUG_EVERY == 0):
                print(
                    f"[PosDebug] step={step} | "
                    f"context_len={last_pos_debug['context_len']} | "
                    f"target_ctx_len={last_pos_debug['target_ctx_len']} | "
                    f"noise_len={last_pos_debug['noise_len']} | "
                    f"block_size={last_pos_debug['block_size']} | "
                    f"pos_len={last_pos_debug['pos_len']} | "
                    f"expected_kv_len={last_pos_debug['expected_kv_len']} | "
                    f"pos_match_kv={last_pos_debug['pos_match_kv']} | "
                    f"use_mrope={last_pos_debug['use_mrope']} | "
                    f"ctx_mismatch={last_pos_debug.get('ctx_token_mismatch', -1)}/"
                    f"{last_pos_debug.get('checked_ctx_tokens', -1)} | "
                    f"noise_mismatch={last_pos_debug.get('noise_token_mismatch', -1)}/"
                    f"{last_pos_debug.get('checked_noise_tokens', -1)} | "
                    f"visual_tokens={last_pos_debug.get('visual_tokens_total', -1)} | "
                    f"visual_samples={last_pos_debug.get('samples_with_visual', -1)} | "
                    f"visual_axes_equal_all={last_pos_debug.get('visual_axes_equal_all', None)} | "
                    f"visual_unique(t/h/w)="
                    f"{last_pos_debug.get('visual_t_unique', -1)}/"
                    f"{last_pos_debug.get('visual_h_unique', -1)}/"
                    f"{last_pos_debug.get('visual_w_unique', -1)} | "
                    f"visual_range_t={last_pos_debug.get('visual_t_range', None)} | "
                    f"visual_range_h={last_pos_debug.get('visual_h_range', None)} | "
                    f"visual_range_w={last_pos_debug.get('visual_w_range', None)}"
                )
            if window_anchor_total > 0:
                mean_anchor_rel = window_anchor_sum_rel / float(window_anchor_total)
                print(
                    f"[AnchorDist] step={step} | n={window_anchor_total} | "
                    f"mean_rel={mean_anchor_rel:.3f} | "
                    f"bins(0-20/20-40/40-60/60-80/80-100)="
                    f"{window_anchor_bins[0]}/{window_anchor_bins[1]}/{window_anchor_bins[2]}/"
                    f"{window_anchor_bins[3]}/{window_anchor_bins[4]}"
                )
            window_tokens = 0
            window_loss = 0.0
            window_loss_ce = 0.0
            window_loss_kl = 0.0
            window_loss_l2 = 0.0
            window_micro_steps = 0
            window_anchor_total = 0
            window_anchor_sum_rel = 0.0
            window_anchor_bins = [0, 0, 0, 0, 0]
            window_start = time.perf_counter()

        if step % SAVE_EVERY == 0:
            eval_acc = evaluate_acceptance_10_10(
                draft=draft,
                target=target,
                processor=processor,
                image_paths=eval_image_paths,
                step=step,
                detail_log_path=eval_detail_path,
            )
            log_entry.update(eval_acc)
            print(
                f"[EvalAcc] step={step} | "
                f"image_acc={eval_acc['eval_image_acc_len']:.3f} | "
                f"image_tps={eval_acc['eval_image_tps']:.1f}"
            )

            eval_image_acc = float(eval_acc["eval_image_acc_len"])
            image_gain_vs_baseline = eval_image_acc - baseline_image_acc
            log_entry["image_gain_vs_baseline"] = float(image_gain_vs_baseline)

            if QUICK_STOP_ENABLED and image_gain_vs_baseline <= IMAGE_ACC_MIN_GAIN:
                print(
                    f"[Quick Stop] step={step} | image_acc gain={image_gain_vs_baseline:.3f} "
                    f"<= {IMAGE_ACC_MIN_GAIN:.3f}. Objective/sampling not effective."
                )
                ckpt_path = ckpt_dir / f"step_{step}.pt"
                save_checkpoint(
                    ckpt_path=ckpt_path,
                    step=step,
                    draft=draft,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_ce=ce_value,
                    loss_kl=kl_value,
                    loss_l2sp=l2_value,
                    train_state=_train_state(),
                )
                append_jsonl(log_path, log_entry)
                break

            ckpt_path = ckpt_dir / f"step_{step}.pt"
            save_checkpoint(
                ckpt_path=ckpt_path,
                step=step,
                draft=draft,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_ce=ce_value,
                loss_kl=kl_value,
                loss_l2sp=l2_value,
                train_state=_train_state(),
            )

            best_ce = min(best_ce, ce_value)
            if image_gain_vs_baseline > (best_image_gain + 1e-6):
                best_image_gain = image_gain_vs_baseline
                best_image_acc = eval_image_acc
                best_image_step = step
                no_improve_saves = 0
                save_checkpoint(
                    ckpt_path=best_ckpt_path,
                    step=step,
                    draft=draft,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_ce=ce_value,
                    loss_kl=kl_value,
                    loss_l2sp=l2_value,
                    train_state=_train_state(),
                )
                print(
                    f"[Best] step={step} | image_acc={best_image_acc:.3f} | "
                    f"gain_vs_baseline={best_image_gain:.3f}"
                )
            else:
                no_improve_saves += 1

            print_gpu_memory("[Info]")

            if no_improve_saves >= EARLY_STOP_PATIENCE:
                print(
                    f"[Early Stop] image_acc did not improve for {EARLY_STOP_PATIENCE} eval saves."
                )
                final_path = ckpt_dir / f"step_{step}_final.pt"
                save_checkpoint(
                    ckpt_path=final_path,
                    step=step,
                    draft=draft,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_ce=ce_value,
                    loss_kl=kl_value,
                    loss_l2sp=l2_value,
                    train_state=_train_state(),
                )
                append_jsonl(log_path, log_entry)
                break

        append_jsonl(log_path, log_entry)

    elapsed_total = time.perf_counter() - start_time
    summary = {
        "phase": "A",
        "total_steps": int(step),
        "best_loss_ce": float(best_ce),
        "best_image_acc": float(best_image_acc) if best_image_acc != float("-inf") else None,
        "best_image_gain_vs_baseline": float(best_image_gain) if best_image_gain != float("-inf") else None,
        "best_image_step": int(best_image_step),
        "train_fc_hidden_only": False,
        "lora_lr": float(LORA_LR),
        "fc_hidden_lr": float(FC_HIDDEN_LR),
        "lambda_kl_stage1": float(LAMBDA_KL_STAGE1),
        "lambda_l2sp_stage1": float(LAMBDA_L2SP_STAGE1),
        "image_ratio": float(IMAGE_RATIO),
        "anchor_stratified_sampling": bool(ANCHOR_STRATIFIED_SAMPLING),
        "anchor_stratified_bins": int(ANCHOR_STRATIFIED_BINS),
        "image_acc_min_gain": float(IMAGE_ACC_MIN_GAIN),
        "baseline_image_acc": float(baseline_image_acc),
        "num_blocks_per_sample": int(NUM_BLOCKS_PER_SAMPLE),
        "current_image_ratio_end": float(current_image_ratio),
        "training_mode": "block_verify_aligned_answer_only",
        "block_size": int(block_size),
        "block_context_len": int(last_context_len),
        "use_full_context": bool(USE_FULL_CONTEXT),
        "loss_decay_gamma": float(gamma),
        "trainable_params": int(n_trainable),
        "total_params": int(n_total),
        "training_time_hours": float(elapsed_total / 3600.0),
        "gpu": torch.cuda.get_device_name(0),
        "target_model": TARGET_MODEL_ID,
        "coco_root": COCO_ROOT,
        "coco_ann": COCO_ANN,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
        f.flush()

    print(f"[Done] Training finished. Checkpoints at: {ckpt_dir}")
    print(f"[Done] Logs: {log_path} | Summary: {summary_path}")


if __name__ == "__main__":
    main()
