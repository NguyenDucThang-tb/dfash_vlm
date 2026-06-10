import json
import math
import random
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
MAX_SEQ_LEN = 16384
BATCH_SIZE = 1
ACCUMULATION_STEPS = 1
MAX_STEPS = 5000
WARMUP_STEPS = 50
LORA_LR = 2e-5
FC_HIDDEN_LR = 5e-7
LR_MIN = 1e-6
NUM_VIDEO_FRAMES = 4
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
VIDEO_RATIO = 1.0
VIDEO_TRAIN_SAMPLES = 2000
VIDEO_EPOCHS = 5
LAMBDA_KL_STAGE1 = 0.0
LAMBDA_L2SP_STAGE1 = 0.0
NUM_BLOCKS_PER_SAMPLE = 4
VAL_LOSS_SAMPLES = 100
VAL_NUM_BLOCKS_PER_SAMPLE = 8
TEST_VIDEO_SAMPLES = 50
EVAL_VIDEO_SAMPLES = TEST_VIDEO_SAMPLES  # Backward-compatible name for acceptance eval.
EVAL_MAX_NEW_TOKENS = 128
BLOCK_CONTEXT_LEN = 512
USE_FULL_CONTEXT = True
LOSS_DECAY_GAMMA = None  # None => auto by block size
LOG_EVERY = 1
SAVE_EVERY = 50
POSITION_DEBUG = False
POSITION_DEBUG_EVERY = 200
STEP_DEBUG = True
STEP_DEBUG_MAX_STEPS = 3
DATASET_DEBUG = True
DATA_LOADER_WORKERS = 0
EARLY_STOP_PATIENCE = 5
QUICK_STOP_ENABLED = False
VIDEO_ACC_MIN_GAIN = 0.05
ANCHOR_STRATIFIED_SAMPLING = False  # paper uses random anchor sampling
ANCHOR_STRATIFIED_BINS = 5
RUN_PERIODIC_BENCHMARK = True
CHECKPOINT_DIR = "/content/drive/MyDrive/dflash_phaseB_msrvtt_video4_mrope"
PHASE0_CKPT = "/content/drive/MyDrive/dflash_phaseA_20k_fullctx_answer128_mrope/best_checkpoint.pt"
VIDEO_RAW_MANIFEST = "/content/phaseB_raw_videos.jsonl"
PHASEB_DATASET_JSONL = "/content/phaseB_target_answers.jsonl"
PHASEB_DATASET_DRIVE_JSONL = ""
TARGET_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DRAFT_MODEL_ID = "z-lab/Qwen3-4B-DFlash-b16"
DRAFT_CONFIG_PATH = "./config.json"

SEED = 42
IGNORE_INDEX = -100
DEVICE = "cuda"
VIDEO_PROMPT = "Describe the main events in this video."


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


def compute_epoch_state(step: int, steps_per_epoch: int, total_epochs: int) -> Dict[str, float]:
    steps_per_epoch = max(1, int(steps_per_epoch))
    completed_steps = max(0, int(step))
    if completed_steps == 0:
        epoch_idx = 0
        step_in_epoch = 0
    else:
        epoch_idx = min((completed_steps - 1) // steps_per_epoch, max(0, int(total_epochs) - 1))
        step_in_epoch = ((completed_steps - 1) % steps_per_epoch) + 1
    epoch_progress = min(1.0, float(step_in_epoch) / float(steps_per_epoch))
    return {
        "epoch": int(epoch_idx + 1),
        "total_epochs": int(total_epochs),
        "step_in_epoch": int(step_in_epoch),
        "steps_per_epoch": int(steps_per_epoch),
        "epoch_progress": float(epoch_progress),
    }


def print_gpu_memory(prefix: str) -> None:
    allocated_gb = torch.cuda.memory_allocated() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"{prefix} GPU memory: {allocated_gb:.1f}GB / {total_gb:.1f}GB")


def step_debug_enabled(step: int) -> bool:
    return bool(STEP_DEBUG) and int(step) < int(STEP_DEBUG_MAX_STEPS)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        f.flush()


def ensure_local_phaseB_dataset() -> None:
    local_path = Path(PHASEB_DATASET_JSONL)
    if local_path.exists():
        return
    if not PHASEB_DATASET_DRIVE_JSONL:
        return
    drive_path = Path(PHASEB_DATASET_DRIVE_JSONL)
    if not drive_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Init] Copying Phase B dataset from Drive to local: {drive_path} -> {local_path}")
    shutil.copy2(drive_path, local_path)


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
    video_grid_thw: Optional[torch.Tensor],
) -> torch.Tensor:
    # Use Qwen3-VL native RoPE index builder when available.
    if hasattr(target, "model") and hasattr(target.model, "get_rope_index"):
        pos_ids, _ = target.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=None,
            video_grid_thw=video_grid_thw,
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

        ctx_slice = full_pos_ids[:, i, ctx_start:anchor]
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
        ctx_slice = full_pos_ids[:, i, ctx_start:anchor]
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

        # Runtime draft starts from a token already produced by target, then predicts later answer tokens.
        # Therefore both anchor and anchor+1 should be inside the answer span when possible.
        answer_pair = (answer_mask[i, :-1] > 0) & (answer_mask[i, 1:] > 0)
        ans_next_idx = torch.nonzero(answer_pair, as_tuple=True)[0]
        candidates = [int(c.item()) for c in ans_next_idx if first_valid <= int(c.item()) <= (last_valid - 1)]

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

        ctx_start = max(first_valid, anchor - context_len)
        block_ctx_start[i] = ctx_start
        ctx_slice = target_hidden[i, ctx_start:anchor]
        ctx_n = int(ctx_slice.shape[0])
        if ctx_n > 0:
            block_hidden_ctx[i, context_len - ctx_n : context_len] = ctx_slice

        max_pred = min(block_size - 1, last_valid - anchor)
        for k in range(max_pred):
            target_pos = anchor + 1 + k
            logit_pos = anchor + k
            if bool(answer_mask[i, target_pos].item()):
                t = target_tokens[i, logit_pos]
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


class PhaseBTargetAnswerDataset(Dataset):
    def __init__(self, dataset_jsonl: str, processor: AutoProcessor):
        self.processor = processor
        self.dataset_path = Path(dataset_jsonl)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"PHASEB_DATASET_JSONL not found: {self.dataset_path}")

        self.items: List[Dict[str, str]] = []
        with self.dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                video_path = str(row.get("video_path", "")).strip()
                answer = str(row.get("answer", "")).strip()
                prompt = str(row.get("prompt", VIDEO_PROMPT)).strip() or VIDEO_PROMPT
                if video_path and answer and Path(video_path).exists():
                    self.items.append({"video_path": video_path, "answer": answer, "prompt": prompt})

        if not self.items:
            raise RuntimeError("No valid Phase B cached samples found.")
        self.debug_emitted = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        row = self.items[idx]
        video_path = Path(row["video_path"])
        answer = row["answer"]
        prompt = row["prompt"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path), "num_frames": NUM_VIDEO_FRAMES},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path), "num_frames": NUM_VIDEO_FRAMES},
                    {"type": "text", "text": prompt},
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
        full_video_token_count = int((full_input_ids == getattr(self.processor, "video_token_id", 151656)).sum().item())

        full_seq_len = int(full_input_ids.shape[1])
        if full_seq_len <= MAX_SEQ_LEN:
            seq_start = 0
        else:
            # Keep the answer span in-window. For answer-only training we do not
            # need the entire multimodal prefix, but we must retain the tail
            # that contains the generated answer tokens.
            seq_start = max(0, min(answer_start, full_seq_len - MAX_SEQ_LEN))
        seq_end = min(full_seq_len, seq_start + MAX_SEQ_LEN)
        input_ids = full_input_ids[:, seq_start:seq_end]
        attention_mask = full_attention_mask[:, seq_start:seq_end]
        answer_start = max(0, answer_start - seq_start)
        kept_video_token_count = int((input_ids == getattr(self.processor, "video_token_id", 151656)).sum().item())
        if input_ids.shape[1] < 4:
            if DATASET_DEBUG and self.debug_emitted < 8:
                print(f"[DatasetDebug] skip idx={idx} reason=short_seq seq_len={int(input_ids.shape[1])}")
                self.debug_emitted += 1
            return None
        if (not FULL_SEQ_LABELS) and answer_start >= int(input_ids.shape[1] - 1):
            if DATASET_DEBUG and self.debug_emitted < 8:
                print(
                    f"[DatasetDebug] skip idx={idx} reason=answer_start_oob "
                    f"answer_start={answer_start} seq_len={int(input_ids.shape[1])} "
                    f"full_seq_len={full_seq_len} seq_start={seq_start}"
                )
                self.debug_emitted += 1
            return None

        # Some processor code paths drop video tensors once an assistant message is appended.
        # Reuse multimodal tensors from the user-only encoding in that case.
        pixel_values_videos = encoded.get("pixel_values_videos")
        video_grid_thw = encoded.get("video_grid_thw")
        if pixel_values_videos is None or video_grid_thw is None:
            pixel_values_videos = user_encoded.get("pixel_values_videos")
            video_grid_thw = user_encoded.get("video_grid_thw")
        if pixel_values_videos is None or video_grid_thw is None:
            if DATASET_DEBUG and self.debug_emitted < 8:
                print(f"[DatasetDebug] skip idx={idx} reason=missing_video_tensors path={video_path.name}")
                self.debug_emitted += 1
            return None
        if full_seq_len > MAX_SEQ_LEN and 0 < kept_video_token_count < full_video_token_count:
            if DATASET_DEBUG and self.debug_emitted < 8:
                print(
                    f"[DatasetDebug] skip idx={idx} reason=partial_video_crop "
                    f"full_seq_len={full_seq_len} seq_start={seq_start} seq_end={seq_end} "
                    f"video_tokens_kept={kept_video_token_count}/{full_video_token_count}"
                )
                self.debug_emitted += 1
            return None
        if DATASET_DEBUG and self.debug_emitted < 8:
            print(
                f"[DatasetDebug] keep idx={idx} seq_len={int(input_ids.shape[1])} "
                f"full_seq_len={full_seq_len} seq_start={seq_start} "
                f"answer_start={answer_start} video_shape={tuple(pixel_values_videos.shape)} "
                f"grid_shape={tuple(video_grid_thw.shape)}"
            )
            self.debug_emitted += 1
        return {
            "input_ids": input_ids.squeeze(0).long(),
            "attention_mask": attention_mask.squeeze(0).long(),
            "answer_start": int(answer_start),
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }


def build_video_collate_fn(pad_token_id: int):
    def collate_fn(samples: List[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
        max_len = max(s["input_ids"].shape[0] for s in samples)
        bs = len(samples)
        input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        labels = torch.full((bs, max_len), IGNORE_INDEX, dtype=torch.long)
        pixel_values_videos_list = []
        video_grid_list = []
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
            pixel_values_videos_list.append(sample["pixel_values_videos"])
            video_grid_list.append(sample["video_grid_thw"])
        try:
            pixel_values_videos = torch.cat(pixel_values_videos_list, dim=0)
            video_grid_thw = torch.cat(video_grid_list, dim=0)
        except RuntimeError:
            return None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

    return collate_fn


def build_eval_video_paths(video_ds: "PhaseBTargetAnswerDataset", limit: int) -> List[Path]:
    paths: List[Path] = []
    for row in video_ds.items:
        paths.append(Path(row["video_path"]))
        if len(paths) >= limit:
            break
    return paths


def build_eval_video_paths_from_items(items: List[Dict[str, str]], limit: int) -> List[Path]:
    paths: List[Path] = []
    for row in items:
        paths.append(Path(row["video_path"]))
        if len(paths) >= limit:
            break
    return paths


@torch.no_grad()
def evaluate_validation_ce(
    *,
    draft: DFlashDraftModel,
    target: Qwen3VLForConditionalGeneration,
    val_loader: DataLoader,
    embed_tokens: nn.Module,
    lm_head: nn.Module,
    mask_token_id: int,
    block_size: int,
    context_len: int,
    gamma: float,
    num_blocks_per_sample: int,
) -> Dict[str, float]:
    was_training = draft.training
    draft.eval()
    target.eval()

    total_num = torch.zeros((), device=DEVICE, dtype=torch.float32)
    total_den = torch.zeros((), device=DEVICE, dtype=torch.float32)
    total_blocks = 0
    total_batches = 0

    for batch in val_loader:
        if batch is None:
            continue
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        answer_mask = labels != IGNORE_INDEX
        if input_ids.shape[1] < 4:
            continue

        pixel_values_videos = batch.get("pixel_values_videos")
        video_grid_thw = batch.get("video_grid_thw")
        if pixel_values_videos is None or video_grid_thw is None:
            continue
        pixel_values_videos = pixel_values_videos.to(DEVICE)
        video_grid_thw = video_grid_thw.to(DEVICE)

        target_out = target(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            output_hidden_states=True,
        )
        target_tokens = torch.argmax(target_out.logits, dim=-1)
        target_hidden = extract_context_feature(target_out.hidden_states, draft.target_layer_ids)
        full_pos_ids = _build_full_rope_position_ids(
            target=target,
            input_ids=input_ids,
            attention_mask=attention_mask,
            video_grid_thw=video_grid_thw,
        )

        for _ in range(num_blocks_per_sample):
            (
                block_input_ids,
                block_hidden_ctx,
                block_labels,
                block_weights,
                block_valid,
                block_anchor_pos,
                _block_anchor_rel,
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

            pos_ids = _build_block_position_ids(
                full_pos_ids=full_pos_ids,
                block_valid=block_valid,
                block_ctx_start=block_ctx_start,
                block_anchor_pos=block_anchor_pos,
                block_last_valid=block_last_valid,
                context_len=context_len,
                block_size=block_size,
            )
            pos_ids_for_draft = pos_ids if bool(getattr(draft, "use_mrope", False)) else pos_ids[0]

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
                ).view(block_labels.shape)
                total_num = total_num + (per_tok_ce * block_weights).sum().float()
                total_den = total_den + block_weights.sum().float()
                total_blocks += 1
        total_batches += 1

    if was_training:
        draft.train()

    if float(total_den.item()) <= 0.0:
        return {"val_loss_ce": float("nan"), "val_blocks": int(total_blocks), "val_batches": int(total_batches)}
    return {
        "val_loss_ce": float((total_num / total_den.clamp(min=1e-6)).item()),
        "val_blocks": int(total_blocks),
        "val_batches": int(total_batches),
    }


@torch.no_grad()
def evaluate_video_acceptance(
    *,
    draft: DFlashDraftModel,
    target: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    video_paths: List[Path],
    step: Optional[int] = None,
    detail_log_path: Optional[Path] = None,
) -> Dict[str, float]:
    draft.eval()
    stop_ids = [processor.tokenizer.eos_token_id] if processor.tokenizer.eos_token_id is not None else None

    video_accept: List[int] = []
    video_tokens = 0
    video_decode_time = 0.0
    per_answer_rows: List[Dict[str, Any]] = []

    for idx, video_path in enumerate(video_paths):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path), "num_frames": NUM_VIDEO_FRAMES},
                    {"type": "text", "text": VIDEO_PROMPT},
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
        pixel_values_videos = encoded.get("pixel_values_videos")
        video_grid_thw = encoded.get("video_grid_thw")
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(DEVICE)
            video_grid_thw = video_grid_thw.to(DEVICE)
        stats = dflash_generate(
            draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            stop_token_ids=stop_ids,
            temperature=0.0,
            return_stats=True,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        video_accept.extend(stats.acceptance_lengths)
        video_tokens += int(stats.num_output_tokens)
        video_decode_time += float(stats.time_per_output_token) * int(stats.num_output_tokens)
        acc_seq = [int(x) for x in stats.acceptance_lengths]
        per_answer_rows.append(
            {
                "step": int(step) if step is not None else None,
                "sample_type": "video",
                "sample_index": int(idx),
                "video_path": str(video_path),
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

    video_mean = float("nan") if not video_accept else float(torch.tensor(video_accept, dtype=torch.float32).mean().item())
    video_tps = float("nan") if video_decode_time <= 0 else float(video_tokens / video_decode_time)
    return {
        "eval_video_acc_len": video_mean,
        "eval_video_tps": video_tps,
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

    print("[Init] Loading draft model...")
    if PHASE0_CKPT:
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
        print(f"[Init] Loaded Phase 0 checkpoint: {phase0_path}")
    else:
        print(f"[Init] Loading original draft model: {DRAFT_MODEL_ID}")
        draft = DFlashDraftModel.from_pretrained(
            DRAFT_MODEL_ID,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(DEVICE)
        phase0_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in draft.state_dict().items()}
    old_draft = deepcopy(draft).eval() if LAMBDA_KL_STAGE1 > 0.0 else None
    if old_draft is not None:
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
    print(f"[Phase B] Trainable: {n_trainable:,} / {n_total:,} params")
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
    ensure_local_phaseB_dataset()
    video_ds = PhaseBTargetAnswerDataset(PHASEB_DATASET_JSONL, processor)
    # Keep file order for append-based splits:
    # first VIDEO_TRAIN_SAMPLES rows = train, appended rows = val/test.
    all_items = list(video_ds.items)
    train_limit = min(len(all_items), VIDEO_TRAIN_SAMPLES) if VIDEO_TRAIN_SAMPLES > 0 else len(all_items)
    train_items = all_items[:train_limit]
    val_start = train_limit
    val_end = val_start + VAL_LOSS_SAMPLES
    test_end = val_end + TEST_VIDEO_SAMPLES
    val_items = all_items[val_start:val_end]
    test_items = all_items[val_end:test_end]
    if len(val_items) < VAL_LOSS_SAMPLES:
        print(
            f"[Warn] Only {len(val_items)} validation-loss videos available after "
            f"reserving {len(train_items)} train videos."
        )
    if len(test_items) < TEST_VIDEO_SAMPLES:
        print(
            f"[Warn] Only {len(test_items)} test-acceptance videos available after "
            f"reserving {len(train_items)} train + {len(val_items)} val videos."
        )
    video_ds.items = train_items
    val_ds = PhaseBTargetAnswerDataset(PHASEB_DATASET_JSONL, processor)
    val_ds.items = val_items
    eval_video_paths = build_eval_video_paths_from_items(test_items, TEST_VIDEO_SAMPLES)
    if not eval_video_paths:
        raise RuntimeError(
            "No test acceptance videos available. Generate more cached samples than "
            f"VIDEO_TRAIN_SAMPLES + VAL_LOSS_SAMPLES = {VIDEO_TRAIN_SAMPLES + VAL_LOSS_SAMPLES}, "
            "or lower one of these values."
        )
    effective_videos_per_step = max(1, BATCH_SIZE * ACCUMULATION_STEPS)
    steps_per_epoch = max(1, math.ceil(len(video_ds) / effective_videos_per_step))
    train_max_steps = max(1, math.ceil((len(video_ds) * VIDEO_EPOCHS) / effective_videos_per_step))
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=min(WARMUP_STEPS, train_max_steps),
        num_training_steps=train_max_steps,
    )
    print(
        f"[Init] Split: train={len(video_ds)} | val_loss={len(val_items)} | "
        f"test_acceptance={len(eval_video_paths)} | split_mode=file_order_append"
    )
    print(f"[Init] Eval set: video={len(eval_video_paths)} split=test held-out=True")
    print(f"[Init] Eval detail log: {eval_detail_path}")
    print(f"[Init] Phase B cached dataset: {PHASEB_DATASET_JSONL}")
    print(
        f"[Init] Train video-only=True | video_samples={len(video_ds)} | "
        f"val_loss_samples={len(val_items)} | test_acceptance_samples={len(eval_video_paths)} | "
        f"video_epochs={VIDEO_EPOCHS} | steps/epoch={steps_per_epoch} | train_steps={train_max_steps}"
    )
    print(f"[Init] Mix ratio video={VIDEO_RATIO:.1f} | blocks/sample={NUM_BLOCKS_PER_SAMPLE}")
    baseline_video_acc = float("nan")

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "collate_fn": build_video_collate_fn(pad_id),
        "drop_last": False,
        "num_workers": DATA_LOADER_WORKERS,
        "pin_memory": True,
    }
    if DATA_LOADER_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    video_loader = DataLoader(video_ds, **loader_kwargs)
    video_iter = iter(video_loader)
    val_loader = None
    if len(val_ds) > 0:
        val_loader_kwargs = dict(loader_kwargs)
        val_loader_kwargs["shuffle"] = False
        val_loader_kwargs["drop_last"] = False
        val_loader = DataLoader(val_ds, **val_loader_kwargs)

    step = 0
    best_ce = float("inf")
    best_video_acc = float("-inf")
    best_video_gain = float("-inf")
    best_video_step = 0
    no_improve_saves = 0
    grad_norm_value = 0.0
    last_loss_ce = float("nan")
    last_loss_kl = float("nan")
    last_loss_l2sp = float("nan")
    current_video_ratio = VIDEO_RATIO
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
        baseline_video_acc = float(train_state.get("baseline_video_acc", float("nan")))
        current_video_ratio = VIDEO_RATIO
        current_lambda_kl = LAMBDA_KL_STAGE1
        current_lambda_l2sp = LAMBDA_L2SP_STAGE1
        best_ce = float(train_state.get("best_ce", best_ce))
        best_video_acc = float(train_state.get("best_video_acc", best_video_acc))
        best_video_gain = float(train_state.get("best_video_gain", best_video_gain))
        best_video_step = int(train_state.get("best_video_step", best_video_step))
        no_improve_saves = int(train_state.get("no_improve_saves", no_improve_saves))
        resume_epoch_state = compute_epoch_state(step, steps_per_epoch, VIDEO_EPOCHS)
        print(
            f"[Resume] epoch={int(resume_epoch_state['epoch'])}/{int(resume_epoch_state['total_epochs'])} "
            f"({int(resume_epoch_state['step_in_epoch'])}/{int(resume_epoch_state['steps_per_epoch'])}) | "
            f"step={step} | "
            f"loss_ce={last_loss_ce:.4f}, loss_kl={last_loss_kl:.4f}, loss_l2sp={last_loss_l2sp:.4f}"
        )
        print(
            f"[Resume] single_stage=True | KL={current_lambda_kl:.2f} "
            f"| L2SP={current_lambda_l2sp:.1e} | video_ratio={current_video_ratio:.1f}"
        )
    else:
        print("[Fresh] Starting from step 0")

    if math.isnan(baseline_video_acc):
        print("[Init] Running baseline acceptance eval...")
        baseline_eval = evaluate_video_acceptance(
            draft=draft,
            target=target,
            processor=processor,
            video_paths=eval_video_paths,
            step=0,
            detail_log_path=eval_detail_path,
        )
        baseline_video_acc = float(baseline_eval["eval_video_acc_len"])
    print(f"[Baseline] video_acc={baseline_video_acc:.3f}")

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
        f"[Phase B] Block training enabled | block_size={block_size}, "
        f"context_mode={context_mode}, loss_decay_gamma={gamma:.2f}, mask_token_id={mask_token_id}"
    )
    stage_tag = "Stage"
    stage_span = f"0..{train_max_steps}"
    print(
        f"[{stage_tag}] steps={stage_span} | KL={current_lambda_kl:.2f} | "
        f"L2SP={current_lambda_l2sp:.1e} | video_ratio={current_video_ratio:.1f}"
    )
    if FULL_SEQ_LABELS:
        print("[TrainMode] full-seq labels from cached target-generated continuations")
    else:
        print("[TrainMode] answer-only labels from cached target-generated continuations")

    def _train_state() -> Dict[str, Any]:
        return {
            "baseline_video_acc": float(baseline_video_acc),
            "current_video_ratio": float(current_video_ratio),
            "current_lambda_kl": float(current_lambda_kl),
            "current_lambda_l2sp": float(current_lambda_l2sp),
            "best_ce": float(best_ce),
            "best_video_acc": float(best_video_acc),
            "best_video_gain": float(best_video_gain),
            "best_video_step": int(best_video_step),
            "no_improve_saves": int(no_improve_saves),
            "video_epochs": int(VIDEO_EPOCHS),
            "steps_per_epoch": int(steps_per_epoch),
            "train_max_steps": int(train_max_steps),
            "effective_videos_per_step": int(effective_videos_per_step),
            "train_samples": int(len(video_ds)),
            "val_loss_samples": int(len(val_items)),
            "test_acceptance_samples": int(len(eval_video_paths)),
            "split_mode": "file_order_append",
        }

    while step < train_max_steps:
        cur_video_ratio = current_video_ratio
        debug_this_step = step_debug_enabled(step)
        if debug_this_step:
            print(f"[StepDebug] step={step + 1} | begin fetch batch")
        batch, video_iter = next_batch(video_loader, video_iter)
        if batch is None:
            if debug_this_step:
                print(f"[StepDebug] step={step + 1} | batch is None")
            continue
        if debug_this_step:
            print(f"[StepDebug] step={step + 1} | fetched batch keys={sorted(batch.keys())}")

        if debug_this_step:
            print(f"[StepDebug] step={step + 1} | move ids/masks to {DEVICE}")
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        answer_mask = (labels != IGNORE_INDEX)
        bs, seq_len = input_ids.shape
        if debug_this_step:
            print(f"[StepDebug] step={step + 1} | bs={bs} seq_len={seq_len}")
        if seq_len < 4:
            if debug_this_step:
                print(f"[StepDebug] step={step + 1} | skipped seq_len<4")
            continue
        context_len = int(seq_len) if USE_FULL_CONTEXT else int(base_context_len)
        last_context_len = context_len

        pixel_values_videos = batch.get("pixel_values_videos")
        video_grid_thw = batch.get("video_grid_thw")
        if pixel_values_videos is not None:
            if debug_this_step:
                print(
                    f"[StepDebug] step={step + 1} | video tensor cpu shape={tuple(pixel_values_videos.shape)} "
                    f"grid shape={tuple(video_grid_thw.shape) if video_grid_thw is not None else None}"
                )
            pixel_values_videos = pixel_values_videos.to(DEVICE)
            video_grid_thw = video_grid_thw.to(DEVICE)
            if debug_this_step:
                print(
                    f"[StepDebug] step={step + 1} | moved video tensor to gpu shape={tuple(pixel_values_videos.shape)} "
                    f"grid shape={tuple(video_grid_thw.shape)}"
                )
        if pixel_values_videos is None or video_grid_thw is None:
            if debug_this_step:
                print(f"[StepDebug] step={step + 1} | missing video tensors")
            continue

        target_kwargs = {}
        target_kwargs["pixel_values_videos"] = pixel_values_videos
        target_kwargs["video_grid_thw"] = video_grid_thw

        with torch.no_grad():
            if debug_this_step:
                print(f"[StepDebug] step={step + 1} | target forward begin")
            target_out = target(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **target_kwargs,
            )
            if debug_this_step:
                print(
                    f"[StepDebug] step={step + 1} | target forward done logits={tuple(target_out.logits.shape)}"
                )
            target_tokens = torch.argmax(target_out.logits, dim=-1)
            target_hidden = extract_context_feature(target_out.hidden_states, draft.target_layer_ids)
            full_pos_ids = _build_full_rope_position_ids(
                target=target,
                input_ids=input_ids,
                attention_mask=attention_mask,
                video_grid_thw=video_grid_thw,
            )
            if debug_this_step:
                print(
                    f"[StepDebug] step={step + 1} | target_hidden={tuple(target_hidden.shape)} "
                    f"full_pos_ids={tuple(full_pos_ids.shape)}"
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

        for block_idx in range(NUM_BLOCKS_PER_SAMPLE):
            if debug_this_step:
                print(f"[StepDebug] step={step + 1} | block={block_idx + 1}/{NUM_BLOCKS_PER_SAMPLE} begin")
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
                if debug_this_step:
                    print(f"[StepDebug] step={step + 1} | block={block_idx + 1} no valid anchors")
                continue
            if debug_this_step:
                print(
                    f"[StepDebug] step={step + 1} | block={block_idx + 1} "
                    f"input={tuple(block_input_ids.shape)} ctx={tuple(block_hidden_ctx.shape)} "
                    f"labels={tuple(block_labels.shape)}"
                )

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
                if old_draft is None:
                    raise RuntimeError("KL loss is enabled but old_draft was not initialized.")
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
            if debug_this_step:
                print(f"[StepDebug] step={step + 1} | block={block_idx + 1} draft forward done")

        if used_blocks == 0:
            if debug_this_step:
                print(f"[StepDebug] step={step + 1} | used_blocks=0")
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
        if debug_this_step:
            print(
                f"[StepDebug] step={step + 1} | losses ce={float(loss_ce.item()):.4f} "
                f"kl={float(loss_kl.item()):.4f} l2={float(loss_l2sp.item()):.4f} "
                f"total={float(loss_total.item()):.4f}"
            )
            print(f"[StepDebug] step={step + 1} | backward begin")
        scaled_loss.backward()
        if debug_this_step:
            print(f"[StepDebug] step={step + 1} | backward done")
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
        if debug_this_step:
            print(f"[StepDebug] step={step + 1} | optimizer step begin grad_norm={grad_norm_value:.4f}")
        optimizer.step()
        scheduler.step()
        for group in optimizer.param_groups:
            group["lr"] = max(group["lr"], LR_MIN)
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if debug_this_step:
            print(f"[StepDebug] step={step} | optimizer step done")

        step_now = time.perf_counter()
        step_elapsed = max(step_now - update_start, 1e-6)
        step_tps = update_tokens / step_elapsed
        update_start = step_now
        update_tokens = 0
        ce_value = float(loss_ce.item())
        kl_value = float(loss_kl.item())
        l2_value = float(loss_l2sp.item())
        total_value = float(loss_total.item())

        has_fc_norm_group = len(fc_norm_params) > 0
        epoch_state = compute_epoch_state(step, steps_per_epoch, VIDEO_EPOCHS)
        log_entry = {
            "step": int(step),
            "epoch": int(epoch_state["epoch"]),
            "total_epochs": int(epoch_state["total_epochs"]),
            "step_in_epoch": int(epoch_state["step_in_epoch"]),
            "steps_per_epoch": int(epoch_state["steps_per_epoch"]),
            "epoch_progress": float(epoch_state["epoch_progress"]),
            "loss": total_value,
            "loss_ce": ce_value,
            "loss_kl": kl_value,
            "loss_l2sp": l2_value,
            "lr": float(optimizer.param_groups[0]["lr"]),
            # Backward-compatible keys:
            "lr_lora": float(optimizer.param_groups[0]["lr"]),
            "lr_fc_hidden": float(optimizer.param_groups[1]["lr"]) if has_fc_norm_group else None,
            # Preferred explicit keys:
            "lr_non_fc": float(optimizer.param_groups[0]["lr"]),
            "lr_fc_norm": float(optimizer.param_groups[1]["lr"]) if has_fc_norm_group else None,
            "grad_norm": float(grad_norm_value),
            "tokens_per_sec": float(step_tps),
            "is_video_batch": True,
            "video_ratio_used": float(cur_video_ratio),
            "lambda_kl": float(current_lambda_kl),
            "lambda_l2sp": float(current_lambda_l2sp),
            "elapsed_sec": float(time.perf_counter() - start_time),
            "val_loss_ce": None,
            "val_blocks": None,
            "val_batches": None,
            "eval_video_acc_len": None,
            "eval_video_tps": None,
        }

        if step % LOG_EVERY == 0:
            elapsed_window = max(time.perf_counter() - window_start, 1e-6)
            tps = window_tokens / elapsed_window
            avg_loss = window_loss / max(1, window_micro_steps)
            avg_ce = window_loss_ce / max(1, window_micro_steps)
            avg_kl = window_loss_kl / max(1, window_micro_steps)
            avg_l2 = window_loss_l2 / max(1, window_micro_steps)
            lora_lr = optimizer.param_groups[0]["lr"]
            fc_lr = optimizer.param_groups[1]["lr"] if has_fc_norm_group else None
            total_elapsed = format_elapsed(time.perf_counter() - start_time)
            lr_display = f"{lora_lr:.2e}/disabled" if fc_lr is None else f"{lora_lr:.2e}/{fc_lr:.2e}"
            epoch_display = (
                f"Epoch {int(epoch_state['epoch'])}/{int(epoch_state['total_epochs'])} "
                f"({int(epoch_state['step_in_epoch'])}/{int(epoch_state['steps_per_epoch'])})"
            )
            print(
                f"{epoch_display} | Step {step:5d}/{train_max_steps} | "
                f"Loss: {avg_loss:.4f} (CE:{avg_ce:.4f} KL:{avg_kl:.4f} L2:{avg_l2:.4f}) | "
                f"Batch: VID | LR(non_fc/fc_norm): {lr_display} | "
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
            print(
                f"[Checkpoint] epoch={int(epoch_state['epoch'])}/{int(epoch_state['total_epochs'])} "
                f"step={step} | saved={ckpt_path}"
            )
            print_gpu_memory("[Info]")
            if RUN_PERIODIC_BENCHMARK:
                if val_loader is not None:
                    val_metrics = evaluate_validation_ce(
                        draft=draft,
                        target=target,
                        val_loader=val_loader,
                        embed_tokens=embed_tokens,
                        lm_head=lm_head,
                        mask_token_id=mask_token_id,
                        block_size=block_size,
                        context_len=last_context_len,
                        gamma=gamma,
                        num_blocks_per_sample=VAL_NUM_BLOCKS_PER_SAMPLE,
                    )
                    log_entry.update(val_metrics)
                    print(
                        f"[ValLoss] epoch={int(epoch_state['epoch'])}/{int(epoch_state['total_epochs'])} "
                        f"step={step} | val_ce={val_metrics['val_loss_ce']:.4f} | "
                        f"blocks={val_metrics['val_blocks']} | batches={val_metrics['val_batches']}"
                    )

                eval_acc = evaluate_video_acceptance(
                    draft=draft,
                    target=target,
                    processor=processor,
                    video_paths=eval_video_paths,
                    step=step,
                    detail_log_path=eval_detail_path,
                )
                log_entry.update(eval_acc)
                print(
                    f"[EvalAcc] epoch={int(epoch_state['epoch'])}/{int(epoch_state['total_epochs'])} "
                    f"step={step} | "
                    f"video_acc={eval_acc['eval_video_acc_len']:.3f} | "
                    f"video_tps={eval_acc['eval_video_tps']:.1f}"
                )

                eval_video_acc = float(eval_acc["eval_video_acc_len"])
                video_gain_vs_baseline = eval_video_acc - baseline_video_acc
                log_entry["video_gain_vs_baseline"] = float(video_gain_vs_baseline)

                if QUICK_STOP_ENABLED and video_gain_vs_baseline <= VIDEO_ACC_MIN_GAIN:
                    print(
                        f"[Quick Stop] epoch={int(epoch_state['epoch'])}/{int(epoch_state['total_epochs'])} "
                        f"step={step} | video_acc gain={video_gain_vs_baseline:.3f} "
                        f"<= {VIDEO_ACC_MIN_GAIN:.3f}. Objective/sampling not effective."
                    )
                    append_jsonl(log_path, log_entry)
                    break

                best_ce = min(best_ce, ce_value)
                if video_gain_vs_baseline > (best_video_gain + 1e-6):
                    best_video_gain = video_gain_vs_baseline
                    best_video_acc = eval_video_acc
                    best_video_step = step
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
                        f"[Best] epoch={int(epoch_state['epoch'])}/{int(epoch_state['total_epochs'])} "
                        f"step={step} | video_acc={best_video_acc:.3f} | "
                        f"gain_vs_baseline={best_video_gain:.3f}"
                    )
                else:
                    no_improve_saves += 1

                if no_improve_saves >= EARLY_STOP_PATIENCE:
                    print(
                        f"[Early Stop] epoch={int(epoch_state['epoch'])}/{int(epoch_state['total_epochs'])} "
                        f"step={step} | video_acc did not improve for {EARLY_STOP_PATIENCE} eval saves."
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
    final_epoch_state = compute_epoch_state(step, steps_per_epoch, VIDEO_EPOCHS)
    summary = {
        "phase": "B",
        "total_steps": int(step),
        "video_epochs": int(VIDEO_EPOCHS),
        "final_epoch": int(final_epoch_state["epoch"]),
        "final_step_in_epoch": int(final_epoch_state["step_in_epoch"]),
        "steps_per_epoch": int(steps_per_epoch),
        "train_max_steps": int(train_max_steps),
        "video_train_samples": int(len(video_ds)),
        "val_loss_samples": int(len(val_items)),
        "val_num_blocks_per_sample": int(VAL_NUM_BLOCKS_PER_SAMPLE),
        "test_acceptance_samples": int(len(eval_video_paths)),
        "eval_video_samples": int(len(eval_video_paths)),
        "eval_split": "test",
        "eval_heldout": True,
        "split_mode": "file_order_append",
        "effective_videos_per_step": int(effective_videos_per_step),
        "batch_size": int(BATCH_SIZE),
        "accumulation_steps": int(ACCUMULATION_STEPS),
        "best_loss_ce": float(best_ce),
        "best_video_acc": float(best_video_acc) if best_video_acc != float("-inf") else None,
        "best_video_gain_vs_baseline": float(best_video_gain) if best_video_gain != float("-inf") else None,
        "best_video_step": int(best_video_step),
        "train_fc_hidden_only": False,
        "lora_lr": float(LORA_LR),
        "fc_hidden_lr": float(FC_HIDDEN_LR),
        "lambda_kl_stage1": float(LAMBDA_KL_STAGE1),
        "lambda_l2sp_stage1": float(LAMBDA_L2SP_STAGE1),
        "video_ratio": float(VIDEO_RATIO),
        "anchor_stratified_sampling": bool(ANCHOR_STRATIFIED_SAMPLING),
        "anchor_stratified_bins": int(ANCHOR_STRATIFIED_BINS),
        "video_acc_min_gain": float(VIDEO_ACC_MIN_GAIN),
        "baseline_video_acc": float(baseline_video_acc),
        "num_blocks_per_sample": int(NUM_BLOCKS_PER_SAMPLE),
        "data_loader_workers": int(DATA_LOADER_WORKERS),
        "current_video_ratio_end": float(current_video_ratio),
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
        "draft_model": DRAFT_MODEL_ID,
        "draft_init": "phase0_checkpoint" if PHASE0_CKPT else "original_draft_model",
        "phase0_ckpt": PHASE0_CKPT,
        "phaseB_dataset_jsonl": PHASEB_DATASET_JSONL,
        "phaseB_dataset_drive_jsonl": PHASEB_DATASET_DRIVE_JSONL,
        "video_raw_manifest": VIDEO_RAW_MANIFEST,
        "num_video_frames": int(NUM_VIDEO_FRAMES),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
        f.flush()

    print(f"[Done] Training finished. Checkpoints at: {ckpt_dir}")
    print(f"[Done] Logs: {log_path} | Summary: {summary_path}")


if __name__ == "__main__":
    main()
