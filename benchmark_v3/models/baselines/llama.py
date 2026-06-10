"""
Baseline: Llama-3.1-8B-Vision-378 (qresearch)
Built on SigLIP + Llama-3.1-8B-Instruct, image-only natively.
Video support: query từng frame, tổng hợp kết quả.
Frame sampling/pruning được xử lý hoàn toàn ở tầng data — adapter không cắt gì thêm.

API đặc biệt: model.answer_question(image, question, tokenizer, ...)
Không dùng processor hay TextIteratorStreamer như LLaVA-OneVision.

Fix OOM:
  - 4-bit NF4 quantization, skip vision modules khỏi quantize
  - Xóa GPU cache trước mỗi lần generate
  - Bắt và propagate exception đúng cách

Fix load errors:
  - Bỏ device_map="auto" → dùng .to(device) thủ công
  - Patch transformers warmup/byte_count để bypass lỗi với custom model
  - Patch instance để có all_tied_weights_keys
  - Dùng dtype thay torch_dtype
  - trust_remote_code=True cho cả tokenizer

Fix visualization:
  - generate() trả về "used_frames" — đúng frames model thực sự thấy
  - preview.py dùng used_frames để figure hiển thị đồng nhất với inference
"""

import time
import torch
from models.base import BaseModelAdapter, get_device


MODEL_ID = "qresearch/llama-3.1-8B-vision-378"


def _make_all_tied_weights_keys(model) -> dict:
    """
    Transformers mới expect model.all_tied_weights_keys là object có .keys().
    Custom Llamavision chỉ có _tied_weights_keys.
    """
    keys = getattr(model, "_tied_weights_keys", [])
    if keys is None:
        keys = []
    if isinstance(keys, dict):
        return keys
    return {k: k for k in keys}


class LlamaVisionAdapter(BaseModelAdapter):
    """Adapter cho qresearch/llama-3.1-8B-vision-378."""

    MODEL_NAME   = "Llama-3.1-8B-Vision-378"
    MODEL_PARAMS = "8B"
    MODALITY     = "image+video"

    def __init__(self, device: str = "cuda", dtype: str = "float16"):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import transformers.modeling_utils as _mu

        self.device       = get_device(device)
        compute_dtype     = torch.float16 if dtype == "float16" else torch.bfloat16

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            llm_int8_skip_modules=["mm_projector", "vision_model"],
        )

        print(f"[{self.MODEL_NAME}] Loading {MODEL_ID} on {self.device} ({dtype}, 4-bit NF4)...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            use_fast=True,
            trust_remote_code=True,
        )

        # ------------------------------------------------------------
        # Patch transformers internals để tránh crash ở load/finalize
        # ------------------------------------------------------------
        _orig_warmup     = getattr(_mu, "caching_allocator_warmup", None)
        _orig_byte_count = getattr(_mu, "get_total_byte_count", None)
        _orig_finalize   = getattr(_mu.PreTrainedModel, "mark_tied_weights_as_initialized", None)

        if _orig_warmup is not None:
            _mu.caching_allocator_warmup = lambda *a, **kw: None

        if _orig_byte_count is not None:
            _mu.get_total_byte_count = lambda *a, **kw: 0

        def _patched_mark_tied_weights_as_initialized(model_self, loading_info):
            if not hasattr(model_self, "all_tied_weights_keys"):
                model_self.all_tied_weights_keys = _make_all_tied_weights_keys(model_self)
            return _orig_finalize(model_self, loading_info)

        if _orig_finalize is not None:
            _mu.PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark_tied_weights_as_initialized

        try:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    trust_remote_code=True,
                    dtype=compute_dtype,
                    quantization_config=bnb_config,
                    device_map=None,
                ).eval()
            except Exception as e:
                print(f"[{self.MODEL_NAME}] 4-bit load failed, retrying without quantization. Error: {e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    trust_remote_code=True,
                    dtype=compute_dtype,
                    device_map=None,
                ).eval()
        finally:
            if _orig_warmup is not None:
                _mu.caching_allocator_warmup = _orig_warmup
            if _orig_byte_count is not None:
                _mu.get_total_byte_count = _orig_byte_count
            if _orig_finalize is not None:
                _mu.PreTrainedModel.mark_tied_weights_as_initialized = _orig_finalize

        # Patch lại trên instance cho chắc
        if not hasattr(self.model, "all_tied_weights_keys"):
            self.model.all_tied_weights_keys = _make_all_tied_weights_keys(self.model)

        # Nếu model vẫn ở CPU thì move thủ công
        try:
            first_param = next(self.model.parameters())
            if first_param.device.type == "cpu" and self.device != "cpu":
                self.model = self.model.to(self.device)
        except StopIteration:
            pass

        print(f"[{self.MODEL_NAME}] Model loaded successfully.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clear_gpu_cache(self):
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _answer_single_image(self, image, question: str, max_new_tokens: int = 512) -> str:
        """
        Gọi model.answer_question() — API đặc thù của qresearch vision models.
        """
        with torch.inference_mode():
            output = self.model.answer_question(
                image,
                question,
                self.tokenizer,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        if isinstance(output, str):
            return output

        return self.tokenizer.decode(output, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Main generate
    # ------------------------------------------------------------------

    def generate(self, sample: dict) -> dict:
        task        = sample.get("task", "short_caption")
        prompt_text = sample.get("prompt", "Describe this image in detail.")

        self._clear_gpu_cache()
        t_start = time.perf_counter()

        # used_frames: frames model thực sự thấy — trả về để caller visualize đúng.
        # Không sample thêm ở đây: data đã lo việc cắt frame trước khi gọi generate().
        used_frames = None

        try:
            if task in ("video_caption", "video_qa", "short_qa", "exhaustive_qa"):
                used_frames = sample.get("frames") or []

                if not used_frames:
                    decoded = "[No frames provided]"

                elif len(used_frames) == 1:
                    decoded = self._answer_single_image(used_frames[0], prompt_text)

                else:
                    # Describe mỗi frame ngắn gọn
                    per_frame_q   = f"Frame description (for later summarization): {prompt_text}"
                    frame_answers = []

                    for i, frame in enumerate(used_frames):
                        ans = self._answer_single_image(frame, per_frame_q, max_new_tokens=128)
                        frame_answers.append(f"Frame {i + 1}: {ans.strip()}")

                    # Tổng hợp tất cả frame descriptions → trả lời câu hỏi chính
                    summary_context = "\n".join(frame_answers)
                    summary_q = (
                        f"Based on these frame descriptions from a video:\n"
                        f"{summary_context}\n\n"
                        f"Answer this question about the video: {prompt_text}"
                    )
                    decoded = self._answer_single_image(used_frames[-1], summary_q, max_new_tokens=512)

            else:
                image = sample.get("image")
                if image is None:
                    decoded = "[No image provided]"
                else:
                    used_frames = [image]   # wrap vào list để figure dùng được
                    decoded     = self._answer_single_image(image, prompt_text)

        except torch.cuda.OutOfMemoryError as e:
            self._clear_gpu_cache()
            raise RuntimeError(
                f"[{self.MODEL_NAME}] CUDA OOM during generate(). "
                f"Thử giảm số frames ở tầng data (dataset num_frames) hoặc max_new_tokens."
            ) from e

        ttft       = time.perf_counter() - t_start
        num_tokens = self.estimate_tokens(decoded)

        return {
            "text":                   decoded,
            "num_tokens":             num_tokens,
            "time_to_first_token_s":  round(ttft, 4),
            "acceptance_length":      None,
            "draft_rounds":           None,
            # ← frames model thực sự dùng để inference, để preview.py visualize đúng
            "used_frames":            used_frames or [],
        }