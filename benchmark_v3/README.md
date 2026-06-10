# VLM Speculative Decoding Benchmark v2

## Cấu trúc

```
Benchmark_v2/
├── run_benchmark.py              ← entry point
├── Requirements.txt
│
├── models/
│   ├── base.py                   ← BaseModelAdapter interface
│   ├── spec_vlm.py               ← ★ FILE DUY NHẤT CẦN SỬA khi nhóm kia xong
│   └── baselines/
│       ├── qwen3vl.py            ← Qwen2-VL-2B  (image+video, 3B)
│       ├── internvl3.py          ← InternVL3-2B  (image, 2B — SOTA nhỏ)
│       ├── llava_onevision.py    ← LLaVA-OV-0.5B (image+video, 0.5B)
│       ├── videollama3.py        ← VideoLLaMA3-2B (video specialist, 2B)
│       └── phi35_vision.py       ← Phi-3.5-Vision-4B (image, 4B)
│
├── my_data/
│   ├── base.py                   ← BaseDataset + token bucket logic
│   ├── sharegpt4v.py             ← image, GPT-4V captions dài
│   ├── llava_instruct.py         ← image, diverse lengths
│   └── videocc3m.py              ← video, temporal understanding
│
└── metrics/
    └── tracker.py                ← MetricsTracker v2 (TTFT, memory, per-bucket)
```

---

## Models được chọn

| Model | Params | Chuyên về | Lý do chọn |
|-------|--------|-----------|------------|
| Qwen2-VL-2B | 3B | image+video | SOTA mạnh nhất ở size nhỏ, OCR, document |
| InternVL3-2B | 2B | image | Top benchmark MMMU, MMStar, visual reasoning |
| LLaVA-OneVision-0.5B | 0.5B | image+video | Nhẹ nhất, native video support |
| VideoLLaMA3-2B | 2B | video | Temporal reasoning, action recognition |
| Phi-3.5-Vision-4B | 4B | image | Multi-image, document, OCR |

---

## Datasets & Token Buckets

### Image datasets

| Dataset | Modality | Token buckets |
|---------|----------|---------------|
| ShareGPT4V | image | `<100` / `<300` / `>500` |
| LLaVA-Instruct-150K | image | `<100` / `<300` / `>500` |

### Video dataset

| Dataset | Modality | Token buckets | Video buckets |
|---------|----------|---------------|---------------|
| VideoCC3M / WebVid | video | `<100` / `<300` / `>500` | `video_short (<10f)` / `video_medium (10-30f)` / `video_long (>30f)` |

---

## Metrics đo được

| Metric | Mô tả |
|--------|-------|
| `tokens_per_sec` | Throughput trung bình |
| `tps_p50 / p95` | Throughput percentile |
| `latency_p50_s / p95_s` | End-to-end latency |
| `ttft_mean_s` | Time to first token (TTFT) |
| `ttft_p50 / p95` | TTFT percentile |
| `acceptance_length (α)` | Token accepted / draft step — đo hiệu quả speculative |
| `draft_rounds_mean` | Số vòng draft trung bình |
| `speedup` | So với Qwen2-VL-2B baseline |
| `ttft_speedup` | TTFT speedup so với baseline |
| `peak_memory_mb` | GPU peak memory |
| `by_bucket` | Tất cả metrics trên chia theo token bucket |

---

## Chạy

```bash
pip install -r Requirements.txt

# Test pipeline (mock, không cần GPU)
python run_benchmark.py

# Chọn dataset + model
python run_benchmark.py --datasets sharegpt4v llava_instruct --models spec qwen3vl internvl3

# Full benchmark (cần GPU)
python run_benchmark.py --real --datasets sharegpt4v videocc3m --models spec qwen3vl videollama3

# Chỉ video
python run_benchmark.py --real --datasets videocc3m --models spec videollama3

# Debug nhanh
python run_benchmark.py --num-samples 10 --datasets sharegpt4v
```

### Trên Kaggle (2x T4)

```bash
%%bash
cp -r /kaggle/input/.../Benchmark_v2 /kaggle/working/
cd /kaggle/working/Benchmark_v2
pip install -r Requirements.txt -q
python run_benchmark.py --real \
    --datasets sharegpt4v llava_instruct videocc3m \
    --models spec qwen3vl internvl3 llava_ov videollama3 \
    --output-dir /kaggle/working/results
```

---

## Khi nhóm kia xong → sửa đúng 1 chỗ trong `models/spec_vlm.py`

```python
# Xóa:
self._mock = True
self._model = _MockSpecModel()

# Thêm (tùy API nhóm kia):
import sys
sys.path.insert(0, "./spec_vlm_repo")   # clone repo nhóm kia vào đây
from spec_vlm import SpecVLM
self._model = SpecVLM.load("path/to/weights", device=self.device)
self.MODEL_NAME  = "SpecVLM-Qwen3-4B"
self.MODEL_PARAMS = "4B"
self._mock = False
```

---

## Thêm model mới

Tạo file mới trong `models/baselines/`, kế thừa `BaseModelAdapter`:

```python
from models.base import BaseModelAdapter, get_device, get_dtype

class MyNewModelAdapter(BaseModelAdapter):
    MODEL_NAME   = "MyModel-1B"
    MODEL_PARAMS = "1B"
    MODALITY     = "image+video"

    def __init__(self, device="cuda", dtype="bfloat16"):
        ...

    def generate(self, sample: dict) -> dict:
        ...
        return {
            "text": ...,
            "num_tokens": ...,
            "time_to_first_token_s": ...,
            "acceptance_length": None,
            "draft_rounds": None,
        }
```

Sau đó đăng ký trong `models/baselines/__init__.py`:
```python
from models.baselines.my_new_model import MyNewModelAdapter
BASELINE_REGISTRY["my_new_model"] = MyNewModelAdapter
```

Chạy ngay:
```bash
python run_benchmark.py --real --models my_new_model --datasets sharegpt4v
```
