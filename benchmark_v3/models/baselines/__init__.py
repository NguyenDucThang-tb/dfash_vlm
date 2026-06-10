from models.baselines.qwen3vl import Qwen3VLAdapter

# Registry: tên CLI → class
BASELINE_REGISTRY = {
    "qwen3vl": Qwen3VLAdapter,
}
