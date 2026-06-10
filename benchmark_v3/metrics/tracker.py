"""
MetricsTracker v3 — đo toàn diện cho speculative VLM benchmark.

Thêm mới (Câu 2 + Câu 3):
  - caption_sim : đo độ tương đồng caption giữa full (base) vs pruned input
                  Dùng ROUGE-L (luôn khả dụng) + BERTScore (nếu cài được)
  - accuracy    : đánh giá đáp án A/B/C/D cho VideoMME (Câu 3)
                  So sánh predicted letter vs ground-truth answer

Metrics đo được:
  - tokens/s              throughput
  - time_to_first_token   TTFT (s)
  - latency p50/p95       end-to-end latency
  - acceptance_length     α — trung bình token accepted / draft step (spec decoding)
  - draft_rounds          số vòng draft trung bình
  - speedup               so với baseline tương ứng
  - memory_mb             GPU peak memory (nếu có)
  - caption_sim_rouge_l   ROUGE-L giữa caption gốc vs caption pruned (YouTube)
  - caption_sim_bert_f1   BERTScore F1 (YouTube, nếu cài bert_score)
  - accuracy              % đúng A/B/C/D (VideoMME)
"""

import gc
import re
import statistics
import time
from typing import Optional, Dict, List


# ─────────────────────────────────────────────────────────────────
# MEMORY HELPERS
# ─────────────────────────────────────────────────────────────────

def get_gpu_memory_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return None


def reset_gpu_memory_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────
# CAPTION SIMILARITY (CÂU 2)
# ─────────────────────────────────────────────────────────────────

def _rouge_l(reference: str, hypothesis: str) -> float:
    """
    Tính ROUGE-L F1 giữa reference và hypothesis.
    Không cần thư viện ngoài — tự implement LCS.
    """
    if not reference or not hypothesis:
        return 0.0

    ref_tokens  = reference.lower().split()
    hyp_tokens  = hypothesis.lower().split()

    n, m = len(ref_tokens), len(hyp_tokens)
    if n == 0 or m == 0:
        return 0.0

    # LCS bằng dynamic programming
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len   = dp[n][m]
    precision = lcs_len / m if m > 0 else 0.0
    recall    = lcs_len / n if n > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(f1, 4)


def _bert_score_f1(reference: str, hypothesis: str) -> Optional[float]:
    """
    Tính BERTScore F1.
    Cần: pip install bert_score
    Trả về None nếu không cài được.
    """
    try:
        from bert_score import score as bs_score
        _, _, F = bs_score(
            [hypothesis], [reference],
            lang="en",
            verbose=False,
            rescale_with_baseline=False,
        )
        return round(float(F[0]), 4)
    except Exception:
        return None


def compute_caption_similarity(
    caption_original: str,
    caption_pruned: str,
) -> dict:
    """
    Tính độ tương đồng giữa caption từ đầu vào gốc vs đầu vào sau prune.

    Args:
        caption_original : caption sinh từ bộ frame đầy đủ (Qwen3 base / full video)
        caption_pruned   : caption sinh từ bộ frame đã prune

    Returns:
        {
          "rouge_l"  : float (0–1),
          "bert_f1"  : float (0–1) hoặc None nếu bert_score chưa cài,
          "composite": float — trung bình có sẵn (luôn có giá trị)
        }
    """
    rouge = _rouge_l(caption_original, caption_pruned)
    bert  = _bert_score_f1(caption_original, caption_pruned)

    if bert is not None:
        composite = round((rouge + bert) / 2, 4)
    else:
        composite = rouge

    return {
        "rouge_l":   rouge,
        "bert_f1":   bert,
        "composite": composite,
    }


# ─────────────────────────────────────────────────────────────────
# ACCURACY HELPER (CÂU 3 — VideoMME)
# ─────────────────────────────────────────────────────────────────

def _normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_predicted_letter(text: str, options: Optional[dict] = None) -> Optional[str]:
    """
    Trích xuất chữ cái A/B/C/D từ output của model.
    Ưu tiên bắt explicit letter; nếu không có thì thử map theo nội dung option.
    """
    if not text:
        return None
    text = text.strip()
    text_upper = text.upper()

    for pat in [
        r"^\s*([A-D])\s*$",
        r"FINAL\s+ANSWER\s*(?:IS|[:\-])\s*([A-D])\b",
        r"CORRECT\s+ANSWER\s*(?:IS|[:\-])\s*([A-D])\b",
        r"ANSWER\s*(?:IS|[:\-])\s*([A-D])\b",
        r"OPTION\s*(?:IS|[:\-])\s*([A-D])\b",
        r"LETTER\s*(?:IS|[:\-])\s*([A-D])\b",
        r"LETTER\s*\(([A-D])\)",
    ]:
        m = re.search(pat, text_upper)
        if m:
            return m.group(1).upper()

    # Fallback: lấy chữ cái standalone ở đoạn cuối câu trả lời, tránh bắt nhầm
    # các bullet "A./B./C./D." trong phần liệt kê option/phân tích.
    tail = text_upper[-200:]
    tail_letters = re.findall(r"(?:^|[\s\(\[\-:])([A-D])(?:[\s\)\].,;:!]|$)", tail)
    if tail_letters:
        return tail_letters[-1].upper()

    first = text[:1].upper()
    if first in "ABCD":
        return first

    if options:
        if isinstance(options, list):
            options = {k: v for k, v in zip(["A", "B", "C", "D"], options)}
        pred_norm = _normalize_text(text)
        candidates = []
        for letter, opt_text in (options or {}).items():
            opt_norm = _normalize_text(str(opt_text))
            if opt_norm and opt_norm in pred_norm:
                candidates.append((str(letter).upper(), len(opt_norm)))
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

    return None


# ─────────────────────────────────────────────────────────────────
# TRACKER
# ─────────────────────────────────────────────────────────────────

class MetricsTracker:
    """
    Tracker cho 1 (model, dataset) pair.
    Lưu từng record và tính summary theo bucket.
    """

    def __init__(self, model_name: str, model_params: str, dataset_name: str):
        self.model_name   = model_name
        self.model_params = model_params
        self.dataset_name = dataset_name
        self._records: List[dict] = []
        reset_gpu_memory_peak()

    # ── Record ──────────────────────────────────────────────────

    def record(
        self,
        elapsed_s: float,
        num_tokens: int,
        sample_id: str,
        token_bucket: str,
        time_to_first_token_s: Optional[float] = None,
        acceptance_length: Optional[float] = None,
        draft_rounds: Optional[int] = None,
        video_bucket: Optional[str] = None,
        # ── Câu 3: accuracy cho VideoMME ────────────────────────
        predicted_text: Optional[str] = None,
        ground_truth_answer: Optional[str] = None,
        answer_options: Optional[dict] = None,
        # ── Câu 2: caption similarity ────────────────────────────
        # Điền sau khi có đủ cả caption gốc + caption pruned
        caption_sim: Optional[dict] = None,
    ):
        predicted_letter = _extract_predicted_letter(predicted_text, answer_options) if predicted_text else None
        gt_letter        = ground_truth_answer.strip().upper() if ground_truth_answer else None
        is_correct       = (
            predicted_letter is not None
            and gt_letter is not None
            and predicted_letter == gt_letter
        )

        self._records.append({
            "sample_id":          sample_id,
            "elapsed_s":          elapsed_s,
            "num_tokens":         num_tokens,
            "tps":                num_tokens / elapsed_s if elapsed_s > 0 else 0.0,
            "token_bucket":       token_bucket,
            "video_bucket":       video_bucket,
            "ttft":               time_to_first_token_s,
            "acceptance_length":  acceptance_length,
            "draft_rounds":       draft_rounds,
            # Accuracy
            "predicted_text":     predicted_text,
            "predicted_letter":   predicted_letter,
            "ground_truth":       gt_letter,
            "answer_options":    answer_options,
            "is_correct":         is_correct if gt_letter else None,
            # Caption similarity
            "caption_sim":        caption_sim,
        })

    def update_caption_sim(self, sample_id: str, caption_sim: dict):
        """
        Cập nhật caption_sim cho một record đã tồn tại.
        Dùng khi có 2 pass (gốc → pruned) và tính sim sau.
        """
        for r in self._records:
            if r["sample_id"] == sample_id:
                r["caption_sim"] = caption_sim
                return

    # ── Running summary (in-progress) ───────────────────────────

    def running_summary(self) -> dict:
        if not self._records:
            return {}
        tps    = [r["tps"] for r in self._records]
        alphas = [r["acceptance_length"] for r in self._records if r["acceptance_length"]]
        ttfts  = [r["ttft"] for r in self._records if r["ttft"]]
        return {
            "tps":   round(statistics.mean(tps), 2),
            "alpha": round(statistics.mean(alphas), 2) if alphas else None,
            "ttft":  round(statistics.mean(ttfts), 3) if ttfts else None,
        }

    # ── Finalize ────────────────────────────────────────────────

    def finalize(self) -> dict:
        if not self._records:
            return {
                "model": self.model_name,
                "model_params": self.model_params,
                "dataset": self.dataset_name,
                "num_samples": 0,
            }

        overall   = self._compute_stats(self._records)
        by_bucket = self._compute_by_bucket()
        memory_mb = get_gpu_memory_mb()

        return {
            # Identity
            "model":        self.model_name,
            "model_params": self.model_params,
            "dataset":      self.dataset_name,
            "num_samples":  len(self._records),
            "total_tokens": sum(r["num_tokens"] for r in self._records),

            # Overall metrics
            **overall,

            # Memory
            "peak_memory_mb": round(memory_mb, 1) if memory_mb else None,

            # Per-bucket breakdown
            "by_bucket": by_bucket,

            # Speedup (filled later by compute_speedups)
            "speedup": None,
        }

    def _compute_stats(self, records: List[dict]) -> dict:
        if not records:
            return {}

        tps_vals  = [r["tps"] for r in records]
        lat_vals  = [r["elapsed_s"] for r in records]
        ttft_vals = [r["ttft"] for r in records if r["ttft"] is not None]
        alphas    = [r["acceptance_length"] for r in records if r["acceptance_length"] is not None]
        drafts    = [r["draft_rounds"] for r in records if r["draft_rounds"] is not None]

        stats = {
            # Throughput
            "tokens_per_sec": round(statistics.mean(tps_vals), 2),
            "tps_p50":        round(statistics.median(tps_vals), 2),
            "tps_p95":        round(_pct(tps_vals, 95), 2),

            # fix 2: trung bình số token sinh ra
            "avg_tokens":     round(statistics.mean([r["num_tokens"] for r in records]), 2),

            # End-to-end latency
            "latency_mean_s": round(statistics.mean(lat_vals), 3),
            "latency_p50_s":  round(statistics.median(lat_vals), 3),
            "latency_p95_s":  round(_pct(lat_vals, 95), 3),

            # Time to first token
            "ttft_mean_s": round(statistics.mean(ttft_vals), 4) if ttft_vals else None,
            "ttft_p50_s":  round(statistics.median(ttft_vals), 4) if ttft_vals else None,
            "ttft_p95_s":  round(_pct(ttft_vals, 95), 4) if ttft_vals else None,

            # Speculative decoding
            "acceptance_length": round(statistics.mean(alphas), 3) if alphas else None,
            "draft_rounds_mean": round(statistics.mean(drafts), 2) if drafts else None,
        }

        # ── Câu 3: Accuracy (VideoMME) ──────────────────────────
        labeled = [r for r in records if r.get("is_correct") is not None]
        if labeled:
            correct = sum(1 for r in labeled if r["is_correct"])
            stats["accuracy"]       = round(correct / len(labeled), 4)
            stats["accuracy_n"]     = len(labeled)
            stats["accuracy_correct"] = correct
        else:
            stats["accuracy"]         = None
            stats["accuracy_n"]       = 0
            stats["accuracy_correct"] = 0

        # ── Câu 2: Caption similarity (YouTube) ─────────────────
        sim_records = [r["caption_sim"] for r in records if r.get("caption_sim")]
        if sim_records:
            rouge_vals = [s["rouge_l"] for s in sim_records if s.get("rouge_l") is not None]
            bert_vals  = [s["bert_f1"] for s in sim_records if s.get("bert_f1") is not None]
            comp_vals  = [s["composite"] for s in sim_records if s.get("composite") is not None]
            stats["caption_sim_rouge_l"]  = round(statistics.mean(rouge_vals), 4) if rouge_vals else None
            stats["caption_sim_bert_f1"]  = round(statistics.mean(bert_vals), 4) if bert_vals else None
            stats["caption_sim_composite"] = round(statistics.mean(comp_vals), 4) if comp_vals else None
            stats["caption_sim_n"]        = len(sim_records)
        else:
            stats["caption_sim_rouge_l"]   = None
            stats["caption_sim_bert_f1"]   = None
            stats["caption_sim_composite"] = None
            stats["caption_sim_n"]         = 0

        return stats

    def _compute_by_bucket(self) -> dict:
        bucket_map: Dict[str, List[dict]] = {}
        for r in self._records:
            b = r.get("token_bucket", "unknown")
            bucket_map.setdefault(b, []).append(r)

        result = {}
        for bucket, records in bucket_map.items():
            result[bucket] = {
                "num_samples": len(records),
                **self._compute_stats(records),
            }
            vbuckets = set(r.get("video_bucket") for r in records if r.get("video_bucket"))
            if vbuckets:
                result[bucket]["video_buckets"] = list(vbuckets)
        return result


# ─────────────────────────────────────────────────────────────────
# SUMMARY ACROSS MODELS
# ─────────────────────────────────────────────────────────────────

def compute_speedups(all_results: dict, baseline_model: str = "Qwen3-VL-8B-Instruct-4bit"):
    for ds_results in all_results.values():
        base      = ds_results.get(baseline_model, {})
        base_tps  = base.get("tokens_per_sec", 0)
        base_ttft = base.get("ttft_mean_s")

        for model_name, res in ds_results.items():
            if base_tps and base_tps > 0:
                res["speedup"]       = round(res.get("tokens_per_sec", 0) / base_tps, 3)
                res["speedup_label"] = baseline_model
            else:
                res["speedup"]       = 1.0
                res["speedup_label"] = "N/A"

            if base_ttft and res.get("ttft_mean_s"):
                res["ttft_speedup"] = round(base_ttft / res["ttft_mean_s"], 3)


# ─────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────────────

def print_result(r: dict):
    tps    = r.get("tokens_per_sec", 0)
    spd    = r.get("speedup") or 1.0
    alph   = r.get("acceptance_length")
    lat    = r.get("latency_p50_s", 0)
    ttft   = r.get("ttft_mean_s")
    mem    = r.get("peak_memory_mb")
    params = r.get("model_params", "?")
    acc    = r.get("accuracy")
    csim   = r.get("caption_sim_composite")
    avg_tk = r.get("avg_tokens")   # fix 2

    line = (
        f"    [{params}] tok/s={tps:.1f}  speedup={spd:.3f}×"
        f"  lat_p50={lat:.3f}s"
    )
    if avg_tk is not None:
        line += f"  avg_tok={avg_tk:.1f}"   # fix 2
    if ttft:
        line += f"  TTFT={ttft:.3f}s"
    if alph:
        line += f"  α={alph:.2f}"
    if mem:
        line += f"  mem={mem:.0f}MB"
    if acc is not None:
        n_correct = r.get("accuracy_correct", 0)
        n_total   = r.get("accuracy_n", 0)
        line += f"  acc={acc*100:.1f}%({n_correct}/{n_total})"
    if csim is not None:
        rouge = r.get("caption_sim_rouge_l", 0)
        bert  = r.get("caption_sim_bert_f1")
        bert_str = f",bert={bert:.3f}" if bert is not None else ""
        line += f"  cap_sim={csim:.3f}(rougeL={rouge:.3f}{bert_str})"
    print(line)


def print_table(all_results: dict):
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    for ds_name, ds_res in all_results.items():
        print(f"\n📊 Dataset: {ds_name.upper()}")
        header = (
            f"  {'Model':<28} {'Params':>6} {'tok/s':>7} {'speedup':>9}"
            f" {'TTFT':>8} {'α':>6} {'lat_p50':>9} {'mem(MB)':>9}"
            f" {'avg_tok':>8} {'Acc':>7} {'CapSim':>8}"   # fix 2: thêm avg_tok
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        for model_name, r in ds_res.items():
            alph   = f"{r['acceptance_length']:.2f}" if r.get("acceptance_length") else "—"
            ttft   = f"{r['ttft_mean_s']:.3f}s"      if r.get("ttft_mean_s")       else "—"
            mem    = f"{r['peak_memory_mb']:.0f}"     if r.get("peak_memory_mb")    else "—"
            avg_tk = f"{r['avg_tokens']:.1f}"         if r.get("avg_tokens") is not None else "—"  # fix 2
            acc    = f"{r['accuracy']*100:.1f}%"      if r.get("accuracy") is not None else "—"
            csim   = f"{r['caption_sim_composite']:.3f}" if r.get("caption_sim_composite") is not None else "—"
            print(
                f"  {model_name:<28} {r.get('model_params','?'):>6}"
                f" {r.get('tokens_per_sec', 0):>7.1f}"
                f" {(r.get('speedup') or 1.0):>8.3f}×"
                f" {ttft:>8}"
                f" {alph:>6}"
                f" {r.get('latency_p50_s', 0):>8.3f}s"
                f" {mem:>9}"
                f" {avg_tk:>8}"   # fix 2
                f" {acc:>7}"
                f" {csim:>8}"
            )

        # Per-bucket breakdown
        print(f"\n  📂 Per-token-bucket breakdown:")
        bucket_header = (
            f"    {'Model':<28} {'bucket':<10} {'n':>4} {'tok/s':>7}"
            f" {'TTFT':>8} {'α':>6} {'avg_tok':>8} {'Acc':>7} {'CapSim':>8}"  # fix 2
        )
        print(bucket_header)
        for model_name, r in ds_res.items():
            for bucket, bstats in (r.get("by_bucket") or {}).items():
                alph   = f"{bstats['acceptance_length']:.2f}" if bstats.get("acceptance_length") else "—"
                ttft   = f"{bstats['ttft_mean_s']:.3f}s"      if bstats.get("ttft_mean_s")       else "—"
                avg_tk = f"{bstats['avg_tokens']:.1f}"         if bstats.get("avg_tokens") is not None else "—"  # fix 2
                acc    = f"{bstats['accuracy']*100:.1f}%"      if bstats.get("accuracy") is not None else "—"
                csim   = f"{bstats['caption_sim_composite']:.3f}" if bstats.get("caption_sim_composite") is not None else "—"
                print(
                    f"    {model_name:<28} {bucket:<10}"
                    f" {bstats.get('num_samples', 0):>4}"
                    f" {bstats.get('tokens_per_sec', 0):>7.1f}"
                    f" {ttft:>8}"
                    f" {alph:>6}"
                    f" {avg_tk:>8}"   # fix 2
                    f" {acc:>7}"
                    f" {csim:>8}"
                )


# ─────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────

def save_results(all_results: dict, output_dir):
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    json_path = output_dir / f"results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    md_lines = [f"# Benchmark Results\n", f"Generated: {time.ctime(ts)}\n"]
    for ds_name, ds_res in all_results.items():
        md_lines.append(f"\n## {ds_name}\n")
        md_lines.append(
            "| Model | Params | token/s | speedup | TTFT | α | lat_p50 | mem(MB) | avg_tok | Accuracy | CapSim |"  # fix 2
        )
        md_lines.append(
            "|-------|-------:|--------:|--------:|-----:|--:|--------:|--------:|--------:|---------:|-------:|"
        )
        for mn, r in ds_res.items():
            alph   = f"{r['acceptance_length']:.2f}" if r.get("acceptance_length") else "—"
            ttft   = f"{r['ttft_mean_s']:.3f}s"      if r.get("ttft_mean_s")       else "—"
            mem    = f"{r['peak_memory_mb']:.0f}"     if r.get("peak_memory_mb")    else "—"
            avg_tk = f"{r['avg_tokens']:.1f}"         if r.get("avg_tokens") is not None else "—"  # fix 2
            acc    = f"{r['accuracy']*100:.1f}%"      if r.get("accuracy") is not None else "—"
            csim   = f"{r['caption_sim_composite']:.3f}" if r.get("caption_sim_composite") is not None else "—"
            md_lines.append(
                f"| {mn} | {r.get('model_params','?')} | {r.get('tokens_per_sec',0):.1f}"
                f" | {(r.get('speedup') or 1.0):.3f}× | {ttft} | {alph}"
                f" | {r.get('latency_p50_s',0):.3f}s | {mem} | {avg_tk} | {acc} | {csim} |"  # fix 2
            )

    (output_dir / "README.md").write_text("\n".join(md_lines))
    print(f"\n✅ Saved → {json_path}")
    print(f"✅ Saved → {output_dir}/README.md")


# ─────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────

def _pct(data: list, p: float) -> float:
    if not data:
        return 0.0
    s  = sorted(data)
    k  = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
