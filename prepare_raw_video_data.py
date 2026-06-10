import argparse
import json
import random
import subprocess
import urllib.request
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_DATASET = "Leyo/ActivityNet_Captions"
DEFAULT_OUTPUT = "/content/phaseB_raw_videos.jsonl"
DEFAULT_VIDEO_DIR = "/content/activitynet_captions/videos"
ACTIVITYNET_CAPTIONS_ZIP = "https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip"


def iter_activitynet_rows(cache_dir: Path) -> Iterable[Dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "captions.zip"
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        print(f"[Init] Downloading ActivityNet captions: {ACTIVITYNET_CAPTIONS_ZIP}")
        urllib.request.urlretrieve(ACTIVITYNET_CAPTIONS_ZIP, zip_path)

    split_files = [("train", "train.json"), ("validation", "val_1.json"), ("test", "val_2.json")]
    with zipfile.ZipFile(zip_path) as zf:
        for split_name, filename in split_files:
            with zf.open(filename) as f:
                infos = json.load(f)
            for video_id, info in infos.items():
                timestamps = info.get("timestamps", [])
                yield {
                    "video_id": video_id,
                    "video_path": "https://www.youtube.com/watch?v=" + str(video_id)[2:],
                    "duration": float(info.get("duration") or 0.0),
                    "captions_starts": [float(ts[0]) for ts in timestamps],
                    "captions_ends": [float(ts[1]) for ts in timestamps],
                    "en_captions": [str(c) for c in info.get("sentences", [])],
                    "_source_split": split_name,
                }


def normalize_video_id(video_id: str) -> str:
    video_id = str(video_id).strip()
    if video_id.startswith("v_"):
        video_id = video_id[2:]
    return video_id


def captions_to_text(row: Dict[str, Any]) -> str:
    captions = row.get("en_captions") or row.get("captions") or []
    if isinstance(captions, str):
        return captions.strip()
    return " ".join(str(c).strip() for c in captions if str(c).strip()).strip()


def load_done_ids(output_path: Path) -> set[str]:
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
            video_id = row.get("video_id")
            video_path = row.get("video_path")
            if video_id and video_path and Path(video_path).exists():
                done.add(str(video_id))
    return done


def download_video(url: str, out_path: Path, height: int, cookies_path: Optional[str] = None) -> tuple[bool, str]:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, ""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_template = str(out_path.with_suffix(".%(ext)s"))
    fmt = f"bv*[height<={height}][ext=mp4]/bv*[height<={height}]/best[height<={height}]/best"
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "-f",
        fmt,
        "--merge-output-format",
        "mp4",
        "-o",
        tmp_template,
        url,
    ]
    if cookies_path:
        cmd[1:1] = ["--cookies", cookies_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        if isinstance(exc, FileNotFoundError):
            return False, "yt-dlp executable not found. Install it with: pip install yt-dlp"
        stderr = getattr(exc, "stderr", "") or ""
        stdout = getattr(exc, "stdout", "") or ""
        return False, (stderr.strip() or stdout.strip() or repr(exc))[-2000:]

    candidates = sorted(out_path.parent.glob(out_path.stem + ".*"))
    for candidate in candidates:
        if candidate.suffix.lower() == ".mp4" and candidate.stat().st_size > 0:
            if candidate != out_path:
                candidate.replace(out_path)
            return True, ""
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, ""
    return False, "yt-dlp finished but no mp4 output was produced"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ActivityNet-Captions raw videos to /content for Phase B.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--video-dir", type=str, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--cache-dir", type=str, default="/content/activitynet_captions")
    parser.add_argument("--num-videos", type=int, default=1200)
    parser.add_argument("--min-duration", type=float, default=45.0)
    parser.add_argument("--max-duration", type=float, default=180.0)
    parser.add_argument("--max-height", type=int, default=480)
    parser.add_argument("--download-workers", type=int, default=1)
    parser.add_argument("--cookies-path", type=str, default="")
    parser.add_argument("--max-fail-streak", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    if args.dataset != DEFAULT_DATASET:
        print(f"[Warn] --dataset is kept for compatibility; using ActivityNet captions zip for {args.dataset}")
    for row in iter_activitynet_rows(Path(args.cache_dir)):
        duration = float(row.get("duration") or 0.0)
        if duration < args.min_duration or duration > args.max_duration:
            continue
        video_id = normalize_video_id(str(row.get("video_id", "")))
        url = str(row.get("video_path") or row.get("url") or "").strip()
        caption_text = captions_to_text(row)
        if not video_id or not url or not caption_text or video_id in seen_ids:
            continue
        seen_ids.add(video_id)
        rows.append(
            {
                "video_id": video_id,
                "source_url": url,
                "duration": duration,
                "source_caption": caption_text,
                "source_split": row.get("_source_split"),
            }
        )

    rnd = random.Random(args.seed)
    rnd.shuffle(rows)

    done_ids = load_done_ids(output_path) if args.resume else set()
    target_success = int(args.num_videos) if args.num_videos > 0 else len(rows)
    mode = "a" if args.resume else "w"
    ok = 0
    failed = 0
    fail_streak = 0
    print(
        f"[Init] candidates={len(rows)} | target_success={target_success} | done={len(done_ids)} | "
        f"duration=[{args.min_duration},{args.max_duration}] | output={output_path}"
    )
    workers = max(1, int(args.download_workers))

    def submit_next(
        executor: ThreadPoolExecutor,
        row_iter,
        pending,
    ) -> Optional[bool]:
        for idx, row in row_iter:
            if len(done_ids) + ok + len(pending) >= target_success:
                return None
            video_id = row["video_id"]
            if video_id in done_ids:
                continue
            out_path = video_dir / f"{video_id}.mp4"
            print(
                f"[Queue] success={len(done_ids) + ok}/{target_success} | "
                f"pending={len(pending)} | candidate={idx}/{len(rows)} | "
                f"{video_id} duration={row['duration']:.1f}s"
            )
            future = executor.submit(
                download_video,
                str(row["source_url"]),
                out_path,
                args.max_height,
                args.cookies_path or None,
            )
            pending[future] = (idx, row, out_path)
            return True
        return None

    with output_path.open(mode, encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as executor:
        row_iter = iter(enumerate(rows, start=1))
        pending = {}
        while len(pending) < workers and len(done_ids) + ok + len(pending) < target_success:
            if submit_next(executor, row_iter, pending) is None:
                break

        while pending and len(done_ids) + ok < target_success:
            done_futures, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done_futures:
                idx, row, out_path = pending.pop(future)
                video_id = row["video_id"]
                try:
                    success, error_text = future.result()
                except Exception as exc:
                    success, error_text = False, repr(exc)

                if not success:
                    failed += 1
                    fail_streak += 1
                    print(f"[Warn] failed: {video_id} {row['source_url']}")
                    if error_text:
                        print(f"[Warn][yt-dlp] {error_text}")
                    if args.max_fail_streak > 0 and fail_streak >= args.max_fail_streak and (len(done_ids) + ok) == 0:
                        raise RuntimeError(
                            f"Stopped after {fail_streak} consecutive download failures and 0 successful videos. "
                            "This usually means YouTube blocks the runtime/IP, yt-dlp is missing/outdated, "
                            "or cookies are required. Use a direct-mp4 dataset or provide yt-dlp cookies."
                        )
                else:
                    fail_streak = 0
                    payload = dict(row)
                    payload["video_path"] = str(out_path)
                    f.write(json.dumps(payload, ensure_ascii=True) + "\n")
                    f.flush()
                    ok += 1
                    print(
                        f"[DoneOne] success={len(done_ids) + ok}/{target_success} | "
                        f"candidate={idx}/{len(rows)} | {video_id}"
                    )

                while len(pending) < workers and len(done_ids) + ok + len(pending) < target_success:
                    if submit_next(executor, row_iter, pending) is None:
                        break
    total_success = len(done_ids) + ok
    print(
        f"[Done] success_total={total_success}/{target_success} "
        f"new_downloaded={ok} failed={failed} manifest={output_path} video_dir={video_dir}"
    )
    if total_success < target_success:
        print(
            f"[Warn] Exhausted candidates before reaching target_success. "
            f"Need {target_success}, got {total_success}."
        )


if __name__ == "__main__":
    main()
