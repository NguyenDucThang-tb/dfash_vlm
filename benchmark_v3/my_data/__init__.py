from my_data.videomme_short import VideoMMEShortDataset
from my_data.mscoco import MSCOCODataset
from my_data.msrvtt import MSRVTTDataset
from my_data.youtube_video import YouTubeVideoDataset
from my_data.long_youtube import LongYouTubeVideoDataset
from my_data.base import BUCKET_SHORT, BUCKET_MEDIUM, BUCKET_LONG, ALL_BUCKETS

DATASET_REGISTRY = {
    # ── Image datasets ────────────────────────────────────────
    "mscoco":         MSCOCODataset,        # image, captioning 2 bucket (short + exhaustive)
    # ── Video datasets ────────────────────────────────────────
    "msrvtt":         MSRVTTDataset,        # short video captioning
    "videomme_short": VideoMMEShortDataset, # video QA, short clips <2min
    "youtube":        YouTubeVideoDataset,  # video captioning, 100 YouTube videos (60s clip)
    "long_youtube" : LongYouTubeVideoDataset
}
