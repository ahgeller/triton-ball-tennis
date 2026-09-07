"""Read explicit label-review status; never infer correctness from source or filenames alone."""
import json
import re
from pathlib import Path


def read_review_status(workspace: Path):
    path = workspace / "review_status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def is_verified(clip: str, status) -> bool:
    if status is None:
        return True  # external archives without a review policy retain their contract
    if clip in status.get("verified_clips", []):
        return True
    match = re.fullmatch(r"video(\d+)", clip)
    minimum = status.get("verified_video_min")
    return bool(match and minimum is not None and int(match.group(1)) >= int(minimum))
