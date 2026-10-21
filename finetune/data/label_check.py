"""Checking a label file against its video's facts: ``check_label_file`` reads one label file
next to the video's frame rate, size and length and lists what does not add up (frames past the
end, points off the image, a 60 FPS clip labelled on both phases, gaps). ``matches_any`` is the
clip-name glob every command that takes clip patterns uses. Nothing here needs the detector."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from finetune.data.labels import LabelFormatError, read_label_rows
from tennis_tracker.frame_rate import phase_stride

LABEL_COVERAGE = 0.9         # fewer labelled frames than this share of the span is a gap


def matches_any(clip: str, patterns: Optional[List[str]]) -> bool:
    """Is `clip` one of the stems or globs in `patterns`? No patterns means every clip."""
    return not patterns or any(re.fullmatch(pattern.replace("*", ".*"), clip) for pattern in patterns)


def _row_issues(csv_path: Path, width: int, height: int) -> Tuple[List[str], List[int]]:
    """What is wrong with the rows themselves, and the frames they name (in file order). Reading
    stops at the first point off the image."""
    issues: List[str] = []
    frames: List[int] = []
    try:
        for row in read_label_rows(csv_path):
            frames.append(row.frame)
            if not (0 <= row.x <= width - 1 and 0 <= row.y <= height - 1):
                issues.append(f"frame {row.frame}: ({row.x:.0f}, {row.y:.0f}) outside {width}x{height}")
                break
    except LabelFormatError as error:
        issues.append(str(error))
    return issues, frames


def _cadence_issues(frames: List[int], fps: float, stride: Optional[int], total: int) -> List[str]:
    """What is wrong with where the labelled frames sit: past the video, closer than the clip's
    phase allows, on both phases, repeated, or too sparse."""
    issues: List[str] = []
    if max(frames) >= total:
        issues.append(f"label frame {max(frames)} beyond the video's {total} frames")
    if stride:
        steps = {b - a for a, b in zip(frames, frames[1:])}
        if steps and min(steps) < stride:
            issues.append(f"labels every {min(steps)} frame(s) but a {fps:.0f} FPS clip needs every {stride}")
        if len(set(frame % stride for frame in frames)) > 1:
            issues.append("labels mix even and odd frames of a 60 FPS clip")
    if len(frames) != len(set(frames)):
        issues.append("duplicate frame rows")
    expected = len(range(min(frames), max(frames) + 1, stride or 1))
    if len(frames) < expected * LABEL_COVERAGE:
        issues.append(f"only {len(frames)} of {expected} frames between first and last label are labelled")
    return issues


def check_label_file(csv_path: Path, fps: float, width: int, height: int, total: int) -> Tuple[List[str], int]:
    """What is wrong with a label file against its video's facts, and how many rows it has."""
    stride = phase_stride(fps)
    issues = [] if stride else [f"{fps:.2f} FPS is neither 30 nor 60"]
    row_issues, frames = _row_issues(csv_path, width, height)
    issues += row_issues
    if not frames:
        issues.append("no label rows")
        return issues, 0
    return issues + _cadence_issues(frames, fps, stride, total), len(frames)
