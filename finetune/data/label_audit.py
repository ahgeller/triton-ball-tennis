"""Checking a clip's labels: against its video's facts, and against the raw detector.

``check_label_file`` reads one label file next to its video's frame rate, size and length and
lists what does not add up (frames past the end, points off the image, a 60 FPS clip labelled on
both phases, gaps). ``audit_labels`` goes further and runs the detector over each clip: labels
that disagree with it as they stand but agree once renumbered were made on a different cut of
the video, and ``shift_labels`` renumbers them (``ft.py check --audit --fix``); labels that
agree with nothing go into ``exclude.txt`` (``add_exclusion``).

Two things are "suspect" in different senses, so they are named apart: a *clip* whose labels
disagree with the detector (``ClipAudit.verdict``, ``audit_labels`` counts them), and a *frame*
an audit flagged for a person to look at. The flags live in the flag sidecar
(``<clip>_ball.suspect.json``, see ``finetune.data.labels``) and are read by the label tool:
``flag_frames`` records them, ``open_flags`` reads them and ``outstanding_flags`` is what is
still worth a look. Only ``audit_labels`` needs the detector; it imports it when called, so
``ft.py`` stays importable without cv2 or torch.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from finetune.data.fileio import backup_copy
from finetune.data.labels import (FLAG_ANSWER_DECIMALS, FlagKind, Flags, LabelFormatError, Labels, read_flags, read_label_rows,
                                  read_labels, write_flags, write_labels)
from finetune.data.scoring import AGREE_PX, DETECTION_CONF, MIN_FAST, REFERENCE_HEIGHT, REFERENCE_WIDTH, scale_for_width
from finetune.data import workspace as ws
from finetune.data.workspace import (VideoMeta, add_exclusion, clip_of, label_csv, label_files, read_manifest, suspect_json,
                                     video_meta, videos_in, write_manifest)
from tennis_tracker.frame_rate import phase_stride

if TYPE_CHECKING:
    from finetune.data.alignment import Evidence, Tracks

LABEL_COVERAGE = 0.9         # check: fewer labelled frames than this share of the span is a gap

AUDIT_MAX_SHIFT = 3000        # temporal shifts (label frames) tried when a clip disagrees: the whole clip
AUDIT_SUSPECT = 0.60         # below this agreement (with enough detector coverage) a clip is flagged
AUDIT_FIXABLE = 0.85         # a shifted alignment this good replaces the labels with --fix
AUDIT_MIN_COVERAGE = 0.3     # the detector must fire on this share of the judged labels to judge a clip
AUDIT_MIN_SHARE = 0.5        # a shifted alignment must cover this share of the unshifted one's frames
SAME_POINT_PX = 10.0 ** -FLAG_ANSWER_DECIMALS   # an answered flag's label and the current one agree within this


# --------------------------------------------------------------------------- the label file itself

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


# --------------------------------------------------------------------------- the flag sidecar

def manifest_frame_size(row: Dict[str, str]) -> Tuple[float, float]:
    """A clip's frame size from its clips.csv row, 1080p when the row does not say."""
    return float(row.get("width") or REFERENCE_WIDTH), float(row.get("height") or REFERENCE_HEIGHT)


def open_flags(clip: str) -> Dict[str, List[int]]:
    """Frames of this clip somebody still has to look at, in two kinds.

    unplaceable  we could not work out which video frame this label belongs to - the clip is only
                 partly repaired, and these are the doubtful part of it
    check        the detector disagrees with the label here. Usually that is the detector losing a
                 ball at the racket rather than a bad label, so it is a prompt, not a defect
    """
    flags = read_flags(suspect_json(clip))
    return {"unplaceable": flags["unplaceable"], "check": flags["check"]}


def flag_frames(clip: str, kind: FlagKind, frames: List[int], note: str) -> None:
    """Record (or clear) one kind of flag, leaving the other kind alone."""
    path = suspect_json(clip)
    flags: Flags = read_flags(path)
    flags[kind] = sorted(set(int(f) for f in frames))
    flags["notes"][kind] = note
    flags["generated"] = str(dt.date.today())
    answered = flags["answered"]
    if answered:
        # A frame you already answered (SPACE or a fix in the label tool) is not asked about again
        # while its label is still the one you answered at; move the label and it can be flagged anew.
        labels = clip_labels(clip)
        flags[kind] = [f for f in flags[kind]
                       if not (str(f) in answered and same_point(answered[str(f)], labels.get(f, "missing")))]
    if not (flags["unplaceable"] or flags["check"] or answered):
        path.unlink(missing_ok=True)
        return
    write_flags(path, flags)


def clip_labels(clip: str) -> Labels:
    """frame -> (x, y), or None where the label says no ball, from the clip's label file (the
    frame size taken from clips.csv, 1080p when the clip is not listed)."""
    path = label_csv(clip)
    if not path.is_file():
        return {}
    width, height = manifest_frame_size(read_manifest().get(clip, {}))
    return read_labels(path, int(width), int(height))


def same_point(answered: Optional[List[float]], label) -> bool:
    if answered is None or label is None:
        return answered is None and label is None
    return label != "missing" and abs(answered[0] - label[0]) < SAME_POINT_PX and abs(answered[1] - label[1]) < SAME_POINT_PX


def outstanding_flags(clip: str) -> Dict[str, List[int]]:
    """The flagged frames still to deal with.

    The label tool takes a flag out of the file once you answer it - SPACE answers it as fully as
    a fix does - so what is in the file is still open, even on a frame you had reviewed before the
    audit raised it. A frame with no ball on it is dropped from 'unplaceable' too - there is no
    position to put in the wrong place, so which side of a cut it lands on cannot hurt anything.
    """
    flagged = open_flags(clip)
    if not (flagged["unplaceable"] or flagged["check"]):
        return flagged
    blind = {frame for frame, point in clip_labels(clip).items() if point is None} if flagged["unplaceable"] else set()
    return {"unplaceable": [f for f in flagged["unplaceable"] if f not in blind],
            "check": list(flagged["check"])}


# --------------------------------------------------------------------------- against the detector

class ClipAudit:
    """One clip's labels against the raw detector: how well they agree as they stand, and the
    temporal shift (in label frames) that would make them agree best. The agreement itself is
    ``finetune.data.alignment``'s, the same one realign.py solves with, so the two never disagree about
    a clip."""

    def __init__(self, visible: Dict[int, Tuple[float, float]], confident: Dict[int, Tuple[float, float]],
                 stride: int, scale: float, total: int):
        from finetune.data.alignment import Evidence, Tracks, agreement, errors_at, judges_fast   # numpy: not at ft.py's import

        self.stride, self.scale = stride, scale
        frames = sorted(visible)
        self.tracks: Tracks = Tracks.of(frames, [visible[f][0] for f in frames], [visible[f][1] for f in frames], scale)
        self.evidence: Evidence = Evidence.of(confident, total, scale)
        self._agreement = lambda shift: agreement(self.tracks, self.evidence, shift * stride)
        # Only a fast-moving ball can separate one temporal offset from the next: a slow ball stays
        # inside AGREE_PX for several frames either way, so judging every label equally lets a
        # shifted clip score "ok" on the strength of its slow frames (grid_match92, +2 frames, found
        # 2026-09-02 at 0.73 overall but 0.28 on its fast labels). Judge on the fast ones when the
        # detector saw enough of them, and fall back to all of them when it did not.
        self.fast = set(self.tracks.frames[self.tracks.fast].tolist())
        seen0, self.errors0 = errors_at(self.tracks, self.evidence, 0)
        self.judged = self.fast if judges_fast(self.tracks, seen0) else set(frames)
        self.best = self.agreement(0)
        self.agree0, n0 = self.best[0], self.best[1]
        self.best_shift = 0
        self.max_shift = min(AUDIT_MAX_SHIFT, total // stride)
        if self.agree0 < AUDIT_SUSPECT:   # only search when needed: labels may belong to another segment of the video
            for shift in range(-self.max_shift, self.max_shift + 1):
                candidate = self.agreement(shift)
                if candidate[0] > self.best[0] + 1e-9 and candidate[1] >= AUDIT_MIN_SHARE * max(n0, 1):
                    self.best_shift, self.best = shift, candidate
        self.coverage = n0 / max(len(self.judged), 1)
        self.at_edge = abs(self.best_shift) >= self.max_shift

    def agreement(self, shift: int) -> Tuple[float, int, float]:
        """(share of judged labels the detector agrees with, labels it fired on, median error) at
        `shift` label frames."""
        return self._agreement(shift)

    @property
    def fixable(self) -> bool:
        return self.best_shift != 0 and self.best[0] >= AUDIT_FIXABLE and not self.at_edge

    def disputed_frames(self) -> List[int]:
        """Labels the detector confidently contradicts on a clip that lines up: about that frame, not
        the alignment. Flagged for a human; never rewritten from the detector."""
        return [int(f) for f, error in zip(self.tracks.frames.tolist(), self.errors0.tolist())
                if not math.isnan(error) and error > AGREE_PX]      # NaN where the detector said nothing

    def verdict(self, fix: bool) -> Tuple[str, bool]:
        """(what to print, whether the clip's labels disagree with the detector)."""
        if self.coverage < AUDIT_MIN_COVERAGE:
            return "detector rarely fires here; cannot judge", False
        if self.agree0 >= AUDIT_SUSPECT:
            return "ok", False
        if self.fixable:
            return f"labels are {self.best_shift:+d} label frame(s) off" + (" -> fixed" if fix else " (use --fix)"), True
        text = "labels disagree with the detector" + (" -> excluded" if fix else " (use --fix to exclude)")
        if self.at_edge:
            text += f" (best shift at the {self.best_shift:+d} search limit)"
        return text, True


def _audit_clips(patterns: Optional[List[str]], manifest: Dict[str, dict]) -> List[str]:
    """The labelled clips an audit covers: those matching `patterns`, or every tennis clip."""
    clips = [clip_of(p) for p in label_files()]
    if patterns:
        return [c for c in clips if matches_any(c, patterns)]
    # tennis clips only by default: the tennis detector cannot audit shuttlecock labels
    return [c for c in clips if not (manifest.get(c, {}).get("source") or "").endswith("badminton")]


def _audit_stride(clip: str, fps: float) -> int:
    stride = phase_stride(fps)
    if stride is None:
        # The detector's cadence is 30 FPS; a clip at another rate is audited frame for frame,
        # which is only right if it is close to 30 FPS (check reports the rate as a problem).
        print(f"  [warn] {clip}: {fps:.2f} FPS is neither 30 nor 60; audited as if 30 FPS")
        return 1
    return stride


def _flag_disputed(clip: str, audit: ClipAudit) -> str:
    """Flag the frames the detector disputes on a clip that lines up; the verdict's suffix."""
    disputed = audit.disputed_frames()
    flag_frames(clip, "check", disputed,
                f"detector disagrees by more than {AGREE_PX:.0f} px - often it simply lost "
                f"the ball at the racket, so confirm before changing anything")
    return f"; {len(disputed)} frame(s) to eyeball" if disputed else ""


def _fix_clip(clip: str, audit: ClipAudit, stride: int, meta: VideoMeta) -> None:
    """--fix on a clip the detector disagrees with: shift it when a shift lines it up, exclude it otherwise."""
    if audit.fixable:
        shift_labels(clip, audit.best_shift * stride, meta.width, meta.height, meta.total)
    else:
        add_exclusion(clip, f"audit {dt.date.today()}: detector agrees on {audit.agree0:.0%} of frames "
                            f"(best shift {audit.best_shift:+d}: {audit.best[0]:.0%})")


def _audit_one(clip: str, video: Path, detector, fix: bool, mark: bool) -> bool:
    """Audit one clip against the detector and print its row; True when its labels disagree with it."""
    from finetune.training.detect import detect_points

    meta = video_meta(video)
    stride = _audit_stride(clip, meta.fps)
    labels = read_labels(label_csv(clip), meta.width, meta.height)
    visible = {f: p for f, p in labels.items() if p is not None}
    confident = {f: (x, y) for f, (x, y, conf) in detect_points(detector, video, meta).items() if conf >= DETECTION_CONF}
    audit = ClipAudit(visible, confident, stride, scale_for_width(meta.width), meta.total)
    verdict, disagrees = audit.verdict(fix)
    if mark and audit.agree0 >= AUDIT_SUSPECT:
        verdict += _flag_disputed(clip, audit)
    print(f"  {clip:36s} {len(visible):7d} {len(audit.fast):5d} {audit.coverage:7.2f} {audit.agree0:6.2f} {audit.best[0]:6.2f} "
          f"{audit.best_shift:+5d} {audit.best[2]:7.1f}  {verdict}", flush=True)
    if fix and audit.coverage >= AUDIT_MIN_COVERAGE and audit.agree0 < AUDIT_SUSPECT:
        _fix_clip(clip, audit, stride, meta)
    return disagrees


def audit_labels(patterns: Optional[List[str]], weights: Optional[str], fix: bool, mark: bool = False) -> int:
    """Cross-check every label file against the raw detector: agreement at offset 0, and the best temporal
    shift. Catches public clips whose labels were made on a different cut of the video. Returns how
    many clips disagree."""
    from finetune.training.detect import ball_detector   # the lazy GPU import: ft.py must load without torch

    videos = videos_in()
    clips = _audit_clips(patterns, read_manifest())
    weights_path = Path(weights).resolve() if weights else ws.REPO_WEIGHTS
    detector = ball_detector(weights_path)
    print(f"\naudit: {len(clips)} clips against {weights_path.name} "
          f"(agree = detector within {AGREE_PX:.0f} px of the label, measured on the fast labels; "
          f"shift = label frames)")
    print(f"  {'clip':36s} {'visible':>7s} {'fast':>5s} {'covered':>7s} {'agree':>6s} {'best':>6s} {'shift':>5s} {'med px':>7s}  verdict")
    disagreeing = sum(_audit_one(clip, videos[clip], detector, fix, mark) for clip in clips if clip in videos)
    if disagreeing:
        print(f"audit: {disagreeing} clip(s) flagged" + ("" if fix else "; re-run with --fix to shift or exclude them"))
    else:
        print("audit: every clip agrees with the detector")
    return disagreeing


# --------------------------------------------------------------------------- repairs

def shift_labels(clip: str, delta_frames: int, width: int, height: int, total: int) -> None:
    """Renumber a clip's label rows by delta_frames (dropping rows that leave the video) and record it."""
    path = label_csv(clip)
    if delta_frames == 0:
        return
    if any(path.with_suffix(suffix).exists() for suffix in (".review.json", ".suspect.json")):
        raise ValueError(f"Cannot bulk-shift {clip}: review metadata needs an explicit frame remapping")
    shifted = {frame + delta_frames: point for frame, point in read_labels(path, width, height).items()
               if 0 <= frame + delta_frames < total}
    backup_copy(path, "shift")
    rows, _ = write_labels(path, shifted, width, height)
    manifest = read_manifest()
    if clip in manifest and ws.MANIFEST.is_file():
        manifest[clip]["label_frames"] = str(rows)
        note = manifest[clip].get("note", "")
        manifest[clip]["note"] = (note + "; " if note else "") + f"audit {dt.date.today()}: shifted {delta_frames:+d} frames"
        write_manifest(manifest)
    print(f"  [fix] {clip}: labels shifted by {delta_frames:+d} frames ({rows} rows kept)")
