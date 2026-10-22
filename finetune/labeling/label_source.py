"""One clip of a labelling run: its video, its rows, the files its work goes back to - and the
decoding, because frame identity is the whole point of a label file.

Everything that is about a particular video lives on ``Source`` - the capture, where it is
positioned, the frame fingerprints, the decoded-frame cache - so that working through several
clips in one sitting cannot mix them up, and edits always go back to the clip a row came from.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from finetune.labeling.label_store import Landings, Row, apply_flags, apply_review, load_landings, load_rows
from finetune.data.labels import LabelFormatError
from finetune.data.workspace import (bounce_csv, clip_of, draft_csv, draft_files, find_video, label_csv, label_files,
                                     natural_key, review_json, suspect_json, video_meta)

CACHE_BYTES = 400_000_000  # decoded frames kept in memory, as a budget: 1080p frames are 6 MB each
CACHE_FRAMES_MAX = 96      # ... and never more than this many, nor fewer than CACHE_FRAMES_MIN
CACHE_FRAMES_MIN = 2
SEEK_BACKUP = 32          # a seek aims this many frames before the one wanted, and decodes through
                          # them: the label frames among them are cached, so stepping back keeps
                          # hitting the cache instead of seeking again
SEEK_VERIFY = 64          # at most this many frames decoded after a seek to prove where it landed
SEEK_SHIFTS = 4           # a seek may have landed this many frames either side of where it claims
MATCH_RECENT = 16         # only the newest decoded frames are matched against the fingerprints
LANDMARK = 8              # every Nth frame walked past is fingerprinted, so a later seek into that
                          # stretch has something to prove itself against
DECODE_RETRIES = 3        # re-seeks tried before a frame is given up on as undecodable
SIGNATURE_SIZE = (96, 54) # a frame is fingerprinted at this size
DEFAULT_ZOOM = 2.0
PREFETCH = 6              # label frames decoded ahead while the tool waits for the next keypress
PREFETCH_BUDGET_S = 0.02  # about a screen refresh of decoding per idle tick

if TYPE_CHECKING:
    from finetune.labeling.label_session import LabelSession


def frame_signature(image: np.ndarray) -> bytes:
    """A cheap fingerprint of a frame, enough to recognise one we have decoded before."""
    # INTER_LINEAR samples rather than averages: 20-50x cheaper than INTER_AREA on 1080p, where
    # INTER_AREA cost as much as the decode itself, and just as distinct between frames.
    return hashlib.blake2b(cv2.resize(image, SIGNATURE_SIZE, interpolation=cv2.INTER_LINEAR).tobytes(),
                           digest_size=8).digest()


def ran_out(source: "Source") -> None:
    """The capture gave nothing: remember where the video ends, so a label past it fails fast
    (not four times), unless this was frame 0 - a capture that did not open, which a reopen
    may yet fix."""
    if source.position > 0:
        source.end = source.position
    source.position = -1


def walk_to(source: "Source", index: int) -> bool:
    """Grab forward from the capture's position to just before `index`, fingerprinting every
    LANDMARK-th frame and keeping the label frames close to the target. False when the video
    ran out first."""
    while 0 <= source.position < index:
        if not source.capture.grab():
            ran_out(source)
            return False
        at = source.position
        landmark = at % LANDMARK == 0 and at not in source.sig_of
        # A label frame just short of the target is worth keeping too: stepping back from here
        # lands on it, and near the start of a clip (no seek allowed) that used to reopen the
        # clip and walk from frame 0 on every press of `a`.
        keep = (at in source.label_frames and index - at <= SEEK_BACKUP
                and at not in source.frame_cache)
        if landmark or keep:
            ok, passed = source.capture.retrieve()     # a landmark for a later seek to prove itself
            if ok:                                     # against; one frame in eight costs little
                source.sig_of.setdefault(at, frame_signature(passed))
                if keep:
                    source.remember_frame(at, passed)
        source.position += 1
    return True


class Source:
    """One clip in a run: its video, its rows, its landings, and the files its work goes back to."""

    def __init__(self, clip: str, video: Path, rows: List[Row], width: int, height: int,
                 out_csv: Path, review: Path, suspects: Optional[Path] = None, bounces: Optional[Path] = None):
        self.clip, self.video, self.rows = clip, video, rows
        self.width, self.height = width, height
        self.out_csv, self.review = out_csv, review
        self.suspects = suspect_json(clip, review.parent) if suspects is None else suspects
        self.bounces = bounce_csv(clip, review.parent) if bounces is None else bounces
        self.landings: Landings = {}  # the frames where the ball hit the ground (label_store.load_landings)
        for row in rows:
            row.clip = clip
        self.capture = cv2.VideoCapture(str(video))
        self.frame_cache: Dict[int, np.ndarray] = {}
        self.cache_limit = max(CACHE_FRAMES_MIN, min(CACHE_FRAMES_MAX, CACHE_BYTES // max(width * height * 3, 1)))
        self.position = 0             # next frame the capture will decode, so stepping can avoid seeks
        self.sig_of: Dict[int, bytes] = {}     # ... and back again, to check a frame is the one wanted
        self.warned_seek = False
        self.dirty = False
        self.cursor = 0               # where you were in this clip, for the review file
        self.saved_cursor = 0         # ... and what the review file last said, so a plain move is kept
        self.start = 0                # where its rows begin in the run's flat row list
        self.zoom = DEFAULT_ZOOM
        self.end: Optional[int] = None    # first frame index the video ran out at, once that is known
        self.missing: Set[int] = set()    # frames that could not be decoded; drawn black, never labelled
        self.label_frames = {row.frame for row in rows}   # the frames worth keeping when passed
        self.flagged = 0              # how many rows the last audit flagged, from apply_flags

    def release(self) -> None:
        """Let go of the decoder and the cache: each open capture holds 75-140 MB once it has
        decoded, and a long run otherwise keeps every clip it passed. decode_at reopens on demand."""
        self.frame_cache.clear()
        self.capture.release()
        self.position = -1

    # ---------------------------------------------------------------- frames
    def frame(self, index: int) -> np.ndarray:
        """Decode one frame, walking forward where possible: seeking a long GOP costs far more than
        decoding through it, and stepping frame by frame is what the tool does all day."""
        cached = self.frame_cache.get(index)
        if cached is not None:
            self.frame_cache.pop(index)
            self.frame_cache[index] = cached  # keep recently revisited frames, not only newly decoded ones
            return cached
        if index in self.missing:
            return self.placeholder(index)
        image = self.decode_at(index)
        for _ in range(DECODE_RETRIES):         # lost the thread: re-seek, now knowing where we are
            if image is not None:
                break
            image = self.decode_at(index)
        if image is None:
            # Not a picture to label: drawn black with a warning, and the edit keys refuse it. Better
            # than the crash this used to be, which threw the session out mid-clip.
            self.missing.add(index)
            print(f"[warn] {self.clip}: frame {index} could not be decoded"
                  + (f" (the video ends at frame {self.end})" if self.end is not None else ""))
            return self.placeholder(index)
        self.remember_frame(index, image)
        return image

    def decode_at(self, index: int) -> Optional[np.ndarray]:
        """Decode exactly `index`, or None if we ended up somewhere else.

        Neither counter can be trusted. On a clip cut and merged from two recordings, a seek past
        the join lands one frame early while CAP_PROP_POS_FRAMES still reports the frame asked for
        (POS_MSEC comes back negative, so that is no help either). The old code took the seek on
        faith, so one bad seek silently shifted everything decoded after it: the tool drew frame
        N-1 while the row being edited was N, and clicks made exactly on the ball were stored
        against the next row.

        Forwards, frames are walked sequentially from wherever the capture is, so identity is by
        count. Backwards, seek_and_resync tries a seek that is then proven by fingerprint against
        frames already walked past (see verified_seek); only when that cannot be proven does it
        fall back to reopening the clip and walking from the start. Fingerprints check consistency
        at an already known index; a lone one cannot identify a timestamp, which is why a seek
        needs two that differ. A failed decode is never returned as a valid annotation image.
        """
        if self.end is not None and index >= self.end:
            return None                                  # the video is known to stop before this
        if self.position < 0 or index < self.position:
            self.seek_and_resync(index)
            cached = self.frame_cache.get(index)         # a proven seek decodes through the frame
            if cached is not None:
                return cached
        if not walk_to(self, index):
            return None
        ok, image = self.capture.read()
        if not ok:
            ran_out(self)
            return None
        at, self.position = self.position, self.position + 1
        signature = frame_signature(image)
        if self.sig_of.get(index) not in (None, signature) and at == index:
            self.position = -1                           # not the frame we know sits here
            return None
        self.sig_of[at] = signature
        return image if at == index else None

    def seek_and_resync(self, index: int) -> None:
        """Get the capture to somewhere at or before `index` that we can vouch for.

        A seek that proves itself is tried first. Failing that, reopen at the start and let
        decode_at walk forward by count - slow on a long clip, but never wrong: compressed-video
        seeks can report a false index, and duplicate pixels cannot identify a timestamp.
        """
        if self.capture.isOpened() and self.verified_seek(index):
            return
        self.capture.release()
        self.capture = cv2.VideoCapture(str(self.video))
        self.position = 0

    def verified_seek(self, index: int) -> bool:
        """Seek to just before `index` and decode through it, believing the landing only once the
        decoded frames line up with fingerprints taken on an earlier pass. True when they did.

        Holding `a` used to reopen the clip and decode from frame 0 for every step outside the
        cache - seconds a press, deep into a long clip - because a seek could not be trusted: on
        a clip cut and merged from two recordings one lands a frame early while claiming
        otherwise. So the claim is checked: with the decoded run in hand, look for the one shift
        (a few frames either way) under which every frame whose fingerprint is on record matches,
        and insist on at least two matching frames that differ from each other, since a static
        stretch of identical frames fits any shift. The label frames decoded on the way are
        cached, which is what makes the next dozen presses of `a` free.
        """
        target = index - SEEK_BACKUP
        if target < SEEK_BACKUP:
            return False                                 # this near the start, walking is as quick
        if not any(k in self.sig_of for k in range(target, target + SEEK_VERIFY)):
            return False                                 # nothing on record here to prove it by
        if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, target):
            return False
        decoded, offset = self.decode_run(target, index)
        if offset is None:
            return False
        for claimed_at, signature, image in decoded:
            truly = claimed_at + offset
            if truly < 0:
                continue
            self.sig_of.setdefault(truly, signature)
            if image is not None and truly in self.label_frames:
                self.remember_frame(truly, image)
        self.position = decoded[-1][0] + offset + 1
        return True

    def decode_run(self, target: int, index: int) -> Tuple[List[Tuple[int, bytes, Optional[np.ndarray]]], Optional[int]]:
        """Read from the seek's claimed landing `target` through `index` and on until the run is
        proven (at most SEEK_VERIFY frames): [(claimed index, fingerprint, image or None)] and the
        proven shift, None when no shift fits. Only the newest cache_limit images are held."""
        decoded, claimed, offset = [], target, None
        while len(decoded) < SEEK_VERIFY:
            ok, image = self.capture.read()
            if not ok:
                break
            decoded.append((claimed, frame_signature(image), image))
            if len(decoded) > self.cache_limit:          # only the newest could be cached anyway:
                old = decoded[-self.cache_limit - 1]     # do not hold 64 full frames to find out
                decoded[-self.cache_limit - 1] = (old[0], old[1], None)
            claimed += 1
            if claimed > index:                          # through the frame wanted: proven yet?
                offset = self.match_offset(decoded)
                if offset is not None:
                    break
        return decoded, offset

    def match_offset(self, decoded) -> Optional[int]:
        """The shift that makes the tail of a decoded run agree with the fingerprints on record, or
        None if no shift does, or more than one does. Only the most recent frames are consulted,
        so the odd frame that decodes differently right after a seek cannot spoil the answer."""
        found = None
        for shift in range(-SEEK_SHIFTS, SEEK_SHIFTS + 1):
            matched = set()
            for claimed_at, signature, _ in decoded[-MATCH_RECENT:]:
                known = self.sig_of.get(claimed_at + shift)
                if known is None:
                    continue
                if known != signature:
                    matched = None
                    break
                matched.add(signature)
            if matched and len(matched) >= 2:
                if found is not None:
                    return None                          # two shifts fit: repeated content, read on
                found = shift
        return found

    def remember_frame(self, index: int, image: np.ndarray) -> None:
        """Keep a decoded frame, dropping the least recently used one when the budget is spent."""
        if index in self.frame_cache:
            self.frame_cache.pop(index)
        elif len(self.frame_cache) >= self.cache_limit:
            self.frame_cache.pop(next(iter(self.frame_cache)))
        self.frame_cache[index] = image

    def placeholder(self, index: int) -> np.ndarray:
        """A black frame saying the video has no picture here; never a frame to label."""
        image = np.zeros((self.height, self.width, 3), np.uint8)
        text = f"FRAME {index} NOT IN VIDEO"
        scale = max(1.0, self.width / 900.0)
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(image, text, ((self.width - text_w) // 2, (self.height + text_h) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2, cv2.LINE_AA)
        return image


def expand_clip_ranges(names: List[str], labels: Path) -> List[str]:
    """Turn what was typed into a clip list: 'a..b' means every clip with labels (or a draft) in
    `labels` from a to b in order, and repeats are dropped. Raises ValueError for a range whose
    ends have no labels."""
    available = sorted({clip_of(path) for path in label_files(labels) + draft_files(labels)}, key=natural_key)
    out: List[str] = []
    for name in names:
        if ".." in name:
            first, last = (part.strip() for part in name.split("..", 1))
            if first not in available or last not in available:
                raise ValueError(f"{first}..{last}: no labels for one of those in {labels}")
            lo, hi = available.index(first), available.index(last)
            out.extend(available[min(lo, hi):max(lo, hi) + 1])
        else:
            out.append(name)
    seen, unique = set(), []
    for name in out:                                  # keep the order, drop repeats
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def build_source(clip: str, videos: Path, labels: Path) -> Optional[Source]:
    """Everything one clip needs, or None with a message if it is not ready."""
    video = find_video(clip, videos)
    if video is None:
        print(f"[skip] {clip}: no video in {videos}")
        return None
    try:
        width, height = video_meta(video)[1:3]
    except RuntimeError as error:
        print(f"[skip] {clip}: {error}")
        return None                   # with a 0x0 frame every label would have read as invisible
    final, draft = label_csv(clip, labels), draft_csv(clip, labels)
    from_file = final if final.is_file() else draft
    if not from_file.is_file():
        print(f"[skip] {clip}: no labels; run  python finetune/training/pretrack.py --clips {clip}")
        return None
    try:
        rows = load_rows(from_file, width, height, "det" if draft.is_file() else "csv")
    except (OSError, LabelFormatError) as error:
        print(f"[skip] {clip}: {from_file.name} could not be read: {error}")
        return None
    if not rows:
        print(f"[skip] {clip}: {from_file.name} has no label rows")
        return None
    review = review_json(clip, labels)
    source = Source(clip, video, rows, width, height, final, review, suspect_json(clip, labels), bounce_csv(clip, labels))
    try:
        source.flagged = apply_flags(source.suspects, rows)
        state = apply_review(review, rows)
        source.landings = load_landings(source.bounces)
    except LabelFormatError as error:
        # Somebody's record of what they checked, or of where the ball landed: never started
        # afresh, never written over.
        print(f"[skip] {clip}: {error}")
        return None
    if state is not None:
        resume_from(source, state)
    return source


def resume_from(source: Source, state: dict) -> None:
    """Put the source's cursor and zoom where the review sidecar says the last session stopped."""
    try:
        cursor = int(state.get("cursor") or 0)
    except (TypeError, ValueError):
        cursor = 0
    index_of = {row.frame: i for i, row in enumerate(source.rows)}
    if state.get("frame") in index_of:            # rows can be added or dropped (check --fix, realign)
        cursor = index_of[state["frame"]]         # between sessions: the frame is what to come back to
    source.cursor = max(0, min(cursor, len(source.rows) - 1))
    source.saved_cursor = source.cursor
    try:
        source.zoom = float(state.get("zoom") or DEFAULT_ZOOM)
    except (TypeError, ValueError):
        source.zoom = DEFAULT_ZOOM


class Prefetcher:
    """Decodes the label frames ahead of the cursor while the tool is idle, so the next SPACE
    lands on a frame that is already in memory. Runs ahead the way the last move went, at the
    session's step, along the uncertain walk when there is one."""

    def __init__(self, session: "LabelSession") -> None:
        self.session = session

    def run(self) -> bool:
        """Decode not-yet-cached frames ahead of the cursor. True when there was work to do.

        Keeps going for about a screen refresh rather than one frame per idle tick: a run of SPACE
        presses at two or three a second was outrunning one decode per 30 ms wait."""
        session = self.session
        started, worked = time.monotonic(), False
        for index in self.upcoming_rows():
            if session.rows[index].clip != session.rows[session.cursor].clip:
                break                                    # do not disturb the next clip's capture yet
            frame = session.rows[index].frame
            if session.filter_name == "motion":
                self.warm_frame_before(index)
            if frame in session.source.frame_cache or frame in session.source.missing:
                continue
            session.source.frame(frame)
            worked = True
            if time.monotonic() - started > PREFETCH_BUDGET_S:
                break
        return worked

    def upcoming_rows(self) -> Sequence[int]:
        """The next PREFETCH row indices the way the last move went, at the current step."""
        session = self.session
        s = session.step                                    # n/d/p/a land every step-th row
        if session.direction >= 0:
            return (session.walk[session.walk_pos + s::s][:PREFETCH] if session.walk
                    else range(session.cursor + s, min(session.cursor + s * PREFETCH + 1, len(session.rows)), s))
        # heading back: run ahead of `a` instead
        return ((session.walk[session.walk_pos - s::-s][:PREFETCH] if session.walk_pos - s >= 0 else [])
                if session.walk else range(session.cursor - s, max(-1, session.cursor - s * PREFETCH - 1), -s))

    def warm_frame_before(self, index: int) -> None:
        """The motion view diffs a row against the row before it: decode that one too while it is
        still ahead of the capture (no seek), unless it is cached or known to be missing."""
        session = self.session
        if index > 0 and session.rows[index - 1].clip == session.rows[index].clip:
            before = session.rows[index - 1].frame
            if (before not in session.source.frame_cache and before not in session.source.missing
                    and before >= session.source.position >= 0):
                session.source.frame(before)
