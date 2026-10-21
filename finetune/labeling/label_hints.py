"""The label tool's suggestions: what ``refine.suggest`` measured for each clip's label frames,
kept current in the background as labels change, and the rules for when a suggestion is shown
beside a label, offered as the mark SPACE takes, or judged to disagree with a label."""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Tuple

from finetune.labeling import cnn_refiner, refine
from finetune.labeling.label_store import Row
from finetune.labeling.refine import Suggestion
from finetune.data.scoring import REFERENCE_HEIGHT

if TYPE_CHECKING:
    from finetune.labeling.label_source import Source

SUGGEST_SETTLE = 0.3      # seconds a clip's labels stay put before its suggestions are re-measured
SUGGEST_MEASURE = 16      # after an edit, label frames either side re-measured ...
SUGGEST_REPLACE = 8       # ... and either side replaced (the path reaches 8 frames for its points)
OFFER_UNSURE = True       # offer an unsure suggestion as the mark SPACE takes on an unreviewed frame
                          # when it agrees with the draft (refine.AGREE_PX_1080); a sure one always
                          # is. See Suggestions.offer_for for the measurement behind this
OFFER_MIN_PX = 0.25       # a suggestion this close to the draft is not offered: SPACE confirms the draft


class Suggestions:
    """Per clip, frame -> refine.Suggestion, with the labels they were measured from."""

    def __init__(self, sources: Iterable) -> None:
        self.sources = list(sources)
        self.found: Dict[str, Dict[int, Suggestion]] = {}   # clip -> frame -> suggestion
        self.measured_state: Dict[str, tuple] = {}          # ... and the labels they were measured from
        self.version = 0              # bumped as suggestions arrive, so the view redraws with them
        self.placer: Optional[cnn_refiner.Placer] = None   # the trained network placing suggestions, if any
        self.placer_model: Optional[Path] = None   # ... and the version it was loaded from
        self.closing = False          # tells the background thread to stop

    # ---------------------------------------------------------------- measuring
    def start(self, current: Callable[[], object], wait: bool = False) -> None:
        """Measure suggestions for every clip, the one you are in first, and then keep the clip you
        are in up to date: a background worker re-measures it shortly after its labels or what you
        have confirmed change, so a fix on one frame improves the suggestions for its neighbours.
        `current` says which clip that is. `wait` measures every clip once in the caller instead
        (tests, headless use)."""
        if wait:
            for source in self.order(current()):
                self.suggest_clip(source)
            return
        threading.Thread(target=self.worker, args=(current,), daemon=True, name="suggestions").start()

    def order(self, first: "Source") -> List["Source"]:
        return sorted(self.sources, key=lambda source: source is not first)

    @staticmethod
    def label_state(source: "Source") -> Tuple[Tuple[float, float, bool, bool], ...]:
        """Everything a clip's suggestions depend on: each label and whether you have checked it."""
        return tuple((row.x, row.y, row.visible, row.settled) for row in source.rows)

    def worker(self, current: Callable[[], object]) -> None:
        for source in self.order(current()):
            if self.closing:
                return
            self.suggest_clip(source)
        waiting: Dict[str, tuple] = {}
        while not self.closing:
            time.sleep(SUGGEST_SETTLE)
            source = current()
            state = self.label_state(source)
            if state == self.measured_state.get(source.clip):
                continue
            if waiting.get(source.clip) == state:     # unchanged for a moment: you have moved on
                self.suggest_clip(source, state)
            else:
                waiting[source.clip] = state

    def suggest_clip(self, source, state: Optional[tuple] = None) -> None:
        """Measure one clip's suggestions from its labels as they are now. Once a clip has them, only
        a stretch around the frames that changed is re-measured: an edit reaches the suggestions at
        most a few label frames away, and Pomona's full pass (1,837 1080p frames) takes 9 s."""
        state = state or self.label_state(source)
        seeds = {row.frame: (row.x, row.y) if row.visible else None for row in source.rows}
        confirmed = [row.frame for row in source.rows if row.settled and row.visible]
        before = self.measured_state.get(source.clip)
        only = keep = None
        if before is not None and len(before) == len(state) and source.clip in self.found:
            changed = [i for i, (old, new) in enumerate(zip(before, state)) if old != new]
            if not changed:
                return
            frames = [row.frame for row in source.rows]
            only = {frames[j] for i in changed for j in range(max(0, i - SUGGEST_MEASURE), min(len(frames), i + SUGGEST_MEASURE + 1))}
            keep = {frames[j] for i in changed for j in range(max(0, i - SUGGEST_REPLACE), min(len(frames), i + SUGGEST_REPLACE + 1))}
        try:
            found = refine.suggest(source.video, seeds, refine.PassOptions(
                refine.label_spacing(seeds), confirmed=confirmed, only=only, placer=self.current_placer(),
                stop=lambda: self.closing))
        except Exception as error:            # a hint is never worth losing the session over
            print(f"[hint] {source.clip}: no suggestions ({error})")
            return
        if self.closing:
            return
        first = source.clip not in self.found
        if keep is not None:                  # replace the stretch around the edit, keep the rest
            merged = {f: hint for f, hint in self.found[source.clip].items() if f not in keep}
            merged.update({f: hint for f, hint in found.items() if f in keep})
            found = merged
        self.found[source.clip] = found
        self.measured_state[source.clip] = state
        self.version += 1
        if first:
            trusted = sum(1 for hint in found.values() if hint["trusted"])
            print(f"[hint] {source.clip}: suggestions for {len(found)} of {len(source.rows)} frames, "
                  f"{trusted} trusted")

    def current_placer(self) -> Optional[cnn_refiner.Placer]:
        """The current cnn_refiner version, loaded on first use and again after a retrain switches
        versions; None (the motion centroid alone) when there is none or it cannot load."""
        model = cnn_refiner.current()
        if model != self.placer_model:
            self.placer_model = model
            self.placer = cnn_refiner.load_current() if model is not None else None
            if self.placer is not None:
                print(f"[hint] placing suggestions with {self.placer.name}")
        return self.placer

    # ---------------------------------------------------------------- what a row is shown
    def for_row(self, row: Row, height: int) -> Optional[Suggestion]:
        """The suggestion for a row, or None when there is none worth showing: nothing ball-like moved
        there, the label has moved since it was measured (a fresh one follows within a second or
        so), or the frame is labelled "no ball" and the suggestion is not trusted - a ball where the
        label says none is only offered when the video clearly shows one on the ball's path."""
        hint = self.found.get(row.clip, {}).get(row.frame)
        if hint is None or not hint["ball_here"]:
            return None
        seed = hint["seed"]
        if not row.visible:
            return hint if seed is None and hint["trusted"] else None
        if seed is None:
            return None
        if math.hypot(row.x - seed[0], row.y - seed[1]) > refine.centroid_radius_px(height) / 2.0:
            return None
        return hint

    def offer_for(self, row: Row, height: int) -> Optional[Suggestion]:
        """The suggestion that stands in for a frame's label until you have checked the frame: it is
        drawn as the mark, the mouse parks on it, and SPACE takes it - so an unreviewed frame asks
        one question, "is the mark on the ball?", instead of which of two marks is closer.

        Measured on 63 reviewed clips seeded from their original drafts, each with a network that
        never saw the clip (1080p px, against the hand label): a sure suggestion is within 1 px on
        91% of frames and over 5 px on 0.1%, against 26% and 5% for the draft, so it is always
        offered. An unsure one is closer than the draft on 78% of frames, but on the GridTrackNet
        clips (drafts about 1 px off) one that moved far from the draft is usually wrong - over 5 px
        on 28% once it moved 4-6 px, 1% within 3 px - so an unsure suggestion is offered only when
        it agrees with the draft (within refine.AGREE_PX_1080, OFFER_UNSURE). On those clips that
        puts the mark within 1 px of the hand label on 73% of frames (the draft: 39%) and over 5 px
        on 3.7% (the draft: 5.3%). An unsure suggestion further away stays a dull hint beside the
        draft, which y takes. None once you have reviewed or edited the frame (your label is then
        the mark), when there is no suggestion worth showing, or when it is within OFFER_MIN_PX of
        the draft (nothing to choose: SPACE confirms the draft)."""
        if row.settled:
            return None
        hint = self.for_row(row, height)      # on a no-ball row, only ever a trusted one
        if hint is None or not row.visible:
            return hint
        gap = math.hypot(hint["x"] - row.x, hint["y"] - row.y)
        if gap < OFFER_MIN_PX:
            return None
        if not hint["trusted"] and not (OFFER_UNSURE and gap * REFERENCE_HEIGHT / height <= refine.AGREE_PX_1080):
            return None
        return hint

    def verdict(self, row: Row, height: int) -> Optional[str]:
        """"agrees" when the label is within AGREE_PX_1080 of a trusted suggestion, "differs" when it
        is further, None when there is no trusted suggestion to judge by. Measured on the verified
        clips, trusted suggestions are more than 3 px off 2% of the time; "differs" is a frame worth
        a second look, not a verdict on it."""
        hint = self.for_row(row, height)
        if hint is None or not hint["trusted"] or not row.visible:
            return None
        gap = math.hypot(hint["x"] - row.x, hint["y"] - row.y) * REFERENCE_HEIGHT / height
        return "agrees" if gap <= refine.AGREE_PX_1080 else "differs"
