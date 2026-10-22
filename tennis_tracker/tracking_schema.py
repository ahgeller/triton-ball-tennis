"""The tracking JSON's shape.  ``export.write_tracking_json`` writes a ``TrackingJson``; the
scorers (``evaluation.validate``, ``evaluation.pipeline_score``) and ``finetune.training.pretrack``
read its ``frames`` table of ``FrameRow``s, and ``evaluation.court_events_score`` its
``court_events`` block.  Keys are listed in the order they are written.

Only ``typing`` is imported, so a reader can use these types at runtime without loading the
tracker's OpenCV, CUDA and TensorRT stack.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict, Union


JsonValue = Union[None, bool, int, float, str, List["JsonValue"], Dict[str, "JsonValue"]]
# ``ball_in_play_selector.models.PositionKind``, spelled again so this module stays typing-only;
# ``tests/test_export.py`` pins the two to each other.
PositionKind = Literal["measurement", "filtered", "recovered"]


class MeasurementJson(TypedDict):
    """The detector's own location of the ball, kept apart from the selected (possibly refitted) position."""

    x: float
    y: float
    conf: float
    bbox: Optional[List[float]]


class SearchJson(TypedDict):
    """Where the selector looked for a motion blob this frame (radius 0 when it did not)."""

    x: float
    y: float
    radius: float


class SelectionJson(TypedDict):
    source: str
    reason: str


class _FrameRowAlways(TypedDict):
    frame: int
    present: bool


class FrameRow(_FrameRowAlways, total=False):
    """One row of the ``frames`` table.  ``frame`` and ``present`` are always there; the
    position keys only when the ball was selected that frame (``frame_result_to_json``);
    ``court_keypoints`` and ``player_boxes`` for every frame pass 1 saw (``RunResult.frame_rows``).
    """

    x: float
    y: float
    conf: float
    source: str
    interpolated: bool
    bbox: Optional[List[float]]
    measurement: Optional[MeasurementJson]
    position_kind: PositionKind
    track_id: Optional[int]             # the selector track whose observation this is; None for a recovered row
    search: SearchJson
    selection: SelectionJson
    court_keypoints: Optional[List[float]]
    player_boxes: Dict[str, Optional[List[float]]]


class ObservationJson(TypedDict):
    frame: int
    x: float
    y: float
    bbox: List[float]
    conf: float
    area: float
    on_motion: bool


class TrackStartJson(TypedDict):
    """Why the detection that started a track joined no live track: ``reason`` is "first" when
    none was alive, "outbid" when ``rival_track`` accepted it but took another detection, else
    the association gate that turned it away from ``rival_track``."""

    reason: str
    rival_track: Optional[int]


class TrackJson(TypedDict):
    """A selector track: its score, how it began and ended, what the selection made of it and
    every detection it holds (``track_to_json``)."""

    track_id: int
    score: float
    num_obs: int
    span: int
    first_frame: int
    last_obs_frame: int
    start: TrackStartJson
    end_reason: str                     # "max_gap" or "clip_end"
    selection: Dict[str, bool]          # core.selected_tracks' tests: established, reacquired, airborne, rolling
    in_timeline: bool                   # whether the timeline chain used it
    score_breakdown: Dict[str, float]
    observations: List[ObservationJson]


class MotionCandidate(TypedDict):
    """One blob of a frame's motion mask, ranked for the diagnostics (``_extract_motion_candidates``)."""

    frame: int
    mask: str
    x: float
    y: float
    bbox: List[int]
    area: float
    compactness: float
    aspect_ratio: float
    fill_ratio: float
    distance_to_search: Optional[float]
    distance_to_selected: Optional[float]
    score: float


class MotionSearchJson(TypedDict):
    x: Optional[float]
    y: Optional[float]
    radius: float
    source: str


class MotionPositionJson(TypedDict):
    x: Optional[float]
    y: Optional[float]


class MotionDiagnosticFrame(TypedDict):
    frame: int
    selected_source: Optional[str]
    reason: str
    has_detections: bool
    mask_source: str
    has_boost_mask: bool
    has_raw_motion: bool
    search: MotionSearchJson
    selected_position: MotionPositionJson
    candidate_count_exported: int
    candidates: List[MotionCandidate]


class MotionDiagnosticsSummary(TypedDict):
    frames_analyzed: int
    detector_gap_frames: int
    detector_gap_frames_with_motion_candidate: int
    reason_counts: Dict[str, int]
    mask_source_counts: Dict[str, int]
    selected_source_counts: Dict[str, int]
    max_candidates_per_frame: int


class MotionDiagnostics(TypedDict, total=False):
    """``build_motion_diagnostics``'s record; empty (``{}``) when the run did not ask for it."""

    schema_version: int
    summary: MotionDiagnosticsSummary
    frames: List[MotionDiagnosticFrame]


# ``tennis_tracker.court.court_scene``'s EventOrigin and EventPosition, spelled again here for
# the same reason as PositionKind; ``tests/test_export.py`` pins them to each other.
EventOrigin = Literal["model", "kink", "gap"]
EventPosition = Literal["measurement", "filtered", "recovered", "extrapolated"]


class BallPointJson(TypedDict):
    """A ball position in native pixels and where it came from (``court_scene.BallPoint``)."""

    x: float
    y: float
    position: EventPosition


class BounceEventJson(TypedDict):
    """A ball landing in native pixels (``court_scene.BounceEvent``): ``origin`` names the stage
    that found it, ``position`` where its point came from."""

    frame: int
    x: float
    y: float
    score: float
    origin: EventOrigin
    position: EventPosition


class ContactEventJson(TypedDict):
    """A racket contact at the player's feet (``court_scene.ContactEvent``); ``ball`` is the
    ball sample it was judged at, None when the stage had none."""

    frame: int
    x: float
    y: float
    player_id: JsonValue
    score: float
    source: str
    ball: Optional[BallPointJson]


class RallyLegJson(TypedDict):
    """One contact-to-landing leg in unit court coordinates (``court_scene.RallyLeg``), its
    points as ``[u, v]`` lists."""

    start_frame: int
    end_frame: int
    start: List[float]
    end: List[float]
    phase: str
    outcome: str
    quality: Optional[str]
    start_quality: str
    start_source: str
    landing_origin: Optional[EventOrigin]
    landing_position: Optional[EventPosition]


class CourtEventsJson(TypedDict, total=False):
    """``export.court_events_to_json``'s record of the court-event stages' output; empty (``{}``)
    when the run did not compute them, every key present (lists possibly empty) when it did."""

    bounces: List[BounceEventJson]              # confirmed by the bounce model
    bounce_candidates: List[BounceEventJson]    # the model's low-score candidates
    kinematic_bounces: List[BounceEventJson]    # kinks and gap fits the model missed
    contacts: List[ContactEventJson]
    rally_legs: List[RallyLegJson]


class VideoJson(TypedDict):
    input: str
    output: str
    fps: float
    width: int
    height: int
    total_frames: int


class SummaryJson(TypedDict):
    filled_frames: int
    filled_percent: float
    elapsed_sec: float
    effective_fps: float
    pass2_frames_rendered: int
    chosen_track_id: Optional[int]
    track_count: int
    timeline_track_ids: List[int]


class TrackingFrames(TypedDict):
    """The part of the tracking JSON a scorer reads: the ``frames`` table and the court as
    last seen (``headless.tracking_payload`` builds just this from a ``RunResult``)."""

    frames: List[FrameRow]
    last_valid_court_keypoints: Optional[List[float]]


class TrackingJson(TrackingFrames):
    """The whole file ``write_tracking_json`` writes."""

    schema_version: int
    video: VideoJson
    summary: SummaryJson
    config: Dict[str, JsonValue]
    timing: Dict[str, float]
    chosen_track: Optional[TrackJson]
    tracks: List[TrackJson]
    motion_diagnostics: MotionDiagnostics
    court_events: CourtEventsJson
