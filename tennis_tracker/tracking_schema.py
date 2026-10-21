"""The tracking JSON's shape.  ``export.write_tracking_json`` writes a ``TrackingJson``; the
scorers (``evaluation.validate``, ``evaluation.pipeline_score``) and ``finetune.training.pretrack``
read its ``frames`` table of ``FrameRow``s.  Keys are listed in the order they are written.

Only ``typing`` is imported, so a reader can use these types at runtime without loading the
tracker's OpenCV, CUDA and TensorRT stack.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict, Union


JsonValue = Union[None, bool, int, float, str, List["JsonValue"], Dict[str, "JsonValue"]]
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


class TrackJson(TypedDict):
    """A selector track: its score and every detection it holds (``track_to_json``)."""

    track_id: int
    score: float
    num_obs: int
    span: int
    first_frame: int
    last_obs_frame: int
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
