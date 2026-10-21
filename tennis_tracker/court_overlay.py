"""Court events and the optional top-right court minibar.

The stages run in this order (see ``pipeline.run``):

1. ``predict_bounce_events`` scores ball landings with the upstream CatBoost
   bounce model and returns confirmed bounces plus low-score candidates.
2. ``detect_player_contacts`` finds racket contacts near player boxes.
3. ``infer_kinematic_bounces`` finds strict down-to-up kinks the model missed.
4. ``assemble_rally_legs`` projects all of them onto the court plane and links
   each contact to its landing.
5. ``draw_court_minimap`` draws the latest completed leg on every frame.

Every event stage takes ``(per_frame, player_boxes_by_frame, fps, width,
height, ...)`` in that order.  Events are ``BounceEvent`` / ``ContactEvent``
named tuples, so plain ``(frame, x, y, score)`` and ``(frame, x, y, player,
score, source)`` tuples are accepted wherever events go in.  Everything is
image-space evidence projected through the court homography; nothing here is
a measured 3D position.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

import cv2
import numpy as np

from ball_in_play_selector import FrameResult

from .court_geometry import (
    COURT_LENGTH_M,
    DOUBLES_WIDTH_M,
    NET_V,
    SERVICE_DEPTH,
    SINGLES_INSET,
    build_ground_projection_model,
    court_keypoints_key,
    homography_apply,
)


Box = Sequence[float]
PlayerBoxes = Mapping[Hashable, Optional[Box]]
PlayerBoxesByFrame = Sequence[Optional[PlayerBoxes]]
BallTrack = Sequence[Optional[FrameResult]]
CourtKeypoints = Sequence[float]
CourtPoint = Tuple[float, float]
Color = Tuple[int, int, int]


# --- Bounce model -----------------------------------------------------------
_BOUNCE_THRESHOLD = 0.45  # CatBoost score above which a landing is confirmed
_INFERRED_BOUNCE_THRESHOLD = 0.20  # lower bar for guarded landing candidates
_EPS = 1e-15  # keeps the model's ratio features finite
# The model was trained on 1280x720 tracks sampled at 30 FPS.
_MODEL_FPS = 30.0
_MODEL_WIDTH = 1280.0
_MODEL_HEIGHT = 720.0
# Two scored landings closer than this are the same landing.
_BOUNCE_MIN_SEPARATION_S = 0.10

# --- Image-space event detection ---------------------------------------------
# Pixel thresholds were tuned on 1920x1080 video; this is its diagonal.
_REFERENCE_DIAGONAL_PX = 2203.0
# Gap-fit velocities are per source frame; their thresholds were tuned at 60 FPS.
_GAP_SPEED_REFERENCE_FPS = 60.0
_MIN_DET_IN_WINDOW = 3  # detections required in a five-sample window
_MIN_COS_DENOMINATOR = 1e-6
_SHORT_GAP_MIN_S = 0.05  # shortest selector gap that may hide a turn
_SHORT_GAP_MAX_S = 0.30  # longest selector gap that may hide a turn
_EVENT_MERGE_S = 0.15  # events of one kind closer than this are one event
# Gap contacts are decisive evidence, so they outrank any kink score when
# contacts by the same player are merged.
_GAP_CONTACT_SCORE_BONUS = 1000.0
# A turn this far down a player's box is a landing at the feet, not a stroke.
_LOWER_BODY_FRACTION = 0.40
_PLAYER_PAD_MIN_PX = 24.0  # horizontal padding around a player box

# --- Rally legs ---------------------------------------------------------------
_MAX_FLIGHT_S = 2.50  # longest contact-to-landing flight that forms a leg
_RALLY_RESET_S = 2.50  # pause after which the next leg may be a serve
_CONTACT_BOUNCE_COLLISION_S = 0.12  # a contact this close to a bounce is the bounce
_SAME_SIDE_BOUNCE_S = 1.20  # second same-side bounce within this is not a new leg
_RESCUE_RADIUS_DIAGONAL = 0.10  # ball-to-player reach for a rescued contact
_SIDE_DEAD_ZONE = 0.03  # court v within this of the net has no side (0.47..0.53)
_VOLLEY_MAX_NET_DISTANCE = 0.27  # a volley is struck inside the service boxes
# Court-plane margins (unit court) for accepting projected events.
_CONFIRMED_BOUNCE_BOUNDS = (-0.08, 1.08)
_INFERRED_BOUNCE_BOUNDS = (-0.05, 1.05)
_CONTACT_BOUNDS = (-0.12, 1.12)

# --- Court geometry (unit-court fractions come from court_geometry) ------------
_SERVE_BASELINE_BAND = 0.15  # a serve starts within this of a baseline

# --- Minimap drawing ----------------------------------------------------------
_PANEL_BG: Color = (42, 42, 42)
_PANEL_COURT: Color = (28, 72, 42)
_PANEL_LINE: Color = (235, 235, 235)
_PANEL_NET: Color = (90, 180, 255)
_PANEL_BORDER: Color = (230, 230, 230)
_SHADOW: Color = (12, 12, 12)
_MARKER_OUTLINE: Color = (15, 15, 15)
_INFERRED_BOUNCE_COLOR: Color = (0, 165, 255)
_CONFIRMED_BOUNCE_COLOR: Color = (0, 255, 255)
_DEFAULT_PHASE_COLOR: Color = (70, 220, 255)
_PHASE_COLORS: Dict[str, Color] = {
    "serve": (255, 210, 60),
    "rally": _DEFAULT_PHASE_COLOR,
    "fault": (70, 70, 255),
}
_PLAYER_COLORS: Tuple[Color, ...] = (
    (255, 110, 60), (70, 90, 255), (255, 80, 220), (80, 230, 120),
)
_PANEL_POINT_BOUNDS = (-0.08, 1.08)
_PANEL_OPACITY = 0.90
_FRAME_OPACITY = 0.10  # video showing through the panel
_ARC_STEPS = 10  # dots along the drawn leg chord
_DASH_PX = 7.0


class BounceEvent(NamedTuple):
    """A ball landing in image pixels; ``score`` is model or kink strength."""

    frame: int
    x: float
    y: float
    score: float


class ContactEvent(NamedTuple):
    """A racket contact anchored at the player's feet.

    ``source`` is ``"kink"`` (visible direction change) or ``"gap"`` (a
    reversing occlusion gap next to the player).
    """

    frame: int
    x: float
    y: float
    player_id: Hashable
    score: float
    source: str


class BouncePrediction(NamedTuple):
    """Bounce-model output: confirmed landings and guarded low-score candidates."""

    confirmed: List[BounceEvent]
    candidates: List[BounceEvent]


class _RallyLegRequired(TypedDict):
    start_frame: int
    end_frame: int
    start: CourtPoint
    end: CourtPoint
    phase: str  # "serve" | "rally" | "fault"
    outcome: str  # "bounce" | "volley"
    quality: Optional[str]  # landing: "confirmed" | "inferred" | None (volley)


class RallyLeg(_RallyLegRequired, total=False):
    """One completed contact-to-landing leg in unit court coordinates."""

    start_quality: str  # origin: "confirmed" | "inferred"


class _PlacedBounce(NamedTuple):
    frame: int
    point: CourtPoint
    quality: Optional[str]
    score: float


class _PlacedContact(NamedTuple):
    frame: int
    point: CourtPoint
    player: Hashable
    score: float
    source: str  # "kink" | "gap" | "inferred"


_Sample = Tuple[int, FrameResult]


class _GapTurn(NamedTuple):
    start: int
    end: int
    before: List[_Sample]  # two samples before the gap, oldest first
    after: List[_Sample]  # two samples after the gap, oldest first
    history: List[_Sample]  # up to 6 model samples before the gap, oldest first
    incoming: np.ndarray
    outgoing: np.ndarray

    @property
    def midpoint(self) -> int:
        return (self.start + self.end) // 2


def _coerce_bounces(events: Optional[Iterable[Sequence[Any]]]) -> List[BounceEvent]:
    return [
        event if isinstance(event, BounceEvent) else BounceEvent(*event)
        for event in (events or ())
    ]


def _coerce_contacts(events: Optional[Iterable[Sequence[Any]]]) -> List[ContactEvent]:
    return [
        event if isinstance(event, ContactEvent) else ContactEvent(*event)
        for event in (events or ())
    ]


def _frames(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


def _model_stride(fps: float) -> int:
    """Source frames between the 30 FPS samples the models were tuned on."""
    return max(1, int(round(float(fps) / _MODEL_FPS)))


def _resolution_scale(width: int, height: int) -> Tuple[float, float]:
    """Return (frame diagonal, pixel scale relative to 1080p)."""
    diagonal = float(np.hypot(width, height))
    return diagonal, diagonal / _REFERENCE_DIAGONAL_PX


def _usable_point(per_frame: BallTrack, frame: int) -> Optional[FrameResult]:
    if frame < 0 or frame >= len(per_frame):
        return None
    return per_frame[frame]


def _det_windows(
    per_frame: BallTrack, stride: int
) -> Iterator[Tuple[int, List[FrameResult]]]:
    """Yield (centre, five samples at +/-2*stride) windows with enough detections.

    Centre on every frame but keep the 30 FPS sample spacing: stepping by
    stride would only ever centre on even frames at 60 FPS, so an event on an
    odd frame could never be scored.
    """
    for frame in range(2 * stride, len(per_frame) - 2 * stride):
        points = [_usable_point(per_frame, frame + offset * stride)
                  for offset in (-2, -1, 0, 1, 2)]
        if any(point is None for point in points):
            continue
        # Do not let a mostly synthetic window manufacture an event.
        if sum(point.source == "det" for point in points) < _MIN_DET_IN_WINDOW:
            continue
        yield frame, points


def _window_velocities(points: Sequence[FrameResult]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample velocity into and out of the window centre."""
    centre = points[2]
    incoming = np.array([
        float(centre.cx) - float(points[0].cx),
        float(centre.cy) - float(points[0].cy),
    ]) / 2.0
    outgoing = np.array([
        float(points[4].cx) - float(centre.cx),
        float(points[4].cy) - float(centre.cy),
    ]) / 2.0
    return incoming, outgoing


def _turn_metrics(incoming: np.ndarray, outgoing: np.ndarray) -> Tuple[float, float, float]:
    """Return (incoming speed, outgoing speed, cosine of the turn)."""
    in_speed = float(np.linalg.norm(incoming))
    out_speed = float(np.linalg.norm(outgoing))
    cosine = float(np.dot(incoming, outgoing) / max(in_speed * out_speed, _MIN_COS_DENOMINATOR))
    return in_speed, out_speed, cosine


def _build_bounce_features(
    per_frame: BallTrack, fps: float, width: int, height: int
) -> Tuple[np.ndarray, List[int], int]:
    """Build the upstream CatBoost model's five-point, 30 FPS feature rows."""
    stride = _model_stride(fps)
    x_scale = _MODEL_WIDTH / max(float(width), 1.0)
    y_scale = _MODEL_HEIGHT / max(float(height), 1.0)
    rows = []
    center_frames = []

    for frame, points in _det_windows(per_frame, stride):
        xs = np.asarray([float(point.cx) * x_scale for point in points], dtype=np.float64)
        ys = np.asarray([float(point.cy) * y_scale for point in points], dtype=np.float64)
        if not np.isfinite(xs).all() or not np.isfinite(ys).all():
            continue

        x_p1, x_p2 = abs(xs[1] - xs[2]), abs(xs[0] - xs[2])
        x_n1, x_n2 = abs(xs[3] - xs[2]), abs(xs[4] - xs[2])
        y_p1, y_p2 = ys[1] - ys[2], ys[0] - ys[2]
        y_n1, y_n2 = ys[3] - ys[2], ys[4] - ys[2]
        rows.append([
            x_p1, x_p2, x_n1, x_n2,
            abs(x_p1 / (x_n1 + _EPS)), abs(x_p2 / (x_n2 + _EPS)),
            y_p1, y_p2, y_n1, y_n2,
            y_p1 / (y_n1 + _EPS), y_p2 / (y_n2 + _EPS),
        ])
        center_frames.append(frame)

    return np.asarray(rows, dtype=np.float32).reshape(-1, 12), center_frames, stride


def _cluster_bounce_candidates(
    center_frames: Sequence[int],
    scores: Sequence[float],
    gap: int,
    threshold: float = _BOUNCE_THRESHOLD,
) -> List[Tuple[int, float]]:
    """Collapse scored frames into one event per run separated by > gap."""
    clusters: List[List[Tuple[int, float]]] = []
    for frame, score in zip(center_frames, scores):
        score = float(score)
        if not np.isfinite(score) or score <= threshold:
            continue
        item = (int(frame), score)
        if not clusters or item[0] - clusters[-1][-1][0] > gap:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    return [max(cluster, key=lambda item: item[1]) for cluster in clusters]


def _inside_player(x: float, y: float, boxes: Optional[PlayerBoxes]) -> bool:
    """True when (x, y) is on or just beside a player (feet excluded below y2)."""
    for box in (boxes or {}).values():
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = map(float, box[:4])
        pad_x = max(_PLAYER_PAD_MIN_PX, x2 - x1)
        pad_y = max(_PLAYER_PAD_MIN_PX, 0.25 * (y2 - y1))
        if x1 - pad_x <= x <= x2 + pad_x and y1 - pad_y <= y <= y2:
            return True
    return False


def _catboost_regressor() -> Any:
    """Return the CatBoostRegressor class, failing with an install hint."""
    # CatBoost is only needed for --court, so it is imported on first use
    # rather than making every pipeline import depend on it.
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("--court requires CatBoost: python -m pip install catboost") from exc
    return CatBoostRegressor


def _off_player_events(
    pairs: Iterable[Tuple[int, float]],
    per_frame: BallTrack,
    player_boxes_by_frame: Optional[PlayerBoxesByFrame],
) -> List[BounceEvent]:
    """Attach ball positions to scored frames, dropping those on a player."""
    result = []
    for frame, score in pairs:
        point = per_frame[frame]
        if point is None:
            continue
        x, y = float(point.cx), float(point.cy)
        boxes = (
            player_boxes_by_frame[frame]
            if player_boxes_by_frame is not None and frame < len(player_boxes_by_frame)
            else None
        )
        if not _inside_player(x, y, boxes):
            result.append(BounceEvent(frame, x, y, score))
    return result


def predict_bounce_events(
    per_frame: BallTrack,
    player_boxes_by_frame: Optional[PlayerBoxesByFrame],
    fps: float,
    width: int,
    height: int,
    model_path: str,
) -> BouncePrediction:
    """Score landings with the CatBoost bounce model.

    Returns confirmed bounces (score above 0.45) and guarded candidates (above
    0.20) that ``assemble_rally_legs`` may use when no confirmed landing exists.
    Positions on a player box are dropped from both.
    """
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Bounce model not found: {path}")
    regressor = _catboost_regressor()
    features, center_frames, stride = _build_bounce_features(
        per_frame, fps, width, height
    )
    if not center_frames:
        return BouncePrediction([], [])

    model = regressor()
    model.load_model(str(path))
    scores = np.asarray(model.predict(features, thread_count=1), dtype=np.float64).reshape(-1)

    # Scoring every frame makes candidates dense, so separate events by a
    # physical interval rather than the sample stride: two real landings are
    # never 0.1 s apart, but one landing can easily score over several frames.
    gap = max(stride, _frames(_BOUNCE_MIN_SEPARATION_S, float(fps)))
    confirmed = _off_player_events(
        _cluster_bounce_candidates(center_frames, scores, gap),
        per_frame, player_boxes_by_frame,
    )
    candidates = _off_player_events(
        _cluster_bounce_candidates(center_frames, scores, gap, _INFERRED_BOUNCE_THRESHOLD),
        per_frame, player_boxes_by_frame,
    )
    return BouncePrediction(confirmed, candidates)


def _box_distance(x: float, y: float, box: Box) -> float:
    x1, y1, x2, y2 = map(float, box[:4])
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy))


def _nearest_player(
    x: float, y: float, boxes: Optional[PlayerBoxes]
) -> Optional[Tuple[float, Hashable, Box]]:
    best = None
    for player_id, box in (boxes or {}).items():
        if box is None or len(box) < 4:
            continue
        distance = _box_distance(x, y, box)
        if best is None or distance < best[0]:
            best = (distance, player_id, box)
    return best


def _fit_gap_turn(before: Sequence[_Sample], after: Sequence[_Sample]) -> Optional[tuple]:
    """Intersect incoming/outgoing image-space fits inside a short gap.

    Returns (incoming, outgoing, turn_frame, point, mismatch) or None.
    """
    before_a, before_b = before
    after_a, after_b = after
    incoming = np.array([
        float(before_b[1].cx) - float(before_a[1].cx),
        float(before_b[1].cy) - float(before_a[1].cy),
    ]) / max(1, before_b[0] - before_a[0])
    outgoing = np.array([
        float(after_b[1].cx) - float(after_a[1].cx),
        float(after_b[1].cy) - float(after_a[1].cy),
    ]) / max(1, after_b[0] - after_a[0])
    before_origin = np.array([
        float(before_b[1].cx), float(before_b[1].cy)
    ]) - incoming * before_b[0]
    after_origin = np.array([
        float(after_a[1].cx), float(after_a[1].cy)
    ]) - outgoing * after_a[0]
    offset = before_origin - after_origin
    slope = incoming - outgoing
    if incoming[1] > 0.0 > outgoing[1] and abs(slope[1]) >= 1e-8:
        # For a bounce, impact time is where descending and ascending height
        # branches meet; horizontal disagreement is detector noise.
        turn_frame = float(-offset[1] / slope[1])
    else:
        denominator = float(np.dot(slope, slope))
        if denominator < 1e-8:
            return None
        turn_frame = float(-np.dot(offset, slope) / denominator)
    incoming_point = before_origin + incoming * turn_frame
    outgoing_point = after_origin + outgoing * turn_frame
    point = 0.5 * (incoming_point + outgoing_point)
    mismatch = float(np.linalg.norm(incoming_point - outgoing_point))
    return incoming, outgoing, turn_frame, point, mismatch


def _extrapolate_incoming(samples: Sequence[_Sample], frame: int) -> Optional[np.ndarray]:
    if len(samples) < 2:
        return None
    anchor = float(samples[-1][0])
    times = np.asarray([float(item[0]) - anchor for item in samples])
    degree = min(2, len(samples) - 1)
    target = float(frame) - anchor
    x_coeff = np.polyfit(times, [float(item[1].cx) for item in samples], degree)
    y_coeff = np.polyfit(times, [float(item[1].cy) for item in samples], degree)
    return np.array([
        float(np.polyval(x_coeff, target)),
        float(np.polyval(y_coeff, target)),
    ])


def _usable_samples(
    per_frame: BallTrack, frames: Iterable[int], limit: Optional[int] = None
) -> List[_Sample]:
    samples = []
    for index in frames:
        point = _usable_point(per_frame, index)
        if point is not None:
            samples.append((index, point))
            if limit is not None and len(samples) == limit:
                break
    return samples


def _gap_turns(per_frame: BallTrack, fps: float, stride: int) -> Iterator[_GapTurn]:
    """Yield short selector gaps whose flanking samples fit a turn."""
    min_gap = max(2, _frames(_SHORT_GAP_MIN_S, fps))
    max_gap = max(min_gap, _frames(_SHORT_GAP_MAX_S, fps))
    count = len(per_frame)
    frame = 0
    while frame < count:
        if _usable_point(per_frame, frame) is not None:
            frame += 1
            continue
        start = frame
        while frame < count and _usable_point(per_frame, frame) is None:
            frame += 1
        end = frame - 1
        if not (min_gap <= end - start + 1 <= max_gap):
            continue
        before = _usable_samples(
            per_frame, range(start - 1, max(-1, start - 4 * stride), -1), limit=2
        )
        after = _usable_samples(
            per_frame, range(end + 1, min(count, end + 4 * stride + 1)), limit=2
        )
        if len(before) < 2 or len(after) < 2:
            continue
        before.reverse()
        fit = _fit_gap_turn(before, after)
        if fit is None:
            continue
        history = _usable_samples(
            per_frame, range(start - 1, max(-1, start - 6 * stride - 1), -1)
        )
        history.reverse()
        yield _GapTurn(start, end, before, after, history, fit[0], fit[1])


def _striking_player(
    ball: FrameResult, boxes: Optional[PlayerBoxes], diagonal: float
) -> Optional[Tuple[Hashable, Box]]:
    """The player whose box the ball is touching at or above the feet."""
    nearest = _nearest_player(float(ball.cx), float(ball.cy), boxes)
    if nearest is None or nearest[0] > 0.045 * diagonal:
        return None
    _, player_id, box = nearest
    _, y1, _, y2 = map(float, box[:4])
    if float(ball.cy) > y2 + 0.05 * max(y2 - y1, 1.0):
        return None
    return player_id, box


def _kink_contact_score(points: Sequence[FrameResult], scale: float) -> Optional[float]:
    """Score a racket-like turn (sharp, or a sudden speed-up), else None."""
    incoming, outgoing = _window_velocities(points)
    in_speed, out_speed, cosine = _turn_metrics(incoming, outgoing)
    if in_speed < 6.0 * scale or out_speed < 4.0 * scale:
        return None
    speed_jump = out_speed >= 2.5 * in_speed and out_speed >= 10.0 * scale
    if not any(point.source == "det" for point in points[:2]) and not speed_jump:
        return None
    if cosine > 0.50 and not speed_jump:
        return None
    return (1.0 - cosine) * min(in_speed, out_speed) / max(scale, 1e-6)


def _kink_contacts(
    per_frame: BallTrack,
    player_boxes_by_frame: PlayerBoxesByFrame,
    stride: int,
    diagonal: float,
    scale: float,
) -> List[ContactEvent]:
    """Visible direction changes next to a player."""
    contacts = []
    for frame, points in _det_windows(per_frame, stride):
        if not any(point.source == "det" for point in points[3:]):
            continue
        striker = _striking_player(points[2], player_boxes_by_frame[frame], diagonal)
        if striker is None:
            continue
        score = _kink_contact_score(points, scale)
        if score is None:
            continue
        player_id, box = striker
        x1, _, x2, y2 = map(float, box[:4])
        contacts.append(ContactEvent(frame, 0.5 * (x1 + x2), y2, player_id, score, "kink"))
    return contacts


def _nearest_player_to_either(
    a: FrameResult, b: FrameResult, boxes: Optional[PlayerBoxes]
) -> Optional[Tuple[float, Hashable, Box]]:
    nearest = None
    for player_id, box in (boxes or {}).items():
        distance = min(
            _box_distance(float(a.cx), float(a.cy), box),
            _box_distance(float(b.cx), float(b.cy), box),
        )
        if nearest is None or distance < nearest[0]:
            nearest = (distance, player_id, box)
    return nearest


def _gap_contacts(
    per_frame: BallTrack,
    player_boxes_by_frame: PlayerBoxesByFrame,
    fps: float,
    stride: int,
    diagonal: float,
    scale: float,
) -> List[ContactEvent]:
    """Occluded racket contacts, which often appear as a short reversing gap."""
    contacts = []
    speed_scale = (_GAP_SPEED_REFERENCE_FPS / max(float(fps), 1.0)) * scale
    for gap in _gap_turns(per_frame, fps, stride):
        in_speed, out_speed, cosine = _turn_metrics(gap.incoming, gap.outgoing)
        if in_speed < 4.0 * speed_scale or out_speed < 2.0 * speed_scale:
            continue
        if cosine > -0.30:
            continue
        midpoint = gap.midpoint
        nearest = _nearest_player_to_either(
            gap.before[-1][1], gap.after[0][1], player_boxes_by_frame[midpoint]
        )
        if nearest is None or nearest[0] > 0.025 * diagonal:
            continue
        _, player_id, box = nearest
        x1, y1, x2, y2 = map(float, box[:4])
        impact_point = _extrapolate_incoming(gap.history, midpoint)
        if (
            gap.incoming[1] > 0.0
            and gap.outgoing[1] < 0.0
            and impact_point is not None
            and impact_point[1] >= y2 - _LOWER_BODY_FRACTION * max(y2 - y1, 1.0)
        ):
            continue  # a down-to-up turn at the feet is a bounce
        score = _GAP_CONTACT_SCORE_BONUS + (1.0 - cosine) * min(in_speed, out_speed)
        contacts.append(ContactEvent(midpoint, 0.5 * (x1 + x2), y2, player_id, score, "gap"))
    return contacts


def _merge_contacts(candidates: Sequence[ContactEvent], fps: float) -> List[ContactEvent]:
    """Keep the strongest contact per player within the merge window."""
    contacts: List[ContactEvent] = []
    for candidate in sorted(candidates, key=lambda item: (str(item.player_id), item.frame)):
        if (
            contacts
            and contacts[-1].player_id == candidate.player_id
            and candidate.frame - contacts[-1].frame <= _frames(_EVENT_MERGE_S, fps)
        ):
            if candidate.score > contacts[-1].score:
                contacts[-1] = candidate
        else:
            contacts.append(candidate)
    return sorted(contacts, key=lambda item: item.frame)


def detect_player_contacts(
    per_frame: BallTrack,
    player_boxes_by_frame: PlayerBoxesByFrame,
    fps: float,
    width: int,
    height: int,
) -> List[ContactEvent]:
    """Find racket contacts in image space and anchor them at player feet."""
    stride = _model_stride(fps)
    diagonal, scale = _resolution_scale(width, height)
    candidates = _kink_contacts(per_frame, player_boxes_by_frame, stride, diagonal, scale)
    candidates += _gap_contacts(per_frame, player_boxes_by_frame, fps, stride, diagonal, scale)
    return _merge_contacts(candidates, fps)


def _kink_bounces(
    per_frame: BallTrack,
    player_boxes_by_frame: PlayerBoxesByFrame,
    stride: int,
    scale: float,
) -> List[BounceEvent]:
    """Visible strict down-to-up kinks away from the players."""
    bounces = []
    for frame, points in _det_windows(per_frame, stride):
        incoming, outgoing = _window_velocities(points)
        in_speed, out_speed, cosine = _turn_metrics(incoming, outgoing)
        prominence = min(
            float(points[2].cy) - float(points[0].cy),
            float(points[2].cy) - float(points[4].cy),
        )
        if not (
            incoming[1] >= 4.0 * scale
            and outgoing[1] <= -1.5 * scale
            and in_speed >= 4.0 * scale
            and out_speed >= 2.0 * scale
            and cosine <= 0.766
            and prominence >= 6.0 * scale
        ):
            continue
        if any(
            _inside_player(
                float(points[index].cx), float(points[index].cy),
                player_boxes_by_frame[frame + (index - 2) * stride],
            )
            for index in (1, 2, 3)
        ):
            continue
        score = float(prominence * (1.0 - cosine))
        bounces.append(BounceEvent(frame, float(points[2].cx), float(points[2].cy), score))
    return bounces


def _gap_bounces(
    per_frame: BallTrack,
    player_boxes_by_frame: PlayerBoxesByFrame,
    fps: float,
    stride: int,
    scale: float,
) -> List[BounceEvent]:
    """Landings fully hidden by a short selector gap.

    Fit the incoming and outgoing branches and keep only a strong down-to-up turn.
    """
    bounces = []
    for gap in _gap_turns(per_frame, fps, stride):
        incoming, outgoing = gap.incoming, gap.outgoing
        in_speed, out_speed, cosine = _turn_metrics(incoming, outgoing)
        event_frame = gap.midpoint
        impact_point = _extrapolate_incoming(gap.history, event_frame)
        if impact_point is None:
            continue
        prominence = float(impact_point[1] - max(
            float(gap.before[-1][1].cy), float(gap.after[0][1].cy)
        ))
        if not (
            incoming[1] >= 2.0 * scale
            and outgoing[1] <= -1.0 * scale
            and in_speed >= 3.0 * scale
            and out_speed >= 1.5 * scale
            and cosine <= 0.80
            and prominence >= 6.0 * scale
        ):
            continue
        boxes = player_boxes_by_frame[
            int(np.clip(event_frame, 0, len(player_boxes_by_frame) - 1))
        ]
        impact_x, impact_y = float(impact_point[0]), float(impact_point[1])
        if _inside_player(impact_x, impact_y, boxes):
            # Beside a player only a landing at the feet counts.
            nearest = _nearest_player(impact_x, impact_y, boxes)
            if nearest is None:
                continue
            _, y1, _, y2 = map(float, nearest[2][:4])
            if impact_point[1] < y2 - _LOWER_BODY_FRACTION * max(y2 - y1, 1.0):
                continue
        score = float(prominence * (1.0 - cosine))
        bounces.append(BounceEvent(event_frame, impact_x, impact_y, score))
    return bounces


def infer_kinematic_bounces(
    per_frame: BallTrack,
    player_boxes_by_frame: PlayerBoxesByFrame,
    fps: float,
    width: int,
    height: int,
) -> List[BounceEvent]:
    """Find strict down-to-up ground kinks missed by the bounce model."""
    stride = _model_stride(fps)
    _, scale = _resolution_scale(width, height)
    candidates = _kink_bounces(per_frame, player_boxes_by_frame, stride, scale)
    candidates += _gap_bounces(per_frame, player_boxes_by_frame, fps, stride, scale)

    # Candidates stay in detection order (kinks, then gaps), as tuned.
    clustered: List[List[BounceEvent]] = []
    radius = max(1, _frames(_EVENT_MERGE_S, fps))
    for candidate in candidates:
        if not clustered or candidate.frame - clustered[-1][-1].frame > radius:
            clustered.append([candidate])
        else:
            clustered[-1].append(candidate)
    return [max(group, key=lambda item: item.score) for group in clustered]


class _CourtProjector:
    """Image-to-court projection per frame, caching one homography per keypoint set."""

    def __init__(self, court_keypoints_by_frame: Sequence[Optional[CourtKeypoints]]) -> None:
        self._keypoints = court_keypoints_by_frame
        self._models: Dict[Tuple[int, ...], Optional[Dict[str, np.ndarray]]] = {}

    def __call__(self, frame: int, x: float, y: float) -> Optional[CourtPoint]:
        if frame < 0 or frame >= len(self._keypoints):
            return None
        keypoints = self._keypoints[frame]
        key = court_keypoints_key(keypoints)
        if key is None:
            return None
        if key not in self._models:
            self._models[key] = build_ground_projection_model(keypoints)
        model = self._models[key]
        return None if model is None else homography_apply(model["H_i2c"], x, y)


def _within(point: CourtPoint, bounds: Tuple[float, float]) -> bool:
    low, high = bounds
    return low <= point[0] <= high and low <= point[1] <= high


def _court_side(point: CourtPoint) -> int:
    """-1 for the far half, 1 for the near half, 0 on the net band."""
    if point[1] < NET_V - _SIDE_DEAD_ZONE:
        return -1
    if point[1] > NET_V + _SIDE_DEAD_ZONE:
        return 1
    return 0


def _is_service_leg(start: CourtPoint, end: CourtPoint, first_after_reset: bool) -> bool:
    if not first_after_reset:
        return False
    if not (SINGLES_INSET <= end[0] <= 1.0 - SINGLES_INSET):
        return False
    if (start[0] - 0.5) * (end[0] - 0.5) >= 0.0:
        return False  # a serve crosses diagonally
    if start[1] <= _SERVE_BASELINE_BAND:
        return NET_V <= end[1] <= NET_V + SERVICE_DEPTH
    if start[1] >= 1.0 - _SERVE_BASELINE_BAND:
        return NET_V - SERVICE_DEPTH <= end[1] <= NET_V
    return False


def _place_bounces(
    confirmed: Sequence[BounceEvent],
    candidates: Sequence[BounceEvent],
    kinematic: Sequence[BounceEvent],
    project: _CourtProjector,
    fps: float,
) -> List[_PlacedBounce]:
    """Project bounces; inferred ones near a confirmed landing are dropped."""
    placed = []
    for event in confirmed:
        point = project(event.frame, event.x, event.y)
        if point is not None and _within(point, _CONFIRMED_BOUNCE_BOUNDS):
            placed.append(_PlacedBounce(int(event.frame), point, "confirmed", float(event.score)))

    confirmed_frames = {int(event.frame) for event in confirmed}
    merge = _frames(_EVENT_MERGE_S, fps)
    for event in list(candidates) + list(kinematic):
        if any(abs(int(event.frame) - value) <= merge for value in confirmed_frames):
            continue
        point = project(event.frame, event.x, event.y)
        if point is not None and _within(point, _INFERRED_BOUNCE_BOUNDS):
            placed.append(_PlacedBounce(int(event.frame), point, "inferred", float(event.score)))
    placed.sort(key=lambda event: event.frame)
    return placed


def _place_contacts(
    contacts: Sequence[ContactEvent],
    bounces: Sequence[_PlacedBounce],
    project: _CourtProjector,
    collision: int,
) -> List[_PlacedContact]:
    """Project contacts, dropping any that coincide with a confirmed bounce."""
    placed = []
    for contact in contacts:
        if any(abs(int(contact.frame) - bounce.frame) <= collision
               for bounce in bounces if bounce.quality == "confirmed"):
            continue
        point = project(contact.frame, contact.x, contact.y)
        if point is not None and _within(point, _CONTACT_BOUNDS):
            placed.append(_PlacedContact(
                int(contact.frame), point, contact.player_id,
                float(contact.score), contact.source,
            ))
    placed.sort(key=lambda event: event.frame)
    return placed


def _closest_opposite_player(
    bounce: _PlacedBounce,
    per_frame: BallTrack,
    player_boxes_by_frame: PlayerBoxesByFrame,
    project: _CourtProjector,
    first_frame: int,
    last_frame: int,
    stride: int,
) -> Optional[Tuple[float, int, CourtPoint, Hashable]]:
    """Closest ball-to-player approach by a player on the other side of the net."""
    bounce_side = _court_side(bounce.point)
    best = None
    for frame in range(first_frame, last_frame, stride):
        ball = _usable_point(per_frame, frame)
        if ball is None:
            continue
        for player_id, box in (player_boxes_by_frame[frame] or {}).items():
            if box is None or len(box) < 4:
                continue
            x1, _, x2, y2 = map(float, box[:4])
            foot = project(frame, 0.5 * (x1 + x2), y2)
            if foot is None or _court_side(foot) != -bounce_side:
                continue
            distance = _box_distance(float(ball.cx), float(ball.cy), box)
            if best is None or distance < best[0]:
                best = (distance, frame, foot, player_id)
    return best


def _landing_lacks_contact(
    bounce: _PlacedBounce,
    previous: Optional[_PlacedBounce],
    contacts: Sequence[_PlacedContact],
    fps: float,
    max_flight: int,
    collision: int,
) -> bool:
    """True for a confirmed, sided landing with no opposite-side contact before it.

    A second same-side bounce soon after the previous one continues that leg.
    """
    if bounce.quality != "confirmed":
        return False
    bounce_side = _court_side(bounce.point)
    if bounce_side == 0:
        return False
    if (
        previous is not None
        and _court_side(previous.point) == bounce_side
        and bounce.frame - previous.frame <= _frames(_SAME_SIDE_BOUNCE_S, fps)
    ):
        return False
    return not any(
        bounce.frame - max_flight < contact.frame < bounce.frame - collision
        and _court_side(contact.point) == -bounce_side
        for contact in contacts
    )


def _rescue_missing_contacts(
    bounces: Sequence[_PlacedBounce],
    contacts: List[_PlacedContact],
    per_frame: BallTrack,
    player_boxes_by_frame: PlayerBoxesByFrame,
    project: _CourtProjector,
    fps: float,
    width: int,
    height: int,
) -> None:
    """Infer a contact for each confirmed landing that has none.

    A confirmed landing must not disappear merely because the racket contact
    was missed. Search backward for the closest opposite-side player/ball
    encounter and use only that player's grounded footpoint as the origin.
    Rescued contacts are appended to ``contacts`` in place and count as
    existing contacts for later landings.
    """
    stride = _model_stride(fps)
    diagonal, _ = _resolution_scale(width, height)
    reach = _RESCUE_RADIUS_DIAGONAL * diagonal
    max_flight = _frames(_MAX_FLIGHT_S, fps)
    collision = max(1, _frames(_CONTACT_BOUNCE_COLLISION_S, fps))
    for bounce_index, bounce in enumerate(bounces):
        previous = bounces[bounce_index - 1] if bounce_index else None
        if not _landing_lacks_contact(bounce, previous, contacts, fps, max_flight, collision):
            continue
        best = _closest_opposite_player(
            bounce, per_frame, player_boxes_by_frame, project,
            max(0, bounce.frame - max_flight), max(0, bounce.frame - collision), stride,
        )
        if best is not None and best[0] <= reach:
            contacts.append(_PlacedContact(
                best[1], best[2], best[3], max(0.0, 1.0 - best[0] / reach), "inferred",
            ))


def _find_landing(
    contact: _PlacedContact,
    next_contact: Optional[_PlacedContact],
    bounces: Sequence[_PlacedBounce],
    stop: int,
    collision: int,
) -> Optional[Tuple[_PlacedBounce, str]]:
    """Return (landing, outcome) for a contact, or None when the leg is incomplete.

    Prefer the first confirmed bounce, then the strongest inferred bounce on
    the far side, then a volley at the next opposite-side contact near the net.
    """
    in_flight = [
        event for event in bounces
        if contact.frame + collision < event.frame < stop
    ]
    confirmed = [event for event in in_flight if event.quality == "confirmed"]
    if confirmed:
        return confirmed[0], "bounce"

    start_side = _court_side(contact.point)
    if start_side != 0:
        inferred = [
            event for event in in_flight
            if event.quality == "inferred" and _court_side(event.point) == -start_side
        ]
        if inferred:
            return max(inferred, key=lambda event: event.score), "bounce"

    if next_contact is None:
        return None
    end_side = _court_side(next_contact.point)
    if start_side == 0 or end_side == 0 or start_side == end_side:
        return None
    if abs(next_contact.point[1] - NET_V) > _VOLLEY_MAX_NET_DISTANCE:
        return None
    return _PlacedBounce(next_contact.frame, next_contact.point, None, next_contact.score), "volley"


def _leg_phase(
    contact: _PlacedContact, landing: _PlacedBounce, outcome: str, first_after_reset: bool
) -> str:
    if outcome == "volley":
        return "rally"
    if _court_side(contact.point) == _court_side(landing.point):
        return "fault"
    if _is_service_leg(contact.point, landing.point, first_after_reset):
        return "serve"
    return "rally"


def _link_legs(
    contacts: Sequence[_PlacedContact],
    bounces: Sequence[_PlacedBounce],
    frame_count: int,
    fps: float,
) -> List[RallyLeg]:
    max_flight = _frames(_MAX_FLIGHT_S, fps)
    collision = max(1, _frames(_CONTACT_BOUNCE_COLLISION_S, fps))
    legs: List[RallyLeg] = []
    last_leg_end = None
    for index, contact in enumerate(contacts):
        next_contact = contacts[index + 1] if index + 1 < len(contacts) else None
        if next_contact is not None and next_contact.frame - contact.frame > max_flight:
            next_contact = None
        stop = min(
            contact.frame + max_flight,
            next_contact.frame if next_contact is not None else frame_count,
        )
        found = _find_landing(contact, next_contact, bounces, stop, collision)
        if found is None:
            continue
        landing, outcome = found
        if _court_side(contact.point) == 0 or _court_side(landing.point) == 0:
            continue
        first_after_reset = (
            last_leg_end is None
            or contact.frame - last_leg_end > _frames(_RALLY_RESET_S, fps)
        )
        legs.append({
            "start_frame": contact.frame,
            "end_frame": landing.frame,
            "start": contact.point,
            "end": landing.point,
            "phase": _leg_phase(contact, landing, outcome, first_after_reset),
            "outcome": outcome,
            "quality": landing.quality,
            "start_quality": "inferred" if contact.source == "inferred" else "confirmed",
        })
        last_leg_end = landing.frame
    return legs


def assemble_rally_legs(
    per_frame: Optional[BallTrack],
    player_boxes_by_frame: Optional[PlayerBoxesByFrame],
    fps: float,
    width: int,
    height: int,
    court_keypoints_by_frame: Sequence[Optional[CourtKeypoints]],
    contacts: Iterable[Sequence[Any]],
    bounces: Tuple[Iterable[Sequence[Any]], Iterable[Sequence[Any]]],
    kinematic_bounces: Iterable[Sequence[Any]],
) -> List[RallyLeg]:
    """Build completed, plane-valid player-contact-to-landing rally legs.

    ``bounces`` is the ``BouncePrediction`` (confirmed, candidates) pair.
    ``per_frame`` and ``player_boxes_by_frame`` may be None; contacts are then
    not rescued for confirmed landings that have none.
    """
    project = _CourtProjector(court_keypoints_by_frame)
    collision = max(1, _frames(_CONTACT_BOUNCE_COLLISION_S, fps))
    confirmed, candidates = bounces
    placed_bounces = _place_bounces(
        _coerce_bounces(confirmed),
        _coerce_bounces(candidates),
        _coerce_bounces(kinematic_bounces),
        project, fps,
    )
    placed_contacts = _place_contacts(
        _coerce_contacts(contacts), placed_bounces, project, collision
    )
    if per_frame is not None and player_boxes_by_frame is not None:
        _rescue_missing_contacts(
            placed_bounces, placed_contacts, per_frame, player_boxes_by_frame,
            project, fps, width, height,
        )
        placed_contacts.sort(key=lambda event: event.frame)
    return _link_legs(placed_contacts, placed_bounces, len(court_keypoints_by_frame), fps)


class _PanelRect(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int


def _panel_point(
    point: Tuple[float, float],
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> Tuple[int, int]:
    low, high = _PANEL_POINT_BOUNDS
    u = float(np.clip(point[0], low, high))
    v = float(np.clip(point[1], low, high))
    return (
        int(round(left + u * (right - left))),
        int(round(top + v * (bottom - top))),
    )


def _draw_dashed_line(
    image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Color,
    thickness: int = 1,
) -> None:
    distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    if distance < 1.0:
        return
    steps = max(1, int(distance // _DASH_PX))
    for index in range(steps):
        if index % 2:
            continue
        a = index / steps
        b = min(1.0, (index + 1) / steps)
        p0 = (int(round(start[0] + a * (end[0] - start[0]))),
              int(round(start[1] + a * (end[1] - start[1]))))
        p1 = (int(round(start[0] + b * (end[0] - start[0]))),
              int(round(start[1] + b * (end[1] - start[1]))))
        cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


def _visible_completed_legs(
    rally_legs: Sequence[RallyLeg],
    contact_events: Optional[Iterable[Sequence[Any]]],
    current_frame: int,
    fps: float,
) -> List[RallyLeg]:
    """The latest completed leg, until about a second passes or play moves on."""
    completed = [
        leg for leg in rally_legs
        if leg["end_frame"] <= current_frame
    ]
    if not completed:
        return []

    latest = completed[-1]
    if current_frame - latest["end_frame"] > int(round(float(fps))):
        return []
    if any(
        latest["end_frame"] < int(event.frame) <= current_frame
        for event in _coerce_contacts(contact_events)
    ):
        return []

    newest_started = next(
        (leg for leg in reversed(rally_legs) if leg["start_frame"] <= current_frame),
        None,
    )
    if (
        newest_started is not None
        and newest_started["start_frame"] > latest["end_frame"]
        and newest_started["end_frame"] > current_frame
    ):
        return []
    return [latest]


def _minimap_layout(frame_shape: Tuple[int, ...]) -> Optional[Tuple[int, int, int, int, _PanelRect]]:
    """Return (x0, y0, panel_w, panel_h, court rect) or None if it does not fit."""
    frame_h, frame_w = frame_shape[:2]
    panel_w = int(np.clip(round(frame_w * 0.10), 130, 180))
    panel_h = int(round(panel_w * 2.08))
    margin = max(8, int(round(frame_w * 0.006)))
    x0 = frame_w - panel_w - margin
    y0 = margin
    if x0 < 0 or y0 + panel_h > frame_h:
        return None
    court_w = panel_w - 36
    court_h = int(round(court_w * COURT_LENGTH_M / DOUBLES_WIDTH_M))
    left = (panel_w - court_w) // 2
    top = (panel_h - court_h) // 2
    return x0, y0, panel_w, panel_h, _PanelRect(left, top, left + court_w, top + court_h)


def _draw_court_lines(panel: np.ndarray, rect: _PanelRect) -> None:
    def line(a: CourtPoint, b: CourtPoint, color: Color, thickness: int) -> None:
        cv2.line(panel, _panel_point(a, *rect), _panel_point(b, *rect),
                 color, thickness, cv2.LINE_AA)

    corners = (rect.left, rect.top), (rect.right, rect.bottom)
    cv2.rectangle(panel, *corners, _PANEL_COURT, -1)
    cv2.rectangle(panel, *corners, _PANEL_LINE, 1, cv2.LINE_AA)
    for u in (SINGLES_INSET, 1.0 - SINGLES_INSET):
        line((u, 0.0), (u, 1.0), _PANEL_LINE, 1)
    line((0.0, NET_V), (1.0, NET_V), _PANEL_NET, 2)
    for v in (NET_V - SERVICE_DEPTH, NET_V + SERVICE_DEPTH):
        line((SINGLES_INSET, v), (1.0 - SINGLES_INSET, v), _PANEL_LINE, 1)
    line((0.5, NET_V - SERVICE_DEPTH), (0.5, NET_V + SERVICE_DEPTH), _PANEL_LINE, 1)
    cv2.putText(panel, "COURT", (7, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                _PANEL_LINE, 1, cv2.LINE_AA)


def _draw_leg_landing(
    panel: np.ndarray, leg: RallyLeg, end_point: Tuple[int, int], color: Color
) -> None:
    if leg["outcome"] == "bounce":
        if leg["quality"] == "inferred":
            x, y = end_point
            diamond = np.array(((x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)), np.int32)
            cv2.polylines(panel, [diamond], True, _INFERRED_BOUNCE_COLOR, 2, cv2.LINE_AA)
            cv2.putText(panel, "?", (x + 5, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.30, _INFERRED_BOUNCE_COLOR, 1, cv2.LINE_AA)
        else:
            cv2.circle(panel, end_point, 5, _MARKER_OUTLINE, 2, cv2.LINE_AA)
            cv2.circle(panel, end_point, 4, _CONFIRMED_BOUNCE_COLOR, 2, cv2.LINE_AA)
    elif leg["outcome"] == "volley":
        cv2.putText(panel, "V", (end_point[0] + 4, end_point[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)


def _draw_leg(panel: np.ndarray, leg: RallyLeg, rect: _PanelRect) -> None:
    start_point = _panel_point(leg["start"], *rect)
    end_point = _panel_point(leg["end"], *rect)
    color = _PHASE_COLORS.get(leg["phase"], _DEFAULT_PHASE_COLOR)
    cv2.line(panel, start_point, end_point, _SHADOW, 3, cv2.LINE_AA)
    if leg["quality"] == "inferred" or leg.get("start_quality") == "inferred":
        _draw_dashed_line(panel, start_point, end_point, color, 2)
    else:
        cv2.line(panel, start_point, end_point, color, 2, cv2.LINE_AA)

    # q(s)=4s(1-s) is a relative height cue only. Court coordinates stay
    # on the straight, physically valid plan-view chord.
    for step in range(1, _ARC_STEPS):
        fraction = step / float(_ARC_STEPS)
        relative_height = 4.0 * fraction * (1.0 - fraction)
        point = (
            int(round(start_point[0] + fraction * (end_point[0] - start_point[0]))),
            int(round(start_point[1] + fraction * (end_point[1] - start_point[1]))),
        )
        radius = 1 + int(round(relative_height * (2 if leg["phase"] == "serve" else 1)))
        cv2.circle(panel, point, radius, color, -1, cv2.LINE_AA)

    _draw_leg_landing(panel, leg, end_point, color)


def _draw_phase_label(panel: np.ndarray, leg: RallyLeg, panel_w: int) -> None:
    label = {"serve": "SERVE", "fault": "FAULT"}.get(leg["phase"], "RALLY")
    if leg["outcome"] == "volley":
        label = "VOLLEY"
    cv2.putText(panel, label, (panel_w - 48, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.30, _PHASE_COLORS.get(leg["phase"], _DEFAULT_PHASE_COLOR),
                1, cv2.LINE_AA)


def _draw_player_dots(
    panel: np.ndarray,
    H: np.ndarray,
    player_boxes: Optional[PlayerBoxes],
    rect: _PanelRect,
) -> None:
    for index, (player_id, box) in enumerate(
        sorted((player_boxes or {}).items(), key=lambda item: str(item[0]))
    ):
        if box is None or len(box) < 4:
            continue
        x1, _, x2, y2 = map(float, box[:4])
        projected = homography_apply(H, 0.5 * (x1 + x2), y2)
        if projected is None:
            continue
        center = _panel_point(projected, *rect)
        color = _PLAYER_COLORS[index % len(_PLAYER_COLORS)]
        cv2.circle(panel, center, 5, _MARKER_OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(panel, center, 3, color, -1, cv2.LINE_AA)


def draw_court_minimap(
    frame: Optional[np.ndarray],
    ground_model: Optional[Mapping[str, np.ndarray]],
    player_boxes: Optional[PlayerBoxes],
    rally_legs: Sequence[RallyLeg],
    current_frame: int,
    fps: float,
    contact_events: Optional[Iterable[Sequence[Any]]] = (),
) -> None:
    """Draw the optional top-right 2D court minibar in-place."""
    if frame is None or ground_model is None:
        return
    H = ground_model.get("H_i2c")
    if H is None:
        return
    layout = _minimap_layout(frame.shape)
    if layout is None:
        return
    x0, y0, panel_w, panel_h, rect = layout

    panel = np.full((panel_h, panel_w, 3), _PANEL_BG, dtype=np.uint8)
    _draw_court_lines(panel, rect)
    visible_legs = _visible_completed_legs(rally_legs, contact_events, current_frame, fps)
    for leg in visible_legs:
        _draw_leg(panel, leg, rect)
    if visible_legs:
        _draw_phase_label(panel, visible_legs[-1], panel_w)
    _draw_player_dots(panel, H, player_boxes, rect)

    roi = frame[y0:y0 + panel_h, x0:x0 + panel_w]
    cv2.addWeighted(panel, _PANEL_OPACITY, roi, _FRAME_OPACITY, 0.0, roi)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w - 1, y0 + panel_h - 1),
                  _PANEL_BORDER, 1, cv2.LINE_AA)
