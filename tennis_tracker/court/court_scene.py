"""What every court-event stage reads and returns: the ``CourtScene`` and the event types.

The stages run in this order (see ``pipeline.run``):

1. ``bounce_events.predict_bounce_events`` scores ball landings with the upstream CatBoost
   bounce model and returns confirmed bounces plus low-score candidates.
2. ``contact_events.detect_player_contacts`` finds racket contacts near player boxes.
3. ``bounce_events.infer_kinematic_bounces`` finds strict down-to-up kinks the model missed.
4. ``rally_legs.assemble_rally_legs`` projects all of them onto the court plane and links
   each contact to its landing.
5. ``court_minimap.draw_court_minimap`` draws the latest completed leg on
   every frame.

Every event stage takes the same ``CourtScene``: the selected ball track,
the tracked player boxes and the clip's frame rate and size, which the
pipeline builds once.  Events are the ``BounceEvent`` / ``ContactEvent``
named tuples below.  Everything is image-space evidence projected through
the court homography; nothing here is a measured 3D position.  The scene's
player-box queries (is a point on a player, how far from a box, which player
is nearest) are here too, since the bounce, contact and leg stages all ask them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Hashable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

import numpy as np

from ball_in_play_selector import FrameResult
from ball_in_play_selector.config import REFERENCE_DIAG_PX


# One player's xyxy box, the tracked player id -> box map pass 1 records per
# frame (``PlayerDetector.player_boxes_by_slot``), and that map for every frame.
PlayerBox = Sequence[float]
PlayerBoxes = Mapping[Hashable, Optional[PlayerBox]]
PlayerBoxesByFrame = Sequence[Optional[PlayerBoxes]]
BallTrack = Sequence[Optional[FrameResult]]
CourtKeypoints = Sequence[float]
CourtPoint = Tuple[float, float]
Color = Tuple[int, int, int]

# The event models and thresholds were tuned on tracks sampled at 30 FPS, and their
# pixel thresholds on 1920x1080 video (``CourtScene.scale``; REFERENCE_DIAG_PX is its diagonal).
_MODEL_FPS = 30.0
EVENT_MERGE_S = 0.15  # events of one kind closer than this are one event
GAP_SPEED_REFERENCE_FPS = 60.0  # the frame rate the gap-fit velocity thresholds were tuned at
# A turn this far down a player's box is a landing at the feet, not a stroke.
LOWER_BODY_FRACTION = 0.40
_PLAYER_PAD_MIN_PX = 24.0  # horizontal padding around a player box

# How a contact was found: a visible direction change, a reversing occlusion
# gap next to the player, or (placed contacts only) rescued from a landing
# that had no contact before it.
ContactSource = Literal["kink", "gap"]
PlacedContactSource = Literal["kink", "gap", "inferred"]
# Whether a landing or a leg origin came from the bounce model / a detected
# contact ("confirmed") or from kinematics / a rescue ("inferred").
EventQuality = Literal["confirmed", "inferred"]
LegPhase = Literal["serve", "rally", "fault"]
LegOutcome = Literal["bounce", "volley"]
# Which stage found a landing: the bounce model, a visible kink in the ball
# track, or a fit across a short gap in it.
EventOrigin = Literal["model", "kink", "gap"]
# Where an event's ball position came from: the ball track's own position kind
# (``FrameResult.position_kind``), or a point extrapolated across a gap.
EventPosition = Literal["measurement", "filtered", "recovered", "extrapolated"]


class BallPoint(NamedTuple):
    """A ball position in image pixels and where it came from."""

    x: float
    y: float
    position: EventPosition


def ball_point(result: FrameResult) -> BallPoint:
    """The ball track's position at one frame, as an event reports it."""
    return BallPoint(float(result.cx), float(result.cy), result.position_kind)


class BounceEvent(NamedTuple):
    """A ball landing in image pixels; ``score`` is model or kink strength.

    ``origin`` says which stage found it and ``position`` where its point came
    from; the defaults describe a model bounce on a refitted track.
    """

    frame: int
    x: float
    y: float
    score: float
    origin: EventOrigin = "model"
    position: EventPosition = "filtered"


class ContactEvent(NamedTuple):
    """A racket contact anchored at the player's feet (``x``, ``y``); ``ball`` is
    the ball position the contact was judged at, None when the stage had none."""

    frame: int
    x: float
    y: float
    player_id: Hashable
    score: float
    source: ContactSource
    ball: Optional[BallPoint] = None


class BouncePrediction(NamedTuple):
    """Bounce-model output: confirmed landings and guarded low-score candidates."""

    confirmed: List[BounceEvent]
    candidates: List[BounceEvent]


class _RallyLegRequired(TypedDict):
    start_frame: int
    end_frame: int
    start: CourtPoint
    end: CourtPoint
    phase: LegPhase
    outcome: LegOutcome
    quality: Optional[EventQuality]  # the landing's; None for a volley


class RallyLeg(_RallyLegRequired, total=False):
    """One completed contact-to-landing leg in unit court coordinates."""

    start_quality: EventQuality  # the origin contact's
    start_source: PlacedContactSource  # how the origin contact was found
    landing_origin: Optional[EventOrigin]  # the landing's; None for a volley
    landing_position: Optional[EventPosition]


def seconds_to_frames(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


@dataclass(frozen=True)
class CourtScene:
    """What every event stage reads: the selected ball track and the tracked
    player boxes, one entry per source frame, and the clip's frame rate and
    size.  The pipeline builds one per run and hands it to each stage.
    """

    per_frame: BallTrack
    player_boxes_by_frame: PlayerBoxesByFrame
    fps: float
    width: int
    height: int

    @property
    def stride(self) -> int:
        """Source frames between the 30 FPS samples the models were tuned on."""
        return max(1, int(round(float(self.fps) / _MODEL_FPS)))

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.width, self.height))

    @property
    def scale(self) -> float:
        """Pixel scale relative to the 1080p frame the thresholds were tuned on."""
        return self.diagonal / REFERENCE_DIAG_PX

    @property
    def gap_speed_scale(self) -> float:
        """Scale for a gap fit's velocities (``ball_turns.gap_turns``), which are per source
        frame: their thresholds were tuned at 60 FPS and 1080p, so ``scale`` times 60 / fps."""
        return GAP_SPEED_REFERENCE_FPS / max(float(self.fps), 1.0) * self.scale

    def frames(self, seconds: float) -> int:
        return seconds_to_frames(seconds, self.fps)

    def player_boxes(self, frame: int) -> Optional[PlayerBoxes]:
        """The player boxes of ``frame``, None outside the clip."""
        if 0 <= frame < len(self.player_boxes_by_frame):
            return self.player_boxes_by_frame[frame]
        return None


def usable_point(per_frame: BallTrack, frame: int) -> Optional[FrameResult]:
    if frame < 0 or frame >= len(per_frame):
        return None
    return per_frame[frame]


def inside_player(x: float, y: float, boxes: Optional[PlayerBoxes]) -> bool:
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


def box_distance(x: float, y: float, box: PlayerBox) -> float:
    x1, y1, x2, y2 = map(float, box[:4])
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy))


def nearest_player(
    x: float, y: float, boxes: Optional[PlayerBoxes]
) -> Optional[Tuple[float, Hashable, PlayerBox]]:
    best = None
    for player_id, box in (boxes or {}).items():
        if box is None or len(box) < 4:
            continue
        distance = box_distance(x, y, box)
        if best is None or distance < best[0]:
            best = (distance, player_id, box)
    return best
