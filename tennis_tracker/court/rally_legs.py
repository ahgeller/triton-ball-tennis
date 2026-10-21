"""Schematic rally legs: bounces and contacts projected onto the court plane and each contact
linked to its landing (``assemble_rally_legs``), in unit court coordinates.

A confirmed landing with no contact before it gets one rescued from the ball track and player
boxes in the ``court_scene.CourtScene``.  The legs are image-space evidence projected through the
court homography, not measured 3D flights.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .court_geometry import (
    NET_V,
    SERVICE_DEPTH,
    SINGLES_INSET,
    build_ground_projection_model,
    court_keypoints_key,
    homography_apply,
)
from .court_scene import (
    EVENT_MERGE_S,
    BallTrack,
    BounceEvent,
    BouncePrediction,
    ContactEvent,
    CourtKeypoints,
    CourtPoint,
    CourtScene,
    EventQuality,
    LegOutcome,
    LegPhase,
    PlacedContactSource,
    PlayerBoxesByFrame,
    RallyLeg,
    box_distance,
    seconds_to_frames,
    usable_point,
)


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


class _PlacedBounce(NamedTuple):
    frame: int
    point: CourtPoint
    quality: Optional[EventQuality]
    score: float


class _PlacedContact(NamedTuple):
    frame: int
    point: CourtPoint
    player: Hashable
    score: float
    source: PlacedContactSource


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
    merge = seconds_to_frames(EVENT_MERGE_S, fps)
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
        ball = usable_point(per_frame, frame)
        if ball is None:
            continue
        for player_id, box in (player_boxes_by_frame[frame] or {}).items():
            if box is None or len(box) < 4:
                continue
            x1, _, x2, y2 = map(float, box[:4])
            foot = project(frame, 0.5 * (x1 + x2), y2)
            if foot is None or _court_side(foot) != -bounce_side:
                continue
            distance = box_distance(float(ball.cx), float(ball.cy), box)
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
        and bounce.frame - previous.frame <= seconds_to_frames(_SAME_SIDE_BOUNCE_S, fps)
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
    scene: CourtScene,
    project: _CourtProjector,
) -> None:
    """Infer a contact for each confirmed landing that has none.

    A confirmed landing must not disappear merely because the racket contact
    was missed. Search backward for the closest opposite-side player/ball
    encounter and use only that player's grounded footpoint as the origin.
    Rescued contacts are appended to ``contacts`` in place and count as
    existing contacts for later landings.
    """
    reach = _RESCUE_RADIUS_DIAGONAL * scene.diagonal
    max_flight = scene.frames(_MAX_FLIGHT_S)
    collision = max(1, scene.frames(_CONTACT_BOUNCE_COLLISION_S))
    for bounce_index, bounce in enumerate(bounces):
        previous = bounces[bounce_index - 1] if bounce_index else None
        if not _landing_lacks_contact(bounce, previous, contacts, scene.fps, max_flight, collision):
            continue
        best = _closest_opposite_player(
            bounce, scene.per_frame, scene.player_boxes_by_frame, project,
            max(0, bounce.frame - max_flight), max(0, bounce.frame - collision), scene.stride,
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
) -> Optional[Tuple[_PlacedBounce, LegOutcome]]:
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
    contact: _PlacedContact, landing: _PlacedBounce, outcome: LegOutcome, first_after_reset: bool
) -> LegPhase:
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
    max_flight = seconds_to_frames(_MAX_FLIGHT_S, fps)
    collision = max(1, seconds_to_frames(_CONTACT_BOUNCE_COLLISION_S, fps))
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
            or contact.frame - last_leg_end > seconds_to_frames(_RALLY_RESET_S, fps)
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
    scene: CourtScene,
    court_keypoints_by_frame: Sequence[Optional[CourtKeypoints]],
    contacts: Sequence[ContactEvent],
    bounces: BouncePrediction,
    kinematic_bounces: Sequence[BounceEvent],
) -> List[RallyLeg]:
    """Build completed, plane-valid player-contact-to-landing rally legs.

    A confirmed landing with no contact before it gets one rescued from the
    ball track and player boxes in ``scene`` (none when the track is empty).
    """
    project = _CourtProjector(court_keypoints_by_frame)
    collision = max(1, scene.frames(_CONTACT_BOUNCE_COLLISION_S))
    placed_bounces = _place_bounces(
        bounces.confirmed, bounces.candidates, kinematic_bounces, project, scene.fps,
    )
    placed_contacts = _place_contacts(contacts, placed_bounces, project, collision)
    _rescue_missing_contacts(placed_bounces, placed_contacts, scene, project)
    placed_contacts.sort(key=lambda event: event.frame)
    return _link_legs(placed_contacts, placed_bounces, len(court_keypoints_by_frame), scene.fps)
