"""Racket contacts: visible direction changes and reversing occlusion gaps next to a player,
anchored at the player's feet (``detect_player_contacts``, reading one ``court_scene.CourtScene``).
"""

from __future__ import annotations

from typing import Hashable, List, Optional, Sequence, Tuple

from ball_in_play_selector import FrameResult

from .ball_turns import det_windows, extrapolate_incoming, gap_turns, turn_metrics, window_velocities
from .court_scene import (
    EVENT_MERGE_S,
    LOWER_BODY_FRACTION,
    ContactEvent,
    CourtScene,
    PlayerBox,
    PlayerBoxes,
    box_distance,
    nearest_player,
    seconds_to_frames,
)


# Gap-fit velocities are per source frame; their thresholds were tuned at 60 FPS.
_GAP_SPEED_REFERENCE_FPS = 60.0
# Gap contacts are decisive evidence, so they outrank any kink score when
# contacts by the same player are merged.
_GAP_CONTACT_SCORE_BONUS = 1000.0


def _striking_player(
    ball: FrameResult, boxes: Optional[PlayerBoxes], diagonal: float
) -> Optional[Tuple[Hashable, PlayerBox]]:
    """The player whose box the ball is touching at or above the feet."""
    nearest = nearest_player(float(ball.cx), float(ball.cy), boxes)
    if nearest is None or nearest[0] > 0.045 * diagonal:
        return None
    _, player_id, box = nearest
    _, y1, _, y2 = map(float, box[:4])
    if float(ball.cy) > y2 + 0.05 * max(y2 - y1, 1.0):
        return None
    return player_id, box


def _kink_contact_score(points: Sequence[FrameResult], scale: float) -> Optional[float]:
    """Score a racket-like turn (sharp, or a sudden speed-up), else None."""
    incoming, outgoing = window_velocities(points)
    in_speed, out_speed, cosine = turn_metrics(incoming, outgoing)
    if in_speed < 6.0 * scale or out_speed < 4.0 * scale:
        return None
    speed_jump = out_speed >= 2.5 * in_speed and out_speed >= 10.0 * scale
    if not any(point.source == "det" for point in points[:2]) and not speed_jump:
        return None
    if cosine > 0.50 and not speed_jump:
        return None
    return (1.0 - cosine) * min(in_speed, out_speed) / max(scale, 1e-6)


def _kink_contacts(scene: CourtScene) -> List[ContactEvent]:
    """Visible direction changes next to a player."""
    contacts = []
    for frame, points in det_windows(scene.per_frame, scene.stride):
        if not any(point.source == "det" for point in points[3:]):
            continue
        striker = _striking_player(points[2], scene.player_boxes_by_frame[frame], scene.diagonal)
        if striker is None:
            continue
        score = _kink_contact_score(points, scene.scale)
        if score is None:
            continue
        player_id, box = striker
        x1, _, x2, y2 = map(float, box[:4])
        contacts.append(ContactEvent(frame, 0.5 * (x1 + x2), y2, player_id, score, "kink"))
    return contacts


def _nearest_player_to_either(
    a: FrameResult, b: FrameResult, boxes: Optional[PlayerBoxes]
) -> Optional[Tuple[float, Hashable, PlayerBox]]:
    nearest = None
    for player_id, box in (boxes or {}).items():
        distance = min(
            box_distance(float(a.cx), float(a.cy), box),
            box_distance(float(b.cx), float(b.cy), box),
        )
        if nearest is None or distance < nearest[0]:
            nearest = (distance, player_id, box)
    return nearest


def _gap_contacts(scene: CourtScene) -> List[ContactEvent]:
    """Occluded racket contacts, which often appear as a short reversing gap."""
    contacts = []
    speed_scale = (_GAP_SPEED_REFERENCE_FPS / max(float(scene.fps), 1.0)) * scene.scale
    for gap in gap_turns(scene):
        in_speed, out_speed, cosine = turn_metrics(gap.incoming, gap.outgoing)
        if in_speed < 4.0 * speed_scale or out_speed < 2.0 * speed_scale:
            continue
        if cosine > -0.30:
            continue
        midpoint = gap.midpoint
        nearest = _nearest_player_to_either(
            gap.before[-1][1], gap.after[0][1], scene.player_boxes_by_frame[midpoint]
        )
        if nearest is None or nearest[0] > 0.025 * scene.diagonal:
            continue
        _, player_id, box = nearest
        x1, y1, x2, y2 = map(float, box[:4])
        impact_point = extrapolate_incoming(gap.history, midpoint)
        if (
            gap.incoming[1] > 0.0
            and gap.outgoing[1] < 0.0
            and impact_point is not None
            and impact_point[1] >= y2 - LOWER_BODY_FRACTION * max(y2 - y1, 1.0)
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
            and candidate.frame - contacts[-1].frame <= seconds_to_frames(EVENT_MERGE_S, fps)
        ):
            if candidate.score > contacts[-1].score:
                contacts[-1] = candidate
        else:
            contacts.append(candidate)
    return sorted(contacts, key=lambda item: item.frame)


def detect_player_contacts(scene: CourtScene) -> List[ContactEvent]:
    """Find racket contacts in image space and anchor them at player feet."""
    candidates = _kink_contacts(scene) + _gap_contacts(scene)
    return _merge_contacts(candidates, scene.fps)
