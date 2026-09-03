from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .rendering import _build_ground_projection_model, _homography_apply


_BOUNCE_THRESHOLD = 0.45
_INFERRED_BOUNCE_THRESHOLD = 0.20
_EPS = 1e-15


def _build_bounce_features(per_frame, fps: float, width: int, height: int):
    """Build the upstream CatBoost model's five-point, 30 FPS feature rows."""
    stride = max(1, int(round(float(fps) / 30.0)))
    x_scale = 1280.0 / max(float(width), 1.0)
    y_scale = 720.0 / max(float(height), 1.0)
    rows = []
    center_frames = []

    # Step by one frame but keep the +/-2*stride sample spacing the model was
    # trained on: stepping by stride would only ever centre on even frames at
    # 60 FPS, so a bounce that lands on an odd frame could never be scored.
    for frame in range(2 * stride, len(per_frame) - 2 * stride):
        points = [per_frame[frame + offset * stride] for offset in (-2, -1, 0, 1, 2)]
        if any(point is None or bool(getattr(point, "debug_only", False)) for point in points):
            continue
        # Do not let a fully synthetic five-point window manufacture a bounce.
        if sum(getattr(point, "source", "") == "det" for point in points) < 3:
            continue

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


def _inside_player(x: float, y: float, boxes) -> bool:
    for box in (boxes or {}).values():
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = map(float, box[:4])
        pad_x = max(24.0, x2 - x1)
        pad_y = max(24.0, 0.25 * (y2 - y1))
        if x1 - pad_x <= x <= x2 + pad_x and y1 - pad_y <= y <= y2:
            return True
    return False


def predict_bounces(
    per_frame,
    fps: float,
    width: int,
    height: int,
    model_path: str,
    player_boxes_by_frame=None,
    return_candidates: bool = False,
):
    """Return confirmed bounces and, optionally, guarded low-score candidates."""
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Bounce model not found: {path}")
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("--court requires CatBoost: python -m pip install catboost") from exc

    features, center_frames, stride = _build_bounce_features(
        per_frame, fps, width, height
    )
    if not center_frames:
        return ([], []) if return_candidates else []

    model = CatBoostRegressor()
    model.load_model(str(path))
    scores = np.asarray(model.predict(features, thread_count=1), dtype=np.float64).reshape(-1)
    def accepted(pairs):
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
                result.append((frame, x, y, score))
        return result

    # Scoring every frame makes candidates dense, so separate events by a
    # physical interval rather than the sample stride: two real landings are
    # never 0.1 s apart, but one landing can easily score over several frames.
    gap = max(stride, int(round(0.10 * float(fps))))
    events = accepted(_cluster_bounce_candidates(center_frames, scores, gap))
    if not return_candidates:
        return events
    candidates = accepted(_cluster_bounce_candidates(
        center_frames, scores, gap, _INFERRED_BOUNCE_THRESHOLD
    ))
    return events, candidates


def _usable_point(per_frame, frame: int):
    if frame < 0 or frame >= len(per_frame):
        return None
    point = per_frame[frame]
    if point is None or bool(getattr(point, "debug_only", False)):
        return None
    if getattr(point, "source", "") in ("carry", "guide"):
        return None
    return point


def _box_distance(x: float, y: float, box) -> float:
    x1, y1, x2, y2 = map(float, box[:4])
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy))


def _nearest_player(x: float, y: float, boxes):
    best = None
    for player_id, box in (boxes or {}).items():
        if box is None or len(box) < 4:
            continue
        distance = _box_distance(x, y, box)
        if best is None or distance < best[0]:
            best = (distance, player_id, box)
    return best


def _fit_gap_turn(before, after):
    """Intersect incoming/outgoing image-space fits inside a short gap."""
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


def _extrapolate_incoming(samples, frame):
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


def detect_player_contacts(per_frame, player_boxes_by_frame, fps, width, height):
    """Find racket contacts in image space and anchor them at player feet."""
    stride = max(1, int(round(float(fps) / 30.0)))
    diagonal = float(np.hypot(width, height))
    scale = diagonal / 2203.0
    candidates = []

    # Centre on every frame; the window keeps its 30 FPS +/-2*stride spacing.
    for frame in range(2 * stride, len(per_frame) - 2 * stride):
        points = [_usable_point(per_frame, frame + offset * stride)
                  for offset in (-2, -1, 0, 1, 2)]
        if any(point is None for point in points):
            continue
        if sum(getattr(point, "source", "") == "det" for point in points) < 3:
            continue
        if not any(getattr(point, "source", "") == "det" for point in points[3:]):
            continue

        p0 = points[2]
        nearest = _nearest_player(
            float(p0.cx), float(p0.cy), player_boxes_by_frame[frame]
        )
        if nearest is None or nearest[0] > 0.045 * diagonal:
            continue
        distance, player_id, box = nearest
        x1, y1, x2, y2 = map(float, box[:4])
        if float(p0.cy) > y2 + 0.05 * max(y2 - y1, 1.0):
            continue

        incoming = np.array([
            float(p0.cx) - float(points[0].cx),
            float(p0.cy) - float(points[0].cy),
        ]) / 2.0
        outgoing = np.array([
            float(points[4].cx) - float(p0.cx),
            float(points[4].cy) - float(p0.cy),
        ]) / 2.0
        in_speed = float(np.linalg.norm(incoming))
        out_speed = float(np.linalg.norm(outgoing))
        if in_speed < 6.0 * scale or out_speed < 4.0 * scale:
            continue
        cosine = float(np.dot(incoming, outgoing) / max(in_speed * out_speed, 1e-6))
        speed_jump = out_speed >= 2.5 * in_speed and out_speed >= 10.0 * scale
        if (
            not any(getattr(point, "source", "") == "det" for point in points[:2])
            and not speed_jump
        ):
            continue
        if cosine > 0.50 and not speed_jump:
            continue
        score = (1.0 - cosine) * min(in_speed, out_speed) / max(scale, 1e-6)
        candidates.append((
            frame, 0.5 * (x1 + x2), y2, player_id, score, "kink"
        ))

    # Occluded racket contacts often appear as a short reversing gap.
    frame = 0
    min_gap = max(2, int(round(0.05 * fps)))
    max_gap = max(min_gap, int(round(0.30 * fps)))
    while frame < len(per_frame):
        if _usable_point(per_frame, frame) is not None:
            frame += 1
            continue
        start = frame
        while frame < len(per_frame) and _usable_point(per_frame, frame) is None:
            frame += 1
        end = frame - 1
        if not (min_gap <= end - start + 1 <= max_gap):
            continue
        before = []
        after = []
        for index in range(start - 1, max(-1, start - 4 * stride), -1):
            point = _usable_point(per_frame, index)
            if point is not None:
                before.append((index, point))
                if len(before) == 2:
                    break
        for index in range(end + 1, min(len(per_frame), end + 4 * stride + 1)):
            point = _usable_point(per_frame, index)
            if point is not None:
                after.append((index, point))
                if len(after) == 2:
                    break
        if len(before) < 2 or len(after) < 2:
            continue
        before.reverse()
        history = []
        for index in range(start - 1, max(-1, start - 6 * stride - 1), -1):
            point = _usable_point(per_frame, index)
            if point is not None:
                history.append((index, point))
        history.reverse()
        fit = _fit_gap_turn(before, after)
        if fit is None:
            continue
        incoming, outgoing, turn_frame, turn_point, mismatch = fit
        in_speed = float(np.linalg.norm(incoming))
        out_speed = float(np.linalg.norm(outgoing))
        speed_scale = (60.0 / max(float(fps), 1.0)) * scale
        if in_speed < 4.0 * speed_scale or out_speed < 2.0 * speed_scale:
            continue
        cosine = float(np.dot(incoming, outgoing) / max(in_speed * out_speed, 1e-6))
        if cosine > -0.30:
            continue
        midpoint = (start + end) // 2
        impact_point = _extrapolate_incoming(history, midpoint)
        boxes = player_boxes_by_frame[midpoint]
        a = before[-1][1]
        b = after[0][1]
        nearest = None
        for player_id, box in (boxes or {}).items():
            distance = min(
                _box_distance(float(a.cx), float(a.cy), box),
                _box_distance(float(b.cx), float(b.cy), box),
            )
            if nearest is None or distance < nearest[0]:
                nearest = (distance, player_id, box)
        if nearest is None or nearest[0] > 0.025 * diagonal:
            continue
        distance, player_id, box = nearest
        x1, y1, x2, y2 = map(float, box[:4])
        if (
            incoming[1] > 0.0
            and outgoing[1] < 0.0
            and impact_point is not None
            and impact_point[1] >= y2 - 0.40 * max(y2 - y1, 1.0)
        ):
            continue
        score = 1000.0 + (1.0 - cosine) * min(in_speed, out_speed)
        candidates.append((
            midpoint, 0.5 * (x1 + x2), y2, player_id, score, "gap"
        ))

    contacts = []
    for candidate in sorted(candidates, key=lambda item: (str(item[3]), item[0])):
        if (
            contacts
            and contacts[-1][3] == candidate[3]
            and candidate[0] - contacts[-1][0] <= int(round(0.15 * fps))
        ):
            if candidate[4] > contacts[-1][4]:
                contacts[-1] = candidate
        else:
            contacts.append(candidate)
    return sorted(contacts, key=lambda item: item[0])


def infer_kinematic_bounces(per_frame, player_boxes_by_frame, fps, width, height):
    """Find strict down-to-up ground kinks missed by the bounce model."""
    stride = max(1, int(round(float(fps) / 30.0)))
    scale = float(np.hypot(width, height)) / 2203.0
    candidates = []
    # Centre on every frame; the window keeps its 30 FPS +/-2*stride spacing.
    for frame in range(2 * stride, len(per_frame) - 2 * stride):
        points = [_usable_point(per_frame, frame + offset * stride)
                  for offset in (-2, -1, 0, 1, 2)]
        if any(point is None for point in points):
            continue
        if sum(getattr(point, "source", "") == "det" for point in points) < 3:
            continue
        incoming = np.array([
            float(points[2].cx) - float(points[0].cx),
            float(points[2].cy) - float(points[0].cy),
        ]) / 2.0
        outgoing = np.array([
            float(points[4].cx) - float(points[2].cx),
            float(points[4].cy) - float(points[2].cy),
        ]) / 2.0
        in_speed = float(np.linalg.norm(incoming))
        out_speed = float(np.linalg.norm(outgoing))
        cosine = float(np.dot(incoming, outgoing) / max(in_speed * out_speed, 1e-6))
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
        candidates.append((
            frame, float(points[2].cx), float(points[2].cy), score
        ))

    # A real landing may be fully hidden by a short selector gap. Fit the
    # incoming and outgoing branches and keep only a strong down-to-up turn.
    diagonal = float(np.hypot(width, height))
    frame = 0
    min_gap = max(2, int(round(0.05 * fps)))
    max_gap = max(min_gap, int(round(0.30 * fps)))
    while frame < len(per_frame):
        if _usable_point(per_frame, frame) is not None:
            frame += 1
            continue
        start = frame
        while frame < len(per_frame) and _usable_point(per_frame, frame) is None:
            frame += 1
        end = frame - 1
        if not (min_gap <= end - start + 1 <= max_gap):
            continue
        before = []
        after = []
        for index in range(start - 1, max(-1, start - 4 * stride), -1):
            point = _usable_point(per_frame, index)
            if point is not None:
                before.append((index, point))
                if len(before) == 2:
                    break
        for index in range(end + 1, min(len(per_frame), end + 4 * stride + 1)):
            point = _usable_point(per_frame, index)
            if point is not None:
                after.append((index, point))
                if len(after) == 2:
                    break
        if len(before) < 2 or len(after) < 2:
            continue
        before.reverse()
        history = []
        for index in range(start - 1, max(-1, start - 6 * stride - 1), -1):
            point = _usable_point(per_frame, index)
            if point is not None:
                history.append((index, point))
        history.reverse()
        fit = _fit_gap_turn(before, after)
        if fit is None:
            continue
        incoming, outgoing, turn_frame, turn_point, mismatch = fit
        in_speed = float(np.linalg.norm(incoming))
        out_speed = float(np.linalg.norm(outgoing))
        cosine = float(np.dot(incoming, outgoing) / max(in_speed * out_speed, 1e-6))
        event_frame = (start + end) // 2
        impact_point = _extrapolate_incoming(history, event_frame)
        if impact_point is None:
            continue
        prominence = float(impact_point[1] - max(
            float(before[-1][1].cy), float(after[0][1].cy)
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
        if _inside_player(float(impact_point[0]), float(impact_point[1]), boxes):
            nearest = _nearest_player(float(impact_point[0]), float(impact_point[1]), boxes)
            if nearest is None:
                continue
            _, _, box = nearest
            _, y1, _, y2 = map(float, box[:4])
            if impact_point[1] < y2 - 0.40 * max(y2 - y1, 1.0):
                continue
        score = float(prominence * (1.0 - cosine))
        candidates.append((
            event_frame, float(impact_point[0]), float(impact_point[1]), score
        ))

    clustered = []
    radius = max(1, int(round(0.15 * fps)))
    for candidate in candidates:
        if not clustered or candidate[0] - clustered[-1][-1][0] > radius:
            clustered.append([candidate])
        else:
            clustered[-1].append(candidate)
    return [max(group, key=lambda item: item[3]) for group in clustered]


def _project_event(frame, x, y, court_keypoints_by_frame, width, height, cache):
    if frame < 0 or frame >= len(court_keypoints_by_frame):
        return None
    keypoints = court_keypoints_by_frame[frame]
    if keypoints is None or len(keypoints) < 16:
        return None
    try:
        key = tuple(int(round(float(value) * 4.0)) for value in keypoints[:16])
    except Exception:
        return None
    if key not in cache:
        cache[key] = _build_ground_projection_model(keypoints, width, height)
    model = cache[key]
    return None if model is None else _homography_apply(model["H_i2c"], x, y)


def _court_side(point):
    if point[1] < 0.47:
        return -1
    if point[1] > 0.53:
        return 1
    return 0


def _is_service_leg(start, end, first_after_reset):
    if not first_after_reset:
        return False
    singles = (10.97 - 8.23) / (2.0 * 10.97)
    service = 6.40 / 23.77
    if not (singles <= end[0] <= 1.0 - singles):
        return False
    if (start[0] - 0.5) * (end[0] - 0.5) >= 0.0:
        return False
    if start[1] <= 0.15:
        return 0.5 <= end[1] <= 0.5 + service
    if start[1] >= 0.85:
        return 0.5 - service <= end[1] <= 0.5
    return False


def build_rally_legs(
    contacts,
    confirmed_bounces,
    bounce_candidates,
    kinematic_bounces,
    court_keypoints_by_frame,
    fps,
    width,
    height,
    per_frame=None,
    player_boxes_by_frame=None,
):
    """Build completed, plane-valid player-contact-to-landing rally legs."""
    cache = {}

    projected_bounces = []
    confirmed_frames = {int(event[0]) for event in confirmed_bounces}
    for frame, x, y, score in confirmed_bounces:
        point = _project_event(frame, x, y, court_keypoints_by_frame, width, height, cache)
        if point is not None and -0.08 <= point[0] <= 1.08 and -0.08 <= point[1] <= 1.08:
            projected_bounces.append({
                "frame": int(frame), "point": point, "quality": "confirmed",
                "score": float(score),
            })

    inferred_sources = list(bounce_candidates) + list(kinematic_bounces)
    for frame, x, y, score in inferred_sources:
        if any(abs(int(frame) - value) <= int(round(0.15 * fps))
               for value in confirmed_frames):
            continue
        point = _project_event(frame, x, y, court_keypoints_by_frame, width, height, cache)
        if point is not None and -0.05 <= point[0] <= 1.05 and -0.05 <= point[1] <= 1.05:
            projected_bounces.append({
                "frame": int(frame), "point": point, "quality": "inferred",
                "score": float(score),
            })

    collision = max(1, int(round(0.12 * fps)))
    projected_contacts = []
    for frame, x, y, player_id, score, source in contacts:
        if any(abs(int(frame) - bounce["frame"]) <= collision
               for bounce in projected_bounces if bounce["quality"] == "confirmed"):
            continue
        point = _project_event(frame, x, y, court_keypoints_by_frame, width, height, cache)
        if point is not None and -0.12 <= point[0] <= 1.12 and -0.12 <= point[1] <= 1.12:
            projected_contacts.append({
                "frame": int(frame), "point": point, "player": player_id,
                "score": float(score), "source": source,
            })

    projected_bounces.sort(key=lambda event: event["frame"])
    projected_contacts.sort(key=lambda event: event["frame"])
    max_flight = int(round(2.50 * fps))

    # A confirmed landing must not disappear merely because the racket contact
    # was missed. Search backward for the closest opposite-side player/ball
    # encounter and use only that player's grounded footpoint as the origin.
    if per_frame is not None and player_boxes_by_frame is not None:
        stride = max(1, int(round(float(fps) / 30.0)))
        diagonal = float(np.hypot(width, height))
        for bounce_index, bounce in enumerate(projected_bounces):
            if bounce["quality"] != "confirmed":
                continue
            bounce_side = _court_side(bounce["point"])
            if bounce_side == 0:
                continue
            previous = projected_bounces[bounce_index - 1] if bounce_index else None
            if (
                previous is not None
                and _court_side(previous["point"]) == bounce_side
                and bounce["frame"] - previous["frame"] <= int(round(1.20 * fps))
            ):
                continue
            if any(
                bounce["frame"] - max_flight < contact["frame"] < bounce["frame"] - collision
                and _court_side(contact["point"]) == -bounce_side
                for contact in projected_contacts
            ):
                continue

            best = None
            for frame in range(
                max(0, bounce["frame"] - max_flight),
                max(0, bounce["frame"] - collision),
                stride,
            ):
                ball = _usable_point(per_frame, frame)
                if ball is None:
                    continue
                for player_id, box in (player_boxes_by_frame[frame] or {}).items():
                    if box is None or len(box) < 4:
                        continue
                    x1, _, x2, y2 = map(float, box[:4])
                    foot = _project_event(
                        frame, 0.5 * (x1 + x2), y2,
                        court_keypoints_by_frame, width, height, cache,
                    )
                    if foot is None or _court_side(foot) != -bounce_side:
                        continue
                    distance = _box_distance(float(ball.cx), float(ball.cy), box)
                    if best is None or distance < best[0]:
                        best = (distance, frame, foot, player_id)
            if best is not None and best[0] <= 0.10 * diagonal:
                projected_contacts.append({
                    "frame": best[1], "point": best[2], "player": best[3],
                    "score": max(0.0, 1.0 - best[0] / (0.10 * diagonal)),
                    "source": "inferred",
                })

    projected_contacts.sort(key=lambda event: event["frame"])
    legs = []
    last_leg_end = None

    for index, contact in enumerate(projected_contacts):
        next_contact = (
            projected_contacts[index + 1]
            if index + 1 < len(projected_contacts)
            else None
        )
        if (
            next_contact is not None
            and next_contact["frame"] - contact["frame"] > max_flight
        ):
            next_contact = None
        stop = min(
            contact["frame"] + max_flight,
            next_contact["frame"] if next_contact is not None else len(court_keypoints_by_frame),
        )
        confirmed = [
            event for event in projected_bounces
            if event["quality"] == "confirmed"
            and contact["frame"] + collision < event["frame"] < stop
        ]
        landing = confirmed[0] if confirmed else None
        outcome = "bounce"

        if landing is None:
            start_side = _court_side(contact["point"])
            inferred = [
                event for event in projected_bounces
                if event["quality"] == "inferred"
                and contact["frame"] + collision < event["frame"] < stop
                and start_side != 0
                and _court_side(event["point"]) == -start_side
            ]
            if inferred:
                landing = max(inferred, key=lambda event: event["score"])

        if landing is None and next_contact is not None:
            start_side = _court_side(contact["point"])
            end_side = _court_side(next_contact["point"])
            if start_side == 0 or end_side == 0 or start_side == end_side:
                continue
            if abs(next_contact["point"][1] - 0.5) <= 0.27:
                landing = {
                    "frame": next_contact["frame"],
                    "point": next_contact["point"],
                    "quality": None,
                    "score": next_contact["score"],
                }
                outcome = "volley"

        if landing is None:
            continue
        start_side = _court_side(contact["point"])
        end_side = _court_side(landing["point"])
        if start_side == 0 or end_side == 0:
            continue
        first_after_reset = (
            last_leg_end is None
            or contact["frame"] - last_leg_end > int(round(2.50 * fps))
        )
        if outcome == "volley":
            phase = "rally"
        elif start_side == end_side:
            phase = "fault"
        elif _is_service_leg(contact["point"], landing["point"], first_after_reset):
            phase = "serve"
        else:
            phase = "rally"
        legs.append({
            "start_frame": contact["frame"],
            "end_frame": landing["frame"],
            "start": contact["point"],
            "end": landing["point"],
            "phase": phase,
            "outcome": outcome,
            "quality": landing["quality"],
            "start_quality": (
                "inferred" if contact["source"] == "inferred" else "confirmed"
            ),
        })
        last_leg_end = landing["frame"]
    return legs


def _panel_point(
    point: Tuple[float, float],
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> Tuple[int, int]:
    u = float(np.clip(point[0], -0.08, 1.08))
    v = float(np.clip(point[1], -0.08, 1.08))
    return (
        int(round(left + u * (right - left))),
        int(round(top + v * (bottom - top))),
    )


def _draw_dashed_line(image, start, end, color, thickness=1):
    distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    if distance < 1.0:
        return
    steps = max(1, int(distance // 7.0))
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


def _visible_completed_legs(rally_legs, contact_events, current_frame, fps):
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
        latest["end_frame"] < int(event[0]) <= current_frame
        for event in (contact_events or ())
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


def draw_court_minimap(
    frame,
    ground_model,
    player_boxes,
    rally_legs,
    current_frame: int,
    fps: float,
    contact_events=(),
) -> None:
    """Draw the optional top-right 2D court minibar in-place."""
    if frame is None or ground_model is None:
        return
    H = ground_model.get("H_i2c")
    if H is None:
        return

    frame_h, frame_w = frame.shape[:2]
    panel_w = int(np.clip(round(frame_w * 0.10), 130, 180))
    panel_h = int(round(panel_w * 2.08))
    margin = max(8, int(round(frame_w * 0.006)))
    x0 = frame_w - panel_w - margin
    y0 = margin
    if x0 < 0 or y0 + panel_h > frame_h:
        return

    panel = np.full((panel_h, panel_w, 3), (42, 42, 42), dtype=np.uint8)
    court_w = panel_w - 36
    court_h = int(round(court_w * 23.77 / 10.97))
    left = (panel_w - court_w) // 2
    top = (panel_h - court_h) // 2
    right, bottom = left + court_w, top + court_h
    white = (235, 235, 235)

    cv2.rectangle(panel, (left, top), (right, bottom), (28, 72, 42), -1)
    cv2.rectangle(panel, (left, top), (right, bottom), white, 1, cv2.LINE_AA)
    singles = (10.97 - 8.23) / (2.0 * 10.97)
    service = 6.40 / 23.77
    for u in (singles, 1.0 - singles):
        cv2.line(panel, _panel_point((u, 0.0), left, top, right, bottom),
                 _panel_point((u, 1.0), left, top, right, bottom), white, 1, cv2.LINE_AA)
    cv2.line(panel, _panel_point((0.0, 0.5), left, top, right, bottom),
             _panel_point((1.0, 0.5), left, top, right, bottom), (90, 180, 255), 2, cv2.LINE_AA)
    for v in (0.5 - service, 0.5 + service):
        cv2.line(panel, _panel_point((singles, v), left, top, right, bottom),
                 _panel_point((1.0 - singles, v), left, top, right, bottom), white, 1, cv2.LINE_AA)
    cv2.line(panel, _panel_point((0.5, 0.5 - service), left, top, right, bottom),
             _panel_point((0.5, 0.5 + service), left, top, right, bottom), white, 1, cv2.LINE_AA)
    cv2.putText(panel, "COURT", (7, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (235, 235, 235), 1, cv2.LINE_AA)

    visible_legs = _visible_completed_legs(
        rally_legs, contact_events, current_frame, fps
    )
    phase_colors = {
        "serve": (255, 210, 60),
        "rally": (70, 220, 255),
        "fault": (70, 70, 255),
    }
    for leg in visible_legs:
        start_point = _panel_point(leg["start"], left, top, right, bottom)
        end_point = _panel_point(leg["end"], left, top, right, bottom)
        base_color = phase_colors.get(leg["phase"], (70, 220, 255))
        color = base_color
        cv2.line(panel, start_point, end_point, (12, 12, 12), 3, cv2.LINE_AA)
        if leg["quality"] == "inferred" or leg.get("start_quality") == "inferred":
            _draw_dashed_line(panel, start_point, end_point, color, 2)
        else:
            cv2.line(panel, start_point, end_point, color, 2, cv2.LINE_AA)

        # q(s)=4s(1-s) is a relative height cue only. Court coordinates stay
        # on the straight, physically valid plan-view chord.
        for step in range(1, 10):
            fraction = step / 10.0
            relative_height = 4.0 * fraction * (1.0 - fraction)
            point = (
                int(round(start_point[0] + fraction * (end_point[0] - start_point[0]))),
                int(round(start_point[1] + fraction * (end_point[1] - start_point[1]))),
            )
            radius = 1 + int(round(relative_height * (2 if leg["phase"] == "serve" else 1)))
            cv2.circle(panel, point, radius, color, -1, cv2.LINE_AA)

        if leg["outcome"] == "bounce":
            if leg["quality"] == "inferred":
                x, y = end_point
                diamond = np.array(((x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)), np.int32)
                cv2.polylines(panel, [diamond], True, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.putText(panel, "?", (x + 5, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.30, (0, 165, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(panel, end_point, 5, (15, 15, 15), 2, cv2.LINE_AA)
                cv2.circle(panel, end_point, 4, (0, 255, 255), 2, cv2.LINE_AA)
        elif leg["outcome"] == "volley":
            cv2.putText(panel, "V", (end_point[0] + 4, end_point[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    if visible_legs:
        latest = visible_legs[-1]
        label = {"serve": "SERVE", "fault": "FAULT"}.get(latest["phase"], "RALLY")
        if latest["outcome"] == "volley":
            label = "VOLLEY"
        cv2.putText(panel, label, (panel_w - 48, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.30, phase_colors.get(latest["phase"], (70, 220, 255)),
                    1, cv2.LINE_AA)

    player_colors = ((255, 110, 60), (70, 90, 255), (255, 80, 220), (80, 230, 120))
    for index, (player_id, box) in enumerate(
        sorted((player_boxes or {}).items(), key=lambda item: str(item[0]))
    ):
        if box is None or len(box) < 4:
            continue
        x1, _, x2, y2 = map(float, box[:4])
        projected = _homography_apply(H, 0.5 * (x1 + x2), y2)
        if projected is None:
            continue
        center = _panel_point(projected, left, top, right, bottom)
        color = player_colors[index % len(player_colors)]
        cv2.circle(panel, center, 5, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(panel, center, 3, color, -1, cv2.LINE_AA)

    roi = frame[y0:y0 + panel_h, x0:x0 + panel_w]
    cv2.addWeighted(panel, 0.90, roi, 0.10, 0.0, roi)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w - 1, y0 + panel_h - 1),
                  (230, 230, 230), 1, cv2.LINE_AA)
