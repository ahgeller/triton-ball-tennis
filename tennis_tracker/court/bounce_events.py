"""Ball landings: the CatBoost bounce model's confirmed bounces and low-score candidates
(``predict_bounce_events``) and the strict down-to-up kinks it missed (``infer_kinematic_bounces``).
Both read one ``court_scene.CourtScene`` and return ``BounceEvent``s in image pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

from .ball_turns import det_windows, extrapolate_incoming, gap_turns, turn_metrics, window_velocities
from .court_scene import (
    EVENT_MERGE_S,
    LOWER_BODY_FRACTION,
    BounceEvent,
    BouncePrediction,
    CourtScene,
    inside_player,
    nearest_player,
)


# --- Bounce model -----------------------------------------------------------
_BOUNCE_THRESHOLD = 0.45  # CatBoost score above which a landing is confirmed
_INFERRED_BOUNCE_THRESHOLD = 0.20  # lower bar for guarded landing candidates
_EPS = 1e-15  # keeps the model's ratio features finite
# The model was trained on 1280x720 tracks sampled at 30 FPS.
_MODEL_WIDTH = 1280.0
_MODEL_HEIGHT = 720.0
# Two scored landings closer than this are the same landing.
_BOUNCE_MIN_SEPARATION_S = 0.10


def _build_bounce_features(scene: CourtScene) -> Tuple[np.ndarray, List[int]]:
    """Build the upstream CatBoost model's five-point, 30 FPS feature rows and
    the source frame each row is centred on."""
    x_scale = _MODEL_WIDTH / max(float(scene.width), 1.0)
    y_scale = _MODEL_HEIGHT / max(float(scene.height), 1.0)
    rows = []
    center_frames = []

    for frame, points in det_windows(scene.per_frame, scene.stride):
        xs = np.asarray([point.cx * x_scale for point in points], dtype=np.float64)
        ys = np.asarray([point.cy * y_scale for point in points], dtype=np.float64)
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

    return np.asarray(rows, dtype=np.float32).reshape(-1, 12), center_frames


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


def _catboost_regressor() -> Any:
    """Return the CatBoostRegressor class, failing with an install hint."""
    # CatBoost is only needed for --court, so it is imported on first use
    # rather than making every pipeline import depend on it.
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("--court requires CatBoost: python -m pip install catboost") from exc
    return CatBoostRegressor


def _off_player_events(pairs: Iterable[Tuple[int, float]], scene: CourtScene) -> List[BounceEvent]:
    """Attach ball positions to scored frames, dropping those on a player."""
    result = []
    for frame, score in pairs:
        point = scene.per_frame[frame]
        if point is None:
            continue
        x, y = float(point.cx), float(point.cy)
        if not inside_player(x, y, scene.player_boxes(frame)):
            result.append(BounceEvent(frame, x, y, score))
    return result


def predict_bounce_events(scene: CourtScene, model_path: str) -> BouncePrediction:
    """Score landings with the CatBoost bounce model.

    Returns confirmed bounces (score above 0.45) and guarded candidates (above
    0.20) that ``assemble_rally_legs`` may use when no confirmed landing exists.
    Positions on a player box are dropped from both.
    """
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Bounce model not found: {path}")
    regressor = _catboost_regressor()
    features, center_frames = _build_bounce_features(scene)
    if not center_frames:
        return BouncePrediction([], [])

    model = regressor()
    model.load_model(str(path))
    scores = np.asarray(model.predict(features, thread_count=1), dtype=np.float64).reshape(-1)

    # Scoring every frame makes candidates dense, so separate events by a
    # physical interval rather than the sample stride: two real landings are
    # never 0.1 s apart, but one landing can easily score over several frames.
    gap = max(scene.stride, scene.frames(_BOUNCE_MIN_SEPARATION_S))
    confirmed = _off_player_events(_cluster_bounce_candidates(center_frames, scores, gap), scene)
    candidates = _off_player_events(
        _cluster_bounce_candidates(center_frames, scores, gap, _INFERRED_BOUNCE_THRESHOLD), scene,
    )
    return BouncePrediction(confirmed, candidates)


def _kink_bounces(scene: CourtScene) -> List[BounceEvent]:
    """Visible strict down-to-up kinks away from the players."""
    bounces = []
    stride, scale = scene.stride, scene.scale
    for frame, points in det_windows(scene.per_frame, stride):
        incoming, outgoing = window_velocities(points)
        in_speed, out_speed, cosine = turn_metrics(incoming, outgoing)
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
            inside_player(
                float(points[index].cx), float(points[index].cy),
                scene.player_boxes_by_frame[frame + (index - 2) * stride],
            )
            for index in (1, 2, 3)
        ):
            continue
        score = float(prominence * (1.0 - cosine))
        bounces.append(BounceEvent(frame, float(points[2].cx), float(points[2].cy), score))
    return bounces


def _gap_bounces(scene: CourtScene) -> List[BounceEvent]:
    """Landings fully hidden by a short selector gap.

    Fit the incoming and outgoing branches and keep only a strong down-to-up turn.
    """
    bounces = []
    scale = scene.scale
    for gap in gap_turns(scene):
        incoming, outgoing = gap.incoming, gap.outgoing
        in_speed, out_speed, cosine = turn_metrics(incoming, outgoing)
        event_frame = gap.midpoint
        impact_point = extrapolate_incoming(gap.history, event_frame)
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
        boxes = scene.player_boxes_by_frame[
            int(np.clip(event_frame, 0, len(scene.player_boxes_by_frame) - 1))
        ]
        impact_x, impact_y = float(impact_point[0]), float(impact_point[1])
        if inside_player(impact_x, impact_y, boxes):
            # Beside a player only a landing at the feet counts.
            nearest = nearest_player(impact_x, impact_y, boxes)
            if nearest is None:
                continue
            _, y1, _, y2 = map(float, nearest[2][:4])
            if impact_point[1] < y2 - LOWER_BODY_FRACTION * max(y2 - y1, 1.0):
                continue
        score = float(prominence * (1.0 - cosine))
        bounces.append(BounceEvent(event_frame, impact_x, impact_y, score))
    return bounces


def infer_kinematic_bounces(scene: CourtScene) -> List[BounceEvent]:
    """Find strict down-to-up ground kinks missed by the bounce model."""
    # Candidates stay in detection order (kinks, then gaps), as tuned.
    candidates = _kink_bounces(scene) + _gap_bounces(scene)
    clustered: List[List[BounceEvent]] = []
    radius = max(1, scene.frames(EVENT_MERGE_S))
    for candidate in candidates:
        if not clustered or candidate.frame - clustered[-1][-1].frame > radius:
            clustered.append([candidate])
        else:
            clustered[-1].append(candidate)
    return [max(group, key=lambda item: item.score) for group in clustered]
