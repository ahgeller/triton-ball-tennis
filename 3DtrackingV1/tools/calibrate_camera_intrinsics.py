"""Calibrate camera intrinsics/distortion from checkerboard images or video.

The output JSON is accepted by reconstruct_3d_v2.py --camera-json.

Important: --board-cols and --board-rows are the number of inner corners, not
the number of printed squares.  A board with 10 by 7 squares has 9 by 6 inner
corners.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from trajectory3d_v1 import _import_cv2, _json_safe


def build_checkerboard_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    if cols <= 1 or rows <= 1:
        raise ValueError("checkerboard must have at least 2 inner corners in each direction")
    if square_size_m <= 0:
        raise ValueError("square_size_m must be positive")
    obj = np.zeros((rows * cols, 3), dtype=np.float32)
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    obj[:, 0] = grid_x.reshape(-1) * float(square_size_m)
    obj[:, 1] = grid_y.reshape(-1) * float(square_size_m)
    return obj


def expand_image_patterns(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    unique = sorted({p.resolve() for p in paths if p.exists() and p.is_file()})
    return unique


def iter_video_frames(video_path: Path, sample_every: int, max_frames: int) -> Iterable[Tuple[str, np.ndarray]]:
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    sample_every = max(1, int(sample_every))
    max_frames = max(1, int(max_frames))
    emitted = 0
    idx = 0
    while emitted < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            yield f"{video_path.name}:frame{idx}", frame
            emitted += 1
        idx += 1
    cap.release()


def iter_image_frames(paths: Sequence[Path]) -> Iterable[Tuple[str, np.ndarray]]:
    cv2 = _import_cv2()
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            yield str(path), img


def find_checkerboard_corners(
    image: np.ndarray,
    cols: int,
    rows: int,
) -> Tuple[bool, Optional[np.ndarray]]:
    cv2 = _import_cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern_size = (int(cols), int(rows))

    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if ok:
            return True, corners.astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        return False, None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.001,
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners.astype(np.float32)


def draw_debug_corners(image: np.ndarray, cols: int, rows: int, corners: np.ndarray, out_path: Path) -> None:
    cv2 = _import_cv2()
    vis = image.copy()
    cv2.drawChessboardCorners(vis, (int(cols), int(rows)), corners, True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def calibrate_from_frames(
    frames: Iterable[Tuple[str, np.ndarray]],
    cols: int,
    rows: int,
    square_size_m: float,
    min_views: int = 12,
    debug_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    cv2 = _import_cv2()
    obj_template = build_checkerboard_object_points(cols, rows, square_size_m)
    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    accepted: List[str] = []
    rejected: List[str] = []
    image_size: Optional[Tuple[int, int]] = None

    for name, frame in frames:
        if frame is None or frame.size == 0:
            rejected.append(name)
            continue
        h, w = frame.shape[:2]
        if image_size is None:
            image_size = (int(w), int(h))
        elif image_size != (int(w), int(h)):
            rejected.append(f"{name} (size {w}x{h} differs from first frame {image_size[0]}x{image_size[1]})")
            continue

        ok, corners = find_checkerboard_corners(frame, cols, rows)
        if not ok or corners is None:
            rejected.append(name)
            continue
        obj_points.append(obj_template.copy())
        img_points.append(corners)
        accepted.append(name)
        if debug_dir is not None:
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in Path(name).stem)
            draw_debug_corners(frame, cols, rows, corners, debug_dir / f"{len(accepted):03d}_{safe_name}.jpg")

    if image_size is None:
        raise RuntimeError("No readable frames/images were found.")
    if len(obj_points) < int(min_views):
        raise RuntimeError(
            f"Only found {len(obj_points)} checkerboard views; need at least {min_views}. "
            "Capture more angles/distances or lower --min-views for a rough diagnostic calibration."
        )

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
    )

    per_view_errors = []
    total_sq = 0.0
    total_points = 0
    for obj, img, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        projected = projected.reshape(-1, 2)
        img2 = img.reshape(-1, 2)
        err = np.linalg.norm(projected - img2, axis=1)
        per_view_errors.append(float(np.sqrt(np.mean(err * err))))
        total_sq += float(np.sum(err * err))
        total_points += int(len(err))

    mean_error = math.sqrt(total_sq / max(1, total_points))
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, image_size, 0.0, image_size)
    quality = "good"
    warnings: List[str] = []
    if mean_error > 0.8:
        quality = "ok"
        warnings.append("Mean reprojection error is above 0.8 px; usable but not great.")
    if mean_error > 1.5:
        quality = "weak"
        warnings.append("Mean reprojection error is above 1.5 px; recapture calibration footage.")
    if len(obj_points) < 20:
        warnings.append("Fewer than 20 accepted views; more views usually improves distortion estimates.")

    return {
        "schema_version": "camera_intrinsics_v1",
        "camera_model": "opencv_pinhole_radtan",
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "checkerboard": {
            "inner_corners": [int(cols), int(rows)],
            "square_size_m": float(square_size_m),
        },
        "K": np.asarray(K, dtype=float).tolist(),
        "dist": np.asarray(dist, dtype=float).reshape(-1).tolist(),
        "optimal_new_K_alpha0": np.asarray(new_K, dtype=float).tolist(),
        "optimal_new_K_roi": [int(v) for v in roi],
        "rms_reprojection_px": float(rms),
        "mean_reprojection_error_px": float(mean_error),
        "per_view_rms_px": per_view_errors,
        "view_count": int(len(obj_points)),
        "rejected_count": int(len(rejected)),
        "accepted_sources": accepted,
        "rejected_sources": rejected[:80],
        "quality": quality,
        "warnings": warnings,
        "use_with": "python 3DtrackingV1/tools/reconstruct_3d_v2.py --camera-json <this file>",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate OpenCV pinhole intrinsics/distortion from checkerboard images or video."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", nargs="+", help="Image paths or glob patterns, e.g. calibration_frames/*.jpg")
    source.add_argument("--video", help="Calibration video path")
    parser.add_argument("--board-cols", type=int, required=True, help="Checkerboard inner corners across columns")
    parser.add_argument("--board-rows", type=int, required=True, help="Checkerboard inner corners down rows")
    parser.add_argument("--square-size-m", type=float, required=True, help="Printed checker square size in meters")
    parser.add_argument("--output-json", required=True, help="Path to write camera intrinsics JSON")
    parser.add_argument("--sample-every", type=int, default=30, help="For --video, sample every N frames")
    parser.add_argument("--max-frames", type=int, default=120, help="For --video, maximum sampled frames to inspect")
    parser.add_argument("--min-views", type=int, default=12, help="Minimum accepted checkerboard views")
    parser.add_argument("--debug-dir", default=None, help="Optional directory of images with detected corners drawn")
    args = parser.parse_args()

    if args.images:
        paths = expand_image_patterns(args.images)
        if not paths:
            raise RuntimeError("No image files matched --images.")
        frames = iter_image_frames(paths)
        source_count = len(paths)
    else:
        video_path = Path(args.video)
        frames = iter_video_frames(video_path, args.sample_every, args.max_frames)
        source_count = 1

    result = calibrate_from_frames(
        frames,
        cols=args.board_cols,
        rows=args.board_rows,
        square_size_m=args.square_size_m,
        min_views=args.min_views,
        debug_dir=None if args.debug_dir is None else Path(args.debug_dir),
    )
    result["source_count"] = int(source_count)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, indent=2)

    print(
        "[calibrate] "
        f"views={result['view_count']} mean_error={result['mean_reprojection_error_px']:.3f}px "
        f"quality={result['quality']}"
    )
    for warning in result.get("warnings", []):
        print(f"[calibrate][warn] {warning}")
    print(f"[calibrate] Output JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
