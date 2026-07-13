"""Strict structural and quality gate for the bundled Pomona benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from validate_tracking import validate


BASELINE = {
    "total_frames": 2022,
    "width": 1920,
    "height": 1080,
    "annotation_sha256": "1e914ed38d4a46ca5f2c142af2a012fbe422ec75091a9387154a6d339b54efe6",
    "video_sha256": "bfd6f73e0ca489111b582040bfc4fe3bba601e67dc93a5a69ac55fec18f9be1a",
    "visible_labels": 76,
    "absent_labels": 4,
    "recall": 1.0,
    "mean_error_px": 1.823580027112843,
    "p90_error_px": 2.8623242433037124,
    "max_error_px": 20.416659863944435,
    "within_5px": 75,
    "false_positive_absent_frames": 3,
    "large_jump_count": 1,
    "present_frames": 687,
}

SMOOTHNESS_LIMITS = {
    "all_p95": 8.0,
    "all_over_20_rate": 0.02,
    "soft_p95": 12.0,
    "soft_over_20_rate": 0.03,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _acceleration_stats(frames: list, soft_only: bool = False) -> dict:
    values = []
    for previous, current, following in zip(frames, frames[1:], frames[2:]):
        if not all(row.get("present") for row in (previous, current, following)):
            continue
        if not (
            previous["frame"] + 1 == current["frame"]
            and current["frame"] + 1 == following["frame"]
        ):
            continue
        if soft_only and current.get("source") == "det":
            continue
        values.append(
            math.hypot(
                following["x"] - 2 * current["x"] + previous["x"],
                following["y"] - 2 * current["y"] + previous["y"],
            )
        )
    _require(bool(values), "no consecutive trajectory triples to check")
    values.sort()
    return {
        "count": len(values),
        "p95": values[math.ceil(0.95 * len(values)) - 1],
        "over_20_rate": sum(value > 20.0 for value in values) / len(values),
    }


def check(prediction_path: Path, annotation_path: Path, video_path: Path) -> dict:
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    _require(_sha256(annotation_path) == BASELINE["annotation_sha256"], "annotation manifest changed")
    _require(_sha256(video_path) == BASELINE["video_sha256"], "benchmark video changed")
    labels = annotation.get("ball") or []
    visible = [row for row in labels if row.get("visible", True)]
    absent = [row for row in labels if not row.get("visible", True)]
    _require(len(visible) == BASELINE["visible_labels"], "visible-label count changed")
    _require(len(absent) == BASELINE["absent_labels"], "absence-label count changed")
    label_frames = [int(row["frame"]) for row in labels]
    _require(len(label_frames) == len(set(label_frames)), "annotation frames must be unique")
    frames = prediction.get("frames") or []
    video = prediction.get("video") or {}
    total = int(video.get("total_frames", -1))
    _require(total == BASELINE["total_frames"], f"expected {BASELINE['total_frames']} frames, found {total}")
    _require(len(frames) == total, f"expected {total} frame rows, found {len(frames)}")
    _require(
        [row.get("frame") for row in frames] == list(range(total)),
        "frames must be contiguous and unique",
    )

    width, height = int(video["width"]), int(video["height"])
    _require(
        (width, height) == (BASELINE["width"], BASELINE["height"]),
        f"expected {BASELINE['width']}x{BASELINE['height']}, found {width}x{height}",
    )
    for row in frames:
        if not row.get("present"):
            continue
        x, y = row.get("x"), row.get("y")
        _require(
            x is not None and y is not None and math.isfinite(x) and math.isfinite(y),
            f"frame {row['frame']} has invalid coordinates",
        )
        _require(
            0.0 <= x < width and 0.0 <= y < height,
            f"frame {row['frame']} is out of bounds",
        )

    report = validate(prediction, annotation, jump_px=120.0, jump_max_gap=1)
    summary = report["summary"]
    all_smoothness = _acceleration_stats(frames)
    soft_smoothness = _acceleration_stats(frames, soft_only=True)
    gates = {
        "recall": summary["recall"] >= BASELINE["recall"],
        "mean error": summary["mean_error_px"] <= BASELINE["mean_error_px"],
        "p90 error": summary["p90_error_px"] <= BASELINE["p90_error_px"],
        "max error": summary["max_error_px"] <= BASELINE["max_error_px"],
        "within 5px": summary["within_5px"] >= BASELINE["within_5px"],
        "absence false positives": summary["false_positive_absent_frames"] <= BASELINE["false_positive_absent_frames"],
        "large jumps": summary["large_jump_count"] <= BASELINE["large_jump_count"],
        "filled frames": BASELINE["present_frames"] <= summary["prediction_present_frames"] <= BASELINE["present_frames"] + round(0.02 * total),
        "overall smoothness p95": all_smoothness["p95"] <= SMOOTHNESS_LIMITS["all_p95"],
        "overall smoothness outliers": all_smoothness["over_20_rate"] <= SMOOTHNESS_LIMITS["all_over_20_rate"],
        "soft smoothness p95": soft_smoothness["p95"] <= SMOOTHNESS_LIMITS["soft_p95"],
        "soft smoothness outliers": soft_smoothness["over_20_rate"] <= SMOOTHNESS_LIMITS["soft_over_20_rate"],
    }
    _require(all(gates.values()), "regressed gates: " + ", ".join(name for name, passed in gates.items() if not passed))
    summary["smoothness"] = {"all": all_smoothness, "soft": soft_smoothness}
    return summary


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Check clean tracker against the frozen baseline")
    parser.add_argument("--predictions", default=str(root / "output" / "tracking.json"))
    parser.add_argument("--annotations", default=str(root / "sample" / "pomona_annotations.json"))
    parser.add_argument("--input", default=str(root / "sample" / "pomona.mp4"))
    parser.add_argument("--existing", action="store_true", help="Check an existing JSON instead of running the tracker first")
    args = parser.parse_args()
    prediction_path = Path(args.predictions).resolve()
    annotation_path = Path(args.annotations).resolve()
    video_path = Path(args.input).resolve()
    if not args.existing:
        subprocess.run(
            [
                sys.executable,
                str(root / "clean_tracker.py"),
                "--input", str(video_path),
                "--tracking-json", str(prediction_path),
                "--no-video",
            ],
            cwd=root,
            check=True,
        )
    summary = check(prediction_path, annotation_path, video_path)
    smoothness = summary["smoothness"]
    print(
        "parity passed: "
        f"recall={summary['recall']:.3f}, mean={summary['mean_error_px']:.3f}px, "
        f"p90={summary['p90_error_px']:.3f}px, absent_fp={summary['false_positive_absent_frames']}, "
        f"jumps={summary['large_jump_count']}, accel_p95={smoothness['all']['p95']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
