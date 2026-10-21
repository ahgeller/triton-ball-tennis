"""Verify saved run completeness and the user's color-off setting."""
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "output/verified_fixes_color_off_20260907"
MEDIA = ROOT / "output/verified_video13_color_off_court.mp4"
RENDER_CHECK = ROOT / "audit/records/render_check.jpg"
RESULT = ROOT / "audit/records/fix_run_verification.json"
# The reviewed standard set: video13 through video53.
EXPECTED = {f"video{i}_gridtracknet.json" for i in range(13, 54)}
# The rendered video13 check: frame size, length and the frame saved for inspection.
RENDER_SHAPE = (720, 1280)
RENDER_FRAMES = 361
RENDER_SAMPLE_FRAME = 180


def require(condition: bool, message: str) -> None:
    """Stop with ``message`` when a check fails (unlike assert, this survives ``python -O``)."""
    if not condition:
        raise SystemExit(f"verification failed: {message}")


def main() -> None:
    require({p.name for p in FOLDER.glob("*.json")} == EXPECTED,
            f"{FOLDER} must hold exactly the video13-video53 exports")
    total = 0
    for name in sorted(EXPECTED):
        data = json.loads((FOLDER / name).read_text())
        require(data["config"]["motion_raw_ball_color_gate"] is False, f"{name}: the color gate was on")
        count = data["video"]["total_frames"]
        frames = data["frames"]
        require(len(frames) == count, f"{name}: {len(frames)} frames exported, the video has {count}")
        require([row["frame"] for row in frames] == list(range(count)), f"{name}: frames are not sequential from 0")
        total += count
    capture = cv2.VideoCapture(str(MEDIA))
    require(capture.isOpened(), f"could not open {MEDIA}")
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        require(frame.shape[:2] == RENDER_SHAPE, f"frame {decoded} is {frame.shape[:2]}, expected {RENDER_SHAPE}")
        if decoded == RENDER_SAMPLE_FRAME:
            cv2.imwrite(str(RENDER_CHECK), frame)
        decoded += 1
    capture.release()
    require(decoded == RENDER_FRAMES, f"decoded {decoded} rendered frames, expected {RENDER_FRAMES}")
    result = {"verified_clips": len(EXPECTED), "exported_source_frames": total,
              "all_color_gates_disabled": True, "rendered_frames_decoded": decoded}
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
