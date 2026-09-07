"""Verify saved run completeness and the user's color-off setting."""
import json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "output/verified_fixes_color_off_20260907"
expected = {f"video{i}_gridtracknet.json" for i in range(13, 54)}
assert {p.name for p in folder.glob("*.json")} == expected
total = 0
for name in sorted(expected):
    data = json.loads((folder / name).read_text())
    assert data["config"]["motion_raw_ball_color_gate"] is False, name
    count = data["video"]["total_frames"]
    frames = data["frames"]
    assert len(frames) == count, name
    assert [row["frame"] for row in frames] == list(range(count)), name
    total += count
media = ROOT / "output/verified_video13_color_off_court.mp4"
capture = cv2.VideoCapture(str(media))
assert capture.isOpened()
decoded = 0
while True:
    ok, frame = capture.read()
    if not ok:
        break
    assert frame.shape[:2] == (720, 1280)
    if decoded == 180:
        cv2.imwrite(str(ROOT / "audit/render_check.jpg"), frame)
    decoded += 1
capture.release()
assert decoded == 361, decoded
result = {"verified_clips": len(expected), "exported_source_frames": total,
          "all_color_gates_disabled": True, "rendered_frames_decoded": decoded}
(ROOT / "audit/fix_run_verification.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
