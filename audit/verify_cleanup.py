"""Compare the cleanup with the saved color-off baseline, using reviewed clips only."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
before = ROOT / "output/verified_fixes_color_off_20260907"
after = ROOT / "output/verified_cleanup_20260907"
expected = {f"video{i}_gridtracknet.json" for i in range(13, 54)}
assert {p.name for p in after.glob("*.json")} == expected
changes = []
presence_changes = []
distances = []
count = 0
for name in sorted(expected):
    old = json.loads((before / name).read_text())
    new = json.loads((after / name).read_text())
    assert len(new["frames"]) == new["video"]["total_frames"]
    assert [row["frame"] for row in new["frames"]] == list(range(len(new["frames"])))
    assert "court_depth" not in new["config"]
    assert "motion_raw_ball_color_gate" not in new["config"]
    assert len(old["frames"]) == len(new["frames"])
    for a, b in zip(old["frames"], new["frames"]):
        if a != b:
            changes.append({"clip": name, "frame": b["frame"],
                            "changed_fields": sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))})
        if a.get("present") != b.get("present"):
            presence_changes.append({"clip": name, "frame": b["frame"]})
        if a.get("present") and b.get("present"):
            distances.append(math.hypot(a["x"]-b["x"], a["y"]-b["y"]))
    count += len(new["frames"])
result = {"clips": len(expected), "frames": count, "changed_frame_records": len(changes),
          "presence_changes": presence_changes, "position_changes_above_1px": sum(d > 1 for d in distances),
          "maximum_position_change_px": max(distances, default=0), "first_20_changes": changes[:20]}
(ROOT / "audit/cleanup_verification.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "first_20_changes"}))
