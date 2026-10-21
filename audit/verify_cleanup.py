"""Compare the cleanup with the saved color-off baseline, using reviewed clips only."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "output/verified_fixes_color_off_20260907"
AFTER = ROOT / "output/verified_cleanup_20260907"
RESULT = ROOT / "audit/records/cleanup_verification.json"
# The reviewed standard set: video13 through video53.
EXPECTED = {f"video{i}_gridtracknet.json" for i in range(13, 54)}
REMOVED_CONFIG_FIELDS = ("court_depth", "motion_raw_ball_color_gate")
CHANGES_KEY = "first_20_changes"
CHANGES_SHOWN = 20


def require(condition: bool, message: str) -> None:
    """Stop with ``message`` when a check fails (unlike assert, this survives ``python -O``)."""
    if not condition:
        raise SystemExit(f"verification failed: {message}")


def main() -> None:
    require({p.name for p in AFTER.glob("*.json")} == EXPECTED,
            f"{AFTER} must hold exactly the video13-video53 exports")
    changes = []
    presence_changes = []
    distances = []
    count = 0
    for name in sorted(EXPECTED):
        old = json.loads((BEFORE / name).read_text())
        new = json.loads((AFTER / name).read_text())
        require(len(new["frames"]) == new["video"]["total_frames"], f"{name}: frame count differs from the video")
        require([row["frame"] for row in new["frames"]] == list(range(len(new["frames"]))),
                f"{name}: frames are not sequential from 0")
        for field in REMOVED_CONFIG_FIELDS:
            require(field not in new["config"], f"{name}: removed config field {field} is still exported")
        require(len(old["frames"]) == len(new["frames"]), f"{name}: frame count changed against the baseline")
        for a, b in zip(old["frames"], new["frames"]):
            if a != b:
                changes.append({"clip": name, "frame": b["frame"],
                                "changed_fields": sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))})
            if a.get("present") != b.get("present"):
                presence_changes.append({"clip": name, "frame": b["frame"]})
            if a.get("present") and b.get("present"):
                distances.append(math.hypot(a["x"] - b["x"], a["y"] - b["y"]))
        count += len(new["frames"])
    result = {"clips": len(EXPECTED), "frames": count, "changed_frame_records": len(changes),
              "presence_changes": presence_changes, "position_changes_above_1px": sum(d > 1 for d in distances),
              "maximum_position_change_px": max(distances, default=0), CHANGES_KEY: changes[:CHANGES_SHOWN]}
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != CHANGES_KEY}))


if __name__ == "__main__":
    main()
