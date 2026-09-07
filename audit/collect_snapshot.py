"""Read-only inventory for the September 2026 review; writes audit/snapshot.json."""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from finetune.train_gridtracknet import find_clips, resolve_val_clips
from tennis_tracker.gridtracknet import GridTrackNet


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    ws = ROOT / "finetune"
    clips = find_clips(ws / "videos", ws / "labels")
    validation = resolve_val_clips([c[0] for c in clips], None)
    sources = {}
    for name, video, labels, source in clips:
        with labels.open(newline="", encoding="utf-8-sig") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        bucket = sources.setdefault(source, {"clips": 0, "label_rows": 0})
        bucket["clips"] += 1
        bucket["label_rows"] += count
    versions = {}
    for package in ("torch", "numpy", "opencv-python", "filterpy", "tensorrt", "boxmot", "catboost", "ultralytics"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    paths = [*ROOT.glob("*.py"), *ROOT.glob("tennis_tracker/*.py"),
             *ROOT.glob("ball_in_play_selector/*.py"), *ROOT.glob("finetune/*.py"),
             ws / "clips.csv", ws / "exclude.txt", *ws.glob("labels/*_ball.csv"),
             ROOT / "models/gridtracknet_weights_torch.npz", ROOT / "models/player.engine",
             ROOT / "models/courtdetection.engine", ROOT / "sample/pomona.mp4",
             ROOT / "sample/pomona_annotations.json"]
    model = GridTrackNet()
    snapshot = {"python": sys.version, "interpreter": sys.executable, "versions": versions,
                "gpu": torch.cuda.get_device_name(0), "sources": sources,
                "default_validation": validation,
                "unchecked_clips_in_default_training": [name for name, _, _, source in clips
                    if source == "custom-uncorrected" and name not in validation],
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "normalization_parameters": sum(p.numel() for p in model.batch_norms.parameters()),
                "sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths}}
    (ROOT / "audit/snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in snapshot.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
