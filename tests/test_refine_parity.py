"""refine.suggest gives the same answer as the recorded one for every way the label tool calls it.

The fixture (tests/fixtures/refine_suggest_parity.json) was recorded on synthetic clips before
suggest() was split into stages, so an edit that changes any field of any suggestion fails here.
Re-record it only for a deliberate change of behaviour:

    python tests/test_refine_parity.py --record
"""
import json
from pathlib import Path
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._helpers import ball_frames, write_frames   # noqa: E402

FIXTURE = Path(__file__).with_name("fixtures") / "refine_suggest_parity.json"
PATH = [(100 + 12 * i, 200 - 18 * i + i * i) for i in range(12)]          # an exact parabola
LONG_PATH = [(60 + 3 * i, 220 - 5 * i + i * i // 8) for i in range(48)]   # a 60 FPS-style clip, labelled every 2nd frame


class Nudge:
    """Stands in for the network: 0.5 px right, frame 5 far off; `sure` as a version without the
    self-check (None), one that doubts every answer (False) or vouches for all (True)."""

    def __init__(self, sure=None):
        from finetune.labeling import cnn_refiner
        self.cut_crop = cnn_refiner.cut_crop
        self.sure = sure

    def place(self, items, height):
        return [(start[0] + (8.0 if abs(start[0] - 160.5) < 2 else 0.5), start[1], self.sure)
                for _, _, start in items]


def seeds_for(path, offset=(0.9, 0.2), step=1):
    return {step * i: (column + offset[0], row + offset[1]) for i, (column, row) in enumerate(path)}


def variants():
    """name -> (frames, the seeds and PassOptions fields for refine.suggest)."""
    plain = seeds_for(PATH)
    off = {**plain, 7: (plain[7][0] + 10.0, plain[7][1])}                 # a label on the wrong thing
    noball = {f: (None if f == 3 else p) for f, p in plain.items() if f != 9}
    return {
        "plain": (PATH, dict(seeds=plain, spacing=1)),
        "confirmed": (PATH, dict(seeds=off, spacing=1, confirmed=[f for f in off if f != 7])),
        "unconfirmed_offlabel": (PATH, dict(seeds=off, spacing=1)),
        "only": (PATH, dict(seeds=plain, spacing=1, only={4, 5, 6, 7})),
        "noball": (PATH, dict(seeds=noball, spacing=1)),
        "placer": (PATH, dict(seeds=plain, spacing=1, placer=Nudge())),
        "placer_sure": (PATH, dict(seeds=plain, spacing=1, placer=Nudge(True))),
        "placer_unsure": (PATH, dict(seeds=plain, spacing=1, placer=Nudge(False))),
        "spacing2": (LONG_PATH, dict(seeds=seeds_for(LONG_PATH[::2], step=2), spacing=None)),
        "stopped": (PATH, dict(seeds=plain, spacing=1, stop=lambda: True)),
    }


def run_variant(name: str) -> dict:
    """One variant's suggestions, in the fixture's shape (frames as text, tuples as lists)."""
    from finetune.labeling import refine
    path, kwargs = variants()[name]
    with tempfile.TemporaryDirectory() as directory:
        video = write_frames(Path(directory) / "clip.avi", ball_frames(path))
        found = refine.suggest(video, kwargs["seeds"], refine.PassOptions(**{k: v for k, v in kwargs.items() if k != "seeds"}))
    return json.loads(json.dumps({str(frame): dict(hint) for frame, hint in sorted(found.items())}))


def run_all() -> dict:
    return {name: run_variant(name) for name in variants()}


def assert_close(expected, actual, where):
    if isinstance(expected, float) or isinstance(actual, float):
        assert actual == pytest.approx(expected, abs=2e-3), where
    elif isinstance(expected, dict):
        assert sorted(expected) == sorted(actual), where
        for key in expected:
            assert_close(expected[key], actual[key], f"{where}.{key}")
    elif isinstance(expected, list):
        assert len(expected) == len(actual), where
        for i, (a, b) in enumerate(zip(expected, actual)):
            assert_close(a, b, f"{where}[{i}]")
    else:
        assert expected == actual, where


def recorded() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_the_recording_covers_every_variant():
    assert sorted(recorded()) == sorted(variants())


@pytest.mark.parametrize("name", sorted(variants()))
def test_every_variant_matches_the_recording(name):
    assert_close(recorded()[name], run_variant(name), name)


if __name__ == "__main__":
    if "--record" in sys.argv:
        FIXTURE.write_text(json.dumps(run_all(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {FIXTURE}")
    else:
        raise SystemExit(pytest.main([__file__]))
