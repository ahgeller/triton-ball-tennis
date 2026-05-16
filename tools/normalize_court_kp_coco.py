"""Re-write a COCO court-keypoint annotation file so every annotation uses a
single, consistent keypoint numbering.

The canonical convention matches the user's reference diagram (top-down view
of the court, FAR baseline at the top of the image):

    0   4 . . . . . 6   1     row 0 - FAR baseline (4 pts, doubles-L,
                                       singles-L, singles-R, doubles-R)
    |   |           |   |
    |   8 . 12 . . 9    |     row 1 - FAR service line (3 pts, singles-L,
    |   |           |   |             center-T, singles-R)
    |   10. 13 . 11     |     row 2 - NEAR service line (3 pts)
    |   |           |   |
    2   5 . . . . . 7   3     row 3 - NEAR baseline (4 pts)

Per-annotation procedure:
  - drop keypoints whose visibility is 0 (treat as unlabeled - output v=0,
    coords (0, 0)).
  - sort the remaining keypoints by image-y ascending.
  - bucket into 4 rows by row-size pattern: 4, 3, 3, 4 (top to bottom).
    If the visible-keypoint pattern doesn't match this, we still try a
    flexible assignment using a small per-row K-means-ish split, and mark
    the annotation as "uncertain" in the audit JSON.
  - within each row, sort by image-x ascending.
  - emit the 14 keypoints in the canonical order described above.

Nothing about the positions changes - we only re-shuffle the indices so the
"bottom-right doubles corner" carries index 3 in every annotation, etc.

Usage:
    python tools/normalize_court_kp_coco.py \
        --in   <path-to>/_annotations.coco.json \
        --out  <path-to>/_annotations.normalized.coco.json \
        [--audit <audit.json>]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

NUM_KP = 14
ROW_SIZES = (4, 3, 3, 4)  # FAR baseline, FAR service, NEAR service, NEAR baseline

# Canonical column index per row -> output keypoint index.
# rows are [FAR baseline, FAR service, NEAR service, NEAR baseline]
# columns within a row are sorted left -> right (image-x ascending).
CANONICAL_MAP: Dict[Tuple[int, int], int] = {
    # FAR baseline: doubles-L, singles-L, singles-R, doubles-R
    (0, 0): 0, (0, 1): 4, (0, 2): 6, (0, 3): 1,
    # FAR service: singles-L, center-T, singles-R
    (1, 0): 8, (1, 1): 12, (1, 2): 9,
    # NEAR service: singles-L, center-T, singles-R
    (2, 0): 10, (2, 1): 13, (2, 2): 11,
    # NEAR baseline: doubles-L, singles-L, singles-R, doubles-R
    (3, 0): 2, (3, 1): 5, (3, 2): 7, (3, 3): 3,
}

KP_NAMES = [str(i) for i in range(NUM_KP)]

# COCO keypoint arrays are 0-based triples, but COCO skeleton edge pairs are
# 1-based keypoint numbers. Roboflow's COCO Keypoint example follows this.
KP_SKELETON = [
    # doubles sidelines
    [1, 3], [2, 4],
    # far baseline: doubles-L -> singles-L -> singles-R -> doubles-R
    [1, 5], [5, 7], [7, 2],
    # near baseline: doubles-L -> singles-L -> singles-R -> doubles-R
    [3, 6], [6, 8], [8, 4],
    # left singles/service column: top -> far service -> near service -> bottom
    [5, 9], [9, 11], [11, 6],
    # right singles/service column: top -> far service -> near service -> bottom
    [7, 10], [10, 12], [12, 8],
    # service-line horizontals
    [9, 13], [13, 10],
    [11, 14], [14, 12],
    # center service line
    [13, 14],
]


def assign_rows(visible: List[Tuple[int, float, float]]) -> Optional[List[List[Tuple[int, float, float]]]]:
    """Sort visible kps by Y, then bucket into the 4 rows (4,3,3,4).

    Returns 4 lists of (orig_idx, x, y) or None if the visible count != 14.
    """
    if len(visible) != NUM_KP:
        return None
    by_y = sorted(visible, key=lambda r: r[2])
    rows: List[List[Tuple[int, float, float]]] = []
    start = 0
    for size in ROW_SIZES:
        rows.append(by_y[start:start + size])
        start += size
    return rows


def assign_rows_partial(visible: List[Tuple[int, float, float]]) -> Optional[List[List[Tuple[int, float, float]]]]:
    """Fallback: when fewer than 14 visible, K-means the Ys into 4 clusters
    and trust that rows are in the right order. Returns 4 lists in Y order.
    """
    if len(visible) < 4:
        return None
    ys = sorted(y for _, _, y in visible)
    k = 4
    # init centers at evenly spaced quantiles
    centers = [ys[int((i + 0.5) * len(ys) / k)] for i in range(k)]
    for _ in range(20):
        groups: List[List[float]] = [[] for _ in range(k)]
        for y in ys:
            j = min(range(k), key=lambda i: abs(y - centers[i]))
            groups[j].append(y)
        new = [(sum(g) / len(g)) if g else centers[i] for i, g in enumerate(groups)]
        if all(abs(a - b) < 0.5 for a, b in zip(centers, new)):
            break
        centers = new
    order = sorted(range(k), key=lambda i: centers[i])
    rows: List[List[Tuple[int, float, float]]] = [[] for _ in range(k)]
    for idx, x, y in visible:
        cluster = min(range(k), key=lambda i: abs(y - centers[i]))
        rows[order.index(cluster)].append((idx, x, y))
    return rows


def remap_one(
    kps_flat: Sequence[float],
) -> Tuple[List[float], str]:
    """Re-shuffle a single keypoint list into canonical order.

    Returns (new_flat_list, status). status in:
       'ok'                - 14 visible, clean 4/3/3/4 row split
       'partial'           - fewer than 14 visible, used K-means fallback
       'unmapped'          - couldn't assign every canonical slot; the
                             remaining slots stay (0, 0, 0). Some kps may
                             have been dropped if a row had too many.
       'malformed'         - input wasn't exactly 14 keypoints (left alone)
    """
    if len(kps_flat) != NUM_KP * 3:
        return list(kps_flat), "malformed"

    src_pts: List[Tuple[float, float, int]] = [
        (float(kps_flat[3 * i]), float(kps_flat[3 * i + 1]), int(kps_flat[3 * i + 2]))
        for i in range(NUM_KP)
    ]
    visible = [(i, x, y) for i, (x, y, v) in enumerate(src_pts) if v > 0]

    status = "ok"
    rows = assign_rows(visible)
    if rows is None:
        rows = assign_rows_partial(visible)
        status = "partial"
    if rows is None:
        return list(kps_flat), "unmapped"

    # Initialise output as all-zero.
    new_pts: List[Tuple[float, float, int]] = [(0.0, 0.0, 0)] * NUM_KP

    for row_idx, row in enumerate(rows):
        if not row:
            continue
        row_sorted = sorted(row, key=lambda r: r[1])  # by X
        expected_size = ROW_SIZES[row_idx]
        # If a row has more visible kps than canonical slots, the extras get
        # mapped to the nearest canonical slot by X-quantile.
        for col_idx, (src_i, x, y) in enumerate(row_sorted):
            if col_idx >= expected_size:
                # extra kp this row; nothing canonical to map to
                if status == "ok":
                    status = "partial"
                continue
            canon_i = CANONICAL_MAP.get((row_idx, col_idx))
            if canon_i is None:
                continue
            src_v = src_pts[src_i][2]
            new_pts[canon_i] = (x, y, src_v)
        if len(row_sorted) < expected_size and status == "ok":
            status = "partial"

    new_flat: List[float] = []
    for x, y, v in new_pts:
        new_flat.extend([float(x), float(y), int(v)])
    if all(v == 0 for *_x, v in new_pts):
        status = "unmapped"
    return new_flat, status


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", required=True,
                   help="Input COCO annotations JSON.")
    p.add_argument("--out", required=True,
                   help="Output COCO JSON with re-numbered keypoints.")
    p.add_argument("--audit", default=None,
                   help="Optional audit JSON listing per-image status.")
    p.add_argument("--drop-malformed", action="store_true",
                   help="Drop annotations whose keypoint list is not 14 x 3.")
    p.add_argument("--drop-empty-images", action="store_true",
                   help="After dropping annotations, also drop images with no annotations.")
    args = p.parse_args()

    inp_path = Path(args.inp)
    out_path = Path(args.out)
    with inp_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images = {im["id"]: im for im in data.get("images", [])}
    anns = data.get("annotations", [])
    print(f"loaded {len(images)} images, {len(anns)} annotations from {inp_path.name}")

    status_counts: Counter[str] = Counter()
    audit_rows: List[Dict[str, object]] = []
    new_anns: List[Dict[str, object]] = []

    for ann in anns:
        kps = ann.get("keypoints")
        if not isinstance(kps, list):
            status_counts["no_keypoints"] += 1
            new_anns.append(ann)
            continue
        new_kps, status = remap_one(kps)
        if args.drop_malformed and status == "malformed":
            status_counts[status] += 1
            status_counts["dropped"] += 1
            audit_rows.append({
                "annotation_id": ann.get("id"),
                "image_id": ann.get("image_id"),
                "file_name": images.get(ann.get("image_id"), {}).get("file_name"),
                "status": status,
                "dropped": True,
            })
            continue
        ann["keypoints"] = new_kps
        ann["num_keypoints"] = sum(
            1
            for i in range(len(new_kps) // 3)
            if new_kps[3 * i + 2] > 0
        )
        status_counts[status] += 1
        if status in ("partial", "unmapped", "malformed"):
            audit_rows.append({
                "annotation_id": ann.get("id"),
                "image_id": ann.get("image_id"),
                "file_name": images.get(ann.get("image_id"), {}).get("file_name"),
                "status": status,
            })
        new_anns.append(ann)
    data["annotations"] = new_anns
    if args.drop_empty_images:
        image_ids_with_anns = {
            ann.get("image_id") for ann in new_anns if ann.get("image_id") is not None
        }
        old_image_count = len(data.get("images", []))
        data["images"] = [
            im for im in data.get("images", [])
            if im.get("id") in image_ids_with_anns
        ]
        dropped_images = old_image_count - len(data["images"])
        if dropped_images:
            status_counts["dropped_images"] += dropped_images

    # Update categories so every category that has a keypoint count gets the
    # canonical keypoint names + skeleton.
    cats = data.get("categories", [])
    for c in cats:
        c["keypoints"] = KP_NAMES
        c["skeleton"] = KP_SKELETON

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"wrote {out_path}")
    print()
    print("status summary:")
    for k in (
        "ok", "partial", "unmapped", "malformed", "no_keypoints",
        "dropped", "dropped_images",
    ):
        if status_counts[k]:
            print(f"  {k:13s} {status_counts[k]}")
    if audit_rows and args.audit:
        audit_path = Path(args.audit)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8") as f:
            json.dump({"counts": dict(status_counts), "rows": audit_rows}, f, indent=2)
        print(f"audit -> {audit_path}")
    elif audit_rows:
        print(f"({len(audit_rows)} annotations are not 'ok'; pass --audit to write a per-row list)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
