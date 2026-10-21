"""Pass-1 stages that run on collected evidence: the batched ghost-ROI erase."""
import numpy as np
from ball_in_play_selector import PackedMask

from tennis_tracker.pass1 import Pass1Result, erase_ghost_rois


def _mask(h=30, w=40, boxes=()):
    mask = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 255
    return mask


def _result(raw_boxes_by_frame, ghost_rois_by_frame):
    result = Pass1Result()
    for boxes, ghosts in zip(raw_boxes_by_frame, ghost_rois_by_frame):
        result.detections.append([])
        result.raw_motions.append(PackedMask.pack(_mask(boxes=boxes), roi=(0, 0, 40, 30)))
        result.boost_masks.append(PackedMask.pack(_mask(boxes=boxes)))
        result.rois.append(None)
        result.ghost_rois.append(list(ghosts))
        result.player_boxes.append({})
        result.court_keypoints.append(None)
    return result


def test_erase_clears_motion_inside_ghost_rois_and_repacks(capsys):
    ball = (5, 5, 9, 9)
    ghost_blob = (20, 20, 26, 26)
    result = _result(
        raw_boxes_by_frame=[[ball, ghost_blob], [ball], [ball, ghost_blob]],
        ghost_rois_by_frame=[[(18, 18, 30, 30)], [], [(18, 18, 30, 30)]],
    )
    before_frame1 = result.raw_motions[1]
    assert erase_ghost_rois(result) == 2
    for fi in (0, 2):
        raw = result.raw_motions[fi].unpack()
        assert raw[22, 22] == 0 and raw[6, 6] == 255           # ghost blob gone, ball kept
        assert result.raw_motions[fi].roi == (0, 0, 40, 30)    # packed rectangle preserved
        assert result.boost_masks[fi].unpack()[22, 22] == 0
    assert result.raw_motions[1] is before_frame1              # untouched frame keeps its object
    assert "cleaned 2 frames" in capsys.readouterr().out


def test_erase_drops_a_mask_that_becomes_empty_and_skips_frames_without_change():
    result = _result(
        raw_boxes_by_frame=[[(20, 20, 26, 26)], [(5, 5, 9, 9)]],
        ghost_rois_by_frame=[[(18, 18, 30, 30)], [(18, 18, 30, 30)]],
    )
    assert erase_ghost_rois(result) == 2          # both frames had a mask to inspect
    assert result.raw_motions[0] is None and result.boost_masks[0] is None
    assert result.raw_motions[1].unpack()[6, 6] == 255


def test_erase_ignores_frames_with_no_masks():
    result = Pass1Result()
    result.detections.append([])
    result.raw_motions.append(None)
    result.boost_masks.append(None)
    result.rois.append(None)
    result.ghost_rois.append([(0, 0, 10, 10)])
    assert erase_ghost_rois(result) == 0
