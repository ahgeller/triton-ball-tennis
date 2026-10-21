"""The comparison sheet's pieces in refiner_sheet: which frames it shows, each tile and the board."""
import numpy as np

from finetune.refiner.refiner_sheet import (SHEET_COLUMNS, SHEET_HEADER, SHEET_PATCH, SHEET_STRIP, SHEET_ZOOM,
                                           frames_furthest_apart, sheet_board, sheet_tile, sheet_tiles)
from tests._helpers import write_clip

SIZE = (160, 90)
TILE = SHEET_PATCH * SHEET_ZOOM


def test_frames_furthest_apart_keeps_labelled_frames_both_versions_placed_in_frame_order():
    seeds = {0: (10.0, 10.0), 1: (20.0, 20.0), 2: None, 3: (30.0, 30.0)}
    a = {0: (10.0, 10.0), 1: (20.0, 20.0), 2: (5.0, 5.0), 3: (30.0, 30.0)}
    b = {0: (10.5, 10.0), 1: (26.0, 20.0), 2: (9.0, 5.0)}
    assert frames_furthest_apart([a, b], seeds, 1.0, 10) == [0, 1]     # 2 has no label, 3 no second mark
    assert frames_furthest_apart([a, b], seeds, 1.0, 1) == [1]          # the one that differs most


def test_sheet_tile_is_zoomed_captioned_and_crossed():
    image = np.full((SIZE[1], SIZE[0], 3), 40, np.uint8)
    tile = sheet_tile(image, 7, [(80.0, 45.0), (81.0, 45.0), (79.0, 45.0)], SIZE)
    assert tile.shape == (TILE + SHEET_STRIP, TILE, 3)
    assert tile[:SHEET_STRIP].any()                                      # the caption
    assert (tile[SHEET_STRIP:] != 40).any()                              # the crosses on the grey patch


def test_sheet_tiles_reads_one_pass_and_the_board_lays_them_out(tmp_path):
    video = write_clip(tmp_path / "clip.avi", 6, SIZE)
    marks = {1: [(80.0, 45.0)] * 3, 4: [(80.0, 45.0)] * 3, 9: [(80.0, 45.0)] * 3}
    tiles = sheet_tiles(video, marks, SIZE)
    assert len(tiles) == 2                                               # frame 9 is past the end
    assert int(tiles[0][SHEET_STRIP + TILE // 2, 3, 0]) == 1 and int(tiles[1][SHEET_STRIP + TILE // 2, 3, 0]) == 4
    board = sheet_board(tiles, "clip", ["a", "b"])
    columns = min(SHEET_COLUMNS, len(tiles))
    assert board.shape == (SHEET_HEADER + TILE + SHEET_STRIP + 4, columns * (TILE + 4), 3)
    assert board[:SHEET_HEADER].any()
