# Pass 1 timing on mains (item 4e of the October 2026 review), 2026-10-21

`clean_tracker.py --no-video --court --info` on `sample/VolyAvw-cut-merged-1771797159395.mp4`
(1920x1080, 59.5 FPS, 3286 frames) on the RTX 5080 Laptop GPU, mains power, Windows power plan
Balanced (clocks sat at 950 to 1900 MHz of 3090; the GPU was not power capped). Logs and the
5-second GPU samples: `output/VolyAvw_mains_info/`. The same run on battery earlier that day,
power capped at 25 to 36 W: `output/VolyAvw/info/`.

| stage | mains | battery |
|---|---|---|
| pass 1 total | 23.6 ms/frame (38.8 fps) | 34.1 ms/frame (26.4 fps) |
| whole run | 89.7 s | 133.8 s |
| pass 1 preprocess (motion, ROIs) | 16.5 ms/frame | |
| of which CUDA total | 4.9 ms/frame | 6.9 ms/frame |
| of which CUDA device-to-host (the mask download) | 0.6 ms/frame | |
| pass 1 store/debug (records, mask packing) | 7.9 ms/frame | |
| player engine launch (aux detect) | 4.5 ms/frame | 9.0 ms/frame |
| player engine wait and result download (aux wait) | 2.35 ms/frame | 2.76 ms/frame |
| decode and window slide | 1.4 ms/frame | |
| GridTrackNet prepass | overlapped, 61 source fps alone | |

What the review suspected (section 2.1): per-frame host syncs serialising pass 1. Measured, the
two syncs it named are small: the player engine's wait is 2.35 ms and the mask download 0.6 ms,
about 3 ms of a 23.6 ms frame. The two large buckets are the CPU side of preprocessing
(16.5 ms, of which the CUDA work is 4.9) and `pass1_store` (7.9 ms: packing the masks and
building the frame records). Those, not the syncs, are where pass 1 time goes; neither was
changed. The battery run shows the cap costs about a third of the throughput.
