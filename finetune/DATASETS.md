# Datasets for training the ball detector (surveyed 2026-08-30/31)

Target: a small temporal detector (GridTrackNet family, 5 frames at 30 FPS
spacing) for a **fixed elevated camera behind the baseline, 1080p60**, where the
far-court ball is 4–6 px. Every link below was checked by fetching the page,
file listing or API (GitHub via `gh api`, Drive via folder view, HF via its
API); items marked *(snippet only)* could not be fetched by a script (Kaggle
pages / Roboflow / ScienceDirect block bots) and need a browser.

## 0. Already on this PC — imported by `ft.py import all`

| source tag | clips | label frames | camera | what it really is |
|---|---:|---:|---|---|
| `custom` | 11 | 4,106 | **ours**, fixed, 1080p60/30 | archive video1–10, 12 — the ground truth that matters. video1–10 match the Kaggle **`tenis-backview`** set (same names, `videoN_ball.csv` / `frame,ball_x,ball_y`, 60 FPS, 86 MB, CC0, June 2025) frame for frame in layout; the archive labels are the click-corrected version. That set also ships `videoN_court.csv` (4 court corners per frame) and `videoN_player.csv` — worth fetching for the court-mask channel |
| `custom-uncorrected` | 1 | 300 | ours, 768×432 | `video11`, labels never click-corrected; hold-out only |
| `prof` | 41 | 9,292 | broadcast, elevated behind baseline, 1280×720 @ 30 | the public **TrackNet tennis** set (§1 #3), copied to `Desktop\prof\Dataset` — games 2, 4, 5, 6, 7, 10 (41 of the 95 clips). Imported as `video13`–`video53` so it lists under the own-camera clips; `game10` is the published **test** split, so hold `video47`–`video53` out of training before comparing with published numbers |
| `grid` | 32 | 17,479 | amateur phone + TV, 1080p | a partial copy of GridTrackNet's public set (the full set is 100 matches / 48 GB and **is downloadable** — see §1). Several matches carried labels that did not line up with the video. All are now resolved: most needed a shift, match49/match50 needed the *right* video (they are rallies at frames 681 and 1405 of match47's), and match26/match55 needed re-timing in segments because our copies drop frames. `exclude.txt` is empty |
| `tracknetv2-badminton` | 201 | 91,214 | broadcast, 1280×720 @ 30 | the **TrackNetV2 badminton** set (V5Test's README called it tennis; the frames show Li-Ning/Yonex badminton courts). Shuttlecock, not ball — cross-sport pre-training at low weight |

The 40 GB `V5Test/tracknetv5_colab_data.tar` is the same three sets converted to
TrackNetV5's PNG layout and is not needed.

## 1. What to fetch, in order

| # | dataset | why | size / access | import |
|---|---|---|---|---|
| 1 | **GridTrackNet Match-Data, full set** | 100 rallies / 54.7k frames (45,923 train + 8,811 test), **74% amateur** court-side phone/YouTube clips + 26 broadcasts; the domain the bundled `.npz` was trained on; we hold only 32 matches | 48.2 GB on Google Drive folder **`1LbmBbLxkmhD3P6UpPAck3NDBbMht2fg7`** (alive — the README's *visible* URL `1gJUn2d6…` is a dead link, the href behind it works). Fetch with `rclone` or `gdown --folder` per `matchN` subfolder (Drive folder downloads cap at 50 files); no data licence stated ("compilation from various sources") | `ft.py import grid --src <Match-Data>` then `ft.py check --audit --fix` (expect a few misaligned matches) |
| 2 | **Kaggle `gastonarielfrancois/tenis-backview`** | the origin of our archive clips, plus **court corners and player points per frame** we do not have; CC0 | 86 MB zip, Kaggle login (Croissant metadata verified: video1–10 × ball/court/player CSVs). Card says "1080x1920, 60fps" | ball CSVs duplicate the archive — do **not** re-import; keep `videoN_court.csv` (column order differs per file — map by name) for the court mask |
| 3 | **TrackNet tennis (NYCU, 10 games)** — *6 games are already here*, see §0 `prof`; the other four (game1, 3, 8, 9) are still worth fetching | dense per-frame labels, elevated behind-baseline broadcast geometry, 720p so the far ball is ~3–6 px; every temporal model (WASB, TOTNet, TrackNetV4) pre-trains on it | 2.39 GB `Dataset.zip`; Google Drive mirror https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut (yastrebksv/TrackNet), official NYCU SharePoint link in WASB's GET_STARTED (browser only), Kaggle mirror `sofuskonglevoll/tracknet-tennis`. No licence stated | the copy we hold imports with `ft.py import prof`; a fuller download imports with `ft.py import tracknet --src <Dataset> --prefix tn --fps 30` (`game1/Clip1/Label.csv` + jpgs is what the importer reads) |
| 4 | **RacketVision** (AAAI 2026, MIT) | the only large modern **1080p** tennis set: 431 clips, 150k frames, 21.5k manual ball labels + `interp_ball/` interpolated trajectories, a median background per clip | 7.5 GB for all three sports; `hf download linfeng302/RacketVision --repo-type dataset` | `tennis/all/matchN/csv/000_ball.csv` (`Frame,Visibility,X,Y`, **sparse**: 1 frame in 5) + `tennis/videos/matchN_000.mp4`; a 20-line rename into `csv/` + `video/` then `ft.py import tracknet`. A 5-frame window needs all five frames, so train on the interpolated tracks and down-weight them |
| 5 | **TrackNetV4 "New Tennis Dataset"** | 87 train + 8 test videos, 30k+ **amateur** frames: night, multiple courts in one frame, multiple balls (primary flagged), doubles — the closest public thing to amateur court footage | request form (research DUA, OneDrive .docx) linked from https://tracknetv4.github.io/ | TrackNet `Label.csv` layout |
| 6 | **PadelTracker100** (Zenodo, CC BY 4.0) | the closest published **viewpoint**: single fixed camera 7.6 m high, 15.5 m behind the court, 1920×1080 @ 30; ~57k frames with a ball box | v1 https://zenodo.org/records/14653706 ships `padel-data-labels.zip` 7.1 GB; v2 (17020011) labels only, videos on YouTube (ids in the description) | COCO JSON (ball + `No-Ball` class) → CSV adapter; padel ball ≈ tennis ball, glass walls instead of fences |
| 7 | **BlurBall** (table tennis, Tübingen, MIT code) | 64k frames from **fixed static cameras**, tiny white ball, blur centre + orientation + length labels (the streak-centre convention the proposal adopts) | ≈1.9 GB https://cloud.cs.uni-tuebingen.de/index.php/s/C3pJEPKWQAkono7 (`matches/NN/{rallies_videos,frames,csv}`) | CSV `Frame,Visibility,X,Y,l,theta`; rename into `csv/` + `video/` and `ft.py import tracknet --prefix bb` |
| 8 | **One-Shot Badminton Shuttle Detection** (ETH RSL, 2026) | 20.5k frames from a **stationary 1920×1200 @ 60 FPS** camera, 11 backgrounds indoor/outdoor — small fast object at our frame rate | Google Drive folder linked from https://github.com/leggedrobotics/shuttle_detection (alive); licence unstated | bbox → centre adapter |
| 9 | **OpenTTGames** (TTNet, CC BY-NC-SA) | fixed side camera, Full-HD **120 FPS**; ~50k labelled samples around events | ≈33 GB, https://lab.osai.ai/datasets/openttgames/ (browser) or the HyperAI mirror | subsample to 30/60 FPS; `ball_markup.json` |
| 10 | **Dryad/Zenodo "high-quality sport ball"** (CC0 labels / CC BY videos) | 16 tennis videos, >10k objects, YOLO boxes | labels https://doi.org/10.5061/dryad.3bk3j9m13 (`tennis.zip` 2.4 MB), videos https://zenodo.org/records/19874311 (1.7 GB) | YOLO → centre adapter |
| 11 | **Kaggle `conradtakasi/tennis-ball`** *(snippet only)* | used by a "camera behind the playground" YOLOv8+SAHI tracker; Apache 2.0, 189 MB, JPEG + JSON/txt | Kaggle login; content unverified | YOLO adapter |
| 12 | **ISSIA-CNR soccer** | 6 **fixed** Full-HD cameras, far tiny ball, WASB re-labelled every frame | https://pspagnolo.jimdofree.com/download/ (non-commercial) | only as a "fixed camera, tiny distant ball" prior |
| 13 | Unlabelled fixed-camera tennis for pseudo-labelling | **Mendeley 75m8vz7jr2**: GoPro 1080p60, camera 1.21 m behind the baseline at 5.15 m — almost our geometry, 472 clips, landing points only (CC BY 4.0). **CalTennis** (HF `demalenk/caltennis`, 104 GB, CC BY-NC): 11M court-level 1080p60 frames, no labels | `ft.py add` → `pretrack.py` → click-correct the doubtful frames |

Court-mask channel: **yastrebksv/TennisCourtDetector** — 8,841 broadcast images at 1280×720 with 14 court keypoints (`data_train.json` / `data_val.json`, Drive `1lhAaeQCmk2y440PmagA0KmIVBIysVMwu`, HF mirror `Gholamreza/tennis_court_keypoints_dataset` 7.3 GB) is the only sizeable one; `tenis-backview`'s 4 corners per frame cover the behind-baseline view. The pipeline's `courtdetection.engine` already produces 14 keypoints, so this is only needed if that detector has to be retrained.

Not worth the time: Roboflow/Kaggle single-image bbox sets (viren-dhanwani 578 CC BY, tennistracking hard-court 4.2k, tennisball-3eqxr 3.9k, rowerup 3.2k — broadcast crops, no temporal context; only for hard-negative mining), Mendeley "Object Detection – Tennis Ball" (150 photos), TenniSet / OSL-loc-tennis (event timestamps only), Where-Is-The-Ball (data "coming soon"), IEEE DataPort multi-view tennis (subscription; small stroke clips), TrackNetV5 (proprietary), SwingVision/PlaySight/Wingfield (no public data), the padel/pickleball TrackNet forks (none ship data).

## 2. Full catalogue

### 2.1 Tennis, per-frame ball position

| dataset | camera | size | resolution / FPS | labels | licence / access | link |
|---|---|---|---|---|---|---|
| GridTrackNet Match-Data | amateur YouTube/phone + broadcasts | 100 rallies, 45,923 train + 8,811 test frames | 1280×720 frames from 1080p @ 30/60 | `matchN/Labels.csv`: `Frame,Visibility,X,Y` at 1280×720 + source mp4 + `frames/N.png` | MIT code; data unlicensed | https://github.com/VKorpelshoek/GridTrackNet (12★) → Drive `1LbmBbLxkmhD3P6UpPAck3NDBbMht2fg7` |
| Kaggle tenis-backview | **behind baseline, 60 fps** | 10 videos, 86 MB | "1080x1920" @ 60 | `videoN_ball.csv` (`frame,ball_x,ball_y`), `videoN_court.csv` (4 corners), `videoN_player.csv` (2 points) | CC0 | https://www.kaggle.com/datasets/gastonarielfrancois/tenis-backview |
| TrackNet (Huang 2019) | broadcast, elevated behind baseline with pans | 10 games, 95 clips, 19,835 frames (WASB split: game1–7 train / game8–10 test) | 1280×720 @ 30 | `Label.csv`: `file name, visibility(0 out,1 easy,2 hard,3 occluded), x, y, status(0 fly,1 hit,2 bounce)`; x,y = leading edge of the blur | none stated | GitLab https://gitlab.nol.cs.nycu.edu.tw/open-source/TrackNet ; mirrors above; HF `owen1233/tracknet_preprocessd_data` (2.57 GB, MIT) |
| RacketVision (2025/26) | pro broadcast | tennis 431 clips / 150,399 frames / 21,544 manual labels | 1920×1080 | `Frame,Visibility,X,Y` (sparse), racket boxes + keypoints, `interp_ball/`, COCO splits | MIT | https://huggingface.co/datasets/linfeng302/RacketVision · https://github.com/OrcustD/RacketVision (88★) |
| TrackNetV4 new set | amateur, multi-court, night | 87 + 8 videos, 30k+ frames | mixed | TrackNet layout, all balls + primary flag | research DUA, request form | https://tracknetv4.github.io/ · https://github.com/TrackNetV4/TrackNetV4 (93★) |
| Dryad/Zenodo sport-ball (Zou & Liu 2026) | online match video, people blurred | 16 tennis videos, >10k objects | mixed | YOLO `x y w h` | CC0 labels, CC BY videos | https://doi.org/10.5061/dryad.3bk3j9m13 |
| FMOv2 (Rozumnyi) | mixed YouTube | `atp_serves, tennis1, tennis2, tennis_serve_back, tennis_serve_side` | mixed | per-frame blur-streak GT | research | https://cmp.felk.cvut.cz/fmo/ (`FMOv2.zip` 3.7 GB) |
| IEEE DataPort multi-view (drone + court cam) | fixed court camera + drone, single strokes | `main.zip` 2.9 GB | unstated | `Tennis_Coordinates.xlsx` per frame | IEEE DataPort subscription | https://ieee-dataport.org/documents/multi-view-tennis-ball-dataset-trajectory-estimation-drone-and-court-cameras-annotated |
| YOLO-Net (PLOS ONE 2026) | broadcast frames | 6,648 images (val subset released) | 1280×1280 | YOLO boxes | Kaggle `taowang1123/yolo-net` | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335558 |
| TennisBallTracking/Tennis_Ball_Tracking | unknown | "20 GB" | — | in-repo GT CSVs only | Mega link dead | https://github.com/TennisBallTracking/Tennis_Ball_Tracking |

### 2.2 Other racket sports (small fast object, similar cadence)

| dataset | sport | camera | size | labels | access |
|---|---|---|---|---|---|
| PadelTracker100 | padel | fixed, 7.6 m high, 15.5 m behind court, 1080p30 | ~100k frames, 57k with ball box | COCO ball boxes (+ `No-Ball`), 17-kp players, shot CSV | CC BY 4.0 — https://zenodo.org/records/14653706 (v1, 7.1 GB) / 17020011 (v2, labels) |
| TrackNetV2 badminton | badminton | broadcast, 720p30 | 26 matches (23 pro + 3 amateur), 78k + 12.7k test frames | `Frame,Visibility,X,Y` | already imported as `tracknetv2-badminton`; https://hackmd.io/@TUIK/rJkRW54cU ; corrected test CSVs in https://github.com/qaz812345/TrackNetV3 (299★) |
| One-Shot Badminton (ETH) | badminton | stationary 1920×1200 @ 60 | 20,510 frames, 11 backgrounds | bbox | https://github.com/leggedrobotics/shuttle_detection |
| BlurBall | table tennis | fixed static, 26 recordings | 64,119 frames (51,423 train / 12,696 test) | `Frame,Visibility,X,Y,l,theta` (blur midpoint + endpoint sets) | https://cogsys-tuebingen.github.io/blurball/ (48★) |
| OpenTTGames | table tennis | fixed side, FHD 120 fps | 12 videos, ~50k samples, labels near events | JSON `ball_markup` | CC BY-NC-SA, https://lab.osai.ai/ (≈33 GB) |
| TTA (TOTNet) | table tennis (Paralympic) | fixed side, 1080p25 | 9–12k samples, occlusion-rich | `train.json` / `test.json` + videos | CC BY-NC, gated: signed agreement to august.xu@research.deakin.edu.au → https://huggingface.co/datasets/AugustRushG123/TTA_Tracking |
| TTHQ (Uplifting TT) | table tennis broadcast | 1080p | 9,092 frames | CSV | https://github.com/KieDani/UpliftingTableTennis |
| TT3D | table tennis | multi-view, 200 Hz GT | 130 trajectories | 2D/3D CSV | CC BY-SA, https://cogsys-tuebingen.github.io/tt3d/ |
| Pickleball / padel forks | — | — | no public per-frame set exists (AndrewDettor/TrackNet-Pickleball, SamuReyes/TrackNetV2-padel, michele98, Juild: none ship data) | | |

### 2.3 Other sports with per-frame ball labels (the WASB-SBDT benchmark)

WASB (Tarashima et al., BMVC 2023, https://github.com/nttcom/WASB-SBDT, MIT, 188★) evaluates on: soccer (ISSIA-CNR, 6 fixed FHD cameras @ 25, 12k + 6k frames, ball re-annotated), tennis (TrackNet), badminton (TrackNetV2), volleyball (Ibrahim 2016 + Perez 2022 ball boxes; 143k + 55k frames, broadcast), basketball (NBA/SAM; source page 404 — contact the SAM authors). `GET_STARTED.md` has the exact download steps; `MODEL_ZOO.md` has pretrained DeepBall / TrackNetV2 / MonoTrack / WASB weights per sport, useful initialisers. SoccerNet-Tracking (225k frames, 215k ball boxes, `pip install SoccerNet`) is the large broadcast alternative for soccer.

### 2.4 Court keypoints and players

| dataset | data | format | access |
|---|---|---|---|
| yastrebksv/TennisCourtDetector (272★) | 8,841 broadcast images, 1280×720, 14 keypoints (hard/clay/grass) | `data_{train,val}.json`: `[{"id", "kps": [[x,y]×14]}]` + `images/<id>.png`; order: 0–1 top baseline L/R, 2–3 bottom baseline L/R, 4–5 left singles top/bottom, 6–7 right singles top/bottom, 8–9 far service L/R, 10–11 near service L/R, 12–13 centre service far/near | Drive `1lhAaeQCmk2y440PmagA0KmIVBIysVMwu`; HF `Gholamreza/tennis_court_keypoints_dataset` |
| Kaggle tenis-backview | 4 court corners + 2 player points per frame, behind baseline | CSV (corner column order varies per file) | CC0 |
| PadelTracker100 | player bbox + 17 keypoints | COCO | CC BY 4.0 |
| TEXflip/tennis-court-detection (32★) | court line training/testing sets, various viewpoints, CVAT | CVAT XML | in-repo |
| Roboflow court sets *(snippet only)* | `pei-ling/tennis-court-keypoint-detection` (535), `tennis-court-segmentation-mynwl` (545), others | YOLO-pose / masks | Roboflow login |

### 2.5 Unlabelled fixed-camera tennis (pseudo-label with the current model)

- Mendeley "Tennis Shot Side-View and Top-View" https://data.mendeley.com/datasets/75m8vz7jr2 — GoPro 1080p60, top-view camera 1.21 m behind the baseline at 5.15 m height, 472 clips (clay + indoor), landing points only. CC BY 4.0.
- CalTennis https://huggingface.co/datasets/demalenk/caltennis — 104 GB, 11M frames, 2–6 synced phones at 1.65 m, 1080p60, faces blurred. CC BY-NC 4.0.
- GridTrackNet repo `Sample Video's/` — 6 unlabelled amateur/pro clips (already under `gridtracknet_finetuning/GridTrackNet_upstream`).

### 2.6 Popular tennis-analysis repos and what data they use (none add labels)

ArtLabss/tennis-tracking (708★, TrackNet weights + YOLOv3, broadcast only), abdullahtarek/tennis_analysis (884★, YOLOv5 on Roboflow viren-dhanwani + TennisCourtDetector), yastrebksv/TennisProject (228★, TrackNet + court + CatBoost bounce), yastrebksv/TrackNet (244★, hosts the TrackNet Drive mirror and `ctb_regr_bounce.cbm`), HarshTomar1234/Tennis-Vision (66★, best written spec of the TrackNet `Label.csv` format), MaximeBataille/tennis_tracking (117★), yo-WASSUP/Good-Tennis (175★, fixed-camera calibration, data = TrackNet), monotrack (56★, badminton 3D), CoachAI ShuttleSet (stroke-level only), THETIS (Kinect strokes).

## 3. Label formats the importer understands

| layout | example | command |
|---|---|---|
| TrackNet: `Label.csv` (`file name, visibility, x-coordinate, y-coordinate[, status]`) beside `0000.jpg…` | TrackNet tennis, TrackNetV4, TrackNetV5 user data | `ft.py import tracknet --src <root> --prefix tn --fps 30` (`--canvas W H` if labels are in another coordinate space) |
| TrackNetV2 / WASB: `csv/<rally>_ball.csv` (`Frame, Visibility, X, Y`) + `video/<rally>.mp4` | TrackNetV2 badminton (imported), RacketVision and BlurBall after renaming | `ft.py import tracknet --src <root> --prefix bb` |
| GridTrackNet: `matchN/Labels.csv` (`Frame, Visibility, X, Y` at 1280×720) + the source mp4 or `frames/*.png` | Match-Data | `ft.py import grid --src <Match-Data>` |
| workspace: `frame,ball_x,ball_y` (native px, invisible parked top-right) | archive, tenis-backview, anything from `label_tool.py` | `ft.py import archive --src <folder>` |

Visibility semantics: 0 = not in frame, 1 = visible, 2 = hard to see (kept),
3 = occluded with an estimated position (dropped unless `--keep-occluded`).
Bbox formats (COCO / YOLO: `class cx cy w h` normalised) need a small adapter
to centre points first (`X = cx·W, Y = cy·H`).

**Label convention caveat.** TrackNet, TrackNetV2, RacketVision and
PadelTracker100 mark the *leading edge / last visible point* of the motion
blur; our archive labels and BlurBall's midpoint set mark the **streak centre**.
At high ball speed these differ by several pixels. Pick one (the proposal
adopts the centre) and be consistent when mixing sources — the 10 px hit
radius in `evaluate_archive.py` is not much larger than the discrepancy.

## 4. Dead or unavailable (do not spend time here)

- GridTrackNet README's visible Drive URL `1gJUn2d6kVji4S_LiZWD1enATZFPs3gRp` — 404; the href `1LbmBbLxkmhD3P6UpPAck3NDBbMht2fg7` works.
- `nol.cs.nctu.edu.tw` (original TrackNet/TrackNetV2 host, referenced by ArtLabss, GridTrackNet, WASB, TrackNet-Pickleball) — DNS dead; use the GitLab pages, SharePoint zips or the Drive mirror.
- Utsav4852/Tennis-Ball-Tracking Drive file `1Dq2ag6a7ESHJm3ZHSJrYcu9_hWNyNkx1` — 404. TennisBallTracking Mega link — blocked. `hgupt3/TRACE` — 404. Tendnesshappy Baidu-pan — unreachable.
- Where Is The Ball (CVPRW 2025) — data, code and simulator "coming soon".
- TrackNetV5 weights and data — proprietary.
- WASB basketball source page (`ruiyan1995.github.io/SAM.html`) — 404.
- Pickleball: Rochester 12k-frame set, hudsong.dev PPA set, HF `sportsgirl/Pickleball*` — not released / empty; padel TrackNet forks — no data.
- IEEE DataPort multi-view tennis, IEEE DataPort cricket — subscription-gated.
- PadelVIC, "Volleyball-1,2", GSTD tennis (Sci Rep 2025), Sci Rep TT 3,300-frame set, eldadoh side-view set — no download.
- Kaggle and Roboflow pages block scripted access; Kaggle's `…/croissant/download` endpoint does return the file list.

## 5. Sources

TrackNet arXiv 1907.03698 · WASB arXiv 2311.05237 · TrackNetV4 arXiv 2409.14543 / https://tracknetv4.github.io/ · TrackNetV5 arXiv 2512.02789 · BlurBall arXiv 2509.18387 · TOTNet arXiv 2508.09650 · RacketVision arXiv 2511.17045 · PadelTracker100 PMC12926558 / Zenodo 14653706 · CalTennis arXiv 2606.20542 · One-Shot Badminton arXiv 2603.06691 · TTNet/OpenTTGames CVPRW 2020 · Uplifting TT WACV 2026 · TT3D arXiv 2504.10035 · YOLO-Net PLOS ONE 0335558 · GridTrackNet thesis https://cs.vu.nl/~versto/VU-CS-BSc-MSc-Theses/VU-CS-BSc-Thesis-Vincent-Korpelshoek-2023.pdf · Kaggle tenis-backview Croissant metadata (2025-06-05, CC0)
