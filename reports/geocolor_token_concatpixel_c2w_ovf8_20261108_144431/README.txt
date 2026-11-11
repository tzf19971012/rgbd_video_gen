Geo/color token concatpixel small-data fitting test
Card7 shared with existing PID841627, not stopped.
Eight fixed training scenes with sixteen jointly cached views each. Corrected c2w manifest pinned in this report directory.
Both camera branches concatpixel: geo pixel_scale14, RGB pixel_scale8 with384x672 virtual pixel grid preserving FOV. Geometry patch remains2x2 and RGB patch2x2 as in original experiment.
Fresh pretrained Wan initialization. LR1e-4, fresh multiplier1, batch1 accumulation4 (effective4), max4000 updates, fixed8 TRAIN-scene eval every200 updates, evalbatch1, latest checkpoint retained.
Original geo/color loss weighting retained to isolate camera-condition changes. This is not a held-out generalization experiment.
Launch/config/output: launch.json.
