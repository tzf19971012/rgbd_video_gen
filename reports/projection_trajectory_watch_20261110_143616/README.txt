Projection GAE/RGB automatic camera probes

Only new complete checkpoints at step2000,3000,...10000 are queued. Source training checkpoints are not modified. A hardlink pin protects each queued model from max_checkpoints=1 deletion. Only the newest successful pin per arm is retained, together with pending/running pins.

Uses the exact frozen3-card training code via PYTHONPATH. Training-code hashes checked before each run. Strict checkpoint loading. Fixed held-out scene fe2fadf89a84e92a, six source views0/15/30/45/60/75, corrected c2w test manifest, TRAIN normalization, seed1234,25step sampling, shared target noise.

Per checkpoint:15-frame real-camera trajectory with RGB/depth metrics +15-position yaw/lateral probe. Virtual-target projection reads only source RGB/depth/features.

NPU7 serial execution, wait if less than26GiB free. Own allocator budget22.4GiB. No training process stopped or reconfigured.

Start: nohup python watch.py >watch.log 2>&1 &
Stop gracefully: python watch.py --stop
Status: status.json; states/*.json; logs/*.log; summary.csv
Results are under each original experiment output: traj_stepN_test/scene and lateral_probe_stepN_test/scene.
Oom jobs retry at most3 times; other failures are recorded without a tight retry loop. Failed jobs do not block later checkpoints.

Interpretation: yawPSNR uses only valid source-overlap pixels, eroded5px. Lateral has no accuracy score without targetGT. Depth references are DA3 pseudo labels; reported depth scores are median-scale aligned.
