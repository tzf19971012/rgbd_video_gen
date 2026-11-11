# Experiment source snapshots

This directory preserves the actual source/configuration used for camera sampling,
projection conditioning, trajectory evaluation and latent-to-depth experiments.
Date-stamped directories are execution snapshots, including historical variants;
use the campaign configuration to choose the intended implementation.

- `projection_ablation_20261109_113836/`: source-depth-based GAE/RGB projection
  conditioning. `resume_3card_20261110_110818/` contains the active three-card
  implementation and configurations. Its `code/` tree overrides repository code
  through `PYTHONPATH`; do not load its checkpoints with the plain root model.
- `projection_trajectory_watch_20261110_143616/`: serial 1000-step trajectory
  probes, checkpoint pinning, strict load verification and comparison metrics.
- `span_ablation_c2w_fresh_20261107_230730/`: corrected-camera training configs,
  manifest conversion snapshots and batch-size/OOM supervision.
- `wan_latent_depth_probe_20261110_221707/`: frozen independently encoded Wan
  RGB latents with a lightweight depth decoder.
- `wan_latent_depth_continue_20261111_103831/`: continuation of the lightweight
  probe; retains validation-based checkpoint selection.
- `wan_pretrained_depth_decoder_20261111_104120/`: pretrained Wan VAE decoder
  adapted to single-view log depth; VAE encoder frozen.
- `wan_depth_multiview_ablation_20261111_153222/`: four observed-view latent
  fusion; optional frozen original Wan DiT blocks 9/19/29 at clean timestep0.
  `groups.json` defines context/target groups; `baseline_snapshot.py` fixes decoder
  initialization. This is not a generated-target-view or denoising-rollout study.

Depth labels in these pilot experiments are estimated depths, not sensor GT.
Scale-aligned depth metrics do not establish absolute-scale accuracy or superiority
against a depth teacher. Scene-disjoint validation selects checkpoints; the test
split is evaluated after selection.

Weights, datasets, cached latents/features, rendered media, running-state JSON,
logs and environment-variable dumps are not source artifacts. Existing absolute
paths describe the shared training server; reconstruct datasets/assets and the
Ascend runtime environment before launching elsewhere. The running experiments
are unaffected by this source commit.
