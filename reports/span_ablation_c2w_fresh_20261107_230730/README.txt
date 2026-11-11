Four-arm camera convention corrected experiment

Start: Wan2.1-T2V-1.3B pretrained backbone, fresh LoRA + GAE patch/head layers. Step 0, no experiment resume. All trainable initialization hashes match.
Arms: control/card1; span5to20/card4; span5to35/card5; span10to35/card6.
All: LR3e-5, fresh multiplier5, batch12 accumulation1, seed42, max_steps10000. Shared fp32 camera geometry. Full7884 cached train scenes.
Training manifest: /mnt/sfs_turbo/t00845730/re10k/manifest_train_c2w_v1.jsonl
Held-out evaluation MUST use: /mnt/sfs_turbo/t00845730/re10k/manifest_test_c2w_v1.jsonl + cache re10k_test.
Online validation is fixed64 TRAIN scenes only, not a held-out benchmark.
Old raw extracted poses.json and NPZ camera fields are legacy w2c; they were not rewritten. RGB/depth and camera-independent GAE latents are reused. New training reads camera poses from fixed manifests exclusively.
The patched extraction worker writes real c2w with a convention marker. Both manifest builders handle unmarked legacy w2c and marked new c2w.
Old runs stopped, latest saved checkpoint step10200 retained. New logs: *.bs12.log; configs: this directory; output paths and PIDs: launches.json.
