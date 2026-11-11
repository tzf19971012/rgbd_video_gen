Span ablation 2026-11-06
Cards 1/4/5/6: control / 5-20 / 5-35 / 10-35.
Shared original step8000 weights; optimizer/scheduler reset identically.
Full RE10K cached training manifest; no 8-sample overfit restriction.
Base LR 3e-5; fresh-layer multiplier 5; single-card batch2 accumulation8 = effective16.
Train step8000 ->18000 (10000 additional optimizer updates).
Every200 updates: checkpoint and fixed64-sample eval,25 denoise steps,seed1234.
Eval samples come from training manifest: diagnostic only, not held-out generalization evidence.
Span probability .5; log attempted/accepted/fallback per worker; control disabled.
Each arm retains only latest checkpoint. Original start weights hardlinked in initial-step-8000.
launches.json records detached processes; checkpoint_cleanup.json records old deletions.
No automatic infinite restarts; jobs end at18000 or stop on errors.
