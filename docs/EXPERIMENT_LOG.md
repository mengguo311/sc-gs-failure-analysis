# Experiment Log

Every run gets an entry: date, command, config, seed, GPU, result summary, output path.
GPU is always Quadro RTX 5000; seed is always 0 (fixed by upstream `safe_state`).

| # | Date | Experiment | Command / script | Output path | Result summary |
|---|------|-----------|------------------|-------------|----------------|
| 1 | 2026-07-27 | Phase 1 train jumpingjacks n512 | `scripts/train_scene.sh jumpingjacks 512` | outputs/jumpingjacks_n512_node | 3893 s wall, peak 1472 MiB, completed 80k iters |
| 2 | 2026-07-27 | Phase 1 eval jumpingjacks n512 | `scripts/eval_scene.sh jumpingjacks 512` | outputs/jumpingjacks_n512_node/test | test PSNR 41.53 / SSIM 0.9975 / LPIPS(vgg-piq) 0.0058 / MS-SSIM 0.9987 / LPIPS(alex) 0.0044 |
| 3 | 2026-07-27 | Failure A, centroid-rotation protocol (supplementary), modes from_init+iterative, angles 5–135° | `scripts/edit_headless.py --rot_center default` (see meta_*.json) | results/failureA/jumpingjacks_centroid | node metrics similar between modes; BOTH shred forearm ≥75° (gs p95 8–12) → propagation-stage artifact |
| 4 | 2026-07-27 | Failure A, shoulder-rotation protocol (main), iterative reference 1°/step | `scripts/edit_headless.py --rot_center 0.22,0,0.55 --edit_mode iterative` | results/failureA/jumpingjacks | clean through 135° (edge p95 ≤0.075, gs p95 ≤0.96); solve 0.5–12.3 s |
| 5 | 2026-07-27 | Failure A, shoulder protocol, from_init | same, `--edit_mode from_init` | results/failureA/jumpingjacks | fails: onset 45–60° (gs p95 1.7→10.0), under-rotation + limb shortening + hand shredding; solve 0.11 s |
| 6 | 2026-07-27 | Phase 3 progressive N=2/4/8, shoulder protocol | same, `--edit_mode progressive --steps N` | results/failureA/jumpingjacks | onset pushed: N=2→60°, N=4→75°, N=8→110°; edge p95 @135°: 0.27/0.21/0.11 vs 0.27 (N=1), 0.07 (iter); solve ≤0.76 s |
| 7 | 2026-07-27 | Failure A image metrics + grids | `scripts/analyze_failure_a.py --dir results/failureA/jumpingjacks` | results/failureA/jumpingjacks/image_metrics.csv, failureA_grid_cam0.png, failureA_diff_cam0.png | PSNR vs reference: from_init 28.5@15° → 16.4@90°; progressive monotone recovery |
| 8 | 2026-07-27 | Phase 1+2B queue: hook, mutant (n512); jumpingjacks n∈{64,128,1024,2048} | `scripts/queue_remaining_training.sh` | outputs/*, outputs/queue.log | running (started 20:40 JST) |
