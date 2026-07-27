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
| 9 | 2026-07-28 | Phase 1 train+eval hook, mutant (queue) | scripts/queue_remaining_training.sh | outputs/{hook,mutant}_n512_node | hook: 39.74/.9963/.0084, 3925 s, 1912 MiB; mutant: 45.03/.9990/.0029, 3809 s, 2082 MiB |
| 10 | 2026-07-28 | Phase 2B train+eval jumpingjacks n∈{64,128,1024,2048} (queue) | same | outputs/jumpingjacks_n*_node | PSNR 40.96/40.85/41.30/41.33 — flat vs n512's 41.53 |
| 11 | 2026-07-28 | Phase 2B FPS benchmark | scripts/bench_fps.py per model | results/failureB/fps.csv | 235.7/233.7/212.2/193.6/169.4 FPS for n=64..2048 |
| 12 | 2026-07-28 | Phase 2B standard edit (45°,90°, both modes) per node count | scripts/edit_headless.py | results/failureB/edit_n* | low extreme: solver-independent rigidity failure (0.42-0.47 edge p95); high extreme: 3 s/solve, iterative leg-ripping |
| 13 | 2026-07-28 | Phase 2B leakage metric from node clouds | inline (results/failureB/leakage.csv) | results/failureB/leakage.csv | iterative @90° p95 leakage 0.09→1.26 monotone in node count |
| 14 | 2026-07-28 | Phase 2B figures | scripts/plot_failure_b.py | results/failureB/failureB_curves.png | 4-panel quality-vs-cost curves |
