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
| 15 | 2026-07-28 | Cross-scene Failure A + improvement: hook (5 modes × 11 angles, drag fwd arm, rot_center 0.2,-0.05,0.55, axis X) | scripts/edit_headless.py | results/failureA/hook | same ordering as jumpingjacks; edge p95 @90°: 0.213 (from_init) → 0.134 (N8) vs 0.088 (iter); renders cam 17 (t=0-frontal) |
| 16 | 2026-07-28 | Cross-scene Failure A + improvement: mutant (claw arm, rot_center 0.35,-0.15,0.45, axis Y) | scripts/edit_headless.py | results/failureA/mutant | edge p95 @90°: 0.237 → 0.135 (N8) vs 0.109 (iter); renders cam 6 |
| 17 | 2026-07-28 | Cross-scene summary table + figure | scripts/cross_scene_summary.py | results/failureA/cross_scene_summary.csv, cross_scene_curves.png | strict monotone ordering in all 3 scenes; N8 closes 63-80% of gap |
| 18 | 2026-07-28 | Train+eval standup, trex n512 (for further validation) | outputs/queue2.log | outputs/{standup,trex}_n512_node | running |
| 19 | 2026-07-28 | Train+eval standup, trex n512 | scripts/train_scene.sh via queue2 | outputs/{standup,trex}_n512_node | standup 47.51/.9991/.0025 (paper 47.89); trex 40.68/.9985/.0038 (paper 41.24, −0.56 flag → investigated: best-ckpt 40.86, SSIM/LPIPS at/above paper) |
| 20 | 2026-07-28 | Cross-scene Failure A + improvement: standup (fwd arm, axis X, cam 7) | scripts/edit_headless.py | results/failureA/standup | monotone ordering; edge p95 @90° 0.362→0.261 (N8) vs 0.165 (iter), 51% gap closed |
| 21 | 2026-07-28 | Cross-scene Failure A + improvement: trex (tail, axis Z, cam 3) | scripts/edit_headless.py | results/failureA/trex | hard case: visual tail-shortening fixed by N4/8 (img PSNR 24.7→27.7 @90°) but edge deltas compressed; N2 ≈ from_init; reference itself stressed |
| 22 | 2026-07-28 | 5-scene cross summary + figures | scripts/cross_scene_summary.py | results/failureA/cross_scene_{summary.csv,curves.png} | N8 gap closure @90°: 70/63/80/51/44% |
| 23 | 2026-07-28 | Academic paper figures (schematic, onset/gap-closure/trade-off, 2-scene qualitative) | scripts/paper_figures.py | paper/figures/fig_*.png | 3 new publication figures |
| 24 | 2026-07-28 | Full academic paper written (abstract, intro, related work, failure study, method, 5-scene experiments, discussion) | paper/paper.md + paper/paper.html | paper/ | 8 figures, 4 tables, 13 references |
| 25 | 2026-07-28 | LaTeX paper (paper.tex) compiled to 9-page two-column PDF via tectonic | paper/paper.tex, paper/paper.pdf | paper/ | 8 figures, 4 tables, algorithm block, 13 refs |
