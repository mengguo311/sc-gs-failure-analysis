# STATUS

## 2026-07-27 (evening) — Phase 1 partially done, Failure A + improvement measured

**Done**
- Phase 0 complete (see docs/SETUP.md).
- **jumpingjacks reproduced**: test PSNR 41.53 vs paper 41.13 (+0.40 dB), SSIM 0.9975,
  LPIPS 0.0058. 65 min train, 1.5 GiB peak GPU. → docs/REPRODUCTION.md
- **Failure A characterized** (docs/FAILURE_ANALYSIS.md): shoulder-rotation drag protocol;
  from_init fails at 45–60° (Gaussian stretch p95 jumps 1.7→10.0, arm under-rotates,
  shortens, hand shreds); iterative reference clean through 135°. Supplementary
  centroid-rotation protocol exposes a second, propagation-stage artifact affecting BOTH
  modes ≥75°. All metrics/renders/node clouds in results/failureA/.
- **Phase 3 improvement measured** (same protocol): progressive drag scheduling N=2/4/8
  pushes failure onset 60°→75°→110°, edge distortion at 135° drops 0.27→0.11 (reference
  0.07), solve cost ≤0.76 s vs 12.3 s for the iterative reference.

**Running**
- Training queue (outputs/queue.log): hook, mutant (Phase 1), then jumpingjacks
  node_num ∈ {64,128,1024,2048} (Failure B). ETA ~7 h from 20:40 JST.

**Next**
- When queue finishes: fill REPRODUCTION.md (hook, mutant), run Failure B analysis
  (metrics + FPS benchmark + standard 45° edit across node counts), figures.
- Write IMPROVEMENT.md; assemble paper figures; draft paper sections 5–8.

**Blockers**: none.
