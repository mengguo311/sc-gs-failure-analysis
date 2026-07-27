# Phase 2 — Failure Analysis

All experiments use the jumpingjacks model trained in Phase 1 (node_num=512, seed 0,
iteration 80000). Raw metrics: `results/failureA/*/metrics.csv`, `image_metrics.csv`;
renders and node clouds alongside. Every number below is copied from those CSVs.

## Failure A — Laplacian-initialized ARAP under large rotational drags

### Hypothesis

`arap_from_init` (the GUI default) initializes every ARAP solve from a **linear
Laplacian-editing least-squares solution** (`utils/arap_deform.py:103`) and runs only
**3 local-global iterations** (`NUM_ITER=3`, line 110). The linear init cannot represent
rotation (classic chord-shortcut / shrinkage artifact), and 3 iterations cannot recover
from it at large angles. `arap_iterative` warm-starts each solve from the previous drag
frame and tracks the same rotation applied incrementally.

### Protocol (scripts/edit_headless.py)

- Scene: jumpingjacks at t=0 (arms up). Drag group = node nearest to the right hand
  (seed (0.6, 0, 0.85)) + 1-ring (5 nodes). Anchors = chest (0,0,0.4) and hips (0,0,0)
  seeds + 2-ring, held fixed (GUI "add keypoints without dragging" workflow).
- Rotational drag: handle targets rotated about the **shoulder joint** (0.22, 0, 0.55),
  axis Y (frontal plane), i.e. dragging the hand along the arc of an arm-lowering motion.
  Intermediate arm nodes are unconstrained — the solver must infer the limb rotation.
- Angle sweep: 5–135°. `iterative` = 1° warm-started sub-steps (reference, emulates GUI
  mouse-move granularity). `from_init` = single solve from Laplacian init, which is exactly
  the state the GUI reaches at accumulated angle θ in this mode (each drag event re-solves
  from init at the current absolute targets).
- Both modes are hard-constrained at the handles (handle_residual = 0 in all runs), so all
  differences appear in the free nodes.

### Node-level results (edge-length distortion of the ARAP graph, drag region)

p95 relative edge stretch, region = drag group + 4 rings:

| angle | iterative | from_init | ratio |
|---|---|---|---|
| 15° | 0.011 | 0.023 | 2.0× |
| 30° | 0.027 | 0.048 | 1.8× |
| 45° | 0.044 | 0.075 | 1.7× |
| 60° | 0.059 | 0.111 | 1.9× |
| 90° | 0.074 | 0.189 | 2.5× |
| 135° | 0.072 | 0.269 | 3.8× |

ARAP energy (`cal_arap_error` between rest and deformed nodes) at 135°: 0.048 (iterative)
vs 0.093 (from_init). The iterative solution plateaus (~0.07 edge stretch) once the arm
rotates rigidly; from_init keeps deforming the free chain instead of rotating it.

### Gaussian-level results (LBS propagation)

p95 of per-Gaussian max neighbor-distance stretch in the edited region (K=8 canonical
neighbors):

| angle | iterative | from_init |
|---|---|---|
| 45° | 0.23 | 1.70 |
| 60° | 0.28 | **9.98** |
| 90° | 0.48 | 10.20 |
| 135° | 0.96 | 12.16 |

**Failure onset: between 45° and 60°** — the Gaussian stretch statistic jumps by ~6× in
that interval (1.70 → 9.98) while the reference stays below 0.3. Mechanism: node rotations
for skinning are re-estimated from the deformed node positions (`p2dR`,
`utils/time_utils.py:1044`); the sheared from_init node field yields inconsistent
rotations, and Gaussians blended across the drag-region boundary are torn apart.

### Image-space results (vs. the iterative reference, test cam 0)

| angle | PSNR | SSIM | LPIPS(alex) |
|---|---|---|---|
| 15° | 28.46 | 0.973 | 0.016 |
| 45° | 19.96 | 0.896 | 0.097 |
| 90° | 16.35 | 0.824 | 0.248 |
| 135° | 16.45 | 0.830 | 0.264 |

(Interpretation caveat: at large angles this measures both artifacts *and* pose deviation —
the visual grids `failureA_grid_cam0.png` / `failureA_diff_cam0.png` show that both are
present: from_init under-rotates the arm (stuck near horizontal at 90–135°), shortens it,
and shreds the hand into a Gaussian blob from ~60°.)

### Supplementary experiment: rotation about the group centroid

`results/failureA/jumpingjacks_centroid/` — same sweep but rotating the (fully
hard-constrained) hand+2-ring group about its own centroid, GUI's R-drag semantics. Here
from_init and iterative differ only mildly at the node level (edge p95 at 135°: 0.40 vs
0.33) because the moving nodes are all handle-constrained; **but both modes shred the
forearm from ~75°** (gs p95 ≈ 8–12). This isolates a second artifact source: the
`p2dR` + Floyd-weight LBS propagation itself fails under large in-place rotations of a
small handle cluster, independent of the ARAP initialization. We report it as a secondary
finding; the improvement (Phase 3) targets the primary (init) failure.

### Solve cost

from_init: 0.11–0.12 s per solve (interactive). iterative at 1°/step: 0.5 s (5°) to
12.3 s (135°) — the robustness of the reference costs ~100× the latency at large angles,
which is precisely why the GUI's default restorable mode is from_init and why a middle
ground (Phase 3) is attractive.

## Failure B — Sensitivity to control point count

### Protocol

jumpingjacks retrained from scratch with node_num ∈ {64, 128, 512, 1024, 2048}, all other
flags identical to Phase 1 (seed 0). Per model: (1) test-set metrics (`render.py`),
(2) render FPS (`scripts/bench_fps.py`, 3 timed rounds over 20 test cams, GPU-synced),
(3) the standard Failure-A shoulder-rotation edit at 45°/90° in both ARAP modes
(`scripts/edit_headless.py`, same seed points), (4) off-region leakage computed from the
saved node clouds (`results/failureB/leakage.csv`). Figure:
`results/failureB/failureB_curves.png`; edit renders under `results/failureB/edit_n*/`.

### Reconstruction quality and cost (test split)

| node_num | PSNR | SSIM | LPIPS | render FPS | #Gaussians | train wall | peak GPU |
|---|---|---|---|---|---|---|---|
| 64 | 40.96 | .99735 | .00620 | 235.7 | 68.7k | 3866 s | 1404 MiB |
| 128 | 40.85 | .99734 | .00729 | 233.7 | 67.9k | 3842 s | 1444 MiB |
| 512 | **41.53** | .99752 | **.00580** | 212.2 | 71.0k | 3893 s | 1472 MiB |
| 1024 | 41.30 | .99756 | .00602 | 193.6 | 72.3k | 3793 s | 1566 MiB |
| 2048 | 41.33 | .99757 | .00634 | 169.4 | 72.1k | 3654 s | 1742 MiB |

**Novel-view synthesis is remarkably insensitive to node count** (spread 0.7 dB over a
32× sweep; 512 best but 64 loses only 0.6 dB). Training cost is flat; FPS drops 28%
monotonically (deform-MLP + skinning cost). Within this range node count is *not* a
reconstruction knob — its real effects are on editability:

### Editing failure regime at the low extreme (64–128 nodes)

At 90°, BOTH solver modes produce nearly identical, heavily distorted results
(edge stretch region p95: 0.42 @64, 0.46–0.47 @128 — 6× the n512 iterative reference;
visually the arm barely articulates and the sleeve smears). The failure is
solver-independent: with ~64 nodes the drag group + free chain spans a handful of graph
hops, deformation granularity is bounded by node spacing, and the limb cannot bend at the
shoulder. Leakage is small (anchors dominate the tiny graph) — the failure is purely local.

### Editing failure regime at the high extreme (2048 nodes)

Two distinct failures:
1. **Interactivity collapse**: one-shot from_init solve takes 3.0 s (25× the n512 cost);
   the 90° iterative drag takes 200 s. The dense-graph solve (2048×2048 pseudo-inverse
   least squares × 3 iterations × steps) is far past interactive rates.
2. **Global leakage** (new metric): p95 displacement of nodes *outside* the drag region and
   anchors reaches **1.26 scene units** for iterative @90° (body height ≈2 units) — the
   legs are visibly ripped sideways (see `edit_n2048/renders/iterative_a090_cam0.png`)
   while all *local* rigidity metrics stay moderate (the drift is smooth, so edge lengths
   barely change — this is why the leakage metric is needed). Leakage grows monotonically
   with node count (iterative @90° p95: 0.09 → 0.21 → 0.38 → 0.55 → 1.26) and is ~2×
   worse in iterative than from_init at 2048: warm-start accumulation across 90 sub-steps
   compounds the drift, whereas the stateless from_init solves once. **The preferable
   solver mode inverts at the high extreme** — the mode that fixes Failure A (iterative)
   is the one that fails here.

### Summary

Quality-vs-cost sweet spot confirms the paper's default (512, or 512–1024): below it,
editing granularity fails; above it, editing latency and global leakage fail — while
reconstruction quality never rewards more nodes in this range.
