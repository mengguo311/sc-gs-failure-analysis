# Phase 3 — Improvement: Progressive Drag Scheduling

## Idea

Failure A showed that `arap_from_init` collapses under large rotational drags because its
Laplacian-editing initialization cannot represent rotation and only 3 local-global ARAP
iterations follow. The `arap_iterative` mode is robust but (a) needs the user to perform
the drag slowly in small increments, (b) costs one full solve per increment (12.3 s
worth of solves for a 135° drag at GUI granularity), and (c) is stateful — the solution
depends on the whole drag history and cannot be re-run from a stored keyframe.

**Progressive drag scheduling** splits a requested drag (θ) into N sub-steps along the
target arc (slerp of the handle rotation): sub-step k solves ARAP with handle targets at
angle θ·k/N, warm-starting from sub-step k−1's solution (`init_verts`). The first sub-step
uses the Laplacian init (small angle → harmless). This brings `arap_iterative`'s
warm-start robustness into the one-shot, restorable `arap_from_init` workflow: the result
is a deterministic function of (rest state, final targets, N).

## Implementation

Implemented entirely in the batch editing driver — `scripts/edit_headless.py
--edit_mode progressive --steps N` (`solve_schedule()` + schedule construction). No
upstream code is modified: each sub-step is a plain `LapDeform.deform_arap` call with
`init_verts` carried over, exactly the call pattern the GUI already uses for
`arap_iterative` (train_gui.py:768-777). A GUI integration would be a ~10-line change in
`callback_keypoint_drag` (loop over interpolated targets); we kept it in the script to
leave upstream untouched and the change isolated, per the project rules.

## Results (Failure A protocol, jumpingjacks, shoulder-rotation drags)

Failure-onset angle (first angle where Gaussian neighbor-stretch p95 exceeds the
visible-artifact level ≈2, see results/failureA/jumpingjacks/failureA_curves.png):

| mode | onset angle | solve time @135° |
|---|---|---|
| from_init (N=1) | 45–60° | 0.11 s |
| progressive N=2 | 45–60° | 0.21 s |
| progressive N=4 | 60–75° | 0.39 s |
| progressive N=8 | 90–110° | 0.76 s |
| iterative 1°/step (reference) | none ≤135° | 12.34 s |

Node edge stretch (region p95) at 135°: 0.269 (N=1) → 0.271 (N=2) → 0.207 (N=4) →
0.112 (N=8) → 0.072 (reference). ARAP energy at 135°: 0.093 → 0.091 → 0.073 → 0.051 →
0.048. Image PSNR vs reference at 60°: 18.59 → 18.73 → 19.19 → 19.97 dB; LPIPS at 45°:
0.097 → 0.094 → 0.083 → 0.066.

**Interpretation.** Robustness scales monotonically with N: each warm-started sub-step
keeps the incremental rotation small enough for 3 ARAP iterations to track. N=8 covers
drags up to ~90° — the practical editing range — at interactive cost (≤0.8 s, 16× cheaper
than the reference at 135°). Beyond its onset each variant fails the same way as
from_init (the last sub-step's increment exceeds what the solver can absorb), so N should
be chosen ∝ the requested angle; a simple adaptive rule (fix the per-substep increment at
~10–15°, N = ceil(θ/15°)) follows directly from the onset table.

**Limitation.** The improvement targets the initialization failure only; the
propagation-stage artifact (supplementary centroid experiment, FAILURE_ANALYSIS.md) is
untouched, and at very large angles (≥110°) even N=8 has entered its failure regime.
Increasing ARAP iterations per sub-step is an orthogonal knob we did not ablate (kept
NUM_ITER=3 to isolate the scheduling effect).

Raw data: results/failureA/jumpingjacks/{metrics.csv,image_metrics.csv}; before/after
visuals: failureA_grid_cam0.png (rows: reference, from_init, N=2/4/8), failureA_diff_cam0.png,
curves: failureA_curves.png.
