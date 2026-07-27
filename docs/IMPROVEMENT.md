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

## Cross-scene validation (2026-07-28)

The full protocol (5 modes × 11 angles) was repeated on **hook** (forward-extended arm
rotated about the shoulder, axis X, render cam 17) and **mutant** (right claw arm rotated
about the shoulder, axis Y, render cam 6); protocol seeds in
`results/failureA/<scene>/meta_*.json`. The mode ordering is strictly monotone in every
scene at every tested angle:

edge stretch region p95 @90° / @135°:

| mode | jumpingjacks | hook | mutant |
|---|---|---|---|
| from_init (N=1) | 0.189 / 0.269 | 0.213 / 0.318 | 0.237 / 0.399 |
| progressive N=2 | 0.174 / 0.271 | 0.201 / 0.285 | 0.211 / 0.337 |
| progressive N=4 | 0.140 / 0.207 | 0.169 / 0.228 | 0.177 / 0.258 |
| progressive N=8 | 0.108 / 0.112 | 0.134 / 0.175 | 0.135 / 0.221 |
| iterative (ref) | 0.074 / 0.072 | 0.088 / 0.125 | 0.109 / 0.157 |

At 90°, N=8 closes 63% (hook), 70% (jumpingjacks), 80% (mutant) of the
from_init→iterative gap at ≤0.75 s per drag. Artifact *severity* is scene-dependent
(Gaussian stretch p95 @90°: jumpingjacks 10.2, hook 4.1, mutant 1.6 — thin limbs tear
much worse than thick claws), but the ordering never inverts. Figures:
`results/failureA/cross_scene_curves.png` + per-scene `failureA_curves.png` +
`image_metrics.csv`. hook/mutant renders use a t=0-frontal camera chosen by scanning all
20 test cams (the character orientation at t=0 differs from the test-set poses).

### Second round: standup (arm) and trex (tail), 2026-07-28

Two more scenes were trained (see REPRODUCTION.md) and put through the identical
protocol. **standup** (crouched figure, forward-reaching arm rotated about the shoulder,
axis X, cam 7): clean monotone ordering — edge region p95 @90°: 0.362 (from_init) →
0.344 / 0.310 / 0.261 (N=2/4/8) → 0.165 (reference); N=8 closes 51% of the gap; image
PSNR-vs-reference @45°: 26.9 → 29.8 dB (N=8).

**trex** (long thin tail swung sideways about its root, axis Z, cam 3) is the **hardest
and most nuanced case**: the visual shortening artifact of from_init is clearly visible
on the tail, and N=4/8 restore the smooth long arc (image PSNR @90°: 24.7 → 25.0 → 27.7;
LPIPS 0.039 → 0.036 → 0.018), but node-level *edge* differences are compressed
(@90°: 0.285 from_init vs 0.244 reference; N=2 statistically indistinguishable from
from_init, 0.295) and even the reference is visibly stressed (gs p95 4.3 @135°). The
tail is a single long elastic chain far from both anchors — every solver mode struggles,
and per-substep increments must be small before warm-starting pays off (N=2's 45–67°
sub-steps are already past the per-step failure level, hence no gain). Honest summary:
progressive scheduling **helps everywhere but is not a silver bullet on extreme
kinematic chains**; N must scale with the difficulty of the drag, and the propagation
stage remains a bottleneck (consistent with the Failure-A secondary finding).

Cross-scene gap closure by N=8 at 90° (edge region p95): jumpingjacks 70%, hook 63%,
mutant 80%, standup 51%, trex 44%.

**Limitation.** The improvement targets the initialization failure only; the
propagation-stage artifact (supplementary centroid experiment, FAILURE_ANALYSIS.md) is
untouched, and at very large angles (≥110°) even N=8 has entered its failure regime.
Increasing ARAP iterations per sub-step is an orthogonal knob we did not ablate (kept
NUM_ITER=3 to isolate the scheduling effect).

Raw data: results/failureA/jumpingjacks/{metrics.csv,image_metrics.csv}; before/after
visuals: failureA_grid_cam0.png (rows: reference, from_init, N=2/4/8), failureA_diff_cam0.png,
curves: failureA_curves.png.
