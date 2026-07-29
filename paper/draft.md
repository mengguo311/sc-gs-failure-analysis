# Failure Analysis and Improvement of SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes

**Course**: Visual Media (The University of Tokyo) — Final Project Report

## 1. Author Information

- Name: `<TODO>`
- Student ID: `<TODO>`
- Department: `<TODO>`
- Laboratory: `<TODO>`
- Own research topic: `<TODO>`

## 2. Summary of SC-GS

SC-GS [Huang et al., CVPR 2024] represents a dynamic scene with dense canonical 3D
Gaussians whose motion is driven by a much sparser set of learned **control points**
(512 for a scene with ~70k Gaussians in our runs). Each control point carries a
time-conditioned MLP that outputs its translation (plus auxiliary rotation/scaling) at any
query time; each Gaussian follows its K nearest control points through linear blend
skinning (LBS) with learned Gaussian-kernel weights. An as-rigid-as-possible (ARAP)
regularizer on control-point trajectories biases the motion field toward locally rigid
deformation. Decoupling appearance (dense Gaussians) from motion (sparse control points)
yields both high-fidelity dynamic novel-view synthesis and, crucially, **motion editing**:
dragging a few control points deforms the scene through an ARAP solve over the control
graph while the LBS propagation preserves rendering quality. On the D-NeRF benchmark the
paper reports state-of-the-art quality (average 43.31 dB PSNR / .997 SSIM / .0063 LPIPS
over 8 scenes), real-time rendering, and interactive editing.

## 3. Understanding of the Method

![Method overview](figures/method_overview.png)
*Figure 1: (a) dense Gaussians reproduce appearance; (b) 512 sparse control nodes carry
motion — editing constrains a few handle nodes (red: dragged along a shoulder-rotation
arc; blue: anchors) and solves ARAP for the rest; (c) the solved node field drives the
Gaussians through LBS.*

Three components interact (file:line references are to the official implementation at the
pinned commit; details verified by reading the code, see docs/METHOD_NOTES.md):

**(a) Sparse control points + LBS interpolation.** Control nodes (positions + learned
radii) are optimized jointly with the Gaussians; a deformation MLP maps `(node, t)` to
per-node translation/rotation/scaling (`utils/time_utils.py`, `ControlNodeWarp`). Each
Gaussian blends its K nearest nodes with Gaussian-kernel weights; learned `hyper_dim`
features separate spatially-close but topologically-distinct parts. With `--local_frame`,
per-node local rotations are applied to the Gaussian's offset in the node frame before
blending (time_utils.py:1148–1154). An ARAP loss on sampled node trajectories regularizes
training toward locally rigid motion.

**(b) Editing = ARAP on the control graph.** At edit time t, node positions
`p = nodes + d_xyz(t)` form a graph (K=16 neighbors, trajectory-aware connectivity). The
user selects keypoint groups (clicked node + 2-ring neighbors); dragging sets absolute
handle targets. The solver (`utils/arap_deform.py:86`) minimizes ARAP energy with handles
as hard constraints via local-global iteration (local: per-node SVD rotation fit; global:
linear solve). **Two GUI modes differ only in initialization** (train_gui.py:768–777):
`arap_from_init` starts every solve from a *linear Laplacian-editing least-squares*
solution; `arap_iterative` warm-starts from the previous drag frame. Only **3
local-global iterations** run per solve (NUM_ITER=3, arap_deform.py:110).

**(c) Propagation to Gaussians.** The solved node displacement field is not paired with
the ARAP rotations; node rotations are re-estimated from displaced node positions by
local SVD (`p2dR`, time_utils.py:1044), then Gaussians are re-expressed relative to their
deformed neighbor nodes (time_utils.py:1196–1213). Rigidity errors in the node field thus
translate directly into inconsistent rotations and stretched Gaussian layouts.

## 4. Execution Environment

| Item | Value |
|---|---|
| GPU | NVIDIA Quadro RTX 5000, 16 GB (Turing, sm_75) |
| Driver / CUDA | 570.133.07 / CUDA 12.4 toolkit (nvcc 12.4.131) |
| OS / compiler | Ubuntu 24.04.1 LTS / gcc 13.3 |
| Python / PyTorch | 3.9.23 / 2.4.1+cu124 |
| Key deps | pytorch3d 0.7.8 (source build); diff-gaussian-rasterization (patched: gcc-13 `<cstdint>` fix); simple-knn |
| Dataset | D-NeRF synthetic; 400×400 train/eval (`--resolution 2`, the paper's comparison convention) |

Full setup log including every build error and fix: `docs/SETUP.md`. All experiments use
upstream's fixed seed (0); every run is logged in `docs/EXPERIMENT_LOG.md`.

## 5. Reproduction

Official command (README) per scene, 80k iterations, no code changes:

| Scene | PSNR paper / ours (Δ) | SSIM paper / ours | LPIPS paper / ours | train time | peak GPU |
|---|---|---|---|---|---|
| jumpingjacks | 41.13 / **41.53** (+0.40) | .998 / .9975 | .0067 / .0058 | 64.9 min | 1.47 GiB |
| hook | 39.87 / 39.74 (−0.13) | .997 / .9963 | .0076 / .0084 | 65.4 min | 1.91 GiB |
| mutant | 45.19 / 45.03 (−0.16) | .999 / .9990 | .0028 / .0029 | 63.5 min | 2.08 GiB |

All three scenes reproduce within ±0.5 dB — no investigation flag triggered. Test-time
rendering runs at 212 FPS (jumpingjacks, 400×400, deform+render, GPU-synced). The small
hook gap is within 3DGS run-to-run noise (nondeterministic densification via rasterizer
atomics despite fixed seeds).

## 6. Failure Case Analysis

Both studies script the GUI editing pipeline headlessly (`scripts/edit_headless.py`
replicates `animation_initialize` → keypoint selection → `deform_arap` →
`deform.step(node_trans_bias)` → render, call-for-call). Protocol: jumpingjacks at t=0;
drag handles = node nearest the right hand + 1-ring; anchors = chest and hip groups;
handle targets rotated about the shoulder joint in the frontal plane — i.e., dragging the
hand along the arc of an arm-lowering motion, leaving intermediate arm nodes free. The
`arap_iterative` mode applied in 1° warm-started sub-steps serves as reference (it
emulates GUI mouse granularity); `arap_from_init` in one shot is exactly the state the
GUI default mode reaches at accumulated angle θ (it re-solves from scratch at the current
absolute targets on every drag event).

### 6.1 Failure A: Laplacian-initialized ARAP under large rotational drags

![Failure A curves](figures/failureA_curves.png)
*Figure 2: (a) control-graph rigidity violation, (b) Gaussian-level tearing (log), (c)
image deviation from the reference, all vs. drag angle.*

**Result: from_init fails between 45° and 60°.** The Gaussian neighbor-stretch statistic
jumps 1.70 → 9.98 across that interval (reference: ≤ 0.28) and node edge stretch reaches
3.8× the reference at 135° (0.269 vs 0.072). Visually (Figure 4) the arm **under-rotates**
(stuck near horizontal at 90–135°), **shortens** toward the chord, and the hand
**shreds** into a Gaussian blob — the classic linear-Laplacian-editing artifact: the init
cannot represent rotation, and 3 local-global ARAP iterations cannot recover from it.
Image deviation from the reference grows accordingly (PSNR 28.5 dB at 15° → 16.4 dB at
90°). Both modes satisfy handle constraints exactly (residual 0), so all differences live
in the free nodes. The robust reference costs 12.3 s of accumulated solves for a 135°
drag vs 0.11 s for from_init — robustness and interactivity trade off directly.

**Secondary finding (supplementary protocol).** Rotating a fully-constrained handle
group about its own centroid (the GUI's R-drag semantics) produces near-identical node
fields in both modes, yet **both** shred the forearm beyond ~75°
(results/failureA/jumpingjacks_centroid/): a second, initialization-independent artifact
in the propagation stage (`p2dR` rotation re-estimation + Floyd-weight LBS across the
handle boundary). The Phase-3 improvement targets the primary (initialization) failure;
this one is orthogonal.

### 6.2 Failure B: Sensitivity to control point count

jumpingjacks retrained with node_num ∈ {64, 128, 512, 1024, 2048} (identical flags/seed);
per model we measure test metrics, render FPS, and the standard 45°/90° edit in both
modes, plus a **leakage** metric: p95 displacement of nodes *outside* the drag region and
anchors (computed from saved node clouds).

![Failure B curves](figures/failureB_curves.png)
*Figure 3: (a) reconstruction quality is flat while FPS falls 28%; (b) edit rigidity is
U-shaped; (c) off-region leakage grows monotonically; (d) solve latency crosses the
interactivity threshold.*

| node_num | 64 | 128 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| test PSNR (dB) | 40.96 | 40.85 | **41.53** | 41.30 | 41.33 |
| render FPS | 235.7 | 233.7 | 212.2 | 193.6 | 169.4 |
| edge stretch region p95, 90° drag (iter / from_init) | 0.42 / 0.42 | 0.46 / 0.47 | 0.07 / 0.19 | 0.04 / 0.30 | 0.21 / 0.31 |
| leakage p95, 90° (iter / from_init) | 0.09 / 0.05 | 0.21 / 0.06 | 0.38 / 0.25 | 0.55 / 0.49 | **1.26** / 0.56 |
| solve time, 90° drag (iter / from_init) | 0.9 / 0.01 s | 1.7 / 0.03 s | 8.2 / 0.12 s | 33 / 0.48 s | 200 / 2.9 s |

**Reconstruction is insensitive** (0.7 dB spread over a 32× node sweep) — node count in
this range is an editability/cost knob, not a quality knob. The failures sit at the
extremes:

- **Low extreme (64–128): articulation failure.** Both solver modes produce nearly
  identical, heavily distorted edits (edge stretch 6× the n512 reference; the arm barely
  bends). The failure is solver-independent: deformation granularity is bounded by node
  spacing.
- **High extreme (2048): interactivity + leakage failure.** A single from_init solve
  takes 3 s (25× n512); the 90° iterative drag takes 200 s. Worse, off-region leakage
  reaches 1.26 scene units (body height ≈ 2): the *legs* are visibly ripped sideways by
  an arm drag, while every *local* rigidity metric stays moderate — the drift is smooth,
  which is exactly why the leakage metric is needed. Leakage is ~2× worse in iterative
  than from_init at 2048 (warm-start accumulation over 90 sub-steps compounds it), i.e.
  **the mode that fixes Failure A is the one that fails here** — the preferable solver
  inverts across the node-count range.

The sweet spot confirms the paper's default (512–1024).

## 7. Improvement: Progressive Drag Scheduling

Failure A's root cause is a rotation-blind initialization plus a hard iteration budget.
**Progressive drag scheduling** splits a requested drag θ into N sub-steps along the
target arc: sub-step k solves with handles at θ·k/N, warm-starting from sub-step k−1
(`init_verts`), the first sub-step falling back to the (now harmless, small-angle)
Laplacian init. This imports `arap_iterative`'s warm-start robustness into the one-shot,
*restorable* `arap_from_init` workflow — the result is a deterministic function of (rest
state, final targets, N), independent of drag history.

Implemented entirely in the batch driver (`scripts/edit_headless.py --edit_mode
progressive --steps N`); zero upstream code modified — each sub-step reuses the exact
`deform_arap(init_verts=...)` call the GUI already makes (a GUI integration would be a
~10-line loop in `callback_keypoint_drag`).

## 8. Before/After Comparison

![Before/after grid](figures/before_after_grid.png)
*Figure 4: SC-GS (original) / Ours N=4 / Ours N=8 / SC-GS iterative (reference), top to
bottom, at 45°, 90°, 135°. The original mode's arm under-rotates and shreds; our
schedules restore the reference pose at a fraction of its cost.*

Failure-onset angle (first angle where Gaussian stretch p95 exceeds the visible-artifact
level ≈ 2; Figure 2b):

| mode | onset | edge p95 @135° | ARAP energy @135° | solve @135° |
|---|---|---|---|---|
| from_init (N=1) | 45–60° | 0.269 | 0.093 | 0.11 s |
| progressive N=2 | 45–60° | 0.271 | 0.091 | 0.21 s |
| progressive N=4 | 60–75° | 0.207 | 0.073 | 0.39 s |
| progressive N=8 | 90–110° | 0.112 | 0.051 | 0.76 s |
| iterative 1°/step (reference) | none ≤135° | 0.072 | 0.048 | 12.34 s |

Robustness scales monotonically with N at interactive cost: N=8 covers the practical
editing range (≤90°) at ≤0.8 s per drag, 16× cheaper than the reference at 135°. Image
metrics vs the reference improve monotonically in N at every angle (e.g. 60°: PSNR
18.59 → 19.97 dB; 45°: LPIPS 0.097 → 0.066). Beyond its onset each variant fails like
from_init (the final sub-step's increment exceeds what 3 ARAP iterations absorb), so N
should scale with θ — a per-substep increment of ~10–15° (N = ⌈θ/15°⌉) follows from the
onset table.

**Cross-scene validation.** The full protocol (5 modes × 11 angles) was repeated on four
more scenes, each with its own limb and rotation joint: **hook** (forward arm, axis X),
**mutant** (claw arm, axis Y), **standup** (crouched forward arm, axis X; trained for
this validation, reproduces at 47.51 dB vs paper 47.89), and **trex** (long thin tail
swung sideways about its root, axis Z; 40.68 dB vs 41.24 — see REPRODUCTION.md for the
brief investigation of this −0.56 dB gap). Node edge stretch (region p95) at 90°/135°:

![Cross-scene curves](figures/cross_scene_curves.png)
*Figure 5: edge-stretch vs angle for all five scenes.*

| mode | jumpingjacks | hook | mutant | standup | trex |
|---|---|---|---|---|---|
| from_init (N=1) | 0.189 / 0.269 | 0.213 / 0.318 | 0.237 / 0.399 | 0.362 / 0.439 | 0.285 / 0.410 |
| progressive N=4 | 0.140 / 0.207 | 0.169 / 0.228 | 0.177 / 0.258 | 0.310 / 0.389 | 0.285 / 0.391 |
| progressive N=8 | 0.108 / 0.112 | 0.134 / 0.175 | 0.135 / 0.221 | 0.261 / 0.307 | 0.267 / 0.371 |
| iterative (ref) | 0.074 / 0.072 | 0.088 / 0.125 | 0.109 / 0.157 | 0.165 / 0.181 | 0.244 / 0.342 |

N=8 closes 70% / 63% / 80% / 51% / 44% of the from_init→iterative gap at 90° on the five
scenes (full table incl. N=2: results/failureA/cross_scene_summary.csv). On four scenes
the ordering from_init > N=2 > N=4 > N=8 > iterative is strictly monotone at every
tested angle. **trex is the honest hard case**: the visual tail-shortening of from_init
is obvious and N=4/8 restore the smooth arc (image PSNR vs reference @90°: 24.7 → 27.7
dB, LPIPS 0.039 → 0.018), but node-level differences compress — N=2 is
indistinguishable from from_init (its 45–67° sub-steps already exceed the per-step
failure level), and even the reference is stressed (the tail is one long elastic chain
far from every anchor). Progressive scheduling helps everywhere but is not a silver
bullet on extreme kinematic chains; N must scale with drag difficulty. Artifact
*severity* is scene-dependent (Gaussian stretch p95 @90°: 10.2 on jumpingjacks' thin
arm, 1.6 on mutant's thick claw) yet the ordering never inverts on any scene at any
angle except the trex N=2 case noted.

## 9. Limitations

- **Coverage**: the Failure-A/improvement protocol is validated on three scenes
  (jumpingjacks, hook, mutant — one limb and one drag protocol per scene, seeds recorded
  in meta_*.json); Failure B's node-count sweep is jumpingjacks-only, and absolute
  artifact severity varies substantially by limb geometry.
- **Improvement scope**: progressive scheduling addresses only the initialization
  failure. The propagation-stage artifact (Sec. 6.1 secondary finding) and the
  high-node-count leakage (Sec. 6.2) are untouched; at ≥110° even N=8 fails. NUM_ITER=3
  was deliberately kept fixed to isolate the scheduling effect — raising it is an
  unablated orthogonal knob.
- **Metrics**: image comparisons vs the iterative reference conflate pose deviation with
  artifacts at large angles; we therefore lean on node/Gaussian-level rigidity and
  leakage metrics, which are geometry-only proxies for perceived quality.
- **Reference is not ground truth**: `arap_iterative` itself drifts at 135° (whole-body
  lean) and catastrophically leaks at n=2048; "reference" means "the GUI's robust mode",
  not an oracle.

## 10. Use of Generative AI

This project was executed by Claude Code (Anthropic Fable 5) as an autonomous research
agent under a human-authored project brief, with all work logged for verification:
environment setup, code reading, experiment design/execution, and drafting of docs and
this report. All quantitative claims are machine-generated from committed CSVs
(`results/`), every run is recorded in `docs/EXPERIMENT_LOG.md`, and protocol choices are
serialized in `meta_*.json` files; the human author reviewed and is responsible for the
final content. The full running log is `docs/AI_USAGE_LOG.md`. No AI-generated text was
copied without review; figures are generated by committed scripts from measured data.

## 11. Discussion

The analysis reframes SC-GS's editing modes as a **robustness–latency–statefulness
trade-off** rather than a simple default choice: from_init is fast and restorable but
rotation-blind at initialization; iterative is robust to rotation but slow, stateful, and
— at high node counts — the *less* safe mode due to drift accumulation. Progressive
scheduling occupies the useful middle: bounded latency, deterministic output, and a
robustness dial (N). Two observations seem general beyond SC-GS. First, editing quality
and reconstruction quality decouple sharply in control-point methods — benchmark tables
(PSNR) are silent about the property users actually experience when editing. Second,
smooth global drift is invisible to local rigidity metrics; an off-region leakage measure
was necessary to quantify the most severe artifact we found. A natural next step is
adaptive scheduling (fix per-substep increment, N = ⌈θ/15°⌉) and increasing ARAP
iterations only on the final sub-step, which would cost little and might push the onset
further; on the propagation side, blending the ARAP solver's own rotations (currently
discarded) instead of re-estimating them from positions (`p2dR`) might remove the
secondary artifact.
