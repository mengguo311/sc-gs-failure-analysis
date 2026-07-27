# Failure Analysis and Improvement of SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes

**Course**: Visual Media (The University of Tokyo) — Final Project Report

## 1. Author Information

- Name: `<TODO>`
- Student ID: `<TODO>`
- Department: `<TODO>`
- Laboratory: `<TODO>`
- Own research topic: `<TODO>`

## 2. Summary of SC-GS

SC-GS [Huang et al., CVPR 2024] represents a dynamic scene with a set of canonical 3D
Gaussians whose motion is driven by a much sparser set of learned **control points**
(typically 512 for a scene with 100k+ Gaussians). Each control point carries a
time-conditioned MLP that outputs its translation (and auxiliary rotation/scaling) at any
query time; each Gaussian follows its K nearest control points through linear blend
skinning (LBS) with learned Gaussian-kernel weights. An ARAP (as-rigid-as-possible)
regularizer on control-point trajectories biases the motion field toward locally rigid
deformations. Decoupling appearance (dense Gaussians) from motion (sparse control points)
yields both high-fidelity novel-view synthesis of dynamic scenes and, crucially,
**motion editing**: dragging a few control points deforms the scene through an ARAP solve
over the control graph, while rendering quality is preserved by the LBS propagation.
On the D-NeRF synthetic benchmark, the paper reports state-of-the-art quality
(average 43.31 dB PSNR / .997 SSIM / .0063 LPIPS over 8 scenes), real-time rendering,
and interactive editing.

## 3. Understanding of the Method

`<TODO: method figure (control points + LBS + ARAP editing diagram)>`

Three components interact (all references are to the official implementation):

**(a) Sparse control points + LBS-style interpolation.** Canonical Gaussians are standard
3DGS parameters. Control nodes (positions + learned radii) are optimized jointly; a
deformation MLP maps `(node, t)` to per-node translation/rotation/scaling
(`utils/time_utils.py`, `ControlNodeWarp`). For each Gaussian, the K nearest nodes (with
learned `hyper_dim` features separating spatially-close but topologically-distinct parts)
define Gaussian-kernel LBS weights; with `--local_frame`, per-node local rotations are
applied to the Gaussian's offset in the node frame before blending
(time_utils.py:1148–1154). An ARAP loss on sampled node trajectories regularizes training.

**(b) Editing = ARAP on the control graph.** At edit time t, node positions
`p = nodes + d_xyz(t)` form a graph (K=16 neighbors, trajectory-aware connectivity).
The user selects a keypoint group (clicked node + 2-ring neighbors); dragging sets
absolute handle targets. The solver (`utils/arap_deform.py:86`) minimizes the ARAP energy
with handles as hard constraints via local-global iterations: local step = per-node SVD
rotation fitting, global step = linear solve with handles substituted. **Two modes differ
only in initialization** (`train_gui.py:768–777`): `arap_from_init` starts every solve
from a Laplacian-editing linear least-squares solution; `arap_iterative` warm-starts from
the previous drag frame's solution. Only **3 local-global iterations** are run per solve.

**(c) Propagation to Gaussians.** The solved node displacement field (`node_trans_bias`)
is not paired with the ARAP rotations; instead node rotations are re-estimated from the
displaced node positions by local SVD (`p2dR`, time_utils.py:1044), then Gaussians are
re-expressed relative to their deformed neighbor nodes with those rotations
(time_utils.py:1196–1213). Rigidity errors in the node field therefore translate directly
into inconsistent rotations and stretched Gaussian layouts.

## 4. Execution Environment

| Item | Value |
|---|---|
| GPU | NVIDIA Quadro RTX 5000, 16 GB (Turing, sm_75) |
| Driver / CUDA | 570.133.07 / CUDA 12.4 toolkit (driver supports 12.8) |
| OS | Ubuntu 24.04.1 LTS |
| Python / PyTorch | 3.9.23 / 2.4.1+cu124 |
| Key deps | pytorch3d 0.7.8 (source build), diff-gaussian-rasterization (patched for gcc 13), simple-knn |
| Dataset | D-NeRF synthetic (jumpingjacks, hook, mutant; 400×400 training resolution, `--resolution 2`) |

Full setup log with all build errors and fixes: `docs/SETUP.md`.

## 5. Reproduction

`<TODO: table paper-vs-ours PSNR/SSIM/LPIPS for 3 scenes; wall-clock; GPU memory>`

## 6. Failure Case Analysis

### 6.1 Failure A: Laplacian-initialized ARAP under large rotational drags

Hypothesis: `arap_from_init` (the GUI default) fails under large rotational drags because
its linear Laplacian-editing initialization cannot represent rotation (classic
shrinkage/shear artifact) and only 3 local-global ARAP iterations follow — too few to
recover; `arap_iterative` tracks the same drag applied in small increments and does not
exhibit the failure.

`<TODO: protocol, angle sweep 15/45/90/135°, node/Gaussian rigidity metrics, image diffs,
visual grid, failure-onset angle>`

### 6.2 Failure B: Sensitivity to control point count

`<TODO: node_num ∈ {64,128,512,1024,2048}: PSNR/SSIM/LPIPS, FPS, training time, edit
quality; quality-vs-cost curves; failure regimes at both extremes>`

## 7. Improvement: Progressive Drag Scheduling

`<TODO: split a large drag into N warm-started sub-steps inside the one-shot
from_init workflow; ablate N ∈ {2,4,8}>`

## 8. Before/After Comparison

`<TODO: tables + figures at each rotation angle>`

## 9. Limitations

`<TODO>`

## 10. Use of Generative AI

`<TODO: summarize from docs/AI_USAGE_LOG.md>`

## 11. Discussion

`<TODO>`
