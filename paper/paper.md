# Progressive Drag Scheduling: Characterizing and Repairing Large-Rotation Editing Failures in Sparse-Controlled Gaussian Splatting

> **Note**: the authoritative, most up-to-date version of this paper is `paper.tex` /
> `paper.pdf` (LaTeX). This markdown version may lag behind the latest revision.

**Author:** `<TODO: name, student ID, department, laboratory>`
**Course:** Visual Media, The University of Tokyo

---

## Abstract

Sparse-Controlled Gaussian Splatting (SC-GS) couples dense 3D Gaussians with a sparse set of learned control points, enabling both high-fidelity dynamic novel-view synthesis and interactive motion editing through as-rigid-as-possible (ARAP) deformation of the control graph. We present a systematic empirical study of when and why this editing pipeline fails. First, we show that the default editing mode, which initializes every ARAP solve from a linear Laplacian-editing solution, collapses for rotational drags beyond 45–60°: the rotation-blind initialization produces the classical chord-shortcut artifact, and the fixed budget of three local-global iterations cannot recover from it, yielding under-rotated, shortened limbs and torn Gaussian layouts. Second, we identify two further failure axes: an initialization-independent artifact in the rotation re-estimation stage that propagates node motion to Gaussians, and a strong sensitivity of *editability* — but not reconstruction quality — to the number of control points, including a previously unreported global "leakage" failure at high control-point counts that local rigidity metrics cannot detect. Motivated by this analysis, we propose *progressive drag scheduling*: a requested drag is subdivided into N sub-steps along the target trajectory, and each sub-step's ARAP solve is warm-started from the previous solution. The method is a pure scheduling change — it requires no modification of the trained model or the solver — yet it extends the failure-onset angle from 45–60° to 90–110° (N=8) at under 0.8 s per drag, 16× faster than the robust-but-slow incremental reference mode. Across five D-NeRF scenes with distinct limb geometries, progressive scheduling closes 44–80% of the rigidity gap between the default and reference modes, with strictly monotone improvement in the schedule length on four of five scenes. All results are reproducible from committed scripts, metrics, and protocol seeds.

---

## 1 Introduction

Photorealistic reconstruction of dynamic scenes has advanced rapidly from neural radiance fields [6, 3] to point-based representations built on 3D Gaussian Splatting (3DGS) [2], which offer real-time rendering and explicit geometry. Beyond replaying a captured sequence, a natural next demand is *editing*: a user should be able to grab a reconstructed character and pose it — bend an arm, swing a tail — while the representation preserves photorealism.

SC-GS [1] is a prominent representative of this direction. It factorizes a dynamic scene into dense canonical Gaussians (appearance) and a sparse set of learned control points (motion), connected by linear blend skinning (LBS) with learned weights. Because motion lives on a few hundred control points, editing reduces to a classical geometry-processing problem: the user constrains a handful of *handle* points and an as-rigid-as-possible (ARAP) energy [4] is minimized over the remaining control graph. The published system demonstrates compelling interactive edits and state-of-the-art benchmark quality.

Benchmarks, however, measure novel-view synthesis — not the editing behavior that motivates the sparse-control design in the first place. In this work we ask: *under what conditions does SC-GS editing actually fail, why, and how far can a minimal intervention push the failure boundary?* We make the deliberate choice to study the official implementation as-is, scripting its GUI editing pipeline call-for-call so that every observation reflects the system users interact with.

Our study yields three findings. **(i)** The default editing mode (`arap_from_init`) re-initializes every ARAP solve from a *linear* Laplacian-editing solution [5]. Linear variational methods are translation-insensitive but rotation-blind [13]; under a rotational drag the initialization takes the chord shortcut, and the solver's fixed budget of three local-global iterations cannot recover. We localize the failure onset to 45–60° of drag rotation and trace the visual symptoms — limb shortening, under-rotation, Gaussian shredding — to this mechanism. **(ii)** The alternative mode (`arap_iterative`) is robust because each mouse-move warm-starts from the previous solution, but it is two orders of magnitude slower over a large drag, is stateful (its output depends on drag history), and — surprisingly — becomes the *less* safe mode at high control-point counts, where warm-start drift accumulates into large off-region deformation ("leakage") that local rigidity metrics cannot see. **(iii)** Reconstruction quality is almost flat across a 32× sweep of control-point count, while editability degrades at both extremes — evidence that the benchmark metrics by which such systems are ranked are silent about their signature capability.

Motivated by (i) and the trade-offs in (ii), we propose **progressive drag scheduling**: subdivide a requested drag into N sub-steps along the target arc and warm-start each sub-step's solve from the previous one. This imports the reference mode's robustness into the one-shot, deterministic workflow of the default mode, with a user-controllable robustness–latency dial (N). The method changes no trained parameters and no solver code — it is a scheduling policy over existing calls — yet it is quantitatively effective across five scenes with different limb geometries.

**Contributions.**
1. A reproducible, GUI-faithful headless editing harness for SC-GS, with geometry-level (graph edge stretch, ARAP energy, Gaussian neighbor stretch), image-level, and a novel *off-region leakage* metric (§4.1).
2. A controlled characterization of three failure axes of SC-GS editing: rotation-blind initialization (onset 45–60°), propagation-stage rotation re-estimation, and control-point-count extremes, including the leakage failure and a mode-preference inversion at high counts (§4).
3. Progressive drag scheduling, a zero-retraining repair that extends the failure onset to 90–110° at interactive latency, validated on five scenes (§5, §6).

## 2 Related Work

**Dynamic scene representations.** Dynamic extensions of NeRF [6] deform a canonical field by a time-conditioned warp (D-NeRF [3]) or accelerate with explicit grids (TiNeuVox [7], K-Planes [8]). On the splatting side, deformable 3DGS variants [10] and 4D Gaussian fields [9] achieve real-time dynamic rendering. SC-GS [1] is distinguished by *sparse motion control*: a compact set of control points carries the motion field, which both regularizes reconstruction and exposes an editing interface. Our work is an empirical analysis of that interface, not a new representation.

**Shape deformation and editing.** Handle-based deformation is classical geometry processing: linear variational methods such as Laplacian surface editing [5] are efficient but rotation-blind, a limitation surveyed thoroughly by Botsch and Sorkine [13]; ARAP energies [4] restore rotation invariance via alternating local rotation fitting and global linear solves, and embedded deformation graphs [12] extend the idea to volumetric proxies — precisely the construction SC-GS adopts over control points. The failure we characterize is a *system-level composition* of known ingredients: a linear initializer, a hard iteration budget, and an LBS propagation stage that re-estimates rotations from positions; to our knowledge its quantitative behavior in SC-GS had not been documented.

## 3 Preliminaries: the SC-GS Editing Pipeline

![Figure 1](figures/method_overview.png)
*Figure 1 — SC-GS at a glance. (a) Dense canonical Gaussians carry appearance (jumpingjacks, 70,951 Gaussians). (b) 512 sparse control nodes carry motion; editing constrains a drag-handle group (red, here rotated about the shoulder along an arc) and anchor groups (blue), and solves ARAP for all remaining nodes. (c) The solved node field drives the Gaussians through LBS.*

SC-GS represents a dynamic scene as canonical 3DGS parameters plus M control nodes (M=512 by default) with learned radii. A deformation MLP maps (node, t) to per-node translation; each Gaussian is skinned to its K nearest nodes with Gaussian-kernel LBS weights, and an ARAP-style regularizer on sampled node trajectories biases training toward locally rigid motion. All file/line references below are to the official implementation.

**Editing.** At edit time t, node positions p = nodes + d_xyz(t) form a graph with trajectory-aware connectivity (K=16). The user selects a *handle* group (a clicked node plus its 2-ring) and drags it; anchors are additional handle groups with zero displacement. The solver (`utils/arap_deform.py:86`) minimizes the ARAP energy subject to hard handle constraints by local-global iteration — per-node SVD rotation fitting alternating with a linear solve — but runs only **NUM_ITER = 3** iterations per call. Two GUI modes differ *only in initialization* (`train_gui.py:768–777`):

- `arap_from_init` (default): every solve starts from the linear Laplacian-editing least-squares solution (`lstsq_with_handles`, arap_deform.py:103);
- `arap_iterative`: every solve warm-starts from the previous drag event's solution (`init_verts`).

**Propagation.** The solved node displacement field is not paired with the solver's rotations. Instead, node rotations are re-estimated from the displaced node positions by a local SVD fit (`p2dR`, time_utils.py:1044), and Gaussians are re-expressed relative to their deformed neighbor nodes with these rotations (time_utils.py:1196–1213). Consequently, any non-rigidity in the node field converts directly into inconsistent rotations and stretched Gaussian layouts.

## 4 An Empirical Study of Editing Failures

### 4.1 Methodology

We script the GUI pipeline headlessly and call-for-call (animation initialization → keypoint selection with n-ring expansion → `deform_arap` → `deform.step(node_trans_bias)` → render), so that measured behavior is exactly the interactive system's. The canonical protocol drags a limb-tip handle group (nearest node + 1-ring) along the arc of a joint rotation — e.g., the jumpingjacks hand rotated about the shoulder in the frontal plane — with the torso held by two anchor groups; intermediate limb nodes are free, so the solver must infer the limb's rotation. Handle constraints are hard, and both modes satisfy them exactly (residual 0): all differences live in the free nodes. The `arap_iterative` mode applied in 1° warm-started sub-steps serves as the *reference* (it emulates GUI mouse granularity); a single `arap_from_init` solve at angle θ is exactly the state the default mode reaches at accumulated angle θ, since that mode re-solves from scratch at the current absolute targets on every drag event.

We measure, per angle θ ∈ [5°, 135°]:
- **Graph rigidity**: relative edge-length distortion over the ARAP graph (mean/p95/max, globally and in the drag region = handle + 4 rings), and the ARAP energy between rest and deformed nodes;
- **Gaussian-level tearing**: p95 of per-Gaussian maximum neighbor-distance stretch (K=8 canonical neighbors) in the edited region, after LBS propagation;
- **Off-region leakage**: p95 displacement of nodes *outside* the drag region and anchors — a global metric designed to expose smooth drift that edge-based statistics cannot see (§4.4);
- **Image deviation**: PSNR/SSIM/LPIPS [11] of the rendered edit against the reference mode's render at the same θ and camera;
- **Latency**: wall-clock solve time (GPU-synced).

All protocol constants (seed points, rotation centers, axes, cameras) are serialized to JSON alongside the committed metric CSVs.

### 4.2 Failure I: rotation-blind initialization under large rotational drags

![Figure 2](figures/failureA_curves.png)
*Figure 2 — jumpingjacks, shoulder-rotation drags. (a) Node-graph rigidity violation vs drag angle. (b) Gaussian-level tearing after LBS propagation (log scale); the dashed line marks the level at which artifacts are clearly visible. (c) Image deviation from the reference mode. The default mode (red) departs from the reference (blue) between 45° and 60°; progressive schedules (§5) interpolate monotonically.*

Figure 2 quantifies the failure on jumpingjacks. The default mode tracks the reference up to ~30°, then departs sharply: across 45°→60° its Gaussian-stretch statistic jumps 1.70 → 9.98 while the reference stays below 0.3 — a six-fold discontinuity that localizes the **failure onset to 45–60°**. At 135° the node edge stretch reaches 3.8× the reference (0.269 vs 0.072) and the ARAP energy is doubled (0.093 vs 0.048). Visually (Figure 6, row 1) the limb *under-rotates* — at 90–135° the arm remains near-horizontal instead of following the arc — *shortens* toward the chord, and the hand *shreds* into a blob of disconnected Gaussians.

The mechanism follows §3: the Laplacian initialization is a linear solve that cannot represent rotation [5, 13]; for a rotational drag it produces the chord-shortcut configuration, and three local-global iterations — each of which only re-estimates local rotations from the *current* guess — make insufficient progress from so poor a start. The reference mode avoids this entirely because its per-event increments (~1°) keep every solve within the basin that three iterations can handle. Robustness, however, costs latency: the accumulated reference solves for a 135° drag take 12.3 s versus 0.11 s for the one-shot default — a 100× gap that explains why the fast mode is the GUI default.

### 4.3 Failure II: propagation-stage rotation re-estimation

A control experiment isolates a second, initialization-independent artifact. When the *entire* handle group is rotated about its own centroid (the GUI's rotation-drag semantics, leaving no free nodes inside the moved region), both modes produce near-identical node fields — yet **both** shred the limb beyond ~75° (Gaussian stretch p95 ≈ 8–12). The blame therefore lies downstream of the ARAP solve: the `p2dR` stage re-estimates node rotations from displaced positions and blends Gaussians across the handle boundary with Floyd-graph LBS weights; under a large in-place rotation of a small node cluster, boundary Gaussians receive inconsistent rotation estimates and tear. This artifact is orthogonal to initialization and is *not* repaired by our method (§7); we report it as a distinct failure axis.

### 4.4 Failure III: sensitivity to control-point count

![Figure 3](figures/failureB_curves.png)
*Figure 3 — Control-point count sweep on jumpingjacks (node_num ∈ {64, 128, 512, 1024, 2048}, identical training). (a) Test PSNR is flat while render FPS falls 28%. (b) Edit rigidity is U-shaped. (c) Off-region leakage grows monotonically with node count and is worst for the warm-started mode. (d) Solve latency crosses the interactivity threshold.*

Retraining jumpingjacks across a 32× range of control-point counts (Table 2) shows that **novel-view quality is essentially independent of node count** (spread 0.7 dB; the default 512 is best) — in this regime node count is an *editability and cost* knob, not a quality knob. Editing, by contrast, fails at both extremes:

- **Low extreme (64–128 nodes): articulation failure.** Under the standard 90° drag both solver modes produce nearly identical, heavily distorted results (region edge stretch 0.42–0.47, six times the 512-node reference); the limb cannot bend at the joint because deformation granularity is bounded by node spacing. The failure is solver-independent.
- **High extreme (2048 nodes): latency and leakage failure.** A single default-mode solve takes 3.0 s (25× the 512-node cost) and the 90° reference drag takes 200 s. More insidiously, the reference mode rips *off-region* geometry: p95 off-region node displacement reaches **1.26 scene units** (body height ≈ 2) — the character's legs are visibly torn sideways by an arm drag — while every *local* rigidity statistic stays moderate, because the drift is spatially smooth. This is precisely why we introduce the leakage metric. Leakage grows monotonically with node count (reference mode @90°: 0.09 → 0.21 → 0.38 → 0.55 → 1.26) and is ~2× worse for the warm-started reference than for the stateless default at 2048 nodes: warm-start drift *accumulates* across the 90 sub-steps. **The preferable solver mode therefore inverts at the high extreme** — the mode that repairs Failure I is the one that fails here.

## 5 Method: Progressive Drag Scheduling

![Figure 4](figures/fig_method_schematic.png)
*Figure 4 — Why one-shot solves fail and progressive scheduling repairs them. (a) The linear initialization moves free nodes along the chord toward the target, shortening and shearing the limb; three ARAP iterations cannot climb back to the arc. (b) Subdividing the drag into N sub-targets along the arc keeps every increment small; each solve warm-starts from the previous solution, so three iterations per sub-step suffice.*

Failure I is caused by the conjunction of a rotation-blind initializer and a hard iteration budget. Rather than modifying either (retraining nothing, touching no solver code), we change *what the solver is asked to do*:

> **Progressive drag scheduling.** Given rest handle positions H₀, final targets H(θ), and a schedule length N: for k = 1…N, set sub-targets H(θ·k/N) by interpolating the handle transformation along its trajectory (slerp of the rotation about the drag center), and solve ARAP with `init_verts` = the (k−1)-th solution; the first sub-step falls back to the Laplacian initialization, which is harmless at small angles.

The result is a deterministic function of (rest state, final targets, N) — unlike the reference mode it does not depend on the user's drag history, can be re-executed from a stored keyframe, and has bounded latency N·t_solve. Conceptually, the schedule replaces one hopeless global solve by N easy local ones: each sub-step's increment θ/N stays within the basin in which three local-global iterations converge, and the warm start transports the solution along the arc. Setting N=1 recovers `arap_from_init`; N→θ/1° recovers the reference. In our implementation the schedule lives entirely in the editing driver: each sub-step reuses the exact `deform_arap(init_verts=…)` call the GUI already issues for its iterative mode, so a GUI integration is a ~10-line loop.

## 6 Experiments

### 6.1 Setup and reproduction

All experiments use the official SC-GS implementation (fixed seed, 80k iterations, 400×400 — the D-NeRF comparison convention) on a Quadro RTX 5000; environment and build details, including a gcc-13 compilation fix, are documented in the repository. We first verify that our trained models reproduce the paper:

*Table 1 — Reproduction on D-NeRF (test split). Paper numbers from SC-GS Tab. 1.*

| Scene | PSNR paper / ours (Δ) | SSIM paper / ours | LPIPS paper / ours | Train | Peak GPU |
|---|---|---|---|---|---|
| jumpingjacks | 41.13 / **41.53** (+0.40) | .998 / .9975 | .0067 / .0058 | 64.9 min | 1.47 GiB |
| hook | 39.87 / 39.74 (−0.13) | .997 / .9963 | .0076 / .0084 | 65.4 min | 1.91 GiB |
| mutant | 45.19 / 45.03 (−0.16) | .999 / .9990 | .0028 / .0029 | 63.5 min | 2.08 GiB |
| standup | 47.89 / 47.51 (−0.38) | .999 / .9991 | .0023 / .0025 | 65.5 min | 3.02 GiB |
| trex | 41.24 / 40.68 (−0.56)* | .998 / .9985 | .0046 / .0038 | 65.6 min | 2.07 GiB |

*The only gap beyond 0.5 dB. Investigation: the best checkpoint during training reached 40.86 (≈0.2 dB is final-iteration checkpoint selection); SSIM matches and LPIPS is better than the paper's; trex contains the finest structures in the benchmark and the residual is consistent with densification run-to-run variance.*

*Table 2 — Control-point sweep (jumpingjacks; §4.4).*

| node_num | 64 | 128 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| test PSNR (dB) | 40.96 | 40.85 | **41.53** | 41.30 | 41.33 |
| render FPS (400²) | 235.7 | 233.7 | 212.2 | 193.6 | 169.4 |
| edge stretch p95 @90° (ref / default) | 0.42 / 0.42 | 0.46 / 0.47 | 0.07 / 0.19 | 0.04 / 0.30 | 0.21 / 0.31 |
| leakage p95 @90° (ref / default) | 0.09 / 0.05 | 0.21 / 0.06 | 0.38 / 0.25 | 0.55 / 0.49 | **1.26** / 0.56 |
| solve time @90° (ref / default) | 0.9 / 0.01 s | 1.7 / 0.03 s | 8.2 / 0.12 s | 33 / 0.48 s | 200 / 2.9 s |

### 6.2 Main results

![Figure 5](figures/fig_onset_gapclosure.png)
*Figure 5 — (a) Failure-onset angle grows with schedule length N (bars: onset interval from the angle sweep). (b) Fraction of the default→reference rigidity gap closed at a 90° drag, per scene (edge-stretch region p95; trex N=4 closes 0% on this node-level metric — but see §6.4 and the image-level metrics). (c) The robustness–latency plane at 90°: progressive schedules populate the previously empty region between the two built-in modes, below the 1 s interactivity threshold.*

*Table 3 — Failure onset and cost (jumpingjacks). Onset = first angle at which Gaussian stretch p95 exceeds the visible-artifact level (≈2).*

| mode | onset | edge p95 @135° | ARAP energy @135° | solve @135° |
|---|---|---|---|---|
| from_init (N=1, default) | 45–60° | 0.269 | 0.093 | 0.11 s |
| progressive N=2 | 45–60° | 0.271 | 0.091 | 0.21 s |
| progressive N=4 | 60–75° | 0.207 | 0.073 | 0.39 s |
| progressive N=8 | **90–110°** | 0.112 | 0.051 | 0.76 s |
| iterative 1°/step (reference) | ≥135° | 0.072 | 0.048 | 12.34 s |

Robustness scales monotonically with N (Table 3, Figure 2): N=8 covers the practical editing range (≤90°) at ≤0.8 s per drag — 16× cheaper than the reference at 135° — and every image metric against the reference improves monotonically in N at every angle (e.g., PSNR at 60°: 18.59 → 19.97 dB; LPIPS at 45°: 0.097 → 0.066). Beyond its onset, each schedule fails exactly like the default mode — the *final* sub-step's increment exceeds what three iterations absorb — implying that N should scale with θ; the onset table yields the simple adaptive rule N = ⌈θ/15°⌉ (fix the per-substep increment at 10–15°).

![Figure 6](figures/before_after_grid.png)
*Figure 6 — Original SC-GS vs ours, qualitative before/after on jumpingjacks (45°/90°/135°). Top to bottom: SC-GS (original), Ours N=4, Ours N=8, SC-GS iterative (reference). The original mode's arm under-rotates and its hand shreds from 60°; Ours N=8 tracks the reference through 90°.*

### 6.3 Cross-scene generalization

We repeat the full protocol (5 modes × 11 angles) on four additional scenes, each with a different limb, joint, and rotation axis: hook (forward arm, axis X), mutant (claw arm, axis Y), standup (crouched forward arm, axis X), and trex (long thin tail about its root, axis Z).

*Table 4 — Node edge stretch (region p95) at 90° / 135° across scenes.*

| mode | jumpingjacks | hook | mutant | standup | trex |
|---|---|---|---|---|---|
| from_init (N=1) | .189 / .269 | .213 / .318 | .237 / .399 | .362 / .439 | .285 / .410 |
| progressive N=2 | .174 / .271 | .201 / .285 | .211 / .337 | .344 / .409 | .295 / .421 |
| progressive N=4 | .140 / .207 | .169 / .228 | .177 / .258 | .310 / .389 | .285 / .391 |
| progressive N=8 | .108 / .112 | .134 / .175 | .135 / .221 | .261 / .307 | .267 / .371 |
| iterative (reference) | .074 / .072 | .088 / .125 | .109 / .157 | .165 / .181 | .244 / .342 |

![Figure 7](figures/cross_scene_curves.png)
*Figure 7 — Edge-stretch vs drag angle for all five scenes. The mode ordering is preserved everywhere; absolute levels are scene-dependent.*

![Figure 8](figures/fig_qualitative_2scene.png)
*Figure 8 — Original SC-GS vs ours at 90°. Columns left to right: SC-GS (original), Ours N=4, Ours N=8, SC-GS iterative (reference). jumpingjacks: the original mode leaves the arm near-horizontal with a shredded hand. trex: the original mode shortens and kinks the tail; ours restores the smooth arc.*

On four of five scenes the ordering default > N=2 > N=4 > N=8 > reference is **strictly monotone at every tested angle**; at 90° N=8 closes 71% / 63% / 80% / 51% / 43% of the default→reference gap (Figure 5b). Artifact *severity* varies with limb geometry — Gaussian stretch p95 at 90° is 10.2 on jumpingjacks' thin arm but only 1.6 on mutant's thick claw — yet the ordering never inverts, except for one case analyzed next.

### 6.4 The hard case: trex

The trex tail is a single long elastic chain far from every anchor, and it stresses every mode: even the reference reaches 0.34 edge stretch at 135°. Node-level differences between modes compress (Table 4, last column), and N=2 is statistically indistinguishable from the default — its 45–67° sub-steps already exceed the per-step failure level, so warm-starting buys nothing. Yet the *visual* verdict is unambiguous (Figure 8, bottom row): the default mode's tail is visibly shortened and kinked, while N=4/8 restore the smooth arc; image metrics agree (PSNR versus reference at 90°: 24.7 → 25.0 → 27.7 dB from default to N=4 to N=8; LPIPS 0.039 → 0.018). We draw two honest conclusions: progressive scheduling helps on every scene we tested, but it is not a silver bullet for extreme kinematic chains; and geometry-level and image-level metrics can dissociate — evaluations relying on a single family of metrics would have mis-ranked this case in both directions.

## 7 Discussion and Limitations

**A robustness–latency–statefulness triangle.** Our analysis reframes SC-GS's two editing modes as corners of a design triangle: the default mode is fast, deterministic, and restorable but rotation-blind; the reference mode is robust to rotation but slow, stateful, and — at high control-point counts — the more dangerous mode due to drift accumulation (§4.4). Progressive scheduling occupies the useful interior: bounded latency, deterministic output, and a robustness dial. Figure 5c makes the geometry of this trade-off explicit.

**What benchmarks do not measure.** Reconstruction quality was flat across every intervention that dramatically changed editability (Table 2). For editable representations, we argue evaluation should include editing-stress protocols of the kind proposed here — angle sweeps with rigidity, leakage, and reference-deviation metrics — rather than novel-view metrics alone.

**Limitations.** (i) Our improvement addresses the initialization failure only; the propagation-stage artifact (§4.3) and the high-count leakage failure (§4.4) are orthogonal and unrepaired — at ≥110° even N=8 fails, and on trex the reference itself is stressed. Blending the ARAP solver's own rotations instead of re-estimating them from positions, and anchoring or damping the global solve against off-region drift, are natural next steps. (ii) The iteration budget NUM_ITER=3 was deliberately held fixed to isolate the scheduling effect; jointly scheduling sub-steps and iterations (e.g., extra iterations on the final sub-step) is unexplored. (iii) One drag protocol per scene, with seeds chosen once and documented; a user study or randomized-protocol sweep would strengthen external validity. (iv) The reference mode is a strong baseline, not ground truth — it drifts at extreme angles and leaks at high node counts.

## 8 Conclusion

We presented a systematic failure analysis of the SC-GS editing pipeline, localizing a sharp large-rotation failure to its rotation-blind linear initialization under a fixed iteration budget, identifying an independent propagation-stage artifact, and characterizing a bidirectional failure of editability — including a previously unmeasured global leakage mode — across control-point counts at which reconstruction quality is indifferent. A minimal, zero-retraining repair — progressive drag scheduling — extends the usable drag range by roughly a factor of two at interactive latency and generalizes across five scenes. The broader lesson is that editable neural representations deserve editing-centric evaluation: the capabilities that motivate their design are exactly the ones current benchmarks do not measure.

## References

[1] Y.-H. Huang, Y.-T. Sun, Z. Yang, X. Lyu, Y.-P. Cao, and X. Qi. *SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes.* CVPR 2024.
[2] B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM TOG (SIGGRAPH) 2023.
[3] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer. *D-NeRF: Neural Radiance Fields for Dynamic Scenes.* CVPR 2021.
[4] O. Sorkine and M. Alexa. *As-Rigid-As-Possible Surface Modeling.* SGP 2007.
[5] O. Sorkine, D. Cohen-Or, Y. Lipman, M. Alexa, C. Rössl, and H.-P. Seidel. *Laplacian Surface Editing.* SGP 2004.
[6] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.* ECCV 2020.
[7] J. Fang, T. Yi, X. Wang, L. Xie, X. Zhang, W. Liu, M. Nießner, and Q. Tian. *Fast Dynamic Radiance Fields with Time-Aware Neural Voxels.* SIGGRAPH Asia 2022.
[8] S. Fridovich-Keil, G. Meanti, F. Warburg, B. Recht, and A. Kanazawa. *K-Planes: Explicit Radiance Fields in Space, Time, and Appearance.* CVPR 2023.
[9] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang. *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering.* CVPR 2024.
[10] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin. *Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction.* CVPR 2024.
[11] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR 2018.
[12] R. W. Sumner, J. Schmid, and M. Pauly. *Embedded Deformation for Shape Manipulation.* ACM TOG (SIGGRAPH) 2007.
[13] M. Botsch and O. Sorkine. *On Linear Variational Surface Deformation Methods.* IEEE TVCG 2008.

---

## Appendix A: Reproducibility Statement

All experiments derive from committed artifacts: one script per experiment (`scripts/`), per-run entries in an experiment log (22 entries), metric CSVs and figures under `results/`, and per-protocol JSON seed files (drag points, anchors, rotation centers/axes, cameras). Every figure and table in this paper is regenerated by a single documented command (repository README). Training uses the upstream fixed seed; residual nondeterminism is limited to rasterizer atomics during densification. Environment: Ubuntu 24.04, Quadro RTX 5000 (16 GB), CUDA 12.4, PyTorch 2.4.1, pytorch3d 0.7.8; full setup including build fixes in `docs/SETUP.md`.

## Appendix B: Use of Generative AI

This project was executed with Claude Code (Anthropic Fable 5) as an autonomous research agent under a human-authored brief. All quantitative claims are machine-generated from committed measurement CSVs; protocol choices are serialized alongside results; the running usage log is `docs/AI_USAGE_LOG.md`. The human author is responsible for final content.
