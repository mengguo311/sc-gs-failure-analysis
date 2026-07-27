# SC-GS Method Notes (code-verified)

Notes from reading the paper and the official implementation. File/line references are to the
upstream repo at the pinned commit. These notes feed the "Understanding of the method" section
of the paper draft.

## 1. Representation

- **Canonical 3D Gaussians**: standard 3DGS parameters (position, rotation quaternion, scale,
  opacity, SH color) in a canonical space (`scene/gaussian_model.py`).
- **Sparse control nodes**: `ControlNodeWarp` (`utils/time_utils.py:770`) holds `nodes`
  (positions + learnable radius). With `--is_blender`, nodes are initialized from random point
  clouds; `node_num` (default 512) sets the count.
- **Node motion MLP**: a deformation MLP maps `(node, t)` → per-node translation `d_xyz`,
  rotation `d_rotation`, scaling `d_scaling` (`node_deform`, time_utils.py:990).

## 2. LBS-style skinning (nodes → Gaussians)

`ControlNodeWarp.forward` (time_utils.py:1133):

- Per-Gaussian K-nearest control nodes with Gaussian-kernel weights
  (`cal_nn_weight`), optionally modulated by learned per-Gaussian `hyper_dim` features
  (separates spatially-close but topologically-distinct parts).
- Translation: weighted blend of node translations. With `--local_frame`, each node also
  carries a learned local rotation applied to the Gaussian's offset in the node frame
  (time_utils.py:1148-1154): `Ax = R_local[k](x - node_k) + node_k + trans_k`, blended over K nodes.
- Rotation/scaling: weighted blends of node values.
- Training regularizer: ARAP-style loss on node trajectories (`arap_loss`, weight scheduled
  by `lambda_arap_landmarks`) keeps node motion locally rigid.

## 3. Interactive editing pipeline (train_gui.py)

1. `animation_initialize` (train_gui.py:237): evaluates node positions at chosen time t
   (`pcl = nodes + d_xyz(t)`), samples 16-step node trajectories, builds
   `LapDeform(init_pcl=pcl, K=4, trajectory, node_radius)` (lap_deform.py:96) which wraps an
   `ARAPDeformer` (utils/arap_deform.py:38). Graph connectivity comes from
   `cal_connectivity_from_points` using trajectory-aware distances (K=16 neighbors).
2. User picks keypoints: click → nearest node → expand by n-ring graph neighbors (n=2)
   (`add_n_ring_nbs`). `DeformKeypoints` stores the handle group.
3. Drag: translation delta, or with R/Q held a **rotation delta** — the handle group is rotated
   around its centroid (`set_rotation_delta`, train_gui_utils.py:90); per mouse event the
   rotation increment is ~0.05°/px, accumulated into absolute handle targets.
4. Solve: `deform_arap(handle_idx, handle_pos, init_verts)` (train_gui.py:768-777):
   - **`arap_from_init`** (default): `init_verts=None` → ARAPDeformer.deform initializes
     `p_prime` from a **Laplacian-editing linear least squares** solve
     (`lstsq_with_handles(L_opt, L_opt @ verts, ...)`, arap_deform.py:103), then runs only
     **NUM_ITER=3** local-global ARAP iterations (arap_deform.py:110-148).
   - **`arap_iterative`**: `init_verts = previous solution` (warm start), same 3 iterations.
5. Propagation to Gaussians (time_utils.py:1164-1213): given `node_trans_bias`
   (= deformed nodes − nodes(t)), node rotations are **re-estimated from the displaced node
   positions by local SVD** (`p2dR`, K=8 neighbors, trajectory mode) — the ARAP solver's own
   rotations are not reused. Gaussians follow via LBS: each Gaussian is re-expressed relative
   to its (K=32 or Floyd-graph K=8) nearest deformed nodes with the estimated rotations.

## 4. Why `arap_from_init` should fail under large rotations (hypothesis, to be tested)

- The Laplacian-editing initialization is a *linear* solve: it preserves Laplacian coordinates
  under translation but **cannot represent rotations** — the classic shrinkage/shear artifact
  of linear Laplacian editing grows with rotation angle.
- Only 3 local-global ARAP iterations follow. ARAP local-global converges slowly from a poor
  init (each iteration only re-estimates local rotations from the current guess); 3 iterations
  cannot recover from a strongly sheared init at large angles.
- `arap_iterative` warm-starts from the previous drag frame; because GUI drags arrive in
  ~sub-degree increments, each solve starts near the optimum and tracks it.
- Downstream amplification: `p2dR` estimates node rotations from the *deformed node positions*;
  a sheared/shrunken node field produces inconsistent per-node rotations, which the LBS blend
  turns into stretched/torn Gaussian layouts.

## 5. Consequence for the improvement (Phase 3)

Progressive drag scheduling = interpolate handle targets over N sub-steps
(slerp of the rotation), warm-starting each ARAP solve from the previous sub-step's solution
(`init_verts`), i.e., bring `arap_iterative`'s warm-start robustness into the one-shot
`arap_from_init` API. Implemented as a batch-script flag (`--progressive_drag_steps N`).
