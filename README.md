# SC-GS Failure Analysis & Improvement — Visual Media Course Project

**Author**: MENG GUO (48-266606), Department of Creative Informatics,
Graduate School of Information Science and Technology, The University of Tokyo.
**Report**: [`paper/paper.pdf`](paper/paper.pdf) · **AI usage log**:
[`docs/AI_USAGE_LOG.md`](docs/AI_USAGE_LOG.md)

Reproduction, controlled failure analysis, and an improvement of
**SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes** (CVPR 2024,
[official repo](https://github.com/yihua7/SC-GS)) on the D-NeRF dataset.

**Headline results**
- Reproduction: 3/3 scenes within ±0.5 dB of the paper (jumpingjacks 41.53 vs 41.13).
- Failure A: the GUI-default `arap_from_init` editing mode collapses for rotational drags
  beyond **45–60°** (limb shortening, Gaussian shredding); the `arap_iterative` reference
  survives 135° at 100× the solve cost.
- Failure B: reconstruction quality is *flat* across node_num 64→2048 (0.7 dB), but
  editing fails at both extremes — articulation failure below 128, latency (200 s/drag) +
  off-region leakage (legs ripped by an arm drag) at 2048.
- Improvement: **progressive drag scheduling** (N warm-started sub-steps) pushes the
  failure onset 60°→110° (N=8) at ≤0.8 s per drag, zero upstream code changes.

## Repository structure

```
├── README.md               this file
├── SC-GS/                  upstream code (branch course-project; gcc-13 build fix only)
├── scripts/                one script per experiment (see below)
├── results/                committed metrics CSVs + figures + edit renders
│   ├── failureA/           rotation-failure study (+ _centroid supplementary)
│   └── failureB/           node-count study (fps.csv, leakage.csv, edit_n*/)
├── docs/                   SETUP, METHOD_NOTES, REPRODUCTION, FAILURE_ANALYSIS,
│                           IMPROVEMENT, EXPERIMENT_LOG, AI_USAGE_LOG, STATUS
├── paper/                  draft.md + figures/
├── data/                   D-NeRF dataset (not committed)
└── outputs/                trained models + logs (not committed)
```

## Setup

```bash
bash scripts/setup_upstream.sh    # clones pinned SC-GS + applies the gcc-13 build patch
# then follow docs/SETUP.md for the conda environment
```

`scripts/setup_upstream.sh` clones the official SC-GS at the commit used for every
result here and applies `patches/diff-gaussian-rasterization-gcc13.patch` (gcc ≥ 13
no longer transitively includes `<cstdint>`, so the CUDA rasterizer fails to build
without it). The upstream tree is deliberately **not** vendored into this repository.

Full environment record in `docs/SETUP.md` (Ubuntu 24.04, CUDA 12.4, Python 3.9,
torch 2.4.1+cu124, pytorch3d 0.7.8 built from source). Data: D-NeRF `data.zip`
(link in the SC-GS README) extracted to `data/<scene>`.

## Reproducing every table / figure

Environment prefix for all commands: `source scripts/env.sh` (or use the conda env
`scgs`). `MODEL_FLAGS` below abbreviates the standard flag set used everywhere:
`--deform_type node --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask
--local_frame --resolution 2 --W 800 --H 800` plus the matching `--node_num`.

| Artifact | Command |
|---|---|
| Paper Tab. reproduction (per scene) | `bash scripts/train_scene.sh <scene> 512 && bash scripts/eval_scene.sh <scene> 512` |
| Failure B trainings | `bash scripts/queue_remaining_training.sh` |
| Failure A metrics + renders (per mode) | `python scripts/edit_headless.py --model_path outputs/jumpingjacks_n512 --source_path data/jumpingjacks --node_num 512 MODEL_FLAGS --edit_time 0 --drag_point 0.6,0,0.85 --n_rings 1 --anchor_point 0,0,0.4 --anchor_point 0,0,0 --rot_axis 0,1,0 --rot_center 0.22,0,0.55 --angles 5,10,15,22,30,45,60,75,90,110,135 --edit_mode {iterative\|from_init\|progressive --steps N} --out_dir results/failureA/jumpingjacks` |
| Fig. 2 (Failure A curves) | `python scripts/plot_failure_a.py --dir results/failureA/jumpingjacks` |
| Failure A image metrics + grids | `python scripts/analyze_failure_a.py --dir results/failureA/jumpingjacks` |
| Failure B FPS column | `python scripts/bench_fps.py --model_path outputs/jumpingjacks_n<N> --source_path data/jumpingjacks --node_num <N> MODEL_FLAGS --label jj_n<N> --out results/failureB/fps.csv` |
| Failure B edit rows | same as Failure A command with `--model_path outputs/jumpingjacks_n<N> --angles 45,90 --out_dir results/failureB/edit_n<N>` |
| Fig. 3 (Failure B curves) | `python scripts/plot_failure_b.py` |
| Improvement (Phase 3) | Failure A command with `--edit_mode progressive --steps {2,4,8} --tag _N{2,4,8}` |

Every historical run is indexed in `docs/EXPERIMENT_LOG.md` (date, command, output path,
result). Seeds are fixed (upstream seed 0); protocol constants are recorded in
`results/*/meta_*.json`.

## Documents

- `docs/REPRODUCTION.md` — Phase 1 tables (paper vs ours), costs
- `docs/FAILURE_ANALYSIS.md` — Failures A & B, full protocols and numbers
- `docs/IMPROVEMENT.md` — progressive drag scheduling design, ablation, limitations
- `paper/paper.tex` → `paper/paper.pdf` — the submitted report (8 pages)
- `docs/PROOFREADING_GUIDE.md` — claim/evidence map used to proofread the report
- `docs/AI_USAGE_LOG.md` — generative-AI usage log (course requirement)
