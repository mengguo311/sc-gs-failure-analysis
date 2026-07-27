# STATUS

## 2026-07-22 — Phase 0 done, Phase 1 running

**Done**
- Full environment from bare Ubuntu 24.04 host: conda (py3.9), torch 2.4.1+cu124, nvcc 12.4,
  both CUDA submodules built (gcc-13 `cstdint` patch needed, committed in SC-GS branch
  `course-project`), pytorch3d 0.7.8 from source. See docs/SETUP.md.
- D-NeRF dataset downloaded (8 scenes).
- Method understanding notes (docs/METHOD_NOTES.md), code-verified with file:line refs.
- Experiment scripts: train/eval runners, headless ARAP editing script
  (scripts/edit_headless.py) implementing from_init / iterative / progressive modes,
  analysis script for Failure A (scripts/analyze_failure_a.py).

**Running**
- jumpingjacks training (node_num=512, paper config, seed 0): started ~12:55 UTC,
  ~18 it/s at 6%, ETA ~80 min. GPU mem so far 1.3 GiB.

**Next**
- Evaluate jumpingjacks (render.py) → validate end-to-end pipeline.
- Queue hook + mutant (Phase 1) and node_num sweep {64,128,1024,2048} (Phase 2B) sequentially.
- Dump node cloud at edit time, pick drag/anchor seed points for Failure A.

**Blockers**: none.
