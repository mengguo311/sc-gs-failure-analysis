# STATUS

## 2026-07-28 — All experimental phases complete; paper drafted

**Done**
- Phase 0: environment (docs/SETUP.md).
- Phase 1: 3/3 scenes reproduced within ±0.5 dB (jumpingjacks +0.40, hook −0.13,
  mutant −0.16). Tags: `repro-done`.
- Phase 2A: Failure A characterized — from_init onset 45–60°, mechanism traced to
  rotation-blind Laplacian init + NUM_ITER=3; secondary propagation-stage artifact
  found (centroid protocol). Tag: `failure-analysis-a-done`.
- Phase 2B: node_num sweep {64,128,512,1024,2048} — reconstruction flat, editing fails
  at both extremes (articulation vs latency+leakage); new leakage metric. Tag:
  `failure-analysis-done`.
- Phase 3: progressive drag scheduling measured (onset 60°→110° for N=1→8, ≤0.8 s);
  docs/IMPROVEMENT.md.
- Phase 4 (mostly): paper/draft.md fully written with real numbers and 4 figures;
  README.md with per-artifact reproduction commands.

**Remaining**
- Author info placeholders in paper §1 (human input required).
- Optional polish: render draft to PDF; edit-test hook/mutant for breadth; 800×800
  final-quality figure renders.

**Blockers**: none.
