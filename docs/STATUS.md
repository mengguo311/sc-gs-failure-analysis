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
- Phase 4: paper/paper.tex (8-page LaTeX report) written with real numbers and 6 figures;
  README.md with per-artifact reproduction commands.

**Remaining**
- Author info placeholders in paper §1 (human input required).
- Optional polish: render draft to PDF; edit-test hook/mutant for breadth; 800×800
  final-quality figure renders.

**Blockers**: none.

## 2026-07-30 — submission preparation

**Done**
- Report compressed to the course limit (8 pages of 4–8); keywords, a "Summary of the
  target paper" section, and explicit "execution environment" / "understanding the
  method" headings added so all ten required items map onto section titles.
- Author block filled (MENG GUO, 48-266606, Department of Creative Informatics).
- Code published: https://github.com/mengguo311/sc-gs-failure-analysis (public, 4 tags),
  URL printed on the report's first page. Upstream SC-GS is no longer vendored; it is
  reconstructed by `scripts/setup_upstream.sh` + `patches/`.
- AI usage log records the AI-drafted / author-proofread workflow the instructor
  confirmed, with a per-section proofreading checklist.

**Remaining (author)**
- Laboratory name and own research topic in the author block (two `<TODO>` fields).
- Proofread the report using `docs/PROOFREADING_GUIDE.md`, tick the checklist in
  `docs/AI_USAGE_LOG.md`, recompile, confirm page count stays 4–8.
- Submit to UTOL by 2026-07-31 23:59 JST: `paper/paper.pdf`, the GitHub URL, and
  `docs/AI_USAGE_LOG.md`.
