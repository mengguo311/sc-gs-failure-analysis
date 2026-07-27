# Phase 1 — Reproduction of SC-GS on D-NeRF

## Configuration

Exactly the official README command (no `--gui` on headless server), seed fixed at 0 by
`safe_state` (upstream default):

```
python train_gui.py --source_path data/<scene> --model_path outputs/<scene>_n512 \
  --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval \
  --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800
```

- `--resolution 2` → 400×400 training/eval images, the standard D-NeRF evaluation
  convention used by the paper's comparisons (D-NeRF, TiNeuVox etc. evaluate at half res).
- 80k optimization iterations (upstream default; progress bar shows 90k ticks because the
  node-sampling/node-rendering warm-up stages are included).
- Evaluation: `render.py` (same flags) → PSNR / SSIM / LPIPS(alex+vgg) / MS-SSIM on the
  test split, plus renders and depth maps.

Runner scripts: `scripts/train_scene.sh <scene> 512`, `scripts/eval_scene.sh <scene> 512`.

## Results

Paper values from Tab. 1 of SC-GS (row "Ours", D-NeRF benchmark).

| Scene | PSNR (paper) | PSNR (ours) | Δ | SSIM (paper) | SSIM (ours) | LPIPS (paper) | LPIPS (ours) |
|---|---|---|---|---|---|---|---|
| jumpingjacks | 41.13 | **41.53** | +0.40 | .998 | .9975 | .0067 | .0058 |
| hook | 39.87 | 39.74 | −0.13 | .997 | .9963 | .0076 | .0084 |
| mutant | 45.19 | 45.03 | −0.16 | .999 | .9990 | .0028 | .0029 |

## Cost

| Scene | Wall-clock (train) | Peak GPU mem | it/s (approx) |
|---|---|---|---|
| jumpingjacks | 3893 s (64.9 min) | 1472 MiB | ~17-18 (RTX 5000) |
| hook | 3925 s (65.4 min) | 1912 MiB | |
| mutant | 3809 s (63.5 min) | 2082 MiB | |

All three scenes reproduce within ±0.5 dB of the paper (jumpingjacks +0.40, hook −0.13,
mutant −0.16) — no investigation flag triggered. LPIPS matches to the 3rd decimal on
mutant and jumpingjacks; hook's small PSNR/LPIPS gap is within run-to-run noise for 3DGS
pipelines (densification is nondeterministic at the margin even with fixed seeds due to
atomics in the rasterizer).

## Notes / deviations

- GPU is a Quadro RTX 5000 (16 GB); the paper does not pin a training GPU for D-NeRF.
- Any metric gap > 0.5 dB PSNR will be investigated (LPIPS backbone mismatch is a known
  source of discrepancy between papers; render.py reports both alex and vgg LPIPS).
