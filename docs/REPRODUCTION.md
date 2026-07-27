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
| jumpingjacks | 41.13 | TODO | | .998 | TODO | .0067 | TODO |
| hook | 39.87 | TODO | | .997 | TODO | .0076 | TODO |
| mutant | 45.19 | TODO | | .999 | TODO | .0028 | TODO |

## Cost

| Scene | Wall-clock (train) | Peak GPU mem | it/s (approx) |
|---|---|---|---|
| jumpingjacks | TODO | TODO | ~17-18 (RTX 5000) |
| hook | TODO | TODO | |
| mutant | TODO | TODO | |

## Notes / deviations

- GPU is a Quadro RTX 5000 (16 GB); the paper does not pin a training GPU for D-NeRF.
- Any metric gap > 0.5 dB PSNR will be investigated (LPIPS backbone mismatch is a known
  source of discrepancy between papers; render.py reports both alex and vgg LPIPS).
