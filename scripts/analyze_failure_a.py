"""Analyze Failure-A results: image-space diffs vs the arap_iterative reference + visual grids.

Reads results/<...>/renders/{mode}_a{angle}_cam{c}.png produced by edit_headless.py,
computes PSNR/SSIM/LPIPS(alex) of every non-reference mode against the iterative reference
at the same angle/camera, writes image_metrics.csv, and assembles a grid figure
(rows = modes, cols = angles) plus per-angle |difference| heatmaps.

Usage: python scripts/analyze_failure_a.py --dir results/failureA/jumpingjacks [--cam 0]
"""
import os
import re
import sys
import argparse
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'SC-GS'))

import torch
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_img(path):
    img = imageio.imread(path).astype(np.float32) / 255.0
    return torch.from_numpy(img[..., :3]).permute(2, 0, 1)[None].cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--cam', type=int, default=0)
    ap.add_argument('--reference', default='iterative')
    ap.add_argument('--fig_prefix', default='failureA')
    args = ap.parse_args()

    rdir = os.path.join(args.dir, 'renders')
    pat = re.compile(r'^(?P<mode>.+?)_a(?P<angle>\d+)_cam(?P<cam>\d+)\.png$')
    entries = {}
    for fn in sorted(os.listdir(rdir)):
        m = pat.match(fn)
        if not m or int(m.group('cam')) != args.cam:
            continue
        entries.setdefault(m.group('mode'), {})[int(m.group('angle'))] = os.path.join(rdir, fn)

    if args.reference not in entries:
        sys.exit(f'reference mode "{args.reference}" not found; have {list(entries)}')

    import lpips as lpips_lib
    lpips_alex = lpips_lib.LPIPS(net='alex').cuda()
    from utils.image_utils import psnr as psnr_fn
    from utils.image_utils import ssim as ssim_fn

    angles = sorted(entries[args.reference].keys())
    modes = [m for m in entries if m != args.reference]

    csv_path = os.path.join(args.dir, 'image_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('mode,angle,cam,psnr_vs_ref,ssim_vs_ref,lpips_alex_vs_ref\n')
        for mode in modes:
            for a in angles:
                if a not in entries[mode]:
                    continue
                img = load_img(entries[mode][a])
                ref = load_img(entries[args.reference][a])
                p = psnr_fn(img, ref).mean().item()
                s = ssim_fn(img, ref).mean().item()
                l = lpips_alex(img * 2 - 1, ref * 2 - 1).item()
                f.write(f'{mode},{a},{args.cam},{p:.4f},{s:.5f},{l:.5f}\n')
                print(f'{mode:>22s} a={a:3d}: PSNR {p:6.2f}  SSIM {s:.4f}  LPIPS {l:.4f}')

    # Visual grid: rows = reference + modes, cols = angles
    all_modes = [args.reference] + modes
    fig, axes = plt.subplots(len(all_modes), len(angles),
                             figsize=(3 * len(angles), 3 * len(all_modes)), squeeze=False)
    for r, mode in enumerate(all_modes):
        for c, a in enumerate(angles):
            ax = axes[r][c]
            ax.axis('off')
            if a in entries[mode]:
                ax.imshow(imageio.imread(entries[mode][a]))
            if r == 0:
                ax.set_title(f'{a}°', fontsize=14)
            if c == 0:
                ax.text(-0.06, 0.5, mode, transform=ax.transAxes, rotation=90,
                        va='center', ha='center', fontsize=12)
    plt.tight_layout()
    grid_path = os.path.join(args.dir, f'{args.fig_prefix}_grid_cam{args.cam}.png')
    plt.savefig(grid_path, dpi=120, bbox_inches='tight')
    print('wrote', grid_path)

    # Difference heatmaps vs reference
    fig, axes = plt.subplots(len(modes), len(angles),
                             figsize=(3 * len(angles), 3 * len(modes)), squeeze=False)
    for r, mode in enumerate(modes):
        for c, a in enumerate(angles):
            ax = axes[r][c]
            ax.axis('off')
            if a in entries[mode]:
                img = imageio.imread(entries[mode][a]).astype(np.float32) / 255.
                ref = imageio.imread(entries[args.reference][a]).astype(np.float32) / 255.
                diff = np.abs(img[..., :3] - ref[..., :3]).mean(-1)
                ax.imshow(diff, cmap='inferno', vmin=0, vmax=0.5)
            if r == 0:
                ax.set_title(f'{a}°', fontsize=14)
            if c == 0:
                ax.text(-0.06, 0.5, f'{mode} vs ref', transform=ax.transAxes, rotation=90,
                        va='center', ha='center', fontsize=12)
    plt.tight_layout()
    diff_path = os.path.join(args.dir, f'{args.fig_prefix}_diff_cam{args.cam}.png')
    plt.savefig(diff_path, dpi=120, bbox_inches='tight')
    print('wrote', diff_path)


if __name__ == '__main__':
    main()
