"""Line plots for Failure A / Phase 3: rigidity metrics and image deviation vs drag angle.

Usage: python scripts/plot_failure_a.py --dir results/failureA/jumpingjacks
Writes failureA_curves.png (3 panels: edge stretch p95, gaussian stretch p95 (log),
PSNR vs reference) into --dir.
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import LABEL, COLOR, ORDER, apply_style
apply_style()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.dir, 'metrics.csv'))
    df['key'] = df['mode'] + df['tag'].fillna('').astype(str)
    im = pd.read_csv(os.path.join(args.dir, 'image_metrics.csv'))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for key in ORDER:
        sub = df[df['key'] == key].sort_values('angle')
        if sub.empty:
            continue
        axes[0].plot(sub['angle'], sub['edge_stretch_region_p95'], 'o-',
                     color=COLOR[key], label=LABEL[key], ms=4)
        axes[1].semilogy(sub['angle'], sub['gs_stretch_p95'], 'o-',
                         color=COLOR[key], label=LABEL[key], ms=4)
    axes[0].set_xlabel('drag rotation angle (deg)')
    axes[0].set_ylabel('node edge stretch, region p95')
    axes[0].set_title('(a) Control-graph rigidity violation')
    axes[0].grid(alpha=.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel('drag rotation angle (deg)')
    axes[1].set_ylabel('Gaussian neighbor stretch p95 (log)')
    axes[1].set_title('(b) Gaussian-level tearing (after LBS propagation)')
    axes[1].grid(alpha=.3, which='both')
    axes[1].axhline(2.0, color='gray', ls='--', lw=1)
    axes[1].text(6, 2.2, 'visible-artifact level', fontsize=8, color='gray')

    for key in ORDER:
        if key == 'iterative':
            continue
        sub = im[(im['mode'] + im['angle'].astype(str).str.replace(r'.*', '', regex=True)) .notna() & (im['mode'] == key)]
        sub = im[im['mode'] == key].sort_values('angle')
        if sub.empty:
            continue
        axes[2].plot(sub['angle'], sub['psnr_vs_ref'], 'o-', color=COLOR[key],
                     label=LABEL[key], ms=4)
    axes[2].set_xlabel('drag rotation angle (deg)')
    axes[2].set_ylabel('PSNR vs iterative reference (dB)')
    axes[2].set_title('(c) Image deviation from reference')
    axes[2].grid(alpha=.3)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(args.dir, 'failureA_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
