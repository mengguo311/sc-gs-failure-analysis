"""Failure B figures: quality-vs-cost curves over node_num.

Panels: (a) test PSNR + render FPS, (b) edit local rigidity (edge stretch region p95 @90°),
(c) off-region leakage p95 @90°, (d) ARAP solve latency (log).

Usage: python scripts/plot_failure_b.py
Reads results/failureB/{fps.csv,leakage.csv,edit_n*/metrics.csv} + hardcoded PSNR table
(values copied from outputs/*_eval.log, see docs/EXPERIMENT_LOG.md).
Writes results/failureB/failureB_curves.png.
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import LABEL, apply_style
apply_style()

NN = [64, 128, 512, 1024, 2048]
# test PSNR from outputs/jumpingjacks_n*_eval.log (iteration 80000)
PSNR = {64: 40.956, 128: 40.846, 512: 41.532, 1024: 41.297, 2048: 41.331}
TRAIN_MIN = {64: 3866/60, 128: 3842/60, 512: 3893/60, 1024: 3793/60, 2048: 3654/60}

fps = pd.read_csv('results/failureB/fps.csv').set_index('node_num')
leak = pd.read_csv('results/failureB/leakage.csv')

edit = []
for n in NN:
    df = pd.read_csv(f'results/failureB/edit_n{n}/metrics.csv')
    df['node_num'] = n
    edit.append(df)
edit = pd.concat(edit)

fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))

ax = axes[0]
ax.plot(NN, [PSNR[n] for n in NN], 'o-', color='#1f77b4', label='test PSNR (dB)')
ax.set_xscale('log', base=2); ax.set_xticks(NN); ax.set_xticklabels(NN)
ax.set_xlabel('node_num'); ax.set_ylabel('test PSNR (dB)', color='#1f77b4')
ax.set_ylim(40.0, 42.0); ax.grid(alpha=.3)
ax2 = ax.twinx()
ax2.plot(NN, [fps.loc[n, 'fps'] for n in NN], 's--', color='#d62728', label='render FPS')
ax2.set_ylabel('render FPS (400×400)', color='#d62728')
ax.set_title('(a) Reconstruction quality & render speed')

ax = axes[1]
for mode, c in [('iterative', '#1f77b4'), ('from_init', '#d62728')]:
    sub = edit[(edit['mode'] == mode) & (edit['angle'] == 90.0)].sort_values('node_num')
    ax.plot(sub['node_num'], sub['edge_stretch_region_p95'], 'o-', color=c, label=LABEL[mode])
ax.set_xscale('log', base=2); ax.set_xticks(NN); ax.set_xticklabels(NN)
ax.set_xlabel('node_num'); ax.set_ylabel('edge stretch region p95 @90° drag')
ax.set_title('(b) Edit local rigidity violation')
ax.grid(alpha=.3); ax.legend()

ax = axes[2]
for mode, c in [('iterative', '#1f77b4'), ('from_init', '#d62728')]:
    sub = leak[(leak['mode'] == mode) & (leak['angle'] == 90)].sort_values('node_num')
    ax.plot(sub['node_num'], sub['leak_p95'], 'o-', color=c, label=LABEL[mode])
ax.set_xscale('log', base=2); ax.set_xticks(NN); ax.set_xticklabels(NN)
ax.set_xlabel('node_num'); ax.set_ylabel('off-region node displacement p95 @90°')
ax.set_title('(c) Edit leakage outside the drag region')
ax.grid(alpha=.3); ax.legend()

ax = axes[3]
for mode, c in [('iterative', '#1f77b4'), ('from_init', '#d62728')]:
    sub = edit[(edit['mode'] == mode) & (edit['angle'] == 90.0)].sort_values('node_num')
    ax.semilogy(sub['node_num'], sub['solve_time_s'], 'o-', color=c, label=LABEL[mode])
ax.set_xscale('log', base=2); ax.set_xticks(NN); ax.set_xticklabels(NN)
ax.axhline(1.0, color='gray', ls='--', lw=1)
ax.text(70, 1.15, 'interactivity threshold (1 s)', fontsize=8, color='gray')
ax.set_xlabel('node_num'); ax.set_ylabel('ARAP solve time for 90° drag (s, log)')
ax.set_title('(d) Editing latency')
ax.grid(alpha=.3, which='both'); ax.legend()

plt.tight_layout()
out = 'results/failureB/failureB_curves.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print('wrote', out)
