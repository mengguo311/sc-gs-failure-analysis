"""Cross-scene validation of progressive drag scheduling.

Aggregates results/failureA/<scene>/metrics.csv over scenes, writes
results/failureA/cross_scene_summary.csv and a per-scene edge-stretch figure
results/failureA/cross_scene_curves.png.

Usage: python scripts/cross_scene_summary.py --scenes jumpingjacks hook mutant [...]
"""
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER = ['from_init', 'progressive_N2', 'progressive_N4', 'progressive_N8', 'iterative']
LABEL = {'from_init': 'from_init (N=1)', 'progressive_N2': 'progressive N=2',
         'progressive_N4': 'progressive N=4', 'progressive_N8': 'progressive N=8',
         'iterative': 'iterative (ref)'}
COLOR = {'from_init': '#d62728', 'progressive_N2': '#ff9896', 'progressive_N4': '#ff7f0e',
         'progressive_N8': '#2ca02c', 'iterative': '#1f77b4'}

ap = argparse.ArgumentParser()
ap.add_argument('--scenes', nargs='+', default=['jumpingjacks', 'hook', 'mutant'])
args = ap.parse_args()

rows = []
fig, axes = plt.subplots(1, len(args.scenes), figsize=(5.2 * len(args.scenes), 4.4), squeeze=False)
for ci, scene in enumerate(args.scenes):
    df = pd.read_csv(f'results/failureA/{scene}/metrics.csv')
    df['key'] = df['mode'] + df['tag'].fillna('').astype(str)
    df = df.drop_duplicates(subset=['key', 'angle'], keep='last')
    ax = axes[0][ci]
    for key in ORDER:
        sub = df[df['key'] == key].sort_values('angle')
        if sub.empty:
            continue
        ax.plot(sub['angle'], sub['edge_stretch_region_p95'], 'o-', color=COLOR[key],
                label=LABEL[key], ms=4)
        for a in [45, 90, 135]:
            r = sub[sub['angle'] == a]
            if len(r):
                rows.append({'scene': scene, 'mode': key, 'angle': a,
                             'edge_region_p95': r['edge_stretch_region_p95'].iloc[0],
                             'gs_p95': r['gs_stretch_p95'].iloc[0],
                             'arap_error': r['arap_error'].iloc[0],
                             'solve_time_s': r['solve_time_s'].iloc[0]})
    ax.set_title(scene)
    ax.set_xlabel('drag rotation angle (deg)')
    if ci == 0:
        ax.set_ylabel('node edge stretch, region p95')
        ax.legend(fontsize=8)
    ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig('results/failureA/cross_scene_curves.png', dpi=150, bbox_inches='tight')
print('wrote results/failureA/cross_scene_curves.png')

out = pd.DataFrame(rows)
out.to_csv('results/failureA/cross_scene_summary.csv', index=False)
piv = out[out['angle'] == 90].pivot(index='mode', columns='scene', values='edge_region_p95')
print('\nedge_region_p95 @90°:')
print(piv.reindex(ORDER).round(3).to_string())
piv2 = out[out['angle'] == 135].pivot(index='mode', columns='scene', values='edge_region_p95')
print('\nedge_region_p95 @135°:')
print(piv2.reindex(ORDER).round(3).to_string())
