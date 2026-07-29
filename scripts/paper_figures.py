"""Additional publication figures for the academic paper.

Generates into paper/figures/:
  fig_method_schematic.png   — one-shot Laplacian init vs progressive warm-started arc
  fig_onset_gapclosure.png   — (a) onset angle vs N, (b) gap closure per scene, (c) latency-robustness
  fig_qualitative_2scene.png — jumpingjacks + trex qualitative strips at 90°
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio.v2 as imageio

OUT = 'paper/figures'

# ---------------------------------------------------------------- schematic
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
shoulder = np.array([0.0, 0.0])
L = 1.0
theta = np.radians(90)
rest = shoulder + np.array([L, 0.0])
target = shoulder + L * np.array([np.cos(-theta), np.sin(-theta)])

for ax, title in zip(axes, ['(a) arap_from_init: one-shot solve from linear init',
                            '(b) Progressive drag scheduling (ours): N warm-started sub-steps']):
    ax.set_xlim(-0.35, 1.25); ax.set_ylim(-1.25, 0.45)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=11)
    arc_t = np.linspace(0, theta, 100)
    ax.plot(L*np.cos(-arc_t), L*np.sin(-arc_t), ':', color='gray', lw=1.2)
    ax.plot(*shoulder, 'ks', ms=9)
    ax.annotate('anchored joint', shoulder, textcoords='offset points', xytext=(-12, 10), fontsize=9)
    ax.plot([shoulder[0], rest[0]], [shoulder[1], rest[1]], '-', color='#1f77b4', lw=5, alpha=.45)
    ax.plot(*rest, 'o', color='#1f77b4', ms=10, alpha=.6)
    ax.annotate('rest limb', rest, textcoords='offset points', xytext=(6, 8), fontsize=9, color='#1f77b4')
    ax.plot(*target, '*', color='#2ca02c', ms=17)
    ax.annotate('drag target θ', target, textcoords='offset points', xytext=(8, -4), fontsize=9, color='#2ca02c')

ax = axes[0]
# chord shortcut: linear (Laplacian) init drags free nodes along the chord, shrinking the limb
chord = np.linspace(rest, target, 6)
ax.plot(chord[:, 0], chord[:, 1], '--', color='#d62728', lw=1.6)
shrunk_dir = (rest + target) / 2 - shoulder
shrunk = shoulder + 0.55 * shrunk_dir / np.linalg.norm(shrunk_dir)
ax.plot([shoulder[0], shrunk[0]], [shoulder[1], shrunk[1]], '-', color='#d62728', lw=5, alpha=.8)
ax.plot(*shrunk, 'o', color='#d62728', ms=9)
ax.annotate('linear init: chord shortcut\n→ shortened, sheared limb', (0.72, -0.42),
            fontsize=9, color='#d62728', ha='center')
ax.annotate('3 ARAP iters cannot recover', (0.62, -0.78), fontsize=9, color='#d62728', ha='center')

ax = axes[1]
N = 4
cols = plt.cm.Greens(np.linspace(0.45, 0.95, N))
prev = rest
for k in range(1, N + 1):
    a = theta * k / N
    p = shoulder + L * np.array([np.cos(-a), np.sin(-a)])
    ax.plot([shoulder[0], p[0]], [shoulder[1], p[1]], '-', color=cols[k-1], lw=4.2, alpha=.85)
    ax.annotate('', xy=p, xytext=prev,
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.4,
                                connectionstyle='arc3,rad=-0.25'))
    prev = p
ax.annotate('sub-step k warm-starts\nfrom sub-step k−1 (init_verts)', (0.72, -0.42),
            fontsize=9, color='#2ca02c', ha='center')
ax.annotate('per-step increment θ/N small\n→ 3 ARAP iters suffice', (0.6, -0.85),
            fontsize=9, color='#2ca02c', ha='center')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_method_schematic.png', dpi=160, bbox_inches='tight')
plt.close()
print('schematic done')

# ------------------------------------------------- onset / gap closure / tradeoff
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))

ax = axes[0]
modes = ['from_init\n(N=1)', 'prog.\nN=2', 'prog.\nN=4', 'prog.\nN=8', 'iterative\n(ref)']
onset_lo = [45, 45, 60, 90, 135]
onset_hi = [60, 60, 75, 110, 135]
mid = [(a+b)/2 for a, b in zip(onset_lo, onset_hi)]
err = [(b-a)/2 for a, b in zip(onset_lo, onset_hi)]
colors = ['#d62728', '#ff9896', '#ff7f0e', '#2ca02c', '#1f77b4']
bars = ax.bar(modes, mid, yerr=err, color=colors, capsize=4, width=.62)
ax.bar_label(bars, labels=[f'{a}–{b}°' if a != b else f'≥{a}°' for a, b in zip(onset_lo, onset_hi)],
             padding=3, fontsize=9)
ax.set_ylabel('failure-onset drag angle (deg)')
ax.set_ylim(0, 155)
ax.set_title('(a) Failure onset vs schedule length (jumpingjacks)')
ax.grid(alpha=.3, axis='y')

ax = axes[1]
scenes = ['jumping-\njacks', 'hook', 'mutant', 'standup', 'trex']
summ = pd.read_csv('results/failureA/cross_scene_summary.csv')
s90 = summ[summ['angle'] == 90].pivot(index='mode', columns='scene', values='edge_region_p95')
order_sc = ['jumpingjacks', 'hook', 'mutant', 'standup', 'trex']
gap4, gap8 = [], []
for sc in order_sc:
    fi, it = s90.loc['from_init', sc], s90.loc['iterative', sc]
    gap4.append(100 * (fi - s90.loc['progressive_N4', sc]) / (fi - it))
    gap8.append(100 * (fi - s90.loc['progressive_N8', sc]) / (fi - it))
x = np.arange(len(scenes)); w = 0.36
b1 = ax.bar(x - w/2, gap4, w, label='progressive N=4', color='#ff7f0e')
b2 = ax.bar(x + w/2, gap8, w, label='progressive N=8', color='#2ca02c')
ax.bar_label(b1, fmt='%.0f%%', padding=2, fontsize=8)
ax.bar_label(b2, fmt='%.0f%%', padding=2, fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(scenes)
ax.set_ylabel('from_init → reference gap closed (%)')
ax.set_ylim(0, 100)
ax.set_title('(b) Gap closure at 90° drag, per scene')
ax.grid(alpha=.3, axis='y'); ax.legend(fontsize=9)

ax = axes[2]
key_map = {'from_init': ('from_init (N=1)', '#d62728', 'o'),
           'progressive_N2': ('N=2', '#ff9896', 's'),
           'progressive_N4': ('N=4', '#ff7f0e', 's'),
           'progressive_N8': ('N=8', '#2ca02c', 's'),
           'iterative': ('iterative (ref)', '#1f77b4', 'D')}
df = pd.read_csv('results/failureA/jumpingjacks/metrics.csv')
df['key'] = df['mode'] + df['tag'].fillna('').astype(str)
df = df.drop_duplicates(subset=['key', 'angle'], keep='last')
sub90 = df[df['angle'] == 90.0]
for key, (label, c, mk) in key_map.items():
    r = sub90[sub90['key'] == key]
    if r.empty:
        continue
    ax.scatter(r['solve_time_s'], r['edge_stretch_region_p95'], s=90, color=c, marker=mk,
               zorder=3, label=label)
ax.set_xscale('log')
ax.axvline(1.0, color='gray', ls='--', lw=1)
ax.text(1.1, 0.165, 'interactivity\nthreshold', fontsize=8, color='gray')
ax.set_xlabel('solve latency for a 90° drag (s, log)')
ax.set_ylabel('edge stretch region p95 @90°')
ax.set_title('(c) Robustness–latency trade-off (jumpingjacks)')
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig(f'{OUT}/fig_onset_gapclosure.png', dpi=160, bbox_inches='tight')
plt.close()
print('onset/gapclosure done')

# ------------------------------------------------------------- 2-scene qualitative
rows_spec = [
    ('jumpingjacks', 0, (20, 330, 60, 380), 'jumpingjacks: arm 90°'),
    ('trex', 3, (60, 360, 40, 420), 'trex: tail 90°'),
]
mode_cols = [('iterative', 'reference (iterative)'), ('from_init', 'from_init (baseline)'),
             ('progressive_N4', 'progressive N=4'), ('progressive_N8', 'progressive N=8')]
fig, axes = plt.subplots(len(rows_spec), len(mode_cols), figsize=(3.6*len(mode_cols), 3.9*len(rows_spec)))
for r, (scene, cam, crop, rlabel) in enumerate(rows_spec):
    for c, (m, clabel) in enumerate(mode_cols):
        img = imageio.imread(f'results/failureA/{scene}/renders/{m}_a090_cam{cam}.png')[..., :3]
        y0, y1, x0, x1 = crop
        axes[r][c].imshow(img[y0:y1, x0:x1])
        axes[r][c].axis('off')
        if r == 0:
            axes[r][c].set_title(clabel, fontsize=12)
    axes[r][0].text(-0.07, 0.5, rlabel, transform=axes[r][0].transAxes, rotation=90,
                    va='center', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_qualitative_2scene.png', dpi=140, bbox_inches='tight')
plt.close()
print('qualitative done')
