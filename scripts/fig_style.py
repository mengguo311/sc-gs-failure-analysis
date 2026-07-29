"""Shared figure style: contrastive naming (SC-GS original vs Ours) + unified typography.

Import as:  from fig_style import LABEL, COLOR, ORDER, apply_style
"""
import matplotlib

# Mode naming convention used in ALL figures and the paper:
#   from_init      -> the original SC-GS one-shot editing mode (the baseline we improve)
#   progressive_N* -> our method
#   iterative      -> SC-GS's slow incremental mode, used as robustness reference
ORDER = ['from_init', 'progressive_N2', 'progressive_N4', 'progressive_N8', 'iterative']
LABEL = {
    'from_init': 'SC-GS (original)',
    'progressive_N2': 'Ours (N=2)',
    'progressive_N4': 'Ours (N=4)',
    'progressive_N8': 'Ours (N=8)',
    'iterative': 'SC-GS iterative (slow ref.)',
}
COLOR = {
    'from_init': '#d62728',
    'progressive_N2': '#ff9896',
    'progressive_N4': '#ff7f0e',
    'progressive_N8': '#2ca02c',
    'iterative': '#1f77b4',
}


def apply_style():
    """Uniform panel typography across every figure in the paper."""
    matplotlib.rcParams.update({
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
    })
