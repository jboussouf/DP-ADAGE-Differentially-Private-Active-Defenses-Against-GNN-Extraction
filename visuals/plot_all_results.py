"""
plot_all_results.py
===================
Publication-Quality Figure Generator for DP-ADAGE (TNNLS Submission)
======================================================================

Reads:
  data/results_all_datasets.csv    -- main evaluation results (all 5 datasets)
  data/results_ablation_budget.csv -- ablation study (epsilon_total sweep)

Generates the following PDF figures into figures/:
  fig2_surrogate_trajectory.pdf -- Surrogate Acc vs Queries (1x5 subplot grid)
  fig3_budget_trajectory.pdf    -- Budget Trajectory for DP-ADAGE (1x5 subplot grid)
  fig4_ablation.pdf             -- Ablation: e_total sweep on DP-ADAGE
  fig5_benign_vs_attacker.pdf   -- Utility Comparison: Benign vs Attacker
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive Agg backend — safe for headless servers / cluster nodes
matplotlib.use('Agg')

# ── Academic style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    # Font
    'font.family':          'sans-serif',
    'font.sans-serif':      ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size':            10,
    'axes.titlesize':       11,
    'axes.labelsize':       10,
    'xtick.labelsize':      9,
    'ytick.labelsize':      9,
    'legend.fontsize':      8.5,
    'legend.title_fontsize':9,
    # Lines & markers
    'lines.linewidth':      1.8,
    'lines.markersize':     5,
    # Grid
    'axes.grid':            True,
    'grid.linestyle':       '--',
    'grid.linewidth':       0.5,
    'grid.alpha':           0.5,
    # Spines
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    # Figure
    'figure.dpi':           150,
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
    'savefig.pad_inches':   0.05,
    # PDF output: embed fonts
    'pdf.fonttype':         42,
    'ps.fonttype':          42,
})

PALETTE = {
    'NoDefense':     '#000000',   # black
    'StaticNoise':   '#0072B2',   # blue
    'ADAGE':         '#E69F00',   # orange/amber
    'OriginalADAGE': '#009E73',   # bluish-green
    'DP-ADAGE':      '#D55E00',   # vermillion (red-orange)
    'PRADA':         '#CC79A7',   # purple/pink
    'DAWN':          '#56B4E9',   # sky blue
}

DEFENSE_LSTYLE = {
    'NoDefense':     '',                 # solid
    'StaticNoise':   (4, 1.5),           # dashed
    'ADAGE':         (4, 1.5, 1, 1.5),   # dash-dot
    'OriginalADAGE': (3, 1, 1, 1),       # dash-dot-dot
    'DP-ADAGE':      (1, 1),             # dotted
    'PRADA':         (2, 2),             # short dash
    'DAWN':          (5, 1),             # long dash
}

DEFENSE_MARKER = {
    'NoDefense':     'o',
    'StaticNoise':   's',
    'ADAGE':         '^',
    'OriginalADAGE': 'v',
    'DP-ADAGE':      'D',
    'PRADA':         'P',
    'DAWN':          'X',
}

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _final_row(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the last (highest Queries) row per (Dataset, Defense) group."""
    return df.loc[df.groupby(['Dataset', 'Defense'])['Queries'].idxmax()].copy()

# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: Surrogate Accuracy Trajectory (1x5 subplots)
# ═════════════════════════════════════════════════════════════════════════════
def plot_fig2(df: pd.DataFrame, out_dir: str):
    datasets = df['Dataset'].unique()
    num_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, num_datasets, figsize=(3 * num_datasets, 3.5), sharey=True)
    if num_datasets == 1: axes = [axes]
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        df_ds = df[df['Dataset'] == dataset]
        sns.lineplot(
            data=df_ds, x='Queries', y='Surrogate_Acc', hue='Defense', style='Defense',
            palette=PALETTE, markers=DEFENSE_MARKER, dashes=DEFENSE_LSTYLE, ax=ax,
            err_style='band', errorbar='sd'
        )
        ax.set_title(dataset)
        ax.set_xlabel('Number of Queries')
        if i == 0:
            ax.set_ylabel('Surrogate Accuracy')
            ax.legend(title='Defense', fontsize=8, loc='best')
        else:
            if ax.get_legend(): ax.get_legend().remove()
            
    fig.suptitle('Surrogate Accuracy vs. Queries (per dataset)', fontsize=11, fontweight='bold', y=1.05)
    plt.tight_layout()
    out = os.path.join(out_dir, 'fig2_surrogate_trajectory.pdf')
    plt.savefig(out)
    plt.close(fig)
    print(f"[✓] Saved: {out}")

# ═════════════════════════════════════════════════════════════════════════════
# Figure 3: Budget Trajectory for DP-ADAGE (1x5 subplots)
# ═════════════════════════════════════════════════════════════════════════════
def plot_fig3(df: pd.DataFrame, out_dir: str, epsilon_total: float):
    datasets = df['Dataset'].unique()
    num_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, num_datasets, figsize=(3 * num_datasets, 3.5), sharey=True)
    if num_datasets == 1: axes = [axes]
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        df_ds = df[(df['Dataset'] == dataset) & (df['Defense'] == 'DP-ADAGE')]
        
        if not df_ds.empty and 'Budget_Consumed' in df_ds.columns:
            sns.lineplot(
                data=df_ds, x='Queries', y='Budget_Consumed', 
                color=PALETTE['DP-ADAGE'], ax=ax, label='DP-ADAGE', marker='D',
                err_style='band', errorbar='sd'
            )
            
        ax.axhline(y=epsilon_total, color='red', linestyle='--', label=rf'Total Budget ($\varepsilon={epsilon_total}$)')
        ax.set_title(dataset)
        ax.set_xlabel('Number of Queries')
        if i == 0:
            ax.set_ylabel('Cumulative Epsilon')
            ax.legend(fontsize=8, loc='best')
        else:
            if ax.get_legend(): ax.get_legend().remove()

    fig.suptitle('Cumulative DP Budget Consumed vs. Queries', fontsize=11, fontweight='bold', y=1.05)
    plt.tight_layout()
    out = os.path.join(out_dir, 'fig3_budget_trajectory.pdf')
    plt.savefig(out)
    plt.close(fig)
    print(f"[✓] Saved: {out}")

# ═════════════════════════════════════════════════════════════════════════════
# Figure 4: Ablation (Epsilon Total)
# ═════════════════════════════════════════════════════════════════════════════
def plot_fig4_ablation(df_abl: pd.DataFrame, out_dir: str):
    if df_abl is None or df_abl.empty:
        print("[WARN] Ablation CSV is empty. Skipping Ablation Plot.")
        return
    
    budget_col = 'Budget_Cap' if 'Budget_Cap' in df_abl.columns else 'Epsilon_Total'
    if budget_col not in df_abl.columns:
        print(f"[WARN] Ablation CSV missing budget column '{budget_col}'.")
        return
        
    df_dp = df_abl[df_abl['Defense'] == 'DP-ADAGE']
    
    eps_values = sorted(df_dp[budget_col].unique())
    cmap = plt.get_cmap('RdYlGn')
    eps_colors = {e: cmap(i / max(len(eps_values) - 1, 1)) for i, e in enumerate(eps_values)}

    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    sns.lineplot(
        data=df_dp, x='Queries', y='Surrogate_Acc', hue=budget_col,
        palette=eps_colors, marker='o', err_style='band', errorbar='sd', ax=ax
    )

    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Surrogate Accuracy')
    ax.legend(title='Budget Config', fontsize=8, framealpha=0.85)

    fig.suptitle('DP-ADAGE Budget Ablation', fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, 'fig4_ablation.pdf')
    plt.savefig(out)
    plt.close(fig)
    print(f"[✓] Saved: {out}")

# ═════════════════════════════════════════════════════════════════════════════
# Figure 5: Benign vs Attacker Downstream/Surrogate Acc
# ═════════════════════════════════════════════════════════════════════════════
def plot_fig5(df: pd.DataFrame, out_dir: str):
    final = _final_row(df)
    
    attacker_col = 'Attacker_Downstream_Acc' if 'Attacker_Downstream_Acc' in final.columns else 'Surrogate_Acc'
    benign_col = 'Downstream_Acc'
    
    if benign_col not in final.columns or attacker_col not in final.columns:
        print("[WARN] Missing required columns for Fig 5 (Downstream_Acc / Surrogate_Acc).")
        return
        
    df_melt = final.melt(
        id_vars=['Dataset', 'Defense'], 
        value_vars=[benign_col, attacker_col],
        var_name='Metric', value_name='Accuracy'
    )
    
    df_melt['Metric'] = df_melt['Metric'].replace({
        benign_col: 'Benign Utility', 
        attacker_col: 'Attacker Utility'
    })

    g = sns.catplot(
        data=df_melt, kind='bar',
        x='Dataset', y='Accuracy', hue='Defense', col='Metric',
        palette=PALETTE, height=4, aspect=1.2, errorbar='sd'
    )
    
    g.set_axis_labels('Dataset', 'Accuracy')
    g.set_titles("{col_name}")
    g.set(ylim=(0, 1.1))
    
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Utility Comparison: Benign vs Attacker', fontsize=12, fontweight='bold')
    
    out = os.path.join(out_dir, 'fig5_benign_vs_attacker.pdf')
    g.savefig(out)
    plt.close(g.fig)
    print(f"[✓] Saved: {out}")

# ═════════════════════════════════════════════════════════════════════════════
# Figure 6: Adaptive Attacker and 500-Query Horizon
# ═════════════════════════════════════════════════════════════════════════════
def plot_fig6_adaptive_attacker(df_adapt: pd.DataFrame, out_dir: str):
    if df_adapt is None or df_adapt.empty:
        print("[WARN] Adaptive Attacker CSV is empty. Skipping Adaptive Plot.")
        return
        
    strategies = df_adapt['Strategy'].unique()
    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 4), sharey=True)
    if len(strategies) == 1: axes = [axes]
    
    for i, strat in enumerate(strategies):
        ax = axes[i]
        df_strat = df_adapt[df_adapt['Strategy'] == strat]
        
        sns.lineplot(
            data=df_strat, x='Queries', y='Surrogate_Acc', hue='Defense', style='Defense',
            palette=PALETTE, markers=DEFENSE_MARKER, dashes=DEFENSE_LSTYLE, ax=ax,
            err_style='band', errorbar='sd'
        )
        ax.set_title(f"Strategy: {strat.replace('_', ' ').title()}")
        ax.set_xlabel('Number of Queries')
        if i == 0:
            ax.set_ylabel('Surrogate Accuracy')
            ax.legend(title='Defense', fontsize=8, loc='best')
        else:
            if ax.get_legend(): ax.get_legend().remove()
            
    fig.suptitle('Attacker Accuracy vs. Long Horizon Query Budget (500 Queries)', fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()
    out = os.path.join(out_dir, 'fig6_adaptive_attacker.pdf')
    plt.savefig(out)
    plt.close(fig)
    print(f"[✓] Saved: {out}")

def parse_args():
    _here       = os.path.dirname(os.path.abspath(__file__))
    _proj_root  = os.path.join(_here, '..')

    parser = argparse.ArgumentParser(description="DP-ADAGE — Publication-Quality Figure Generator")
    parser.add_argument('--csv', default=os.path.join(_proj_root, 'data', 'results_all_datasets.csv'))
    parser.add_argument('--ablation_csv', default=os.path.join(_proj_root, 'data', 'results_ablation_budget.csv'))
    parser.add_argument('--adaptive_csv', default=os.path.join(_proj_root, 'data', 'results_adaptive_attacker.csv'))
    parser.add_argument('--out_dir', default=os.path.join(_proj_root, 'figures'))
    parser.add_argument('--epsilon_total', type=float, default=200.0)
    return parser.parse_args()

def main():
    args    = parse_args()
    out_dir = _ensure_dir(args.out_dir)

    if not os.path.exists(args.csv):
        print(f"[ERROR] Main CSV not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"[✓] Loaded main CSV: {args.csv} ({len(df)} rows)")

    df_abl = None
    if os.path.exists(args.ablation_csv):
        df_abl = pd.read_csv(args.ablation_csv)
        print(f"[✓] Loaded ablation CSV: {args.ablation_csv} ({len(df_abl)} rows)")

    df_adapt = None
    if os.path.exists(args.adaptive_csv):
        df_adapt = pd.read_csv(args.adaptive_csv)
        print(f"[✓] Loaded adaptive CSV: {args.adaptive_csv} ({len(df_adapt)} rows)")

    print(f"\nGenerating figures into: {out_dir}\n")

    plot_fig2(df, out_dir)
    plot_fig3(df, out_dir, args.epsilon_total)
    plot_fig4_ablation(df_abl, out_dir)
    plot_fig5(df, out_dir)
    plot_fig6_adaptive_attacker(df_adapt, out_dir)

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {out_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
