import pandas as pd
import numpy as np
import matplotlib
# Use TkAgg backend to support interactive matplotlib windows (common for desktop GUIs)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator
from matplotlib.gridspec import GridSpec
from matplotlib.cbook import boxplot_stats
from pathlib import Path

# ────────────────────────────────────────────────────────────────
# Load pickled experiment results from two sources
# ────────────────────────────────────────────────────────────────

SAVE_DIR1 = Path(__file__).resolve().parents[1] / "Datasets" / "Single-fidelity" / "CTR23" / "Result"        # Real-world CTR23 dataset
SAVE_DIR2 = Path(__file__).resolve().parents[1] / "Datasets" / "Single-fidelity" / "Synthetic" / "HDR"   # Synthetic high-dimensional regression (HDR)

Results_ctr23_list = pickle.load(open(os.path.join(SAVE_DIR1, 'Results_ctr23_list.pkl'), 'rb'))
Results_simulation_list = pickle.load(open(os.path.join(SAVE_DIR2, 'Results_simulation_list.pkl'), 'rb'))


# ────────────────────────────────────────────────────────────────
# Combine training & prediction times from both sources
# ────────────────────────────────────────────────────────────────
data = []

for source in [Results_ctr23_list, Results_simulation_list]:
    for dataset_idx in range(len(source)):
        for sim_idx in range(len(source[dataset_idx])):
            df = source[dataset_idx][sim_idx]
            if df is None:
                continue
            for _, row in df.iterrows():
                data.append({
                    'Model'      : row['Model'],
                    'Time_Train' : row['Time_Train'],
                    'Time_Pred'  : row['Time_Pred'],
                })

df = pd.DataFrame(data)
df['Time_Total'] = df['Time_Train'] + df['Time_Pred']   # Total runtime = train + predict


# ────────────────────────────────────────────────────────────────
# Global publication-style matplotlib settings
# ────────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 7
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['axes.linewidth']      = 0.6
plt.rcParams['xtick.major.width']   = 0.6
plt.rcParams['ytick.major.width']   = 0.6
plt.rcParams['xtick.minor.width']   = 0.5
plt.rcParams['ytick.minor.width']   = 0.5
plt.rcParams['xtick.major.size']    = 2.0
plt.rcParams['ytick.major.size']    = 2.0
plt.rcParams['xtick.minor.size']    = 1.5
plt.rcParams['ytick.minor.size']    = 1.5


def extract_base_variant(name):
    """
    Parse model name into base model and variant.
    Special handling for 'AutoGluon' and 'Best' → 'Default'.
    """
    if name == 'AutoGluon':
        return 'AutoGluon', 'Tuned + Ensembled'
    if ' (' in name:
        base, var = name.split(' (')
        var = var[:-1]
        if var == 'Best':
            var = 'Default'
        return base, var
    return name, 'Default'


# Add parsed columns
df['Base'] = df['Model'].apply(lambda x: extract_base_variant(x)[0])
df['Variant'] = df['Model'].apply(lambda x: extract_base_variant(x)[1])


# Filter data for visualization: only Default and Tuned + Ensembled variants, exclude AutoGluon
df_plot = df[df['Variant'].isin(['Default', 'Tuned + Ensembled'])].copy()
df_plot = df_plot[df_plot['Base'] != 'AutoGluon']


# ────────────────────────────────────────────────────────────────
# Determine model order: sort by median training time of Tuned + Ensembled variant
# ────────────────────────────────────────────────────────────────
all_bases = sorted(df_plot['Base'].unique())
has_tuned = df_plot[df_plot['Variant'] == 'Tuned + Ensembled']['Base'].unique()

sorted_by_speed = (df_plot[df_plot['Variant'] == 'Tuned + Ensembled']
                   .groupby('Base')['Time_Train'].median()
                   .sort_values().index.tolist())

base_order = [b for b in sorted_by_speed if b in has_tuned] + \
             [b for b in all_bases if b not in has_tuned]


# Color mapping for different base models
model_colors = {
    'TabPFN'     : '#2ca02c', 'RealMLP': '#ff7f0e', 'TabM': '#1b9e77',
    'ExtraTrees' : '#d62728', 'LightGBM': '#e377c2', 'CatBoost': '#7f7f7f',
    'XGBoost'    : '#bcbd22', 'MFTabPFN': '#1f77b4', 'ModernNCA': '#f7b6d2',
    'TabDPT'     : '#17becf', 'EBM': '#8c564b', 'FastaiMLP': '#bcbd88',
    'TorchMLP'   : '#9f9f9f'
}


def plot_all_time_comparison_combined():
    """
    Create a 3×2 grid of violin + box plots comparing runtime (train / predict / total)
    between Default and Tuned + Ensembled configurations across models.
    Uses log scale for time (seconds).
    """
    metrics = [
        ('Time_Train', 'Training time'),
        ('Time_Pred',  'Prediction time'),
        ('Time_Total', 'Total time')
    ]

    fig = plt.figure(figsize=(6.8, 5.0))
    gs = GridSpec(3, 2, figure=fig)   # 3 rows (metrics), 2 columns (variants)

    for row_idx, (metric, title_prefix) in enumerate(metrics):
        for col_idx, variant in enumerate(['Default', 'Tuned + Ensembled']):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            # Filter data for current variant
            data_var = df_plot[df_plot['Variant'] == variant]
            valid_bases = [b for b in base_order if b in data_var['Base'].values]
            positions = [base_order.index(b) for b in valid_bases]

            # Prepare data lists for violin/box
            datasets = [data_var[data_var['Base'] == b][metric].dropna().values for b in valid_bases]

            # ─── Violin plot: shows full data distribution ───
            if datasets:
                parts = ax.violinplot(datasets, positions=positions, widths=0.85,
                                      showmeans=False, showextrema=False, showmedians=False)
                for pc, base in zip(parts['bodies'], valid_bases):
                    color = model_colors.get(base, '#808080')
                    pc.set_facecolor(color)
                    pc.set_edgecolor('none')
                    pc.set_alpha(0.5)

            # ─── Box plot: overlaid on violin for clarity ───
            for i, base in enumerate(base_order):
                if base not in data_var['Base'].values:
                    continue
                color = model_colors.get(base, '#808080')
                y = data_var[data_var['Base'] == base][metric].dropna()
                ax.boxplot([y], positions=[i], widths=0.4,
                           patch_artist=True,
                           boxprops=dict(facecolor=color, edgecolor=color, linewidth=0),
                           whiskerprops=dict(color=color, linewidth=1.5),
                           capprops=dict(color=color, linewidth=1.5),
                           medianprops=dict(color='black', linewidth=1.5),
                           flierprops=dict(marker='o', markerfacecolor=color,
                                           markeredgecolor=color, markersize=1, alpha=1.0))

            # ─── Logarithmic y-scale with custom ticks ───
            ax.set_yscale('log')
            ax.minorticks_on()
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.set_axisbelow(True)

            # Custom y-limits and ticks per metric & variant
            if metric == 'Time_Train':
                if variant == 'Default':
                    ax.set_yticks([0.1, 1, 10, 100, 1000, 10000])
                    ax.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$'])
                    ax.set_ylim(0.1, 10000)
                else:
                    ax.set_yticks([10, 100, 1000, 10000, 100000])
                    ax.set_yticklabels([r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$', r'$10^{5}$'])
                    ax.set_ylim(10, 100000)
            elif metric == 'Time_Pred':
                ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
                ax.set_yticklabels([r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$'])
                ax.set_ylim(0.001, 1000)
            elif metric == 'Time_Total':
                if variant == 'Default':
                    ax.set_yticks([0.1, 1, 10, 100, 1000, 10000])
                    ax.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$'])
                    ax.set_ylim(0.1, 10000)
                else:
                    ax.set_yticks([10, 100, 1000, 10000, 100000])
                    ax.set_yticklabels([r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$', r'$10^{5}$'])
                    ax.set_ylim(10, 100000)

            # Subplot title: e.g. "Training time (D)" or "Total time (T+E)"
            config_name = 'D' if variant == 'Default' else 'T+E'
            ax.set_title(f"{title_prefix} ({config_name})")

            # ─── X-axis labels (only on bottom row) ───
            if row_idx == 2:
                if col_idx == 1:
                    # Right column: skip TabDPT label (often outlier / missing)
                    visible_idx = []
                    visible_labels = []
                    for i, model in enumerate(base_order):
                        if model == 'TabDPT':
                            continue
                        visible_idx.append(i)
                        visible_labels.append(model)
                else:
                    visible_idx = list(range(len(base_order)))
                    visible_labels = base_order.copy()

                ax.set_xticks(visible_idx)
                ax.set_xticklabels(visible_labels, rotation=45, ha='right', rotation_mode='anchor')
            else:
                # Upper rows: hide x-labels
                if col_idx == 1:
                    visible_idx = [i for i, m in enumerate(base_order) if m != 'TabDPT']
                    ax.set_xticks(visible_idx)
                    ax.set_xticklabels([])
                else:
                    ax.set_xticks(range(len(base_order)))
                    ax.set_xticklabels([])

            ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.xaxis.set_minor_locator(NullLocator())

            # Y-label only on left column
            if col_idx == 0:
                ax.set_ylabel('Time (seconds)')

            # ─── Legend ───
            # Legend 1: violin + box explanation (appears only once, in middle-left subplot)
            if row_idx == 1 and col_idx == 0:
                legend_elements = [
                    Patch(facecolor='#1f77b4', alpha=0.5, edgecolor='none',
                          label='Violin plot: data distribution'),
                    Patch(facecolor='#1f77b4', edgecolor='none', alpha=1.0,
                          label='Box: 25th–75th percentile (IQR)'),
                ]
                leg1 = ax.legend(handles=legend_elements,
                                 loc='upper center',
                                 bbox_to_anchor=(0.5, 1.02),
                                 ncol=1,
                                 frameon=True,
                                 columnspacing=1.0,
                                 handletextpad=0.5,
                                 handlelength=0.8,
                                 )
                leg1.get_frame().set_linewidth(0.75)

            # Legend 2: whiskers + outliers + median (appears only once, in bottom-right subplot)
            if row_idx == 2 and col_idx == 1:
                legend_elements = [
                    Line2D([0], [0], color='#1f77b4', lw=1.5, label='Upper and lower whiskers'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=3,
                           label='Outliers beyond 1.5×IQR'),
                    Line2D([0], [0], color='black', lw=1.5, label='Median'),
                ]
                leg1 = ax.legend(handles=legend_elements,
                                 loc='upper center',
                                 bbox_to_anchor=(0.5, 1.02),
                                 ncol=2,
                                 frameon=True,
                                 columnspacing=1.0,
                                 handletextpad=0.5,
                                 handlelength=0.8,
                                 )
                leg1.get_frame().set_linewidth(0.75)

    # Final layout adjustments
    plt.tight_layout(pad=0.01)
    plt.subplots_adjust(left=0.10, right=0.96, top=0.95, bottom=0.12, wspace=0.15, hspace=0.20)

    # Save in multiple formats
    for ext in ['png', 'tiff', 'pdf']:
        output_path = os.path.join(SAVE_DIR2, f'Runtime_Comparison_All_Combined.{ext}')
        plt.savefig(output_path, dpi=400, bbox_inches='tight')

    plt.show()


# Generate the main visualization
plot_all_time_comparison_combined()


# ────────────────────────────────────────────────────────────────
# Print detailed boxplot statistics (median, quartiles, whiskers, N) for console inspection
# ────────────────────────────────────────────────────────────────
def print_boxplot_stats(metric, metric_name):
    """
    Print boxplot statistics (median, Q1, Q3, whiskers, sample size) for each model
    and variant in a readable table format.
    """
    print(f"\n{'=' * 60}")
    print(f"{metric_name}")
    print(f"{'=' * 60}")

    for variant in ['Default', 'Tuned + Ensembled']:
        print(f"\n【{variant}】")
        rows = []
        for base in base_order:
            data = df_plot[(df_plot['Base'] == base) &
                           (df_plot['Variant'] == variant)][metric].dropna()
            if len(data) == 0:
                rows.append([base, '—', '—', '—', '—', '—', 0])
                continue
            stats = boxplot_stats(data)[0]
            rows.append([
                base,
                f"{stats['med']:.4f}",
                f"{stats['q1']:.4f}",
                f"{stats['q3']:.4f}",
                f"{stats['whislo']:.4f}",
                f"{stats['whishi']:.4f}",
                len(data)
            ])

        print(pd.DataFrame(rows, columns=['Model', 'Median', 'Q1', 'Q3', 'Lower Whisker', 'Upper Whisker', 'N'])
              .to_string(index=False))


print_boxplot_stats('Time_Train', 'Training Time')
print_boxplot_stats('Time_Pred', 'Prediction Time')
print_boxplot_stats('Time_Total', 'Total Time')