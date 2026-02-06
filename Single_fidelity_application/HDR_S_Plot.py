import pandas as pd
import numpy as np
import matplotlib
# Use TkAgg backend (commonly used for interactive matplotlib windows on many systems)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import pickle
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from pathlib import Path

# Directory where pickled experiment results are stored
SAVE_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "Single-fidelity" / "CTR23" / "Result"

# Load CTR23 (real-world dataset) results
pkl_file = os.path.join(SAVE_DIR, 'Results_ctr23_list.pkl')
with open(pkl_file, 'rb') as f:
    Results_ctr23_list = pickle.load(f)

pkl_file = os.path.join(SAVE_DIR, 'Results_ctr23_prediction.pkl')
with open(pkl_file, 'rb') as f:
    Results_ctr23_prediction = pickle.load(f)


# Directory for synthetic high-dimensional regression (HDR) results
SAVE_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "Single-fidelity" / "Synthetic" / "HDR"

# Load synthetic HDR simulation results
pkl_file = os.path.join(SAVE_DIR, 'Results_simulation_list.pkl')
with open(pkl_file, 'rb') as f:
    Results_simulation_list = pickle.load(f)

pkl_file = os.path.join(SAVE_DIR, 'Results_simulation_prediction.pkl')
with open(pkl_file, 'rb') as f:
    Results_simulation_prediction = pickle.load(f)


# ────────────────────────────────────────────────────────────────
# Combine all results into a single long-format DataFrame
# ────────────────────────────────────────────────────────────────
data = []

# Process CTR23 real-world results
for dataset_idx in range(len(Results_ctr23_list)):
    for sim_idx in range(len(Results_ctr23_list[dataset_idx])):
        df = Results_ctr23_list[dataset_idx][sim_idx]
        if df is None:
            continue
        for _, row in df.iterrows():
            data.append({
                'Dataset': dataset_idx,
                'Fold': sim_idx,
                'Model': row['Model'],
                'RMSE': row['RMSE'],
                'R2': row['R2'],
                'MAE': row['MAE'],
                'RMSE_N': row['RMSE_N'],
                'MAE_N': row['MAE_N'],
                'Time_Train': row['Time_Train'],
                'Time_Pred': row['Time_Pred'],
            })

# Process synthetic HDR simulation results
for dataset_idx in range(len(Results_simulation_list)):
    for sim_idx in range(len(Results_simulation_list[dataset_idx])):
        df = Results_simulation_list[dataset_idx][sim_idx]
        if df is None:
            continue
        for _, row in df.iterrows():
            data.append({
                'Dataset': dataset_idx,
                'Fold': sim_idx,
                'Model': row['Model'],
                'RMSE': row['RMSE'],
                'R2': row['R2'],
                'MAE': row['MAE'],
                'RMSE_N': row['RMSE_N'],
                'MAE_N': row['MAE_N'],
                'Time_Train': row['Time_Train'],
                'Time_Pred': row['Time_Pred'],
            })

Data_frame = pd.DataFrame(data)


def extract_base_variant(model_name):
    """
    Parse model name into base model and variant suffix.
    Example: "TabPFN (Tuned + Ensembled)" → ('TabPFN', 'Tuned + Ensembled')
    """
    if model_name == 'AutoGluon':
        return 'AutoGluon', 'Tuned + Ensembled'
    if ' (' in model_name:
        base, var = model_name.split(' (')
        var = var[:-1]  # remove closing parenthesis
    else:
        base = model_name
        var = 'Best'
    return base, var


# Add parsed columns for easier grouping
Data_frame['Base'] = Data_frame['Model'].apply(lambda x: extract_base_variant(x)[0])
Data_frame['Variant'] = Data_frame['Model'].apply(lambda x: extract_base_variant(x)[1])


# Confidence interval settings (95%)
confidence = 0.95
alpha = 1 - confidence


# ────────────────────────────────────────────────────────────────
# Global matplotlib style settings
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


# Plot configuration dictionaries
variants = ['Tuned + Ensembled', 'Tuned', 'Default']

model_colors = {
    'TabPFN': '#2ca02c',
    'RealMLP': '#ff7f0e',
    'TabM': '#1b9e77',
    'ExtraTrees': '#d62728',
    'LightGBM': '#e377c2',
    'CatBoost': '#7f7f7f',
    'XGBoost': '#bcbd22',
    'AutoGluon': '#9467bd',
    'MFTabPFN': '#1f77b4',
    'ModernNCA': '#f7b6d2',
    'TabDPT': '#17becf',
    'EBM': '#8c564b',
    'FastaiMLP': '#bcbd88',
    'TorchMLP': '#9f9f9f'
}

variant_hatches = {
    'Default': '',
    'Tuned': '//////',
    'Tuned + Ensembled': 'xxxxx'
}

variant_widths = {
    'Default': 0.30,
    'Tuned': 0.30,
    'Tuned + Ensembled': 0.30
}

variant_afa = {
    'Default': 1.0,
    'Tuned': 0.8,
    'Tuned + Ensembled': 0.6
}

error_bar_properties = {
    'Default': {'linewidth': 1, 'color': 'black', 'capsize': 1, 'alpha': 1.0},
    'Tuned': {'linewidth': 1, 'color': 'green', 'capsize': 1, 'alpha': 1.0},
    'Tuned + Ensembled': {'linewidth': 1, 'color': 'blue', 'capsize': 1, 'alpha': 1.0}
}

offset_dict = {
    'Tuned + Ensembled': +0.25,
    'Tuned':             -0.25,
    'Default':            0.0
}


# Metrics to plot
metrics = ['RMSE_N', 'MAE_N', 'R2']
metric_titles = {'RMSE_N': 'NNRMSE', 'MAE_N': 'NNMAE', 'R2': r'$R^2$'}
metric_ylabels = {'RMSE_N': 'NNRMSE (95% CI)', 'MAE_N': 'NNMAE (95% CI)', 'R2': r'$R^2$ (95% CI)'}


# ────────────────────────────────────────────────────────────────
# Main plotting loop — one figure per metric
# ────────────────────────────────────────────────────────────────
for metric in metrics:
    # Compute mean, std, count, standard error, and 95% CI per base model + variant
    grouped = Data_frame.groupby(['Base', 'Variant'])[metric].agg(['mean', 'std', 'count']).reset_index()
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['ci'] = grouped.apply(
        lambda row: row['se'] * stats.t.ppf(1 - alpha / 2, row['count'] - 1) if row['count'] > 1 else 0,
        axis=1
    )

    # Helper to sort base models by performance (best variant first)
    def get_sort_key(base):
        sub = grouped[grouped['Base'] == base]
        if sub.empty:
            return 0
        for v in ['Tuned + Ensembled', 'Tuned', 'Default']:
            if v in sub['Variant'].values:
                return sub[sub['Variant'] == v]['mean'].iloc[0]
        return sub['mean'].max()

    base_models = sorted(grouped['Base'].unique(), key=get_sort_key, reverse=False)
    base_models = list(filter(lambda x: x != 'AutoGluon', base_models))  # AutoGluon plotted separately

    # Create figure
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(base_models))

    # Plot bars for each variant
    for var in variants:
        sub = grouped[grouped['Variant'] == var]
        means = []
        cis = []
        for b in base_models:
            row = sub[sub['Base'] == b]
            if not row.empty:
                means.append(row['mean'].iloc[0])
                cis.append(row['ci'].iloc[0])
            else:
                means.append(np.nan)
                cis.append(np.nan)

        err_props = error_bar_properties[var]

        # Draw bars (hatch + alpha + color)
        p = ax.bar(
            x + offset_dict[var],
            means,
            width=variant_widths[var],
            color=[model_colors.get(b, '#808080') for b in base_models],
            label=var,
            edgecolor='black',
            linewidth=0.5,
            alpha=variant_afa[var]
        )

        # Apply hatch pattern
        for bar in p:
            bar.set_hatch(variant_hatches[var])
            if var in ['Tuned', 'Tuned + Ensembled']:
                bar._hatch_color = (1.0, 1.0, 1.0, 1.0)
                bar._hatch_linewidth = 0.5

        # Add error bars (95% CI)
        for i, bar in enumerate(p):
            ax.errorbar(
                x[i] + offset_dict[var],
                means[i],
                yerr=cis[i],
                fmt='none',
                color=err_props['color'],
                linewidth=err_props['linewidth'],
                capsize=err_props['capsize'],
                alpha=err_props['alpha']
            )

    # ─── Plot AutoGluon as horizontal reference line ───
    mean_value = grouped[grouped['Base'] == 'AutoGluon']['mean'].iloc[0]
    ax.hlines(
        mean_value,
        xmin=-0.6,
        xmax=len(base_models) - 0.4,
        colors=model_colors['AutoGluon'],
        linestyles='dashed',
        label='AutoGluon',
        linewidth=1,
        zorder=0
    )

    # Add text label for AutoGluon line
    if metric == 'R2':
        ax.text(-0.5, mean_value + 0.025, 'AutoGluon', color=model_colors['AutoGluon'], va='bottom', ha='left')
    else:
        ax.text(-0.5, mean_value + 0.005, 'AutoGluon', color=model_colors['AutoGluon'], va='bottom', ha='left')

    # ─── Axis & title settings ───
    ax.set_title(metric_titles[metric])
    ax.set_ylabel(metric_ylabels[metric])
    ax.set_xticks(x)
    ax.set_xticklabels(base_models, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_xlim(-0.6, len(base_models) - 0.4)

    if metric == 'R2':
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylim(0.8, 1.0)
        ax.yaxis.set_major_locator(MultipleLocator(0.04))

    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # ─── Special legend for RMSE_N (with hatch + error bar demo) ───
    if metric == 'RMSE_N':
        variant_patches = [
            Patch(facecolor='#1f77b4', edgecolor='black', linewidth=0.5, hatch='', label='Mean (D)'),
            Patch(facecolor='#1f77b4', edgecolor='black', linewidth=0.5, hatch=variant_hatches['Tuned'], label='Mean (T)', alpha=variant_afa['Tuned']),
            Patch(facecolor='#1f77b4', edgecolor='black', linewidth=0.5, hatch=variant_hatches['Tuned + Ensembled'], label='Mean (T+E)', alpha=variant_afa['Tuned + Ensembled']),
        ]
        variant_patches[1]._hatch_color = (1.0, 1.0, 1.0, 1.0)
        variant_patches[2]._hatch_color = (1.0, 1.0, 1.0, 1.0)

        # Invisible axis just for legend elements
        ax_hidden = fig.add_axes([0, 0, 0.0001, 0.0001])
        ax_hidden.set_xlim(0, 1)
        ax_hidden.set_ylim(0, 1)
        ax_hidden.axis('off')

        def legend_errorbar(color, label):
            eb = ax_hidden.errorbar(
                0, 0,
                yerr=[[0.4], [0.6]],
                color=color,
                capsize=1.5,
                capthick=1.0,
                lw=1.0,
                fmt='none',
                markersize=0,
                markerfacecolor='none',
                markeredgecolor='none',
                label=label
            )
            return eb

        eb1 = legend_errorbar('black', 'CI (D)')
        eb2 = legend_errorbar('green', 'CI (T)')
        eb3 = legend_errorbar('blue', 'CI (T+E)')

        handles = [
            variant_patches[0], eb1,
            variant_patches[1], eb2,
            variant_patches[2], eb3
        ]

        leg1 = ax.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.01),
            frameon=True,
            ncol=3,
            columnspacing=1.0,
            handletextpad=0.5,
            handlelength=0.8,
        )
        leg1.get_frame().set_linewidth(0.75)

    plt.tight_layout(pad=0.01)

    # ─── Save figure in multiple formats ───
    for ext in ['png', 'tiff', 'pdf']:
        output_path = os.path.join(SAVE_DIR, f'{metric.lower()}_bar_plot.{ext}')
        plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)

    # Display (interactive window) and close
    plt.show(block=True)
    plt.close(fig)