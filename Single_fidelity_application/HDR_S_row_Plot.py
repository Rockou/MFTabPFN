import pandas as pd
import numpy as np
import matplotlib
# Use TkAgg backend to support interactive matplotlib windows (common on many desktop environments)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import pickle
from matplotlib.lines import Line2D
from pathlib import Path


# ────────────────────────────────────────────────────────────────
# Load pickled experiment results from two sources
# ────────────────────────────────────────────────────────────────
SAVE_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "Single-fidelity" / "CTR23" / "Result"

# Real-world CTR23 dataset results
Results_ctr23_list = pickle.load(open(os.path.join(SAVE_DIR, 'Results_ctr23_list.pkl'), 'rb'))
Results_ctr23_prediction = pickle.load(open(os.path.join(SAVE_DIR, 'Results_ctr23_prediction.pkl'), 'rb'))

SAVE_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "Single-fidelity" / "Synthetic" / "HDR"
# Synthetic high-dimensional regression (HDR) simulation results
Results_simulation_list = pickle.load(open(os.path.join(SAVE_DIR, 'Results_simulation_list.pkl'), 'rb'))
Results_simulation_prediction = pickle.load(open(os.path.join(SAVE_DIR, 'Results_simulation_prediction.pkl'), 'rb'))


# ────────────────────────────────────────────────────────────────
# Combine performance metrics from both sources into one DataFrame
# ────────────────────────────────────────────────────────────────
data = []

# Process both CTR23 and HDR results in a unified loop
for source in [Results_ctr23_list, Results_simulation_list]:
    for dataset_idx in range(len(source)):
        for sim_idx in range(len(source[dataset_idx])):
            df = source[dataset_idx][sim_idx]
            if df is None:
                continue
            for _, row in df.iterrows():
                data.append({
                    'Model': row['Model'],
                    'RMSE_N': row['RMSE_N'],
                    'MAE_N': row['MAE_N'],
                    'R2': row['R2'],
                    'RMSE': row['RMSE'],
                    'MAE': row['MAE'],
                })

Data_frame = pd.DataFrame(data)


def extract_base_variant(model_name):
    """
    Parse model name into base model and variant type.
    Examples:
        "TabPFN (Tuned + Ensembled)" → ('TabPFN', 'Tuned + Ensembled')
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


# Add parsed columns for grouping and visualization
Data_frame['Base'] = Data_frame['Model'].apply(lambda x: extract_base_variant(x)[0])
Data_frame['Variant'] = Data_frame['Model'].apply(lambda x: extract_base_variant(x)[1])


# Confidence interval settings (95%)
confidence = 0.95
alpha = 1 - confidence


# ────────────────────────────────────────────────────────────────
# Publication-style matplotlib settings
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


# Color mapping for different base models
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

# Markers to visually distinguish variants
variant_markers = {
    'Default': 'o',               # circle
    'Tuned': 'D',                 # diamond
    'Tuned + Ensembled': 's'      # square
}

# Error bar styles for different variants
error_bar_properties = {
    'Default': {'linewidth': 1, 'color': 'black', 'capsize': 1, 'alpha': 1.0},
    'Tuned': {'linewidth': 1, 'color': 'green', 'capsize': 1, 'alpha': 1.0},
    'Tuned + Ensembled': {'linewidth': 1, 'color': 'blue', 'capsize': 1, 'alpha': 1.0}
}

# Horizontal offset for each variant to avoid overlap
offset_dict = {
    'Tuned + Ensembled': +0.25,
    'Tuned':             -0.25,
    'Default':            0.0
}


# ────────────────────────────────────────────────────────────────
# Plot setup: RMSE and MAE in two vertically stacked subplots
# ────────────────────────────────────────────────────────────────
metrics = ['RMSE', 'MAE']
metric_titles = {'RMSE': 'RMSE', 'MAE': 'MAE'}
metric_ylabels = {'RMSE': 'RMSE (95% CI)', 'MAE': 'MAE (95% CI)'}

# Sort base models by MAE of the best variant (Tuned + Ensembled), descending
grouped_for_sort = Data_frame.groupby(['Base', 'Variant'])['MAE'].agg('mean').reset_index()
base_models = grouped_for_sort[grouped_for_sort['Variant'] == 'Tuned + Ensembled'] \
                .sort_values('MAE', ascending=False)['Base'].tolist()
base_models = [b for b in base_models if b != 'AutoGluon']  # Exclude AutoGluon (reference only)

# Create two vertically aligned subplots sharing x-axis
fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.6), sharex=True)

for ax, metric in zip(axes, metrics):
    # Compute statistics: mean, std, count, SE, 95% CI
    grouped = Data_frame.groupby(['Base', 'Variant'])[metric].agg(['mean', 'std', 'count']).reset_index()
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['ci'] = grouped.apply(
        lambda row: row['se'] * stats.t.ppf(1 - alpha / 2, row['count'] - 1) if row['count'] > 1 else 0,
        axis=1
    )

    x = np.arange(len(base_models))

    # Plot each variant as scattered points + error bars
    for var in ['Default', 'Tuned', 'Tuned + Ensembled']:
        sub = grouped[grouped['Variant'] == var]
        for i, base in enumerate(base_models):
            row = sub[sub['Base'] == base]
            if row.empty:
                continue
            mean_val = row['mean'].iloc[0]
            ci_val = row['ci'].iloc[0]
            pos = x[i] + offset_dict[var]

            color = model_colors.get(base, '#808080')
            marker = variant_markers[var]

            # White-filled marker with colored edge
            ax.scatter(
                pos, mean_val,
                c='white',
                s=10,
                marker=marker,
                edgecolors=color,
                zorder=3
            )

            # 95% confidence interval error bar
            ax.errorbar(
                pos, mean_val, yerr=ci_val,
                **error_bar_properties[var],
                zorder=2
            )

    # Axis limits and ticks (hard-coded for readability)
    if metric == 'RMSE':
        ax.set_yticks([0, 200, 400, 600])
        ax.set_ylim(0, 600)
    else:
        ax.set_yticks([0, 150, 300, 450])
        ax.set_ylim(0, 450)

    ax.set_title(metric_titles[metric])
    ax.set_ylabel(metric_ylabels[metric])
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    # Add legend only on the top subplot (RMSE)
    if metric == 'RMSE':
        variant_markers111 = {
            'Mean (D)': 'o',
            'Mean (T)': 'D',
            'Mean (T+E)': 's'
        }
        marker_handles = []
        for var, marker in variant_markers111.items():
            handle = Line2D(
                [0], [0],
                marker=marker,
                markerfacecolor='white',
                markeredgecolor='#1f77b4',
                markersize=3,
                linestyle='None',
                label=var
            )
            marker_handles.append(handle)

        leg1 = ax.legend(
            handles=marker_handles,
            loc='upper center',
            bbox_to_anchor=(0.625, 1.01),
            frameon=True,
            ncol=3,
            columnspacing=1.0,
            handletextpad=0.5,
            handlelength=0.8,
        )
        leg1.get_frame().set_linewidth(0.75)


# ─── Set x-axis labels only on the bottom subplot ───
axes[-1].set_xticks(x)
axes[-1].set_xticklabels(base_models, rotation=45, ha='right', rotation_mode='anchor')


# Adjust layout to avoid overlap
plt.tight_layout(pad=0.01)
plt.subplots_adjust(hspace=0.35)


# ─── Save figure in multiple formats ───
for ext in ['png', 'tiff', 'pdf']:
    plt.savefig(
        os.path.join(SAVE_DIR, f'RMSE_MAE_combined.{ext}'),
        dpi=400,
        bbox_inches='tight'
    )

# Display interactive figure (if running locally)
plt.show()