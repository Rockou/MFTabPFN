import pandas as pd
import numpy as np
import matplotlib
# Use TkAgg backend for interactive matplotlib windows (common on desktop systems)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import pickle
from scipy.stats import wilcoxon
import seaborn as sns
from collections import defaultdict
from pathlib import Path

# ────────────────────────────────────────────────────────────────
# Load pickled results from two experiment sources
# ────────────────────────────────────────────────────────────────

SAVE_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "Single-fidelity" / "CTR23" / "Result"
pkl_file = os.path.join(SAVE_DIR, 'Results_ctr23_list.pkl')
with open(pkl_file, 'rb') as f:
    Results_ctr23_list = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_ctr23_prediction.pkl')
with open(pkl_file, 'rb') as f:
    Results_ctr23_prediction = pickle.load(f)

SAVE_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "Single-fidelity" / "Synthetic" / "HDR"
pkl_file = os.path.join(SAVE_DIR, 'Results_simulation_list.pkl')
with open(pkl_file, 'rb') as f:
    Results_simulation_list = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_simulation_prediction.pkl')
with open(pkl_file, 'rb') as f:
    Results_simulation_prediction = pickle.load(f)

# ────────────────────────────────────────────────────────────────
# Global publication-style matplotlib settings (small fonts, thin lines)
# ────────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 7
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['xtick.major.size'] = 2.0
plt.rcParams['ytick.major.size'] = 2.0
plt.rcParams['xtick.minor.size'] = 1.5
plt.rcParams['ytick.minor.size'] = 1.5

# Variant list and model color mapping for visualization
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

# ────────────────────────────────────────────────────────────────
# Combine prediction results from CTR23 and HDR experiments
# ────────────────────────────────────────────────────────────────
Results_prediction = Results_ctr23_prediction + Results_simulation_prediction

data = []
for dataset_idx in range(len(Results_prediction)):
    for sim_idx in range(len(Results_prediction[dataset_idx])):
        df = Results_prediction[dataset_idx][sim_idx]
        if df is None:
            continue
        for _, row in df.iterrows():
            data.append({
                'Dataset': dataset_idx,
                'Fold': sim_idx,
                'Model': row['Model'],
                'Performance': row['Performance'],
            })

Data_frame = pd.DataFrame(data)


def extract_base_variant(model_name):
    """
    Parse model name into base model and variant.
    Special case for 'AutoGluon'; 'Best' mapped to 'Tuned + Ensembled'.
    """
    if model_name == 'AutoGluon':
        return 'AutoGluon', 'Tuned + Ensembled'
    if ' (' in model_name:
        base, var = model_name.split(' (')
        var = var[:-1]
    else:
        base = model_name
        var = 'Best'
    return base, var


# Add parsed columns for grouping
Data_frame['Base'] = Data_frame['Model'].apply(lambda x: extract_base_variant(x)[0])
Data_frame['Variant'] = Data_frame['Model'].apply(lambda x: extract_base_variant(x)[1])


def selected_vs_models(selected_model1):
    """
    Perform pairwise Wilcoxon signed-rank test between selected model (MFTabPFN)
    and all other models, across all datasets/folds.
    Computes mean p-value per baseline model + variant.
    """
    # Extract actual (ground truth) performance
    Data_actual = Data_frame[Data_frame['Base'] == 'Actual'].copy()

    # Exclude AutoGluon and Actual from comparison
    df = Data_frame[~Data_frame['Base'].isin(['AutoGluon', 'Actual'])].copy()
    df['P'] = np.nan

    selected_model = selected_model1
    mftab = df[df['Base'] == selected_model].copy()
    mftab = mftab[['Dataset', 'Fold', 'Variant', 'Performance']].rename(
        columns={'Performance': 'Performance_MFTabPFN'})
    mftab['Performance_MFTabPFN'] = mftab['Performance_MFTabPFN'].apply(lambda x: x.ravel())

    # All other models
    others = df[df['Base'] != selected_model].copy()
    others['Performance'] = others['Performance'].apply(lambda x: x.ravel())

    # Merge with actual performance and MFTabPFN performance
    others = others.merge(
        Data_actual[['Dataset', 'Fold', 'Performance']],
        on=['Dataset', 'Fold'],
        how='left',
        suffixes=('', '_Actual')
    )
    others = others.merge(
        mftab,
        on=['Dataset', 'Fold', 'Variant'],
        how='inner'
    )

    # Compute absolute errors
    others['AE_Model'] = np.abs(others['Performance'] - others['Performance_Actual'])
    others['AE_MFTabPFN'] = np.abs(others['Performance_MFTabPFN'] - others['Performance_Actual'])
    others['diff'] = others['AE_Model'] - others['AE_MFTabPFN']

    # Wilcoxon signed-rank test (one-sided: model error > MFTabPFN error)
    for j in range(len(others)):
        diff_row = others['diff'].iloc[j]
        stat, p = wilcoxon(diff_row, alternative='greater')
        others.loc[j, "P"] = p

    # Aggregate mean p-value per dataset + model + variant
    confidence = 0.95
    alpha = 1 - confidence
    grouped = others.groupby(['Dataset', 'Base', 'Variant'])['P'].agg(['mean', 'std', 'count']).reset_index()
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['ci'] = grouped.apply(
        lambda row: row['se'] * stats.t.ppf(1 - alpha / 2, row['count'] - 1) if row['count'] > 1 else 0,
        axis=1
    )
    return grouped


# Compute statistical comparison: MFTabPFN vs all baselines
grouped = selected_vs_models('MFTabPFN')

# Adjust dataset index (start from 1 for readability)
grouped['Dataset'] = grouped['Dataset'] + 1

# Shorten variant names for display
variant_map = {'Default': 'D', 'Tuned': 'T', 'Tuned + Ensembled': 'T+E'}
grouped['Variant_short'] = grouped['Variant'].map(variant_map)
grouped['Model_Variant'] = grouped['Base'] + ' (' + grouped['Variant_short'] + ')'

# ────────────────────────────────────────────────────────────────
# Pivot table: mean p-value of MFTabPFN superiority per baseline
# ────────────────────────────────────────────────────────────────
pivot = grouped.pivot_table(
    values='mean',
    index='Dataset',
    columns='Model_Variant',
    aggfunc='first'
).sort_index(ascending=False)


# Count number of datasets where p < 0.05 (significant superiority)
def count_significance(col):
    p001 = (col < 0.001).sum()
    p01 = (col < 0.01).sum() - p001
    p05 = (col < 0.05).sum() - p01 - p001
    return p01 + p05 + p001


sig_counts = pivot.apply(count_significance, axis=0)


# Convert p-value to significance stars/symbols
def p_to_stars(p):
    if p < 0.001:
        return '#'
    elif p < 0.01:
        return '&'
    elif p < 0.05:
        return '*'
    else:
        return ''

base_to_columns = defaultdict(list)
for col in pivot.columns:
    base = col.split(' (')[0]
    base_to_columns[base].append(col)

base_scores = {}
for base, cols in base_to_columns.items():
    te_col = f"{base} (T+E)"
    if te_col in pivot.columns:
        base_scores[base] = sig_counts[te_col]
    else:
        base_scores[base] = 0

variant_priority = {'D': 0, 'T': 1, 'T+E': 2}

def get_variant(col):
    return col.split('(')[1].split(')')[0]

sorted_columns = []
for base in sorted(base_to_columns.keys(),
                   key=lambda b: base_scores[b], reverse=True):
    cols = base_to_columns[base]
    cols_sorted = sorted(cols, key=lambda c: variant_priority.get(get_variant(c), 99))
    sorted_columns.extend(cols_sorted)

pivot = pivot[sorted_columns]
sig_counts = sig_counts[sorted_columns]
stars = pivot.applymap(p_to_stars)

# ─── Plot heatmap of mean p-values with significance annotations ───
plt.figure(figsize=(6.8, 4.5))
ax = sns.heatmap(
    pivot,
    cmap="RdYlBu_r",  # Red (high p) → Blue (low p)
    annot=stars.values,  # Show * / & / # instead of numbers
    fmt='',
    linewidths=0.5,
    linecolor='white',
    cbar_kws={
        'label': 'Mean p-value',
        'shrink': 1.0,
        'pad': 0.01,
        'aspect': 25
    },
    vmin=0, vmax=1.0,
    annot_kws={'fontsize': 5, 'fontweight': 'bold', 'color': 'white'},
)

# Customize colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=5)
cbar.ax.yaxis.set_label_position('right')
cbar.ax.yaxis.set_label_coords(2.9, 0.5)

# Add significance count (number of datasets where p<0.05) above each column
for i, col in enumerate(pivot.columns):
    count_text = sig_counts.iloc[i]
    ax.text(i + 0.5, 1.025, count_text,
            ha='center', va='top',
            fontsize=7,
            color='blue',
            transform=ax.get_xaxis_transform())

plt.title('MFTabPFN vs Baselines: Wilcoxon p-value', x=0.5, y=1.03)
plt.ylabel('Dataset')
plt.xlabel('')
plt.yticks(rotation=0)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

plt.tight_layout(pad=0.01)

# Save in multiple formats
for ext in ['png', 'tiff', 'pdf']:
    output_path = os.path.join(SAVE_DIR, f'Wilcoxon_plot.{ext}')
    plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)

plt.show()

# ────────────────────────────────────────────────────────────────
# Pairwise win-rate matrix: count datasets where model A beats model B (p<0.05)
# ────────────────────────────────────────────────────────────────

Data_actual = Data_frame[Data_frame['Base'] == 'Actual'].copy()
Data_actual = Data_actual.set_index(['Dataset', 'Fold'])['Performance']
Data_actual = Data_actual.apply(lambda x: np.ravel(x) if isinstance(x, np.ndarray) else np.array([x]))

# List all model-variant combinations (exclude Actual & AutoGluon)
all_model_variants = Data_frame[~Data_frame['Base'].isin(['Actual', 'AutoGluon'])].drop_duplicates(['Base', 'Variant'])
all_model_variants['Model_Variant'] = all_model_variants['Base'] + ' (' + \
                                      all_model_variants['Variant'].map(
                                          {'Default': 'D', 'Tuned': 'T', 'Tuned + Ensembled': 'T+E'}) + ')'
all_models = all_model_variants['Model_Variant'].tolist()

# Initialize full win matrix
win_matrix_full = pd.DataFrame(0, index=all_models, columns=all_models, dtype=int)
variant_full = {'D': 'Default', 'T': 'Tuned', 'T+E': 'Tuned + Ensembled'}

print("Calculating pairwise win rate matrix...")
for i, model_i in enumerate(all_models):
    for j, model_j in enumerate(all_models):
        if i == j:
            continue
        base_i, var_i = model_i.split(' (')
        var_i = variant_full[var_i[:-1]]
        base_j, var_j = model_j.split(' (')
        var_j = variant_full[var_j[:-1]]

        # Extract performance for both models
        perf_i = \
        Data_frame[(Data_frame['Base'] == base_i) & (Data_frame['Variant'] == var_i)].set_index(['Dataset', 'Fold'])[
            'Performance']
        perf_j = \
        Data_frame[(Data_frame['Base'] == base_j) & (Data_frame['Variant'] == var_j)].set_index(['Dataset', 'Fold'])[
            'Performance']
        common = perf_i.index.intersection(perf_j.index)
        if len(common) == 0:
            continue

        win_count = 0
        dataset_pvals = {}
        for idx in common:
            dataset_id, fold_id = idx
            p_i = np.ravel(perf_i.loc[idx])
            p_j = np.ravel(perf_j.loc[idx])
            p_a = np.ravel(Data_actual.loc[idx])
            min_len = min(len(p_i), len(p_j), len(p_a))
            if min_len < 2:
                continue
            # Absolute error difference (positive → model_i worse)
            diff = np.abs(p_i[:min_len] - p_a[:min_len]) - np.abs(p_j[:min_len] - p_a[:min_len])
            _, p = wilcoxon(diff, alternative='greater')
            if dataset_id not in dataset_pvals:
                dataset_pvals[dataset_id] = []
            dataset_pvals[dataset_id].append(p)

        # Count datasets where average p < 0.05 (model_j significantly better)
        for dataset_id, pvals in dataset_pvals.items():
            avg_p = np.mean(pvals)
            if avg_p < 0.05:
                win_count += 1

        win_matrix_full.loc[model_i, model_j] = win_count

win_matrix_full = win_matrix_full.astype(float)
np.fill_diagonal(win_matrix_full.values, np.nan)

# Ensure TabDPT variants are included (even if empty)
new_rows_cols = ['TabDPT (T)', 'TabDPT (T+E)']
win_matrix_full = win_matrix_full.reindex(
    index=win_matrix_full.index.union(new_rows_cols),
    columns=win_matrix_full.columns.union(new_rows_cols),
    fill_value=np.nan
)


def plot_three_matrices_in_one_row():
    """
    Plot three separate win-rate heatmaps side-by-side for:
    - Default (D)
    - Tuned (T)
    - Tuned + Ensembled (T+E)
    Each shows number of datasets where row model beats column model (p<0.05).
    """
    configs = [
        ('D', 'Default', False,
         ['FastaiMLP (D)', 'EBM (D)', 'ExtraTrees (D)', 'XGBoost (D)', 'LightGBM (D)', 'TorchMLP (D)', 'TabDPT (D)',
          'TabM (D)', 'ModernNCA (D)', 'CatBoost (D)', 'RealMLP (D)', 'TabPFN (D)', 'MFTabPFN (D)']),
        ('T', 'Tuned', False,
         ['FastaiMLP (T)', 'EBM (T)', 'ExtraTrees (T)', 'XGBoost (T)', 'LightGBM (T)', 'TorchMLP (T)', 'TabDPT (T)',
          'TabM (T)', 'ModernNCA (T)', 'CatBoost (T)', 'RealMLP (T)', 'TabPFN (T)', 'MFTabPFN (T)']),
        ('T+E', 'Tuned + Ensembled', True,
         ['FastaiMLP (T+E)', 'EBM (T+E)', 'ExtraTrees (T+E)', 'XGBoost (T+E)', 'LightGBM (T+E)', 'TorchMLP (T+E)',
          'TabDPT (T+E)', 'TabM (T+E)', 'ModernNCA (T+E)', 'CatBoost (T+E)', 'RealMLP (T+E)', 'TabPFN (T+E)',
          'MFTabPFN (T+E)'])
    ]

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.602), constrained_layout=True)

    for ax, (v_short, name, show_cbar, custom_order) in zip(axes, configs):
        # Select columns for current variant
        cols = [c for c in win_matrix_full.columns if f'({v_short})' in c]
        if not cols:
            ax.axis('off')
            continue

        mat = win_matrix_full.loc[cols, cols]
        x_order = [m for m in custom_order if m in mat.columns]
        y_order = x_order[::-1]  # Reverse for better visual hierarchy
        mat = mat.loc[y_order, x_order]

        sns.heatmap(
            mat,
            cmap="viridis_r",  # Dark → light: higher wins → darker
            annot=True,
            fmt='.0f',
            linewidths=0.5,
            linecolor='white',
            cbar=show_cbar,
            cbar_kws={'label': 'Number of datasets', 'shrink': 1.0, 'pad': 0.01, 'aspect': 25} if show_cbar else None,
            square=True,
            ax=ax,
            annot_kws={'fontsize': 6, 'color': 'white', 'fontweight': 'bold'}
        )

        clean_labels = [label.split(' (')[0] for label in x_order]
        current_ticks = ax.get_xticks()

        # Hide TabDPT x-labels in Tuned and T+E plots (often missing/sparse)
        if name != 'Default':
            new_labels = []
            new_ticks = []
            for i, (full_name, label) in enumerate(zip(x_order, clean_labels)):
                if 'TabDPT' in full_name:
                    new_labels.append('')
                else:
                    new_labels.append(label)
                    new_ticks.append(current_ticks[i])
        else:
            new_labels = clean_labels
            new_ticks = current_ticks

        ax.set_xticklabels(new_labels, rotation=80, ha='right', rotation_mode='anchor')
        ax.set_xticks(new_ticks)

        # Y-labels only on leftmost plot (Default)
        if name == 'Default':
            clean_y_labels = [label.split(' (')[0] for label in y_order]
            ax.set_yticklabels(clean_y_labels, rotation=0)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax.set_title(f'{v_short} configuration')

        # Customize colorbar (only on rightmost plot)
        if show_cbar:
            cbar = ax.collections[0].colorbar
            max_val = int(mat.max().max())
            ticks = np.arange(0, max_val, 5)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([int(t) for t in ticks])
            cbar.ax.tick_params(labelsize=5)

    # Figure labels
    fig.text(0.51, 0.001, 'Winner', ha='center', va='center', fontsize=8)
    fig.text(0.001, 0.6, 'Loser', ha='center', va='center', rotation='vertical', fontsize=8)

    # Save in multiple formats
    for ext in ['png', 'tiff', 'pdf']:
        output_path = os.path.join(SAVE_DIR, f'Win_Matrix_Combined.{ext}')
        plt.savefig(output_path, dpi=400, bbox_inches='tight')

    plt.show()


# Generate the three-panel win-rate matrix visualization
plot_three_matrices_in_one_row()