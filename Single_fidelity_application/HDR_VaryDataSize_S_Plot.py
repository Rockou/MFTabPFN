import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
from pathlib import Path

SAVE_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "Single-fidelity" / "Synthetic" / "Datasize"
pkl_file = os.path.join(SAVE_DIR, 'Results_simulation_train_list.pkl')
with open(pkl_file, 'rb') as f:
    Results_simulation_train_list = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_simulation_train_prediction.pkl')
with open(pkl_file, 'rb') as f:
    Results_simulation_train_prediction = pickle.load(f)


data_sizes = ['n', '2n', '3n', '4n', '5n']
confidence = 0.95
alpha = 1 - confidence
free = len(Results_simulation_train_list[0][0]) - 1
t_value = stats.t.ppf(1 - alpha / 2, df=free)

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


models = ['MFTabPFN', 'TabPFN', 'AutoGluon']
model_labels = ['MFTabPFN', 'TabPFN', 'AutoGluon']

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

model_markers = {
    'MFTabPFN': '^',
    'TabPFN': 'o',
    'AutoGluon': 's',
}

case_names = [
    'PF (n=100)',
    'PF (n=200)',
    'PF (n=300)',
    'PF (n=400)',
    'PF (n=500)'
]
metrics = ['RMSE_N', 'MAE_N', 'R2']
metric_names = {'RMSE_N': 'NNRMSE', 'MAE_N': 'NNMAE', 'R2': 'R2'}

all_stats_rows = []

all_stats = {}
for dataset_idx in range(len(Results_simulation_train_list)):
    data = []
    for size_idx in range(len(Results_simulation_train_list[0])):
        for sim_idx in range(len(Results_simulation_train_list[0][0])):
            df = Results_simulation_train_list[dataset_idx][size_idx][sim_idx]
            if df is None:
                continue
            for _, row in df.iterrows():
                model_name = row['Model'].replace(' (Default)', '')
                data.append({
                    'DataSize': data_sizes[size_idx],
                    'Model': model_name,
                    'RMSE_N': row['RMSE_N'],
                    'MAE_N': row['MAE_N'],
                    'R2': row['R2']
                })
    Data_frame = pd.DataFrame(data)

    stats_dict = {}
    for metric in metrics:
        grouped = Data_frame.groupby(['DataSize', 'Model'], sort=False)[metric].agg(['mean', 'std']).reset_index()
        grouped['se'] = grouped['std'] / np.sqrt(len(Results_simulation_train_list[0][0]))
        grouped['ci'] = grouped['se'] * t_value
        stats_dict[metric] = grouped
    all_stats[dataset_idx] = stats_dict

plt.rcParams['xtick.major.pad'] = 1.5
plt.rcParams['ytick.major.pad'] = 0.8

fig, axes = plt.subplots(3, 5, figsize=(6.8, 3.5), sharex=True)

ylim_settings = {
    0: {'RMSE_N': (0.60, 1.00, [0.60, 0.68, 0.76, 0.84, 0.92, 1.00]),
        'MAE_N': (0.70, 1.00, [0.70, 0.76, 0.82, 0.88, 0.94, 1.00]),
        'R2': (-0.50, 1.00, [-0.50, -0.20, 0.10, 0.40, 0.70, 1.00])},
    1: {'RMSE_N': (0.80, 1.00, [0.80, 0.84, 0.88, 0.92, 0.96, 1.00]),
        'MAE_N': (0.80, 1.00, [0.80, 0.84, 0.88, 0.92, 0.96, 1.00]),
        'R2': (-0.50, 1.00, [-0.50, -0.20, 0.10, 0.40, 0.70, 1.00])},
    2: {'RMSE_N': (0.75, 0.95, [0.75, 0.79, 0.83, 0.87, 0.91, 0.95]),
        'MAE_N': (0.80, 0.95, [0.80, 0.83, 0.86, 0.89, 0.92, 0.95]),
        'R2': (-0.2, 0.8, [-0.2, -0.0, 0.2, 0.4, 0.6, 0.8])},
    3: {'RMSE_N': (0.75, 1.00, [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]),
        'MAE_N': (0.80, 1.00, [0.80, 0.84, 0.88, 0.92, 0.96, 1.00]),
        'R2': (0.00, 1.00, [0.00, 0.20, 0.40, 0.60, 0.80, 1.00])},
    4: {'RMSE_N': (0.80, 1.00, [0.80, 0.84, 0.88, 0.92, 0.96, 1.00]),
        'MAE_N': (0.80, 1.00, [0.80, 0.84, 0.88, 0.92, 0.96, 1.00]),
        'R2': (0.20, 1.00, [0.20, 0.36, 0.52, 0.68, 0.86, 1.00])}
}

ylabels = {'RMSE_N': 'NNRMSE (95% CI)', 'MAE_N': 'NNMAE (95% CI)', 'R2': r'$R^2$ (95% CI)'}

x_pos = np.arange(len(data_sizes))

for row, metric in enumerate(metrics):
    for col, dataset_idx in enumerate(range(5)):
        ax = axes[row, col]
        stats = all_stats[dataset_idx][metric]

        for i, model in enumerate(models):
            if model in stats['Model'].values:
                model_data = stats[stats['Model'] == model]
                means = model_data['mean'].values
                cis = model_data['ci'].values
                label = model_labels[i] if (row == 0 and col == 0) else None
                ax.plot(x_pos, means, label=label, color=model_colors[model], linewidth=1.5,
                        marker=model_markers[model], markersize=3)
                ax.fill_between(x_pos, means - cis, means + cis, color=model_colors[model], alpha=0.2, edgecolor='none')

        if row == 0:
            ax.set_title(case_names[dataset_idx])
        if col == 0:
            if row == 2:
                ax.set_ylabel(ylabels[metric])
                ax.yaxis.set_label_coords(-0.30, 0.5)
            else:
                ax.set_ylabel(ylabels[metric])
        if row == 2:
            ax.set_xlabel('Total data size')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(data_sizes)
        ax.set_xlim(-0.15, 4.15)

        low, high, ticks = ylim_settings[dataset_idx][metric]
        ax.set_ylim(low, high)
        ax.set_yticks(ticks)

        ax.grid(True, axis='both', linestyle='--', linewidth=0.5, alpha=0.7)

        if row == 0 and col == 0:
            ax.legend(loc='lower center', ncol=1, bbox_to_anchor=(0.7, -0.05),
                      columnspacing=1.0,
                      labelspacing=0.15,
                      handletextpad=0.4,
                      handlelength=1.2,
                      frameon=False,
                      fontsize=6)

plt.tight_layout(pad=0.01)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.30, hspace=0.20)

for ext in ['png', 'tiff', 'pdf']:
    output_path = os.path.join(SAVE_DIR, f'combined_datasize_evolution_plot.{ext}')
    plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)

plt.show()
plt.close(fig)

