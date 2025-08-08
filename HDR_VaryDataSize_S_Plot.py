import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import pickle
from matplotlib.ticker import MultipleLocator

SAVE_DIR = './Datasets/Synthetic/HDR_varying_data_size'
pkl_file = os.path.join(SAVE_DIR, 'Results_simulation_train_list.pkl')
with open(pkl_file, 'rb') as f:
    Results_simulation_train_list = pickle.load(f)

Data_simulation = []
data_sizes = ['n', '2n', '3n', '4n', '5n']
for size_idx in range(len(Results_simulation_train_list[0])):
    data = []
    for dataset_idx in range(len(Results_simulation_train_list)):
        df = Results_simulation_train_list[dataset_idx][size_idx]
        if df is None:
            continue
        for _, row in df.iterrows():
            data.append({
                'Dataset': dataset_idx,
                'DataSize': data_sizes[size_idx],
                'Model': row['Model'],
                'RMSE': row['RMSE'],
                'R2': row['R2'],
                'MAE': row['MAE']
            })
    Data_frame = pd.DataFrame(data)
    Data_simulation.append(Data_frame)

confidence = 0.95
alpha = 1 - confidence
free = len(Results_simulation_train_list) - 1
t_value = stats.t.ppf(1 - alpha / 2, df=free)

metrics = ['RMSE', 'R2', 'MAE']
stats = {}
for metric in metrics:
    grouped = Data_simulation[0].groupby(['DataSize', 'Model'])[metric].agg(['mean', 'std']).reset_index()
    for df in Data_simulation[1:]:
        temp = df.groupby(['DataSize', 'Model'])[metric].agg(['mean', 'std']).reset_index()
        grouped = pd.concat([grouped, temp], ignore_index=True)
    grouped['se'] = grouped['std'] / np.sqrt(len(Results_simulation_train_list))
    grouped['ci'] = grouped['se'] * t_value
    stats[metric] = grouped

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
# plt.rcParams['figure.dpi'] = 500

models = ['MFTabPFN', 'TabPFN', 'AutoGluon']
model_labels = ['MFTabPFN', 'TabPFN', 'AutoGluon']

model_colors = {
    'TabPFN': '#2ca02c',
    'SVR': '#ff7f0e',
    'RandomForest': '#1b9e77',
    'ExtraTrees': '#d62728',
    'KNN': '#17becf',
    'Ridge': '#8c564b',
    'LightGBM': '#e377c2',
    'CatBoost': '#7f7f7f',
    'XGBoost': '#bcbd22',
    'AutoGluon': '#9467bd',
    'MFTabPFN': '#1f77b4',
    'ANN': '#f7b6d2'
}
model_markers = {
    'MFTabPFN': '^',
    'TabPFN': 'o',
    'AutoGluon': 's',
    'ExtraTrees': 'D',
    'CatBoost': '*'
}

for metric in metrics:
    fig, ax = plt.subplots(figsize=(4, 3))
    metric_data = stats[metric]
    for i, model in enumerate(models):
        if model in metric_data['Model'].values:
            model_data = metric_data[metric_data['Model'] == model]
            means = model_data['mean'].values
            cis = model_data['ci'].values
            x = np.arange(len(data_sizes))
            ax.plot(x, means, label=model_labels[i], color=model_colors[model], linewidth=2.0, marker=model_markers[model], markersize=6)
            ax.fill_between(x, means - cis, means + cis, color=model_colors[model], alpha=0.2, edgecolor='none')
        else:
            print(f"{model} is missing in {metric}")

    if metric == 'R2':
        ax.set_title(r'$R^2$', fontsize=12)
        ax.set_ylabel(r'$R^2$'f'(95% CI)', fontsize=12)
    else:
        ax.set_title(f'Normalized negative {metric}', fontsize=12)
        ax.set_ylabel(f'{metric} (95% CI)', fontsize=12)
    ax.set_xlabel('Training data size', fontsize=12)
    ax.set_xticks(np.arange(len(data_sizes)))
    ax.set_xticklabels(data_sizes)
    ax.set_xlim(-0.2, 4.2)
    means = metric_data['mean'].values
    cis = metric_data['ci'].values

    if metric == 'R2':
        ax.set_ylim(0, 1.0)
    elif metric == 'MAE':
        ax.set_yticks([0.85, 0.88, 0.91, 0.94, 0.97, 1.0])
    elif metric == 'RMSE':
        max_y = 1.0
        min_y = 0.8
        ax.set_ylim(min_y, max_y)

    if metric == 'RMSE':
        ax.yaxis.set_major_locator(MultipleLocator(0.04))
    elif metric == 'MAE':
        pass

    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # for ext in ['png', 'tiff']:
    #     output_path = os.path.join(SAVE_DIR, f'{metric.lower()}_evolution_plot.{ext}')
    #     plt.savefig(output_path, dpi=500, bbox_inches='tight', format=ext)

    plt.show(block=True)
    plt.close(fig)

for metric in metrics:
    print(f"\n{metric} results:")
    print(stats[metric][['DataSize', 'Model', 'mean', 'ci']])
