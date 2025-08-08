import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import pickle
from matplotlib.ticker import MultipleLocator

SAVE_DIR = './Datasets/OpenML/'
SAVE_DIR_simulation = './Datasets/Synthetic/HDR_fixed_data_size'
pkl_file = os.path.join(SAVE_DIR, 'Results_list.pkl')
pkl_file_simulation = os.path.join(SAVE_DIR_simulation, 'Results_simulation_list.pkl')
with open(pkl_file, 'rb') as f:
    Results_list = pickle.load(f)
with open(pkl_file_simulation, 'rb') as f:
    Results_simulation_list = pickle.load(f)

data = []
for dataset_idx in range(len(Results_list)):
    df = Results_list[dataset_idx]
    if df is None:
        continue
    for _, row in df.iterrows():
        data.append({'Dataset': dataset_idx, 'Model': row['Model'], 'RMSE': row['RMSE'], 'R2': row['R2'], 'MAE': row['MAE']})

for dataset_idx in range(len(Results_simulation_list)):
    df = Results_simulation_list[dataset_idx]
    if df is None:
        continue
    for _, row in df.iterrows():
        data.append({'Dataset': dataset_idx, 'Model': row['Model'], 'RMSE': row['RMSE'], 'R2': row['R2'], 'MAE': row['MAE']})
Data_frame = pd.DataFrame(data)

confidence = 0.95
alpha = 1 - confidence
free = len(Results_list) - 1
t_value = stats.t.ppf(1 - alpha/2, df=free)

metrics = ['RMSE', 'R2', 'MAE']
stats = {}
for metric in metrics:
    grouped = Data_frame.groupby(['Model'])[metric].agg(['mean', 'std', list]).reset_index()
    grouped['se'] = grouped['std'] / np.sqrt(len(Results_list))
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

models = ['TabPFN', 'SVR', 'RandomForest', 'ExtraTrees', 'KNN', 'Ridge', 'LightGBM', 'CatBoost', 'XGBoost', 'AutoGluon', 'MFTabPFN', 'ANN']
model_labels = ['TabPFN', 'SVM', 'RF', 'ExtraTrees', 'KNN', 'Ridge', 'LightGBM', 'CatBoost', 'XGBoost', 'AutoGluon', 'MFTabPFN', 'MLP']
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

for metric in metrics:
    fig, ax = plt.subplots(figsize=(4, 3))
    metric_data = stats[metric]
    plot_data = []
    for i, model in enumerate(models):
        if model in metric_data['Model'].values:
            mean = metric_data[metric_data['Model'] == model]['mean'].iloc[0]
            ci = metric_data[metric_data['Model'] == model]['ci'].iloc[0]
            plot_data.append({
                'model': model,
                'mean': mean,
                'ci': ci,
                'label': model_labels[i],
                'color': model_colors[model]
            })
        else:
            print(f"{model} is missing in {metric}")

    plot_data.sort(key=lambda x: x['mean'])
    sorted_means = [d['mean'] for d in plot_data]
    sorted_cis = [d['ci'] for d in plot_data]
    sorted_labels = [d['label'] for d in plot_data]
    sorted_colors = [d['color'] for d in plot_data]

    x = np.arange(len(plot_data))
    ax.bar(x, sorted_means, yerr=sorted_cis, capsize=2, color=sorted_colors, edgecolor='black', alpha=0.8)

    if metric == 'R2':
        ax.set_title(r'$R^2$', fontsize=12)
        ax.set_ylabel(r'$R^2$'f'(95% CI)', fontsize=12)
    else:
        ax.set_title(f'Normalized negative {metric}', fontsize=12)
        ax.set_ylabel(f'{metric} (95% CI)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_labels, rotation=45, ha='right')

    ax.set_axisbelow(True)

    if metric == 'R2':
        ax.set_yticks([-0.5, -0.2, 0.1, 0.4, 0.7, 1.0])
    else:
        max_y = 1.0
        min_y = 0.5
        ax.set_ylim(min_y, max_y)

    if metric == 'R2':
        pass
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # for ext in ['png', 'tiff']:
    #     output_path = os.path.join(SAVE_DIR, f'{metric.lower()}_bar_plot.{ext}')
    #     plt.savefig(output_path, dpi=500, bbox_inches='tight', format=ext)

    plt.show(block=True)
    plt.close(fig)

for metric in metrics:
    print(f"\n{metric} results:")
    print(stats[metric][['Model', 'mean', 'ci']])
