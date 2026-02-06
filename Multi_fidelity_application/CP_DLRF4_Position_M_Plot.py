import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.patches import Patch
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.interpolate import interp1d
from pathlib import Path

SAVE_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "Multi-fidelity" / "DLRF4" / "Result"

pkl_file = os.path.join(SAVE_DIR, 'Results_DLRF4_input.pkl')
with open(pkl_file, 'rb') as f:
    Results_DLRF4_input = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_DLRF4_list.pkl')
with open(pkl_file, 'rb') as f:
    Results_DLRF4_list = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_DLRF4_prediction.pkl')
with open(pkl_file, 'rb') as f:
    Results_DLRF4_prediction = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_DLRF4_list_MFGPR.pkl')
with open(pkl_file, 'rb') as f:
    Results_DLRF4_list_MFGPR = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_DLRF4_prediction_MFGPR.pkl')
with open(pkl_file, 'rb') as f:
    Results_DLRF4_prediction_MFGPR = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_DLRF4_list_MAHNN.pkl')
with open(pkl_file, 'rb') as f:
    Results_DLRF4_list_MAHNN = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_DLRF4_prediction_MAHNN.pkl')
with open(pkl_file, 'rb') as f:
    Results_DLRF4_prediction_MAHNN = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_DLRF4_list_FNO.pkl')
with open(pkl_file, 'rb') as f:
    Results_DLRF4_list_FNO = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_DLRF4_prediction_FNO.pkl')
with open(pkl_file, 'rb') as f:
    Results_DLRF4_prediction_FNO = pickle.load(f)

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

model_colors = {
    'MFGPR': '#1b9e77',
    'MFTabPFN': '#1f77b4',
    'NMFGPR': '#17becf',
    'TLFNO': '#8c564b',
    'MAHNN': '#bcbd88',
    'TabPFN-H': '#2ca02c',
    'TabPFN-M': '#2ca02c',
    'AutoGluon-H': '#9467bd',
    'AutoGluon-M': '#9467bd',
}
model_markers = {
    'MFTabPFN': '^',
    'MFGPR': 'o',
    'NMFGPR': 's',
    'TLFNO': '<',
    'MAHNN': '>',
}
confidence = 0.95
alpha = 1 - confidence

n_performances = 4
n_simulations = 3

plt.rcParams['xtick.major.pad'] = 1.5
plt.rcParams['ytick.major.pad'] = 0.8

pearsonr_list = []
cosine_list = []
for k in range(0, n_simulations):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(6.8, 1.6),
    )
    for j in range(0, n_performances):
        YY_total_yuan_high = Results_DLRF4_input[j].loc[0, 'YY_total_yuan_high']
        Y_test_prediction_MFTabPFN = Results_DLRF4_prediction[j][k].query("Model == 'MFTabPFN'")['Y_test_prediction'].iloc[0]
        Y_test_prediction_LinearMFGP = Results_DLRF4_prediction_MFGPR[j][k].query("Model == 'LinearMFGP'")['Y_test_prediction'].iloc[0]
        Y_test_prediction_NonlinearMFGP = Results_DLRF4_prediction_MFGPR[j][k].query("Model == 'NonlinearMFGP'")['Y_test_prediction'].iloc[0]
        Y_test_prediction_FNO = Results_DLRF4_prediction_FNO[j][k].query("Model == 'FNO'")['Y_test_prediction'].iloc[0]
        Y_test_prediction_MAHNN = Results_DLRF4_prediction_MAHNN[j][k].query("Model == 'MA-HNN'")['Y_test_prediction'].iloc[0]

        Y_test_prediction_MFTabPFN_low = Results_DLRF4_prediction[j][k].query("Model == 'MFTabPFN'")['Y_test_prediction_low'].iloc[0]
        Y_test_prediction_LinearMFGP_low = Results_DLRF4_prediction_MFGPR[j][k].query("Model == 'LinearMFGP'")['Y_test_prediction_low'].iloc[0]
        Y_test_prediction_NonlinearMFGP_low = Results_DLRF4_prediction_MFGPR[j][k].query("Model == 'NonlinearMFGP'")['Y_test_prediction_low'].iloc[0]
        Y_test_prediction_FNO_low = Results_DLRF4_prediction_FNO[j][k].query("Model == 'FNO'")['Y_test_prediction_low'].iloc[0]
        Y_test_prediction_MAHNN_low = Results_DLRF4_prediction_MAHNN[j][k].query("Model == 'MA-HNN'")['Y_test_prediction_low'].iloc[0]

        Y_TARGET = Results_DLRF4_input[j].loc[0, 'Y_TARGET']
        XX_total_yuan = Results_DLRF4_input[j].loc[0, 'XX_total_yuan']
        XX_train_yuan_low = Results_DLRF4_input[j].loc[0, 'XX_train_yuan_low']
        YY_train_yuan_low = Results_DLRF4_input[j].loc[0, 'YY_train_yuan_low']
        mask_xia = (XX_train_yuan_low[:, 1] == Y_TARGET[j]) & (XX_train_yuan_low[:, 2] == -1)
        filtered_input111 = XX_train_yuan_low[mask_xia]
        filtered_input_xia = filtered_input111[:, 0]
        filtered_output_xia = YY_train_yuan_low[mask_xia]
        simulation_quadratic = interp1d(filtered_input_xia, filtered_output_xia, kind='quadratic', bounds_error=False, fill_value="extrapolate")#linear
        index_xia = np.where(XX_total_yuan[:, 2] == -1)[0]
        filtered_input111 = XX_total_yuan[index_xia, :]
        filtered_input_test_xia = filtered_input111[:, 0]
        filtered_output_test_xia = simulation_quadratic(filtered_input_test_xia)
        mask_shang = (XX_train_yuan_low[:, 1] == Y_TARGET[j]) & (XX_train_yuan_low[:, 2] == 1)
        filtered_input111 = XX_train_yuan_low[mask_shang]
        filtered_input_shang = filtered_input111[:, 0]
        filtered_output_shang = YY_train_yuan_low[mask_shang]
        simulation_quadratic = interp1d(filtered_input_shang, filtered_output_shang, kind='quadratic', bounds_error=False, fill_value="extrapolate")#linear
        index_shang = np.where(XX_total_yuan[:, 2] == 1)[0]
        filtered_input111 = XX_total_yuan[index_shang, :]
        filtered_input_test_shang = filtered_input111[:, 0]
        filtered_output_test_shang = simulation_quadratic(filtered_input_test_shang)
        YY_total_yuan_high_simulation = YY_total_yuan_high.copy()
        YY_total_yuan_high_simulation[index_xia] = filtered_output_test_xia
        YY_total_yuan_high_simulation[index_shang] = filtered_output_test_shang

        Difference_MFTabPFN = Y_test_prediction_MFTabPFN.ravel() - Y_test_prediction_MFTabPFN_low.ravel()
        Difference_LinearMFGP = Y_test_prediction_LinearMFGP.ravel() - Y_test_prediction_LinearMFGP_low.ravel()
        Difference_NonlinearMFGP = Y_test_prediction_NonlinearMFGP.ravel() - Y_test_prediction_NonlinearMFGP_low.ravel()
        Difference_FNO = Y_test_prediction_FNO.ravel() - Y_test_prediction_FNO_low.ravel()
        Difference_MAHNN = Y_test_prediction_MAHNN.ravel() - Y_test_prediction_MAHNN_low.ravel()

        Difference_MFTabPFN_yuan = YY_total_yuan_high.ravel() - YY_total_yuan_high_simulation.ravel()
        Difference_LinearMFGP_yuan = YY_total_yuan_high.ravel() - YY_total_yuan_high_simulation.ravel()
        Difference_NonlinearMFGP_yuan = YY_total_yuan_high.ravel() - YY_total_yuan_high_simulation.ravel()
        Difference_FNO_yuan = YY_total_yuan_high.ravel() - YY_total_yuan_high_simulation.ravel()
        Difference_MAHNN_yuan = YY_total_yuan_high.ravel() - YY_total_yuan_high_simulation.ravel()

        corr_MFTabPFN = pearsonr(Difference_MFTabPFN_yuan, Difference_MFTabPFN)[0].ravel()
        corr_LinearMFGP = pearsonr(Difference_LinearMFGP_yuan, Difference_LinearMFGP)[0].ravel()
        corr_NonlinearMFGP = pearsonr(Difference_NonlinearMFGP_yuan, Difference_NonlinearMFGP)[0].ravel()
        corr_FNO = pearsonr(Difference_FNO_yuan, Difference_FNO)[0].ravel()
        corr_MAHNN = pearsonr(Difference_MAHNN_yuan, Difference_MAHNN)[0].ravel()

        cosine_MFTabPFN = cosine_similarity(Difference_MFTabPFN_yuan.reshape(1, -1), Difference_MFTabPFN.reshape(1, -1)).ravel()
        cosine_LinearMFGP = cosine_similarity(Difference_LinearMFGP_yuan.reshape(1, -1), Difference_LinearMFGP.reshape(1, -1)).ravel()
        cosine_NonlinearMFGP = cosine_similarity(Difference_NonlinearMFGP_yuan.reshape(1, -1), Difference_NonlinearMFGP.reshape(1, -1)).ravel()
        cosine_FNO = cosine_similarity(Difference_FNO_yuan.reshape(1, -1), Difference_FNO.reshape(1, -1)).ravel()
        cosine_MAHNN = cosine_similarity(Difference_MAHNN_yuan.reshape(1, -1), Difference_MAHNN.reshape(1, -1)).ravel()

        pearsonr_list.append({'Model': 'MFTabPFN', 'Dataset': j, 'Fold': k, 'pearsonr': corr_MFTabPFN})
        pearsonr_list.append({'Model': 'MFGPR', 'Dataset': j, 'Fold': k, 'pearsonr': corr_LinearMFGP})
        pearsonr_list.append({'Model': 'NMFGPR', 'Dataset': j, 'Fold': k, 'pearsonr': corr_NonlinearMFGP})
        pearsonr_list.append({'Model': 'TLFNO', 'Dataset': j, 'Fold': k, 'pearsonr': corr_FNO})
        pearsonr_list.append({'Model': 'MAHNN', 'Dataset': j, 'Fold': k, 'pearsonr': corr_MAHNN})

        cosine_list.append({'Model': 'MFTabPFN', 'Dataset': j, 'Fold': k, 'cosine': cosine_MFTabPFN})
        cosine_list.append({'Model': 'MFGPR', 'Dataset': j, 'Fold': k, 'cosine': cosine_LinearMFGP})
        cosine_list.append({'Model': 'NMFGPR', 'Dataset': j, 'Fold': k, 'cosine': cosine_NonlinearMFGP})
        cosine_list.append({'Model': 'TLFNO', 'Dataset': j, 'Fold': k, 'cosine': cosine_FNO})
        cosine_list.append({'Model': 'MAHNN', 'Dataset': j, 'Fold': k, 'cosine': cosine_MAHNN})

        ax = axes[j]
        all_data = np.concatenate([Difference_MFTabPFN, Difference_LinearMFGP, Difference_NonlinearMFGP, Difference_FNO, Difference_MAHNN,
                                   Difference_MFTabPFN_yuan, Difference_LinearMFGP_yuan, Difference_NonlinearMFGP_yuan,
                                   Difference_FNO_yuan, Difference_MAHNN_yuan])
        x_min, x_max = all_data.min(), all_data.max()
        x_range = x_max - x_min if x_max > x_min else 1.0

        norm = lambda arr: (arr - x_min) / x_range

        ax.plot([0, 1], [0, 1], '--', color='gray', lw=1.0, zorder=0)
        ax.scatter(norm(Difference_MFTabPFN_yuan), norm(Difference_MFTabPFN),
                           color=model_colors['MFTabPFN'], s=2, alpha=1.0, marker=model_markers['MFTabPFN'],
                           label='MFTabPFN', zorder=5)
        ax.scatter(norm(Difference_LinearMFGP_yuan), norm(Difference_LinearMFGP),
                           color=model_colors['MFGPR'], s=2, alpha=1.0, marker=model_markers['MFGPR'],
                           label='MFGPR', zorder=4)
        ax.scatter(norm(Difference_NonlinearMFGP_yuan), norm(Difference_NonlinearMFGP),
                           color=model_colors['NMFGPR'], s=2, alpha=1.0, marker=model_markers['NMFGPR'],
                           label='NMFGPR', zorder=3)
        ax.scatter(norm(Difference_FNO_yuan), norm(Difference_FNO),
                           color=model_colors['TLFNO'], s=2, alpha=1.0, marker=model_markers['TLFNO'],
                           label='TLFNO', zorder=2)
        ax.scatter(norm(Difference_MAHNN_yuan), norm(Difference_MAHNN),
                           color=model_colors['MAHNN'], s=2, alpha=1.0, marker=model_markers['MAHNN'],
                           label='MAHNN', zorder=1)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0.0, 1.0])
        ax.set_yticks([0.0, 1.0])

        Y_TARGET = Results_DLRF4_input[j].loc[0, 'Y_TARGET']
        y_target = Y_TARGET[j]
        ax.set_title(f'y/b = {y_target:.3f}')
        if j == 0:
            ax.set_ylabel('Prediction')
        ax.set_xlabel('Reference')

        if  k == 0 and j == 2:
            leg = ax.legend(bbox_to_anchor=(1.05, -0.05),
                            loc='lower right',
                            ncol=1,
                              columnspacing=1.0,
                              labelspacing=0.15,
                              handletextpad=0.4,
                              handlelength=0.7,
                              frameon=False,
                              fontsize=6)
            leg.get_frame().set_linewidth(0.75)
        elif k == 1 and j == 3:
            leg = ax.legend(bbox_to_anchor=(0.22, 1.04), loc='upper center', ncol=1,
                              columnspacing=1.0,
                              labelspacing=0.15,
                              handletextpad=0.4,
                              handlelength=0.7,
                              frameon=False,
                              fontsize=6)
            leg.get_frame().set_linewidth(0.75)
        elif k == 2 and j == 0:
            leg = ax.legend(bbox_to_anchor=(0.22, 1.04), loc='upper center', ncol=1,
                              columnspacing=1.0,
                              labelspacing=0.15,
                              handletextpad=0.4,
                              handlelength=0.7,
                              frameon=False,
                              fontsize=6)
            leg.get_frame().set_linewidth(0.75)


    plt.tight_layout(pad=0.01)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.15, wspace=0.20)

    for ext in ['png', 'tiff', 'pdf']:
        output_path = os.path.join(SAVE_DIR, f'fold_{k + 1}_comparison.{ext}')
        plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)

    plt.show()

Pearsonr_frame = pd.DataFrame(pearsonr_list)
Cosine_frame = pd.DataFrame(cosine_list)


model_display = {
    'MFTabPFN': 'MFTabPFN',
    'MFGPR': 'MFGPR',
    'NMFGPR': 'NMFGPR',
    'TLFNO': 'TLFNO',
    'MAHNN': 'MAHNN'
}

Pearsonr_frame['pearsonr'] = Pearsonr_frame['pearsonr'].apply(
    lambda x: x.item() if isinstance(x, np.ndarray) else float(x)
)

Cosine_frame['cosine'] = Cosine_frame['cosine'].apply(
    lambda x: x.item() if isinstance(x, np.ndarray) else float(x)
)

pearson_mean_df = (
    Pearsonr_frame.groupby(['Model', 'Dataset'])['pearsonr']
    .mean()
    .unstack()
    .fillna(0)
    .astype(float)
)

cosine_mean_df = (
    Cosine_frame.groupby(['Model', 'Dataset'])['cosine']
    .mean()
    .unstack()
    .fillna(0)
    .astype(float)
)

model_order = (
    pearson_mean_df.mean(axis=1)
    .sort_values(ascending=True)
    .index
    .tolist()
)

pearson_sorted = pearson_mean_df.loc[model_order].T
cosine_sorted  = cosine_mean_df.loc[model_order].T

dataset_labels = {}
for j in range(n_performances):
    y_target = Results_DLRF4_input[j].loc[0, 'Y_TARGET'][j]
    dataset_labels[j] = f'y/b = {y_target:.3f}'

pearson_sorted.index = [dataset_labels.get(int(idx), str(idx)) for idx in pearson_sorted.index]
cosine_sorted.index  = [dataset_labels.get(int(idx), str(idx)) for idx in cosine_sorted.index]

pearson_sorted = pearson_sorted[::-1]
cosine_sorted  = cosine_sorted[::-1]

fig, (ax1, ax2, cax) = plt.subplots(
    1, 3,
    figsize=(3.4, 1.5),
    gridspec_kw={'width_ratios': [1, 1, 0.06]},
)

sns.heatmap(
    pearson_sorted,
    ax=ax1,
    cmap='YlGnBu',
    vmin=0, vmax=1,
    annot=True,
    fmt='.2f',
    cbar=False,
    square=True,
    annot_kws={'size': 6,
               'color': 'black',
               }
)
ax1.set_title('Pearson correlation')
ax1.set_xlabel('')
ax1.tick_params(axis='x', rotation=45)
ax1.tick_params(axis='y', rotation=45)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=45, ha='right', rotation_mode='anchor')

sns.heatmap(
    cosine_sorted,
    ax=ax2,
    cmap='YlGnBu',
    vmin=0, vmax=1,
    annot=True,
    fmt='.2f',
    cbar=False,
    square=True,
    annot_kws={'size': 6,
               'color': 'black',
               }
)
ax2.set_title('Cosine similarity')
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_yticks([])
ax2.set_yticklabels([])
ax2.tick_params(axis='x', rotation=45)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

cbar = fig.colorbar(ax1.collections[0], cax=cax, orientation='vertical')
cbar.set_label('Value')
cbar.ax.tick_params(labelsize=6)
for spine in cbar.ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(pad=0.01)
plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.15, wspace=0.10)

for ext in ['png', 'tiff', 'pdf']:
    output_path = os.path.join(SAVE_DIR, f'corr_position.{ext}')
    plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)

plt.show()


Pearsonr_grouped_all = Pearsonr_frame.groupby(['Model'])['pearsonr'].agg(['mean', 'std', 'count']).reset_index()
Pearsonr_grouped_all['se'] = Pearsonr_grouped_all['std'] / np.sqrt(Pearsonr_grouped_all['count'])
Pearsonr_grouped_all['ci'] = Pearsonr_grouped_all.apply(lambda row: row['se'] * stats.t.ppf(1 - alpha / 2, row['count'] - 1) if row['count'] > 1 else 0, axis=1)

Cosine_grouped_all = Cosine_frame.groupby(['Model'])['cosine'].agg(['mean', 'std', 'count']).reset_index()
Cosine_grouped_all['se'] = Cosine_grouped_all['std'] / np.sqrt(Cosine_grouped_all['count'])
Cosine_grouped_all['ci'] = Cosine_grouped_all.apply(lambda row: row['se'] * stats.t.ppf(1 - alpha / 2, row['count'] - 1) if row['count'] > 1 else 0, axis=1)



fig, axes = plt.subplots(
    ncols=2,
    figsize=(3.4, 1.5),
)
ax = axes[0]
df_stat = Pearsonr_grouped_all.copy()
df_stat = df_stat.sort_values('mean', ascending=True).reset_index(drop=True)
models_sorted = df_stat['Model'].values
means = df_stat['mean'].values
cis = df_stat['ci'].values
bars = ax.bar(
    range(len(models_sorted)),
    means,
    yerr=cis,
    capsize=1,
    color=[model_colors.get(m, '#808080') for m in models_sorted],
    edgecolor='black',
    linewidth=0.5,
    error_kw={'elinewidth': 1.0, 'capthick': 1.0}
)
ax.set_xticks(range(len(models_sorted)))
ax.set_xticklabels(models_sorted, rotation=45, ha='right', rotation_mode='anchor')
ax.set_ylabel('Pearson (95% CI)')
ax.set_title('Pearson correlation')

ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True)

ax.yaxis.set_label_coords(-0.16, 0.5)

variant_patches = [
    Patch(facecolor='#1f77b4', edgecolor='black', linewidth=0.5, hatch='', label='Mean'),
]
ax_hidden = fig.add_axes([0, 0, 0.0001, 0.0001])
ax_hidden.set_xlim(0, 1)
ax_hidden.set_ylim(0, 1)
ax_hidden.axis('off')

def legend_errorbar(color, label):
    eb = ax_hidden.errorbar(0, 0,
                            yerr=[[0.4], [0.6]],
                            color=color,
                            capsize=1.3,
                            capthick=1.0,
                            lw=1.0,
                            fmt='none',
                            markersize=0,
                            markerfacecolor='none',
                            markeredgecolor='none',
                            label=label)
    return eb


eb1 = legend_errorbar('black', 'CI')
handles = [
    variant_patches[0], eb1,
]

leg1 = ax.legend(handles=handles,
                 loc='upper left',
                 bbox_to_anchor=(-0.01, 1.01),
                 frameon=True,
                 ncol=2,
                 columnspacing=0.5,
                 handletextpad=0.3,
                 handlelength=0.8,
                 fontsize=6
                 )
leg1.get_frame().set_linewidth(0.75)

ax = axes[1]
df_stat = Cosine_grouped_all.copy()
df_stat = df_stat.sort_values('mean', ascending=True).reset_index(drop=True)
models_sorted = df_stat['Model'].values
means = df_stat['mean'].values
cis = df_stat['ci'].values
bars = ax.bar(
    range(len(models_sorted)),
    means,
    yerr=cis,
    capsize=1,
    color=[model_colors.get(m, '#808080') for m in models_sorted],
    edgecolor='black',
    linewidth=0.5,
    error_kw={'elinewidth': 1.0, 'capthick': 1.0}
)
ax.set_xticks(range(len(models_sorted)))
ax.set_xticklabels(models_sorted, rotation=45, ha='right', rotation_mode='anchor')
ax.set_ylabel('Cosine (95% CI)')
ax.set_title('Cosine similarity')

ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True)

ax.yaxis.set_label_coords(-0.16, 0.5)

plt.tight_layout(pad=0.01)
plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.15, wspace=0.33)

for ext in ['png', 'tiff', 'pdf']:
        output_path = os.path.join(SAVE_DIR, f'cor_all_position.{ext}')
        plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)

plt.show()




for k in range(0, n_simulations):

    fig, axes = plt.subplots(
        nrows=n_performances,
        ncols=4,
        figsize=(6.8, 6.0),
    )


    for j in range(0, n_performances):
        testing_index = Results_DLRF4_input[j].loc[0, 'testing_index']
        MA = Results_DLRF4_input[j].loc[0, 'MA']
        AFA = Results_DLRF4_input[j].loc[0, 'AFA']
        Y_TARGET = Results_DLRF4_input[j].loc[0, 'Y_TARGET']
        XX_total_yuan = Results_DLRF4_input[j].loc[0, 'XX_total_yuan']
        XX_train_yuan_low = Results_DLRF4_input[j].loc[0, 'XX_train_yuan_low']
        XX_train_yuan_high = Results_DLRF4_input[j].loc[0, 'XX_train_yuan_high']
        YY_train_yuan_low = Results_DLRF4_input[j].loc[0, 'YY_train_yuan_low']
        YY_train_yuan_high = Results_DLRF4_input[j].loc[0, 'YY_train_yuan_high']
        YY_total_yuan_high = Results_DLRF4_input[j].loc[0, 'YY_total_yuan_high']

        Y_test_prediction_TabPFN_high_PLOT = Results_DLRF4_prediction[j][k].query("Model == 'TabPFN-High'")['Y_test_prediction_PLOT'].iloc[0]
        LOWER_SIGMA_TabPFN_high = Results_DLRF4_prediction[j][k].query("Model == 'TabPFN-High'")['LOWER_SIGMA'].iloc[0]
        UPPER_SIGMA_TabPFN_high = Results_DLRF4_prediction[j][k].query("Model == 'TabPFN-High'")['UPPER_SIGMA'].iloc[0]
        Y_test_prediction_TabPFN_multi_PLOT = Results_DLRF4_prediction[j][k].query("Model == 'TabPFN-Multi'")['Y_test_prediction_PLOT'].iloc[0]
        LOWER_SIGMA_TabPFN_multi = Results_DLRF4_prediction[j][k].query("Model == 'TabPFN-Multi'")['LOWER_SIGMA'].iloc[0]
        UPPER_SIGMA_TabPFN_multi = Results_DLRF4_prediction[j][k].query("Model == 'TabPFN-Multi'")['UPPER_SIGMA'].iloc[0]
        Y_test_prediction_MFTabPFN_PLOT = Results_DLRF4_prediction[j][k].query("Model == 'MFTabPFN'")['Y_test_prediction_PLOT'].iloc[0]
        LOWER_SIGMA = Results_DLRF4_prediction[j][k].query("Model == 'MFTabPFN'")['LOWER_SIGMA'].iloc[0]
        UPPER_SIGMA = Results_DLRF4_prediction[j][k].query("Model == 'MFTabPFN'")['UPPER_SIGMA'].iloc[0]
        Y_test_prediction_LinearMFGP_PLOT = Results_DLRF4_prediction_MFGPR[j][k].query("Model == 'LinearMFGP'")['Y_test_prediction_PLOT'].iloc[0]
        LOWER_SIGMA_LinearMFGP = Results_DLRF4_prediction_MFGPR[j][k].query("Model == 'LinearMFGP'")['LOWER_SIGMA'].iloc[0]
        UPPER_SIGMA_LinearMFGP = Results_DLRF4_prediction_MFGPR[j][k].query("Model == 'LinearMFGP'")['UPPER_SIGMA'].iloc[0]
        Y_test_prediction_NonlinearMFGP_PLOT = Results_DLRF4_prediction_MFGPR[j][k].query("Model == 'NonlinearMFGP'")['Y_test_prediction_PLOT'].iloc[0]
        LOWER_SIGMA_NonlinearMFGP = Results_DLRF4_prediction_MFGPR[j][k].query("Model == 'NonlinearMFGP'")['LOWER_SIGMA'].iloc[0]
        UPPER_SIGMA_NonlinearMFGP = Results_DLRF4_prediction_MFGPR[j][k].query("Model == 'NonlinearMFGP'")['UPPER_SIGMA'].iloc[0]
        Y_test_prediction_AutoGluon_high_PLOT = Results_DLRF4_prediction[j][k].query("Model == 'AutoGluon-High'")['Y_test_prediction_PLOT'].iloc[0]
        Y_test_prediction_AutoGluon_multi_PLOT = Results_DLRF4_prediction[j][k].query("Model == 'AutoGluon-Multi'")['Y_test_prediction_PLOT'].iloc[0]
        Y_test_prediction_MAHNN_PLOT = Results_DLRF4_prediction_MAHNN[j][k].query("Model == 'MA-HNN'")['Y_test_prediction_PLOT'].iloc[0]
        Y_test_prediction_FNO_PLOT = Results_DLRF4_prediction_FNO[j][k].query("Model == 'FNO'")['Y_test_prediction_PLOT'].iloc[0]


        Ma = MA[0]
        afa = AFA[0]

        ref1 = Y_test_prediction_LinearMFGP_PLOT
        lower1 = LOWER_SIGMA_LinearMFGP
        upper1 = UPPER_SIGMA_LinearMFGP
        ref2 = Y_test_prediction_NonlinearMFGP_PLOT
        lower2 = LOWER_SIGMA_NonlinearMFGP
        upper2 = UPPER_SIGMA_NonlinearMFGP
        ref3 = Y_test_prediction_MAHNN_PLOT
        ref4 = Y_test_prediction_FNO_PLOT
        ss = 5

        for col, i in enumerate(range(len(Y_TARGET))):
            y_target = Y_TARGET[i]
            ax = axes[j, col]
            condition_lower = (
                    (XX_total_yuan[:, 1] == y_target) &
                    (XX_total_yuan[:, 2] == -1)
            )
            index_lower = np.where(condition_lower)[0]
            condition_upper = (
                    (XX_total_yuan[:, 1] == y_target) &
                    (XX_total_yuan[:, 2] == 1)
            )
            index_upper = np.where(condition_upper)[0]

            condition_simulation_lower = (
                    (XX_train_yuan_low[:, 1] == y_target) &
                    (XX_train_yuan_low[:, 2] == -1)
            )
            index_simulation_lower = np.where(condition_simulation_lower)[0]
            condition_simulation_upper = (
                    (XX_train_yuan_low[:, 1] == y_target) &
                    (XX_train_yuan_low[:, 2] == 1)
            )
            index_simulation_upper = np.where(condition_simulation_upper)[0]

            condition_experiment_lower = ((XX_train_yuan_high[:, 1] == y_target) & (XX_train_yuan_high[:, 2] == -1))
            index_experiment_lower = np.where(condition_experiment_lower)[0]
            condition_experiment_upper = ((XX_train_yuan_high[:, 1] == y_target) & (XX_train_yuan_high[:, 2] == 1))
            index_experiment_upper = np.where(condition_experiment_upper)[0]

            scatter_train_upper = ax.scatter(XX_train_yuan_high[index_experiment_upper, 0],
                                             YY_train_yuan_high[index_experiment_upper], marker='s', s=ss,
                                             color='black')
            scatter_train_lower = ax.scatter(XX_train_yuan_high[index_experiment_lower, 0],
                                             YY_train_yuan_high[index_experiment_lower], marker='s', s=ss, color='black')

            scatter_upper = ax.scatter(XX_total_yuan[index_upper, 0],
                        YY_total_yuan_high[index_upper], marker='o', s=ss,
                        color='black')
            scatter_lower = ax.scatter(XX_total_yuan[index_lower, 0],
                        YY_total_yuan_high[index_lower], marker='o', s=ss,
                        color='black')

            line_simulation_upper = ax.plot(XX_train_yuan_low[index_simulation_upper, 0],
                     YY_train_yuan_low[index_simulation_upper],
                     linestyle='--', color='gray')[0]
            line_simulation_lower = ax.plot(XX_train_yuan_low[index_simulation_lower, 0],
                     YY_train_yuan_low[index_simulation_lower],
                     linestyle='--', color='gray')[0]
            ax.plot(np.array(
                [XX_train_yuan_low[index_simulation_upper, 0][0], XX_train_yuan_low[index_simulation_lower, 0][0]]),
                    np.array([YY_train_yuan_low[index_simulation_upper][0], YY_train_yuan_low[index_simulation_lower][0]]),
                    linestyle='--', color='gray')
            ax.plot(np.array(
                [XX_train_yuan_low[index_simulation_upper, 0][-1], XX_train_yuan_low[index_simulation_lower, 0][-1]]),
                    np.array(
                        [YY_train_yuan_low[index_simulation_upper][-1], YY_train_yuan_low[index_simulation_lower][-1]]),
                    linestyle='--', color='gray')

            line_ref1_upper = ax.plot(XX_train_yuan_low[index_simulation_upper, 0], ref1[index_simulation_upper],
                    linestyle='-', color=model_colors['MFGPR'])[0]
            line_ref1_lower = ax.plot(XX_train_yuan_low[index_simulation_lower, 0], ref1[index_simulation_lower],
                    linestyle='-', color=model_colors['MFGPR'])[0]
            ax.plot(np.array(
                [XX_train_yuan_low[index_simulation_upper, 0][0], XX_train_yuan_low[index_simulation_lower, 0][0]]),
                    np.array([ref1[index_simulation_upper][0],
                              ref1[index_simulation_lower][0]]), linestyle='-', color=model_colors['MFGPR'])
            ax.plot(np.array(
                [XX_train_yuan_low[index_simulation_upper, 0][-1], XX_train_yuan_low[index_simulation_lower, 0][-1]]),
                    np.array([ref1[index_simulation_upper][-1],
                              ref1[index_simulation_lower][-1]]), linestyle='-', color=model_colors['MFGPR'])
            line_ref1_upper_interval = ax.fill_between(XX_train_yuan_low[index_simulation_upper, 0],
                                                           lower1[index_simulation_upper],
                                                           upper1[index_simulation_upper], color=model_colors['MFGPR'],
                                                           edgecolor='none', alpha=0.2)
            line_ref1_lower_interval = ax.fill_between(XX_train_yuan_low[index_simulation_lower, 0],
                                                           lower1[index_simulation_lower],
                                                           upper1[index_simulation_lower], color=model_colors['MFGPR'],
                                                           edgecolor='none', alpha=0.2)


            line_ref2_upper = ax.plot(XX_train_yuan_low[index_simulation_upper, 0], ref2[index_simulation_upper],
                    linestyle='-', color=model_colors['NMFGPR'])[0]
            line_ref2_lower = ax.plot(XX_train_yuan_low[index_simulation_lower, 0], ref2[index_simulation_lower],
                    linestyle='-', color=model_colors['NMFGPR'])[0]
            ax.plot(np.array(
                [XX_train_yuan_low[index_simulation_upper, 0][0], XX_train_yuan_low[index_simulation_lower, 0][0]]),
                    np.array([ref2[index_simulation_upper][0],
                              ref2[index_simulation_lower][0]]), linestyle='-', color=model_colors['NMFGPR'])
            ax.plot(np.array(
                [XX_train_yuan_low[index_simulation_upper, 0][-1], XX_train_yuan_low[index_simulation_lower, 0][-1]]),
                    np.array([ref2[index_simulation_upper][-1],
                              ref2[index_simulation_lower][-1]]), linestyle='-', color=model_colors['NMFGPR'])
            line_ref2_upper_interval = ax.fill_between(XX_train_yuan_low[index_simulation_upper, 0],
                                                           lower2[index_simulation_upper],
                                                           upper2[index_simulation_upper], color=model_colors['NMFGPR'],
                                                           edgecolor='none', alpha=0.2)
            line_ref2_lower_interval = ax.fill_between(XX_train_yuan_low[index_simulation_lower, 0],
                                                           lower2[index_simulation_lower],
                                                           upper2[index_simulation_lower], color=model_colors['NMFGPR'],
                                                           edgecolor='none', alpha=0.2)

            line_ref3_upper = ax.plot(XX_train_yuan_low[index_simulation_upper, 0], ref3[index_simulation_upper],
                    linestyle='-', color=model_colors['MAHNN'])[0]
            line_ref3_lower = ax.plot(XX_train_yuan_low[index_simulation_lower, 0], ref3[index_simulation_lower],
                    linestyle='-', color=model_colors['MAHNN'])[0]
            ax.plot(np.array(
                [XX_train_yuan_low[index_simulation_upper, 0][0], XX_train_yuan_low[index_simulation_lower, 0][0]]),
                    np.array([ref3[index_simulation_upper][0],
                              ref3[index_simulation_lower][0]]), linestyle='-', color=model_colors['MAHNN'])
            ax.plot(np.array(
                [XX_train_yuan_low[index_simulation_upper, 0][-1], XX_train_yuan_low[index_simulation_lower, 0][-1]]),
                    np.array([ref3[index_simulation_upper][-1],
                              ref3[index_simulation_lower][-1]]), linestyle='-', color=model_colors['MAHNN'])

            line_ref4_upper = ax.plot(XX_train_yuan_low[index_simulation_upper, 0], ref4[index_simulation_upper],
                    linestyle='-', color=model_colors['TLFNO'])[0]
            line_ref4_lower = ax.plot(XX_train_yuan_low[index_simulation_lower, 0], ref4[index_simulation_lower],
                    linestyle='-', color=model_colors['TLFNO'])[0]
            ax.plot(np.array(
                [XX_train_yuan_low[index_simulation_upper, 0][0], XX_train_yuan_low[index_simulation_lower, 0][0]]),
                    np.array([ref4[index_simulation_upper][0],
                              ref4[index_simulation_lower][0]]), linestyle='-', color=model_colors['TLFNO'])
            ax.plot(np.array(
                [XX_train_yuan_low[index_simulation_upper, 0][-1], XX_train_yuan_low[index_simulation_lower, 0][-1]]),
                    np.array([ref4[index_simulation_upper][-1],
                              ref4[index_simulation_lower][-1]]), linestyle='-', color=model_colors['TLFNO'])

            line_mftabpfn_upper = ax.plot(XX_train_yuan_low[index_simulation_upper, 0], Y_test_prediction_MFTabPFN_PLOT[index_simulation_upper],
                    linestyle='-', color=model_colors['MFTabPFN'])[0]
            line_mftabpfn_lower = ax.plot(XX_train_yuan_low[index_simulation_lower, 0], Y_test_prediction_MFTabPFN_PLOT[index_simulation_lower],
                    linestyle='-', color=model_colors['MFTabPFN'])[0]
            ax.plot(np.array(
                [XX_train_yuan_low[index_simulation_upper, 0][0], XX_train_yuan_low[index_simulation_lower, 0][0]]),
                    np.array([Y_test_prediction_MFTabPFN_PLOT[index_simulation_upper][0],
                              Y_test_prediction_MFTabPFN_PLOT[index_simulation_lower][0]]), linestyle='-', color=model_colors['MFTabPFN'])
            ax.plot(np.array(
                [XX_train_yuan_low[index_simulation_upper, 0][-1], XX_train_yuan_low[index_simulation_lower, 0][-1]]),
                    np.array([Y_test_prediction_MFTabPFN_PLOT[index_simulation_upper][-1],
                              Y_test_prediction_MFTabPFN_PLOT[index_simulation_lower][-1]]), linestyle='-', color=model_colors['MFTabPFN'])
            line_mftabpfn_upper_interval = ax.fill_between(XX_train_yuan_low[index_simulation_upper, 0],
                                                           LOWER_SIGMA[index_simulation_upper],
                                                           UPPER_SIGMA[index_simulation_upper], color=model_colors['MFTabPFN'],
                                                           edgecolor='none', alpha=0.5)
            line_mftabpfn_lower_interval = ax.fill_between(XX_train_yuan_low[index_simulation_lower, 0],
                                                           LOWER_SIGMA[index_simulation_lower],
                                                           UPPER_SIGMA[index_simulation_lower], color=model_colors['MFTabPFN'],
                                                           edgecolor='none', alpha=0.5)

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-1.5, 1.0)
            ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
            ax.set_yticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
            ax.invert_yaxis()
            ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

            if col == 0:
                ax.set_ylabel('Cp')
            else:
                ax.set_ylabel('')

            if j == n_performances - 1:
                ax.set_xlabel('x/c')
            else:
                ax.set_xlabel('')

            if j == 0:
                ax.set_title(f'y/b = {y_target:.3f}')

    handles = [
            plt.Line2D([0], [0], color=model_colors['MFTabPFN'], lw=1.5, label='MFTabPFN'),
            plt.Line2D([0], [0], color=model_colors['MFGPR'], lw=1.5, label='MFGPR'),
            plt.Line2D([0], [0], color=model_colors['NMFGPR'], lw=1.5, label='NMFGPR'),
            plt.Line2D([0], [0], color=model_colors['MAHNN'], lw=1.5, label='MAHNN'),
            plt.Line2D([0], [0], color=model_colors['TLFNO'], lw=1.5, label='TLFNO'),
            plt.Line2D([], [], color='gray', ls='--', label='Simulation'),
            plt.Line2D([], [], marker='s', color='black', ls='', markersize=2, label='Training experimental data'),
            plt.Line2D([], [], marker='o', color='black', ls='', markersize=2, label='Testing experimental data'),
        ]
    legend_elements = [
        Patch(facecolor=model_colors['MFTabPFN'], alpha=0.5, edgecolor='none',
              label='SD (MFTabPFN)'),
        Patch(facecolor=model_colors['MFGPR'], edgecolor='none', alpha=0.2,
              label='SD (MFGPR)'),
        Patch(facecolor=model_colors['NMFGPR'], edgecolor='none', alpha=0.2,
              label='SD (NMFGPR)'),
    ]

    leg = fig.legend(handles=handles + legend_elements,
                   loc='lower center',
                   ncol=6,
                   bbox_to_anchor=(0.525, -0.00),
                   frameon=True,
                   columnspacing=1.3,
                   handletextpad=0.5,
                   handlelength=1.6,
               )

    plt.tight_layout(pad=0.01)
    plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.12, wspace=0.25, hspace=0.20)

    for ext in ['png', 'tiff', 'pdf']:
            output_path = os.path.join(SAVE_DIR, f'fold_{k+1}_position.{ext}')
            plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)

    plt.show(block=True)


data = []
for dataset_idx in range(len(Results_DLRF4_list)):
    for sim_idx in range(len(Results_DLRF4_list[dataset_idx])):
        df = Results_DLRF4_list[dataset_idx][sim_idx]
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
for dataset_idx in range(len(Results_DLRF4_list_MFGPR)):
    for sim_idx in range(len(Results_DLRF4_list_MFGPR[dataset_idx])):
        df = Results_DLRF4_list_MFGPR[dataset_idx][sim_idx]
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
for dataset_idx in range(len(Results_DLRF4_list_MAHNN)):
    for sim_idx in range(len(Results_DLRF4_list_MAHNN[dataset_idx])):
        df = Results_DLRF4_list_MAHNN[dataset_idx][sim_idx]
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
for dataset_idx in range(len(Results_DLRF4_list_FNO)):
    for sim_idx in range(len(Results_DLRF4_list_FNO[dataset_idx])):
        df = Results_DLRF4_list_FNO[dataset_idx][sim_idx]
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
Data_frame['Time_Total'] = Data_frame['Time_Train'] + Data_frame['Time_Pred']

Data_frame['Model'] = Data_frame['Model'].replace('TabPFN-High', 'TabPFN-H')
Data_frame['Model'] = Data_frame['Model'].replace('TabPFN-Multi', 'TabPFN-M')
Data_frame['Model'] = Data_frame['Model'].replace('LinearMFGP', 'MFGPR')
Data_frame['Model'] = Data_frame['Model'].replace('NonlinearMFGP', 'NMFGPR')
Data_frame['Model'] = Data_frame['Model'].replace('AutoGluon-High', 'AutoGluon-H')
Data_frame['Model'] = Data_frame['Model'].replace('AutoGluon-Multi', 'AutoGluon-M')
Data_frame['Model'] = Data_frame['Model'].replace('MA-HNN', 'MAHNN')
Data_frame['Model'] = Data_frame['Model'].replace('FNO', 'TLFNO')


metrics = ['RMSE_N', 'MAE_N', 'R2', 'RMSE', 'MAE']
metric_titles = {'RMSE_N': 'NNRMSE', 'MAE_N': 'NNMAE', 'R2': r'$R^2$', 'RMSE': 'RMSE', 'MAE': 'MAE'}
metric_ylabels = {'RMSE_N': 'NNRMSE (95% CI)', 'MAE_N': 'NNMAE (95% CI)', 'R2': r'$R^2$ (95% CI)', 'RMSE': 'RMSE (95% CI)', 'MAE': 'MAE (95% CI)'}

model_order_preferred = ['MFTabPFN', 'MFGPR', 'NMFGPR', 'MAHNN', 'TLFNO', 'TabPFN-H', 'TabPFN-M', 'AutoGluon-H', 'AutoGluon-M']

model_colors_list = [model_colors.get(m, '#7f7f7f') for m in model_order_preferred]

stats_dict = {}
for metric in metrics:
    grouped = Data_frame.groupby(['Model'])[metric].agg(['mean', 'std', 'count']).reset_index()
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['ci'] = grouped.apply(lambda row: row['se'] * stats.t.ppf(1 - alpha / 2, row['count'] - 1) if row['count'] > 1 else 0, axis=1)
    stats_dict[metric] = grouped

fig, axes = plt.subplots(
    ncols=len(metrics),
    figsize=(6.8, 1.8),
)

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    df_stat = stats_dict[metric].copy()

    if metric in ['RMSE', 'MAE']:
        df_stat = df_stat.sort_values('mean', ascending=False).reset_index(drop=True)
    else:
        df_stat = df_stat.sort_values('mean', ascending=True).reset_index(drop=True)

    models_sorted = df_stat['Model'].values
    means = df_stat['mean'].values
    cis = df_stat['ci'].values

    bars = ax.bar(
        range(len(models_sorted)),
        means,
        yerr=cis,
        capsize=1,
        color=[model_colors.get(m, '#808080') for m in models_sorted],
        edgecolor='black',
        linewidth=0.5,
        error_kw={'elinewidth': 1.0, 'capthick': 1.0}
    )

    ax.set_xticks(range(len(models_sorted)))
    ax.set_xticklabels(models_sorted, rotation=60, ha='right', rotation_mode='anchor')

    ax.set_ylabel(metric_ylabels[metric])
    ax.set_title(metric_titles[metric])

    ax.yaxis.set_label_coords(-0.27, 0.5)

    if metric == 'R2':
        ax.set_ylim(0.85, 1.0)
        ax.set_yticks([0.85, 0.88, 0.91, 0.94, 0.97, 1.0])
    elif metric == 'RMSE':
        ax.set_ylim(0, 0.21)
        ax.set_yticks([0.00, 0.04, 0.08, 0.12, 0.16, 0.20])
    elif metric == 'MAE':
        ax.set_ylim(0, 0.157)
        ax.set_yticks([0.00, 0.03, 0.06, 0.09, 0.12, 0.15])
    else:
        ax.set_ylim(0.9, 1.0)
        ax.set_yticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.0])

    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    if metric == 'MAE':
        variant_patches = [
            Patch(facecolor='#1f77b4', edgecolor='black', linewidth=0.5, hatch='', label='Mean'),
        ]
        ax_hidden = fig.add_axes([0, 0, 0.0001, 0.0001])
        ax_hidden.set_xlim(0, 1)
        ax_hidden.set_ylim(0, 1)
        ax_hidden.axis('off')
        def legend_errorbar(color, label):
            eb = ax_hidden.errorbar(0, 0,
                                    yerr=[[0.4], [0.6]],
                                    color=color,
                                    capsize=1.3,
                                    capthick=1.0,
                                    lw=1.0,
                                    fmt='none',
                                    markersize=0,
                                    markerfacecolor='none',
                                    markeredgecolor='none',
                                    label=label)
            return eb
        eb1 = legend_errorbar('black', 'CI')
        handles = [
            variant_patches[0], eb1,
        ]

        leg1 = ax.legend(handles=handles,
                         loc='upper center',
                         bbox_to_anchor=(0.62, 1.01),
                         frameon=True,
                         ncol=2,
                         columnspacing=0.5,
                         handletextpad=0.3,
                         handlelength=0.8,
                         fontsize=6
                         )
        leg1.get_frame().set_linewidth(0.75)

plt.tight_layout(pad=0.01)
plt.subplots_adjust(left=0.05, right=0.97, top=0.85, bottom=0.35, wspace=0.45)

for ext in ['png', 'tiff', 'pdf']:
        output_path = os.path.join(SAVE_DIR, f'metric_position.{ext}')
        plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)

plt.show()





time_columns = {
    'Training time': 'Time_Train',
    'Prediction time': 'Time_Pred',
    'Total time': 'Time_Total'
}


fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2),
                         )

for ax_idx, (title, col_name) in enumerate(time_columns.items()):
    ax = axes[ax_idx]

    order = Data_frame.groupby('Model')[col_name].median().sort_values().index

    datasets = [Data_frame[Data_frame['Model'] == m][col_name].dropna().values
                for m in order]

    positions = np.arange(len(order)) + 1

    parts = ax.violinplot(datasets,
                          positions=positions,
                          widths=0.85,
                          showmeans=False,
                          showextrema=False,
                          showmedians=False)

    for pc, model_name in zip(parts['bodies'], order):
        color = model_colors.get(model_name, '#808080')
        pc.set_facecolor(color)
        pc.set_edgecolor('none')
        pc.set_alpha(0.5)

    for i, model_name in enumerate(order):
        color = model_colors.get(model_name, '#808080')
        y = Data_frame[Data_frame['Model'] == model_name][col_name].dropna()
        ax.boxplot([y], positions=[positions[i]], widths=0.4,
                   patch_artist=True,
                   boxprops=dict(facecolor=color, edgecolor=color, linewidth=0),
                   whiskerprops=dict(color=color, linewidth=1.5),
                   capprops=dict(color=color, linewidth=1.5),
                   medianprops=dict(color='black', linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor=color,
                                   markeredgecolor=color, markersize=1, alpha=1.0))

    ax.set_title(title)
    ax.set_xlabel('')
    if ax_idx == 0:
        ax.set_ylabel('Time (seconds)')

    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=45, ha='right', rotation_mode='anchor')

    ax.set_yscale('log')
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', length=0)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    if col_name == 'Time_Train':
        ax.set_yticks([0.1, 1, 10, 100, 1000, 10000])
        ax.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$'])
        ax.set_ylim(0.1, 10000)
    elif col_name == 'Time_Pred':
        ax.set_yticks([0.0001, 0.001, 0.01, 0.1, 1, 10])
        ax.set_yticklabels([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
        ax.set_ylim(0.0001, 10)
    elif col_name == 'Time_Total':
        ax.set_yticks([0.1, 1, 10, 100, 1000, 10000])
        ax.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$'])
        ax.set_ylim(0.1, 10000)

    if ax_idx == 1:
        legend_elements = [
            Patch(facecolor='#1f77b4', edgecolor='none', alpha=1.0,
                  label='Box: 25th–75th percentile (IQR)'),
        ]
        leg1 = ax.legend(handles=legend_elements,
                         loc='lower right',
                         bbox_to_anchor=(1.0, 0.0),
                         ncol=1,
                         frameon=True,
                         columnspacing=1.0,
                         handletextpad=0.5,
                         handlelength=0.8,
                         fontsize=6,
                         )
        leg1.get_frame().set_linewidth(0.75)
    if ax_idx == 2:
        legend_elements = [
            Patch(facecolor='#1f77b4', alpha=0.5, edgecolor='none',
                  label='Violin plot: data distribution'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=3,
                   label='Outliers beyond 1.5×IQR'),
        ]
        leg1 = ax.legend(handles=legend_elements,
                         loc='lower right',
                         bbox_to_anchor=(1.0, 0.0),
                         ncol=1,
                         frameon=True,
                         columnspacing=1.0,
                         handletextpad=0.5,
                         handlelength=0.8,
                         fontsize=6,
                         )
        leg1.get_frame().set_linewidth(0.75)
    if ax_idx == 0:
        legend_elements = [
            Line2D([0], [0], color='black', lw=1.5, label='Median'),
            Line2D([0], [0], color='#1f77b4', lw=1.5, label='Upper and lower whiskers'),
        ]
        leg1 = ax.legend(handles=legend_elements,
                         loc='lower right',
                         bbox_to_anchor=(1.0, 0.0),
                         ncol=1,
                         frameon=True,
                         columnspacing=1.0,
                         handletextpad=0.5,
                         handlelength=0.8,
                         fontsize=6,
                         )
        leg1.get_frame().set_linewidth(0.75)

plt.tight_layout(pad=0.01)
plt.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.29, wspace=0.2)

for ext in ['png', 'tiff', 'pdf']:
        output_path = os.path.join(SAVE_DIR, f'time_position.{ext}')
        plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)

plt.show()
