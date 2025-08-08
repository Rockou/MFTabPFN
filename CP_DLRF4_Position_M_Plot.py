import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pickle

Case = 1 # 1: testing data at y/b=0.185; 2: testing data at y/b=0.331; 3: testing data at y/b=0.512; 4: testing data at y/b=0.844
if Case == 1:
    SAVE_DIR = './Datasets/DLR_F4/Position/0185'
elif Case == 2:
    SAVE_DIR = './Datasets/DLR_F4/Position/0331'
elif Case == 3:
    SAVE_DIR = './Datasets/DLR_F4/Position/0512'
elif Case == 4:
    SAVE_DIR = './Datasets/DLR_F4/Position/0884'

pkl_file = os.path.join(SAVE_DIR, f'testing_index.pkl')
with open(pkl_file, 'rb') as f:
    testing_index = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'MA.pkl')
with open(pkl_file, 'rb') as f:
    MA = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'AFA.pkl')
with open(pkl_file, 'rb') as f:
    AFA = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'Y_TARGET.pkl')
with open(pkl_file, 'rb') as f:
    Y_TARGET = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'XX_total_yuan.pkl')
with open(pkl_file, 'rb') as f:
    XX_total_yuan = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'XX_train_yuan_low.pkl')
with open(pkl_file, 'rb') as f:
    XX_train_yuan_low = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'XX_train_yuan_high.pkl')
with open(pkl_file, 'rb') as f:
    XX_train_yuan_high = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'YY_train_yuan_high.pkl')
with open(pkl_file, 'rb') as f:
    YY_train_yuan_high = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'YY_total_yuan_high.pkl')
with open(pkl_file, 'rb') as f:
    YY_total_yuan_high = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'YY_train_yuan_low.pkl')
with open(pkl_file, 'rb') as f:
    YY_train_yuan_low = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'Y_test_prediction_AutoGluon_PLOT.pkl')
with open(pkl_file, 'rb') as f:
    Y_test_prediction_AutoGluon_PLOT = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'Y_test_prediction_TabPFN_PLOT.pkl')
with open(pkl_file, 'rb') as f:
    Y_test_prediction_TabPFN_PLOT = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'LOWER_SIGMA_TabPFN.pkl')
with open(pkl_file, 'rb') as f:
    LOWER_SIGMA_TabPFN = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'UPPER_SIGMA_TabPFN.pkl')
with open(pkl_file, 'rb') as f:
    UPPER_SIGMA_TabPFN = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'Y_test_prediction_MFTabPFN_PLOT.pkl')
with open(pkl_file, 'rb') as f:
    Y_test_prediction_MFTabPFN_PLOT = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'LOWER_SIGMA.pkl')
with open(pkl_file, 'rb') as f:
    LOWER_SIGMA = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, f'UPPER_SIGMA.pkl')
with open(pkl_file, 'rb') as f:
    UPPER_SIGMA = pickle.load(f)


plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
# plt.rcParams['figure.dpi'] = 500

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

for i in range(0, 1):
    Ma = MA[0]
    afa = AFA[0]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes = axes.flatten()
    subplot_idx = 0
    legend_handles = []
    legend_labels = []
    for j in range(0, len(Y_TARGET)):
        y_target = Y_TARGET[j]
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

        ax = axes[subplot_idx]
        scatter_train_upper = ax.scatter(XX_train_yuan_high[index_experiment_upper, 0],
                                         YY_train_yuan_high[index_experiment_upper], marker='s', s=20,
                                         color='black')
        scatter_train_lower = ax.scatter(XX_train_yuan_high[index_experiment_lower, 0],
                                         YY_train_yuan_high[index_experiment_lower], marker='s', s=20, color='black')

        scatter_upper = ax.scatter(XX_total_yuan[index_upper, 0],
                    YY_total_yuan_high[index_upper], marker='o', s=20,
                    color='black')
        scatter_lower = ax.scatter(XX_total_yuan[index_lower, 0],
                    YY_total_yuan_high[index_lower], marker='o', s=20,
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

        line_autogluon_upper = ax.plot(XX_train_yuan_low[index_simulation_upper, 0], Y_test_prediction_AutoGluon_PLOT[index_simulation_upper],
                linestyle='-', color=model_colors['AutoGluon'])[0]
        line_autogluon_lower = ax.plot(XX_train_yuan_low[index_simulation_lower, 0], Y_test_prediction_AutoGluon_PLOT[index_simulation_lower],
                linestyle='-', color=model_colors['AutoGluon'])[0]
        ax.plot(np.array(
            [XX_train_yuan_low[index_simulation_upper, 0][0], XX_train_yuan_low[index_simulation_lower, 0][0]]),
                np.array([Y_test_prediction_AutoGluon_PLOT[index_simulation_upper][0],
                          Y_test_prediction_AutoGluon_PLOT[index_simulation_lower][0]]), linestyle='-', color=model_colors['AutoGluon'])
        ax.plot(np.array(
            [XX_train_yuan_low[index_simulation_upper, 0][-1], XX_train_yuan_low[index_simulation_lower, 0][-1]]),
                np.array([Y_test_prediction_AutoGluon_PLOT[index_simulation_upper][-1],
                          Y_test_prediction_AutoGluon_PLOT[index_simulation_lower][-1]]), linestyle='-', color=model_colors['AutoGluon'])

        line_tabpfn_upper = ax.plot(XX_train_yuan_low[index_simulation_upper, 0], Y_test_prediction_TabPFN_PLOT[index_simulation_upper],
                linestyle='-', color=model_colors['TabPFN'])[0]
        line_tabpfn_lower = ax.plot(XX_train_yuan_low[index_simulation_lower, 0], Y_test_prediction_TabPFN_PLOT[index_simulation_lower],
                linestyle='-', color=model_colors['TabPFN'])[0]
        ax.plot(np.array(
            [XX_train_yuan_low[index_simulation_upper, 0][0], XX_train_yuan_low[index_simulation_lower, 0][0]]),
                np.array([Y_test_prediction_TabPFN_PLOT[index_simulation_upper][0],
                          Y_test_prediction_TabPFN_PLOT[index_simulation_lower][0]]), linestyle='-', color=model_colors['TabPFN'])
        ax.plot(np.array(
            [XX_train_yuan_low[index_simulation_upper, 0][-1], XX_train_yuan_low[index_simulation_lower, 0][-1]]),
                np.array([Y_test_prediction_TabPFN_PLOT[index_simulation_upper][-1],
                          Y_test_prediction_TabPFN_PLOT[index_simulation_lower][-1]]), linestyle='-', color=model_colors['TabPFN'])
        line_tabpfn_upper_interval = ax.fill_between(XX_train_yuan_low[index_simulation_upper, 0],
                                                       LOWER_SIGMA_TabPFN[index_simulation_upper],
                                                       UPPER_SIGMA_TabPFN[index_simulation_upper], color=model_colors['TabPFN'],
                                                       edgecolor='none', alpha=0.1)
        line_tabpfn_lower_interval = ax.fill_between(XX_train_yuan_low[index_simulation_lower, 0],
                                                       LOWER_SIGMA_TabPFN[index_simulation_lower],
                                                       UPPER_SIGMA_TabPFN[index_simulation_lower], color=model_colors['TabPFN'],
                                                       edgecolor='none', alpha=0.1)

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
        ax.set_ylim(-1.5, 1)

        ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax.set_yticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])

        ax.invert_yaxis()
        ax.set_xlabel('x/c')
        ax.set_ylabel('Cp')
        ax.set_title(f'y/b = {y_target}')
        ax.grid(True)

        subplot_idx += 1
    legend_handles.extend([
            scatter_train_upper, scatter_upper, line_mftabpfn_upper, line_mftabpfn_upper_interval,
            line_tabpfn_upper, line_tabpfn_upper_interval, line_autogluon_upper, line_simulation_upper
            ])
    legend_labels.extend([
                'Training experimental data', 'Testing experimental data', 'MFTabPFN', 'Standard deviation interval',
                'TabPFN', 'Standard deviation interval','AutoGluon', 'Simulation'
            ])

    # fig.subplots_adjust(bottom=0.25)
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.20))
    fig.suptitle(f'Testing experimental data at y/b = {Y_TARGET[testing_index][0]}', fontsize=12)
    fig.tight_layout()

    # for ext in ['png', 'tiff']:
    #     plt.savefig(f'Cp (Testing experimental data at y_b = {Y_TARGET[testing_index][0]}).{ext}', dpi=500, bbox_inches='tight', format=ext)

    plt.show(block=True)
