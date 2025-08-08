import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from scipy.stats import entropy

SAVE_DIR = './Datasets/Synthetic/UQ'
pkl_file = os.path.join(SAVE_DIR, 'YY_real.pkl')
with open(pkl_file, 'rb') as f:
    YY_real = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'YY_list.pkl')
with open(pkl_file, 'rb') as f:
    YY_list = pickle.load(f)

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

js_results = []
for j in range(0, len(YY_list)):
    YY_total_yuan = YY_real[j]
    Y_test_prediction_TabPFN = YY_list[j][0]
    Y_test_prediction_AutoGluon = YY_list[j][1]
    Y_test_prediction_MFTabPFN = YY_list[j][2]

    YY_total_yuan = np.ravel(YY_total_yuan)
    Y_test_prediction_MFTabPFN = np.ravel(Y_test_prediction_MFTabPFN)
    Y_test_prediction_TabPFN = np.ravel(Y_test_prediction_TabPFN)
    Y_test_prediction_AutoGluon = np.ravel(Y_test_prediction_AutoGluon)

    pdf_data = {}
    x_pdf_MFTabPFN, y_pdf_MFTabPFN = sns.kdeplot(data=Y_test_prediction_MFTabPFN, color=model_colors['MFTabPFN']).get_lines()[0].get_data()
    pdf_data['MFTabPFN'] = (x_pdf_MFTabPFN, y_pdf_MFTabPFN)
    plt.close()
    x_pdf_TabPFN, y_pdf_TabPFN = sns.kdeplot(data=Y_test_prediction_TabPFN, color=model_colors['TabPFN']).get_lines()[0].get_data()
    pdf_data['TabPFN'] = (x_pdf_TabPFN, y_pdf_TabPFN)
    plt.close()
    x_pdf_AutoGluon, y_pdf_AutoGluon = sns.kdeplot(data=Y_test_prediction_AutoGluon, color=model_colors['AutoGluon']).get_lines()[0].get_data()
    pdf_data['AutoGluon'] = (x_pdf_AutoGluon, y_pdf_AutoGluon)
    plt.close()

    fig, ax = plt.subplots(figsize=(4, 3))
    bins = min(30, max(10, len(YY_total_yuan) // 10))
    hist, bin_edges = np.histogram(YY_total_yuan, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mcs_pdf = hist
    ax.hist(YY_total_yuan, bins=bins, density=True, alpha=0.3, color='#AED6F1', label='Reference', edgecolor='black')
    for model in models:
        x_pdf, y_pdf = pdf_data[model]
        ax.plot(x_pdf, y_pdf, label=model_labels[models.index(model)], color=model_colors[model], linewidth=2, linestyle='-')

    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('PDF', fontsize=12)
    ax.set_title(f'PDF of case with n={(j + 1) * 100}', fontsize=12)

    all_data = np.concatenate(
        [YY_total_yuan, Y_test_prediction_MFTabPFN, Y_test_prediction_TabPFN, Y_test_prediction_AutoGluon])
    x_min, x_max = np.min(all_data), np.max(all_data)
    ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
    ax.set_ylim(0, None)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    js_values = {}
    x_grid = np.linspace(x_min, x_max, 1000)
    mcs_pdf_grid = np.interp(x_grid, bin_centers, mcs_pdf, left=0, right=0)
    for model in models:
        x_pdf, y_pdf = pdf_data[model]
        if len(x_pdf) == 0:
            continue
        model_pdf_grid = np.interp(x_grid, x_pdf, y_pdf, left=0, right=0)
        mcs_pdf_grid_norm = mcs_pdf_grid / np.trapz(mcs_pdf_grid, x_grid)
        model_pdf_grid_norm = model_pdf_grid / np.trapz(model_pdf_grid, x_grid)
        m = 0.5 * (mcs_pdf_grid_norm + model_pdf_grid_norm)
        kl_p_m = entropy(mcs_pdf_grid_norm + 1e-10, m + 1e-10)
        kl_q_m = entropy(model_pdf_grid_norm + 1e-10, m + 1e-10)
        js_div = 0.5 * (kl_p_m + kl_q_m)
        js_values[model] = js_div
        js_results.append({
            'Dataset': j + 1,
            'Model': model,
            'JS_Divergence': js_div
        })

    plt.tight_layout()
    plt.subplots_adjust(right=0.75, left=0.15)

    # for ext in ['png', 'tiff']:
    #     output_path = os.path.join(SAVE_DIR, f'pdf_comparison_dataset_{j + 1}.{ext}')
    #     plt.savefig(output_path, dpi=500, bbox_inches='tight', format=ext)

    plt.show(block=True)
    plt.close(fig)

# js_results_df = pd.DataFrame(js_results)
# results_path = os.path.join(SAVE_DIR, 'js_divergence_results.csv')
# js_results_df.to_csv(results_path, index=False)
# print(f"JS results are saved to {results_path}")

for j in range(0, len(YY_list)):
    YY_total_yuan = YY_real[j]
    Y_test_prediction_TabPFN = YY_list[j][0]
    Y_test_prediction_AutoGluon = YY_list[j][1]
    Y_test_prediction_MFTabPFN = YY_list[j][2]

    YY_total_yuan = np.ravel(YY_total_yuan)
    Y_test_prediction_MFTabPFN = np.ravel(Y_test_prediction_MFTabPFN)
    Y_test_prediction_TabPFN = np.ravel(Y_test_prediction_TabPFN)
    Y_test_prediction_AutoGluon = np.ravel(Y_test_prediction_AutoGluon)

    all_data = np.concatenate(
        [YY_total_yuan, Y_test_prediction_MFTabPFN, Y_test_prediction_TabPFN, Y_test_prediction_AutoGluon])
    x_min, x_max = np.min(all_data), np.max(all_data)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], color='#CCCCCC', lw=1.0, linestyle='--', zorder=0)
    ax.scatter((YY_total_yuan-x_min)/(x_max-x_min), (Y_test_prediction_MFTabPFN-x_min)/(x_max-x_min), color=model_colors['MFTabPFN'], s=2, alpha=1.0, marker=model_markers['MFTabPFN'], label='MFTabPFN', zorder=3)
    ax.scatter((YY_total_yuan-x_min)/(x_max-x_min), (Y_test_prediction_TabPFN-x_min)/(x_max-x_min), color=model_colors['TabPFN'], s=2, alpha=1.0, marker=model_markers['TabPFN'], label='TabPFN', zorder=2)
    ax.scatter((YY_total_yuan-x_min)/(x_max-x_min), (Y_test_prediction_AutoGluon-x_min)/(x_max-x_min), color=model_colors['AutoGluon'], s=2, alpha=1.0, marker=model_markers['AutoGluon'], label='AutoGluon', zorder=1)
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([0.0, 1.0])

    ax.set_xlabel('Reference value', fontsize=12)
    ax.set_ylabel('Prediction', fontsize=12)
    ax.set_title(f'Case with n={(j + 1) * 100}', fontsize=12)

    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.subplots_adjust(right=0.75, left=0.15)

    # for ext in ['png', 'tiff']:
    #     output_path = os.path.join(SAVE_DIR, f'Comparison_dataset_{j + 1}.{ext}')
    #     plt.savefig(output_path, dpi=500, bbox_inches='tight', format=ext)

    plt.show(block=True)
    plt.close(fig)
