import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

WORK_DIR = "./Datasets/ONERA_M6/Drag"

pkl_file = os.path.join(WORK_DIR, 'i_CD.pkl')
with open(pkl_file, 'rb') as f:
    i = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'Y_test_prediction_Ad_MFTabPFN_CD.pkl')
with open(pkl_file, 'rb') as f:
    Y_test_prediction_Ad_MFTabPFN = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'YY_total_yuan_high_CD.pkl')
with open(pkl_file, 'rb') as f:
    YY_total_yuan_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'RMSE_Ad_MFTabPFN_CD.pkl')
with open(pkl_file, 'rb') as f:
    RMSE_Ad_MFTabPFN = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'MAE_Ad_MFTabPFN_CD.pkl')
with open(pkl_file, 'rb') as f:
    MAE_Ad_MFTabPFN = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'R2_Ad_MFTabPFN_CD.pkl')
with open(pkl_file, 'rb') as f:
    R2_Ad_MFTabPFN = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'JS_Ad_MFTabPFN_CD.pkl')
with open(pkl_file, 'rb') as f:
    JS_Ad_MFTabPFN = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'Index_fidelity_Ad_MFTabPFN_CD.pkl')
with open(pkl_file, 'rb') as f:
    Index_fidelity_Ad_MFTabPFN = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'N_low_call_CD.pkl')
with open(pkl_file, 'rb') as f:
    N_low_call = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'repeated_values_low_CD.pkl')
with open(pkl_file, 'rb') as f:
    repeated_values_low = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'repeated_counts_low_CD.pkl')
with open(pkl_file, 'rb') as f:
    repeated_counts_low = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'N_high_call_CD.pkl')
with open(pkl_file, 'rb') as f:
    N_high_call = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'repeated_values_high_CD.pkl')
with open(pkl_file, 'rb') as f:
    repeated_values_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'repeated_counts_high_CD.pkl')
with open(pkl_file, 'rb') as f:
    repeated_counts_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'Y_test_prediction_TabPFN_high_CD.pkl')
with open(pkl_file, 'rb') as f:
    Y_test_prediction_TabPFN_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'RMSE_TabPFN_high_CD.pkl')
with open(pkl_file, 'rb') as f:
    RMSE_TabPFN_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'MAE_TabPFN_high_CD.pkl')
with open(pkl_file, 'rb') as f:
    MAE_TabPFN_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'R2_TabPFN_high_CD.pkl')
with open(pkl_file, 'rb') as f:
    R2_TabPFN_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'JS_TabPFN_high_CD.pkl')
with open(pkl_file, 'rb') as f:
    JS_TabPFN_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'Y_test_prediction_TabPFN_low_CD.pkl')
with open(pkl_file, 'rb') as f:
    Y_test_prediction_TabPFN_low = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'RMSE_TabPFN_low_CD.pkl')
with open(pkl_file, 'rb') as f:
    RMSE_TabPFN_low = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'MAE_TabPFN_low_CD.pkl')
with open(pkl_file, 'rb') as f:
    MAE_TabPFN_low = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'R2_TabPFN_low_CD.pkl')
with open(pkl_file, 'rb') as f:
    R2_TabPFN_low = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'JS_TabPFN_low_CD.pkl')
with open(pkl_file, 'rb') as f:
    JS_TabPFN_low = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'Y_test_prediction_AutoGluon_high_CD.pkl')
with open(pkl_file, 'rb') as f:
    Y_test_prediction_AutoGluon_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'RMSE_AutoGluon_high_CD.pkl')
with open(pkl_file, 'rb') as f:
    RMSE_AutoGluon_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'MAE_AutoGluon_high_CD.pkl')
with open(pkl_file, 'rb') as f:
    MAE_AutoGluon_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'R2_AutoGluon_high_CD.pkl')
with open(pkl_file, 'rb') as f:
    R2_AutoGluon_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'JS_AutoGluon_high_CD.pkl')
with open(pkl_file, 'rb') as f:
    JS_AutoGluon_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'Y_test_prediction_AutoGluon_low_CD.pkl')
with open(pkl_file, 'rb') as f:
    Y_test_prediction_AutoGluon_low = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'RMSE_AutoGluon_low_CD.pkl')
with open(pkl_file, 'rb') as f:
    RMSE_AutoGluon_low = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'MAE_AutoGluon_low_CD.pkl')
with open(pkl_file, 'rb') as f:
    MAE_AutoGluon_low = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'R2_AutoGluon_low_CD.pkl')
with open(pkl_file, 'rb') as f:
    R2_AutoGluon_low = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'JS_AutoGluon_low_CD.pkl')
with open(pkl_file, 'rb') as f:
    JS_AutoGluon_low = pickle.load(f)

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

x_pdf_ad_mftabpfn, y_pdf_ad_mftabpfn = sns.kdeplot(data=Y_test_prediction_Ad_MFTabPFN).get_lines()[0].get_data()
plt.close()
x_pdf_tabpfn_low, y_pdf_tabpfn_low = sns.kdeplot(data=Y_test_prediction_TabPFN_low).get_lines()[0].get_data()
plt.close()
x_pdf_tabpfn_high, y_pdf_tabpfn_high = sns.kdeplot(data=Y_test_prediction_TabPFN_high).get_lines()[0].get_data()
plt.close()
x_pdf_autogluon_low, y_pdf_autogluon_low = sns.kdeplot(data=Y_test_prediction_AutoGluon_low).get_lines()[0].get_data()
plt.close()
x_pdf_autogluon_high, y_pdf_autogluon_high = sns.kdeplot(data=Y_test_prediction_AutoGluon_high).get_lines()[0].get_data()
plt.close()
x_pdf_mcs, y_pdf_mcs = sns.kdeplot(data=YY_total_yuan_high).get_lines()[0].get_data()
plt.close()

fig, ax_left = plt.subplots(figsize=(4.5, 3))

ax_right = ax_left.twinx()
ax_left.axvline(x=i, color='k', linestyle='--', linewidth=1.0)
ax_left.plot(np.arange(i+1), JS_Ad_MFTabPFN[:i+1], linestyle='-', linewidth=2.0, color=model_colors['XGBoost'], label='JS divergence')

ax_right.plot(np.arange(i+1), RMSE_Ad_MFTabPFN[:i+1], linestyle='-', linewidth=2.0, color=model_colors['LightGBM'], label='Normalized negative RMSE')
ax_right.plot(np.arange(i+1), MAE_Ad_MFTabPFN[:i+1], linestyle='-', linewidth=2.0, color=model_colors['SVR'], label='Normalized negative MAE')
ax_right.plot(np.arange(i+1), R2_Ad_MFTabPFN[:i+1], linestyle='-', linewidth=2.0, color=model_colors['ANN'], label=r'$R^2$')

ax_left.set_xlim(0, 220)
ax_left.set_xticks([0, 55, 110, 165, 220])
ax_left.set_title('Metrics for drag coefficient', fontsize=12)

ax_left.set_yticks([0.0, 0.01, 0.02, 0.03, 0.04])
ax_left.set_ylabel('JS divergence', fontsize=12)
ax_left.xaxis.grid(True)
ax_left.yaxis.grid(True)

ax_right.set_ylim(-0.60, 1.00)
ax_right.set_yticks([-0.60, -0.20, 0.20, 0.60, 1.00])
ax_right.set_ylabel('Metrics (RMSE, MAE, R²)', fontsize=12)
ax_right.yaxis.grid(False)

ax_left.set_xlabel('Iteration', fontsize=12)

handles_left, labels_left = ax_left.get_legend_handles_labels()
handles_right, labels_right = ax_right.get_legend_handles_labels()
handles = handles_left + handles_right
labels = labels_left + labels_right
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15), frameon=True, fontsize=12)
plt.tight_layout()
# plt.subplots_adjust(bottom=0.25)
# for ext in ['png', 'tiff']:
#     plt.savefig(f'Metrics_CD_n.{ext}', dpi=500, bbox_inches='tight', format=ext)
plt.show(block=True)
plt.close(fig)


fig, ax = plt.subplots(figsize=(4, 3))
ax.hist(YY_total_yuan_high, bins=25, density=True, alpha=0.3, color='#AED6F1', label='Reference', edgecolor='black', zorder=0)
ax.plot(x_pdf_ad_mftabpfn, y_pdf_ad_mftabpfn, color=model_colors['MFTabPFN'], linewidth=2, linestyle='-', label="Active MFTabPFN", zorder=3)
ax.plot(x_pdf_tabpfn_low, y_pdf_tabpfn_low, color=model_colors['TabPFN'], linewidth=2, linestyle='--', label="TabPFN-low", zorder=2)
ax.plot(x_pdf_tabpfn_high, y_pdf_tabpfn_high, color=model_colors['TabPFN'], linewidth=2, linestyle='-', label="TabPFN-high", zorder=2)
ax.plot(x_pdf_autogluon_low, y_pdf_autogluon_low, color=model_colors['AutoGluon'], linewidth=2, linestyle='--', label="AutoGluon-low", zorder=1)
ax.plot(x_pdf_autogluon_high, y_pdf_autogluon_high, color=model_colors['AutoGluon'], linewidth=2, linestyle='-', label="AutoGluon-high", zorder=1)

ax.set_xlim(0.00, 0.03)
ax.set_xticks([0.0, 0.01, 0.02, 0.03])
ax.set_yticks([0, 110, 220, 330, 440, 550])
ax.set_title('PDF of drag coefficient', fontsize=12)

ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('PDF', fontsize=12)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# ax.legend()
plt.tight_layout()
# for ext in ['png', 'tiff']:
#     plt.savefig(f'PDF_CD.{ext}', dpi=500, bbox_inches='tight', format=ext)
plt.show(block=True)
plt.close(fig)
