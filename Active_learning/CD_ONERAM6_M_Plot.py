import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from pathlib import Path
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
import scipy.stats as stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.cbook import boxplot_stats


WORK_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "Multi-fidelity" / "ONERAM6" / "ONERA_M6_Data"
pkl_file = os.path.join(WORK_DIR, 'XX_total_yuan.pkl')
with open(pkl_file, 'rb') as f:
    XX_total_yuan = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'CD_total_high.pkl')
with open(pkl_file, 'rb') as f:
    CD_total_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'CD_total_low.pkl')
with open(pkl_file, 'rb') as f:
    CD_total_low = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'Time_high.pkl')
with open(pkl_file, 'rb') as f:
    Time_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'Time_low.pkl')
with open(pkl_file, 'rb') as f:
    Time_low = pickle.load(f)

SAVE_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "Multi-fidelity" / "ONERAM6" / "Result"
stage = 25
pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_list.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_list = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_prediction.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_prediction = pickle.load(f)

pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_list_AMFGPR.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_list_AMFGPR = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_prediction_AMFGPR.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_prediction_AMFGPR = pickle.load(f)

pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_list_ANMFGPR.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_list_ANMFGPR = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_prediction_ANMFGPR.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_prediction_ANMFGPR = pickle.load(f)

pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_list_MFGPR.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_list_MFGPR = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_prediction_MFGPR.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_prediction_MFGPR = pickle.load(f)

pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_list_NMFGPR.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_list_NMFGPR = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_prediction_NMFGPR.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_prediction_NMFGPR = pickle.load(f)

pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_list_ML.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_list_ML = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_prediction_ML.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_prediction_ML = pickle.load(f)

pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_list_MAHNN.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_list_MAHNN = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_prediction_MAHNN.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_prediction_MAHNN = pickle.load(f)

pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_list_FNO.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_list_TLFNO = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_prediction_FNO.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_prediction_TLFNO = pickle.load(f)

YY_total_yuan_low = CD_total_low.ravel()
YY_total_yuan_high = CD_total_high.ravel()

stop0 = np.array([0.06, 0.05, 0.04])
stop = np.zeros((3,len(stop0)), dtype=int)
for k in range(3):
    Stop = Results_ONERAM6_list[k].loc[:, "Stop"].values
    Stop = np.concatenate(Stop)
    for kk in range(len(stop0)):
        index = np.where (Stop[1:] < stop0[kk])[0]
        if index.size > 0:
            stop[k, kk] = index[0] + 1
        else:
            index_min = np.where(Stop[1:] == np.min(Stop[1:]))[0]
            stop[k, kk] = index_min[0] + 1

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
plt.rcParams['xtick.major.pad'] = 1.5
plt.rcParams['ytick.major.pad'] = 0.8
model_colors = {
    'AMFGPR': '#1b9e77',
    'MFGPR': '#1b9e77',
    'AMFTabPFN': '#1f77b4',
    'MFTabPFN': '#1f77b4',
    'ANMFGPR': '#17becf',
    'NMFGPR': '#17becf',
    'TLFNO': '#8c564b',
    'MAHNN': '#bcbd88',
    'TabPFN-H': '#2ca02c',
    'TabPFN-M': '#2ca02c',
    'AutoGluon-H': '#9467bd',
    'AutoGluon-M': '#9467bd',
}
models = ['AMFTabPFN', 'MFTabPFN', 'TabPFN-H', 'TabPFN-M', 'AutoGluon-H', 'AutoGluon-M',
          'AMFGPR', 'ANMFGPR', 'MFGPR', 'NMFGPR', 'MAHNN', 'TLFNO']

confidence = 0.95
alpha = 1 - confidence

class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):
        self.format = '%.1e'
class OOMFormatter(ScalarFormatter):
    def __init__(self, order=0, fformat="%.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % self.format
    def __call__(self, x, pos=None):
        if abs(x) < 1e-12:
            return '0'
        return super().__call__(x, pos)


pdf_data = {}
for k in range(3):
    pdf_data[k] = {}
    for kk in range(len(stop0)):
        pdf_data[k][kk] = {}
        Performance = Results_ONERAM6_prediction[k].loc[stop[k, kk], "Performance"].ravel()
        Performance_MFTabPFN = (Results_ONERAM6_prediction_ML[k][kk].query("Model == 'MFTabPFN'")['Y_test_prediction'].iloc[0]).ravel()
        Performance_TabPFN_H = (Results_ONERAM6_prediction_ML[k][kk].query("Model == 'TabPFN-High'")['Y_test_prediction'].iloc[0]).ravel()
        Performance_TabPFN_M = (Results_ONERAM6_prediction_ML[k][kk].query("Model == 'TabPFN-Multi'")['Y_test_prediction'].iloc[0]).ravel()
        Performance_AutoGluon_H = (Results_ONERAM6_prediction_ML[k][kk].query("Model == 'AutoGluon-High'")['Y_test_prediction'].iloc[0]).ravel()
        Performance_AutoGluon_M = (Results_ONERAM6_prediction_ML[k][kk].query("Model == 'AutoGluon-Multi'")['Y_test_prediction'].iloc[0]).ravel()
        Performance_AMFGPR = Results_ONERAM6_prediction_AMFGPR[k].loc[stop[k, kk], "Performance"].ravel()
        Performance_ANMFGPR = Results_ONERAM6_prediction_ANMFGPR[k].loc[stop[k, kk], "Performance"].ravel()
        Performance_MFGPR = Results_ONERAM6_prediction_MFGPR[k][kk].loc[0, "Y_test_prediction"].ravel()
        Performance_NMFGPR = Results_ONERAM6_prediction_NMFGPR[k][kk].loc[0, "Y_test_prediction"].ravel()
        Performance_MAHNN = Results_ONERAM6_prediction_MAHNN[k][kk].loc[0, "Y_test_prediction"].ravel()
        Performance_TLFNO = Results_ONERAM6_prediction_TLFNO[k][kk].loc[0, "Y_test_prediction"].ravel()

        x_pdf_Performance, y_pdf_Performance = sns.kdeplot(data=Performance).get_lines()[0].get_data()
        plt.close()
        x_pdf_low, y_pdf_low = sns.kdeplot(data=YY_total_yuan_low).get_lines()[0].get_data()
        plt.close()
        x_pdf_Performance_MFTabPFN, y_pdf_Performance_MFTabPFN = sns.kdeplot(data=Performance_MFTabPFN).get_lines()[0].get_data()
        plt.close()
        x_pdf_Performance_TabPFN_H, y_pdf_Performance_TabPFN_H = sns.kdeplot(data=Performance_TabPFN_H).get_lines()[0].get_data()
        plt.close()
        x_pdf_Performance_TabPFN_M, y_pdf_Performance_TabPFN_M = sns.kdeplot(data=Performance_TabPFN_M).get_lines()[0].get_data()
        plt.close()
        x_pdf_Performance_AutoGluon_H, y_pdf_Performance_AutoGluon_H = sns.kdeplot(data=Performance_AutoGluon_H).get_lines()[0].get_data()
        plt.close()
        x_pdf_Performance_AutoGluon_M, y_pdf_Performance_AutoGluon_M = sns.kdeplot(data=Performance_AutoGluon_M).get_lines()[0].get_data()
        plt.close()
        x_pdf_Performance_AMFGPR, y_pdf_Performance_AMFGPR = sns.kdeplot(data=Performance_AMFGPR).get_lines()[0].get_data()
        plt.close()
        x_pdf_Performance_ANMFGPR, y_pdf_Performance_ANMFGPR = sns.kdeplot(data=Performance_ANMFGPR).get_lines()[0].get_data()
        plt.close()
        x_pdf_Performance_MFGPR, y_pdf_Performance_MFGPR = sns.kdeplot(data=Performance_MFGPR).get_lines()[0].get_data()
        plt.close()
        x_pdf_Performance_NMFGPR, y_pdf_Performance_NMFGPR = sns.kdeplot(data=Performance_NMFGPR).get_lines()[0].get_data()
        plt.close()
        x_pdf_Performance_MAHNN, y_pdf_Performance_MAHNN = sns.kdeplot(data=Performance_MAHNN).get_lines()[0].get_data()
        plt.close()
        x_pdf_Performance_TLFNO, y_pdf_Performance_TLFNO = sns.kdeplot(data=Performance_TLFNO).get_lines()[0].get_data()
        plt.close()

        pdf_data[k][kk]['Low-fidelity'] = (x_pdf_low, y_pdf_low)
        pdf_data[k][kk]['AMFTabPFN'] = (x_pdf_Performance, y_pdf_Performance)
        pdf_data[k][kk]['MFTabPFN'] = (x_pdf_Performance_MFTabPFN, y_pdf_Performance_MFTabPFN)
        pdf_data[k][kk]['TabPFN-H'] = (x_pdf_Performance_TabPFN_H, y_pdf_Performance_TabPFN_H)
        pdf_data[k][kk]['TabPFN-M'] = (x_pdf_Performance_TabPFN_M, y_pdf_Performance_TabPFN_M)
        pdf_data[k][kk]['AutoGluon-H'] = (x_pdf_Performance_AutoGluon_H, y_pdf_Performance_AutoGluon_H)
        pdf_data[k][kk]['AutoGluon-M'] = (x_pdf_Performance_AutoGluon_M, y_pdf_Performance_AutoGluon_M)
        pdf_data[k][kk]['AMFGPR'] = (x_pdf_Performance_AMFGPR, y_pdf_Performance_AMFGPR)
        pdf_data[k][kk]['ANMFGPR'] = (x_pdf_Performance_ANMFGPR, y_pdf_Performance_ANMFGPR)
        pdf_data[k][kk]['MFGPR'] = (x_pdf_Performance_MFGPR, y_pdf_Performance_MFGPR)
        pdf_data[k][kk]['NMFGPR'] = (x_pdf_Performance_NMFGPR, y_pdf_Performance_NMFGPR)
        pdf_data[k][kk]['MAHNN'] = (x_pdf_Performance_MAHNN, y_pdf_Performance_MAHNN)
        pdf_data[k][kk]['TLFNO'] = (x_pdf_Performance_TLFNO, y_pdf_Performance_TLFNO)


for k in range(3):
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(6.8, 4.5),
    )
    for kk in range(len(stop0)):
        x_pdf_Performance, y_pdf_Performance = pdf_data[k][kk]['AMFTabPFN']
        x_pdf_low, y_pdf_low = pdf_data[k][kk]['Low-fidelity']
        x_pdf_Performance_MFTabPFN, y_pdf_Performance_MFTabPFN = pdf_data[k][kk]['MFTabPFN']
        x_pdf_Performance_TabPFN_H, y_pdf_Performance_TabPFN_H = pdf_data[k][kk]['TabPFN-H']
        x_pdf_Performance_TabPFN_M, y_pdf_Performance_TabPFN_M = pdf_data[k][kk]['TabPFN-M']
        x_pdf_Performance_AutoGluon_H, y_pdf_Performance_AutoGluon_H = pdf_data[k][kk]['AutoGluon-H']
        x_pdf_Performance_AutoGluon_M, y_pdf_Performance_AutoGluon_M = pdf_data[k][kk]['AutoGluon-M']
        x_pdf_Performance_AMFGPR, y_pdf_Performance_AMFGPR = pdf_data[k][kk]['AMFGPR']
        x_pdf_Performance_ANMFGPR, y_pdf_Performance_ANMFGPR = pdf_data[k][kk]['ANMFGPR']
        x_pdf_Performance_MFGPR, y_pdf_Performance_MFGPR = pdf_data[k][kk]['MFGPR']
        x_pdf_Performance_NMFGPR, y_pdf_Performance_NMFGPR = pdf_data[k][kk]['NMFGPR']
        x_pdf_Performance_MAHNN, y_pdf_Performance_MAHNN = pdf_data[k][kk]['MAHNN']
        x_pdf_Performance_TLFNO, y_pdf_Performance_TLFNO = pdf_data[k][kk]['TLFNO']

        ax = axes[kk, 0]
        ax.hist(YY_total_yuan_high, bins=25, density=True, alpha=0.4, color='#AED6F1', edgecolor='black', label='Reference', zorder=0)
        ax.plot(x_pdf_low, y_pdf_low, color='black', linewidth=1.5, linestyle = '--', label='Low-fidelity', zorder=1)
        ax.plot(x_pdf_Performance_AutoGluon_H, y_pdf_Performance_AutoGluon_H, color=model_colors['AutoGluon-H'], linewidth=1.5, linestyle='--', label='AutoGluon-H', zorder=2)
        ax.plot(x_pdf_Performance_AutoGluon_M, y_pdf_Performance_AutoGluon_M, color=model_colors['AutoGluon-M'], linewidth=1.5, linestyle='-', label='AutoGluon-M', zorder=3)
        ax.plot(x_pdf_Performance_TabPFN_H, y_pdf_Performance_TabPFN_H, color=model_colors['TabPFN-H'], linewidth=1.5, linestyle='--', label='TabPFN-H', zorder=4)
        ax.plot(x_pdf_Performance_TabPFN_M, y_pdf_Performance_TabPFN_M, color=model_colors['TabPFN-M'], linewidth=1.5, linestyle='-', label='TabPFN-M', zorder=5)
        ax.plot(x_pdf_Performance_MFTabPFN, y_pdf_Performance_MFTabPFN, color=model_colors['MFTabPFN'], linewidth=1.5, linestyle='--', label='MFTabPFN', zorder=6)
        ax.plot(x_pdf_Performance, y_pdf_Performance, color=model_colors['AMFTabPFN'], linewidth=1.5, linestyle='-', label='AMFTabPFN', zorder=7)

        ax.set_ylim(0, 300)
        ax.set_yticks([0, 100, 200, 300])
        ax.set_title(f'PDF under Stop {kk + 1}')
        ax.set_ylabel('PDF')

        ax.set_xlim(-0.005, 0.035)
        ax.set_xticks([0.00, 0.01, 0.02, 0.03])
        if kk == 2:
            ax.set_xlabel('Value')

        if kk == 0:
            leg = ax.legend(loc='upper right',
                             frameon=True,
                             columnspacing=0.7,
                             fontsize=5,
                             )
            leg.get_frame().set_linewidth(0.75)

        ax = axes[kk, 1]
        ax.hist(YY_total_yuan_high, bins=25, density=True, alpha=0.4, color='#AED6F1', edgecolor='black', label='Reference', zorder=0)
        ax.plot(x_pdf_low, y_pdf_low, color='black', linewidth=1.5, linestyle = '--', label='Low-fidelity', zorder=1)
        ax.plot(x_pdf_Performance_MFGPR, y_pdf_Performance_MFGPR, color=model_colors['MFGPR'], linewidth=1.5, linestyle='--', label='MFGPR', zorder=2)
        ax.plot(x_pdf_Performance_AMFGPR, y_pdf_Performance_AMFGPR, color=model_colors['AMFGPR'], linewidth=1.5, linestyle='-', label='AMFGPR', zorder=3)
        ax.plot(x_pdf_Performance_NMFGPR, y_pdf_Performance_NMFGPR, color=model_colors['NMFGPR'], linewidth=1.5, linestyle='--', label='NMFGPR', zorder=4)
        ax.plot(x_pdf_Performance_ANMFGPR, y_pdf_Performance_ANMFGPR, color=model_colors['ANMFGPR'], linewidth=1.5, linestyle='-', label='ANMFGPR', zorder=5)
        ax.plot(x_pdf_Performance, y_pdf_Performance, color=model_colors['AMFTabPFN'], linewidth=1.5, linestyle='-', label='AMFTabPFN', zorder=6)
        ax.set_ylim(0, 300)
        ax.set_yticks([0, 100, 200, 300])
        ax.set_title(f'PDF under Stop {kk + 1}')
        ax.set_xlim(-0.005, 0.035)
        ax.set_xticks([0.00, 0.01, 0.02, 0.03])
        if kk == 2:
            ax.set_xlabel('Value')

        if kk == 0:
            leg = ax.legend(loc='upper right',
                             frameon=True,
                             columnspacing=0.7,
                             fontsize=5,
                             )
            leg.get_frame().set_linewidth(0.75)

        ax = axes[kk, 2]
        ax.hist(YY_total_yuan_high, bins=25, density=True, alpha=0.4, color='#AED6F1', edgecolor='black', label='Reference', zorder=0)
        ax.plot(x_pdf_low, y_pdf_low, color='black', linewidth=1.5, linestyle = '--', label='Low-fidelity', zorder=1)
        ax.plot(x_pdf_Performance_MAHNN, y_pdf_Performance_MAHNN, color=model_colors['MAHNN'], linewidth=1.5, linestyle='-', label='MAHNN', zorder=2)
        ax.plot(x_pdf_Performance_TLFNO, y_pdf_Performance_TLFNO, color=model_colors['TLFNO'], linewidth=1.5, linestyle='-', label='TLFNO', zorder=3)
        ax.plot(x_pdf_Performance, y_pdf_Performance, color=model_colors['AMFTabPFN'], linewidth=1.5, linestyle='-', label='AMFTabPFN', zorder=4)

        ax.set_ylim(0, 300)
        ax.set_yticks([0, 100, 200, 300])
        ax.set_title(f'PDF under Stop {kk + 1}')
        ax.set_xlim(-0.005, 0.035)
        ax.set_xticks([0.00, 0.01, 0.02, 0.03])
        if kk == 2:
            ax.set_xlabel('Value')

        if kk == 0:
            leg = ax.legend(loc='upper right',
                             frameon=True,
                             columnspacing=0.7,
                             fontsize=5,
                             )
            leg.get_frame().set_linewidth(0.75)

    plt.tight_layout(pad=0.01)
    plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05, wspace=0.18, hspace=0.35)

    for ext in ['png', 'tiff', 'pdf']:
        output_path = os.path.join(SAVE_DIR, f'fold_{k + 1}_pdf.{ext}')
        plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)

    plt.show()


metrics = ["RMSE_N", "MAE_N", "R2", "RMSE", "MAE"]
metric_titles = {'RMSE_N': 'NNRMSE', 'MAE_N': 'NNMAE', 'R2': r'$R^2$', 'RMSE': 'RMSE', 'MAE': 'MAE'}
U = 300
for k in range(3):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=5,
        figsize=(6.8, 1.4),
        sharex=True,
        sharey=False
    )
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        value = Results_ONERAM6_list[k].loc[:, metric].values
        ax.plot(range(U), value[:U], linewidth=1.0)
        ax.grid(True, axis='both', linestyle='--', linewidth=0.5, alpha=0.7)

        vertical_lines_info = [
            # (stage, 'black', '--', 1.0, 0.8, 'Stage'),
            (stop[k, 0], 'green', '-', 1.2, 0.8, 'Stop 1'),
            (stop[k, 1], 'orange', '-', 1.2, 0.8, 'Stop 2'),
            (stop[k, 2], 'purple', '-', 1.2, 0.8, 'Stop 3'),
        ]
        for x_pos, color, ls, lin, alpha, label in vertical_lines_info:
            ax.axvline(
                x=x_pos,
                color=color,
                linestyle=ls,
                linewidth=lin,
                alpha=alpha,
                label=label if label else None
            )
        if idx == 0:
            leg = ax.legend(loc='lower right',
                             bbox_to_anchor=(0.93, 0.0),
                             columnspacing=0.7,
                             fontsize=5,
                             )
            leg.get_frame().set_linewidth(0.75)

        ax.set_title(metric_titles[metric])
        ax.set_xlabel("Iteration")
        if idx == 0:
            ax.set_ylabel("Metric")

        ax.set_xlim(0, 300)
        ax.set_xticks([0, 100, 200, 300])

        y_min, y_max = ax.get_ylim()
        yyy = np.linspace(y_min, y_max, 5)
        ax.set_yticks(yyy)

        if k == 1:
            if metric == 'RMSE':
                order = -3
            elif metric == 'MAE':
                order = -3
            elif metric == 'R2':
                order = 0
            else:
                order = -1
        else:
            if metric == 'RMSE':
                order = -3
            elif metric == 'MAE':
                order = -3
            else:
                order = -1
        ax.yaxis.set_major_formatter(OOMFormatter(order=order, fformat="%.1f"))
        ax.yaxis.offsetText.set_x(-0.10)
    plt.tight_layout(pad=0.01)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.20, wspace=0.30)
    for ext in ['png', 'tiff', 'pdf']:
        output_path = os.path.join(SAVE_DIR, f'fold_{k + 1}_iteration.{ext}')
        plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)
    plt.show()



uq_index = []
for k in range(3):
    for kk in range(len(stop0)):
        Performance_AMFTabPFN = Results_ONERAM6_prediction[k].loc[stop[k, kk], "Performance"].ravel()
        Performance_MFTabPFN = (Results_ONERAM6_prediction_ML[k][kk].query("Model == 'MFTabPFN'")['Y_test_prediction'].iloc[0]).ravel()
        Performance_TabPFN_H = (Results_ONERAM6_prediction_ML[k][kk].query("Model == 'TabPFN-High'")['Y_test_prediction'].iloc[0]).ravel()
        Performance_TabPFN_M = (Results_ONERAM6_prediction_ML[k][kk].query("Model == 'TabPFN-Multi'")['Y_test_prediction'].iloc[0]).ravel()
        Performance_AutoGluon_H = (Results_ONERAM6_prediction_ML[k][kk].query("Model == 'AutoGluon-High'")['Y_test_prediction'].iloc[0]).ravel()
        Performance_AutoGluon_M = (Results_ONERAM6_prediction_ML[k][kk].query("Model == 'AutoGluon-Multi'")['Y_test_prediction'].iloc[0]).ravel()
        Performance_AMFGPR = Results_ONERAM6_prediction_AMFGPR[k].loc[stop[k, kk], "Performance"].ravel()
        Performance_ANMFGPR = Results_ONERAM6_prediction_ANMFGPR[k].loc[stop[k, kk], "Performance"].ravel()
        Performance_MFGPR = Results_ONERAM6_prediction_MFGPR[k][kk].loc[0, "Y_test_prediction"].ravel()
        Performance_NMFGPR = Results_ONERAM6_prediction_NMFGPR[k][kk].loc[0, "Y_test_prediction"].ravel()
        Performance_MAHNN = Results_ONERAM6_prediction_MAHNN[k][kk].loc[0, "Y_test_prediction"].ravel()
        Performance_TLFNO = Results_ONERAM6_prediction_TLFNO[k][kk].loc[0, "Y_test_prediction"].ravel()

        model_arrays = [Performance_AMFTabPFN, Performance_MFTabPFN, Performance_TabPFN_H, Performance_TabPFN_M,
                        Performance_AutoGluon_H, Performance_AutoGluon_M, Performance_AMFGPR, Performance_ANMFGPR,
                        Performance_MFGPR, Performance_NMFGPR, Performance_MAHNN, Performance_TLFNO]

        all_data = np.concatenate(model_arrays)
        x_grid = np.linspace(all_data.min() - 0.05 * np.ptp(all_data), all_data.max() + 0.05 * np.ptp(all_data),5000)

        kde_ref = gaussian_kde(YY_total_yuan_high)
        pdf_ref = kde_ref(x_grid)
        pdf_ref = pdf_ref / np.trapezoid(pdf_ref, x_grid)
        cdf_ref = cumulative_trapezoid(pdf_ref, x_grid, initial=0)
        cdf_ref = cdf_ref / cdf_ref[-1]

        for name, data_arr in zip(models, model_arrays):
            kde = gaussian_kde(data_arr)
            pdf_model = kde(x_grid)
            pdf_model = pdf_model / np.trapezoid(pdf_model, x_grid)
            cdf_model = cumulative_trapezoid(pdf_model, x_grid, initial=0)
            cdf_model = cdf_model / cdf_model[-1]

            epsilon = 1e-12
            P = pdf_ref + epsilon
            Q = pdf_model + epsilon
            M = 0.5 * (P + Q)
            def js_divergence(p, q, m, x_grid):
                kl_pm = np.sum(p * np.log(p / m)) * (x_grid[1] - x_grid[0])
                kl_qm = np.sum(q * np.log(q / m)) * (x_grid[1] - x_grid[0])
                return 0.5 * (kl_pm + kl_qm)
            js_div = js_divergence(P, Q, M, x_grid)

            cdf_interp = interp1d(x_grid, cdf_model, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))
            cdf_ref_interp = interp1d(x_grid, cdf_ref, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))
            F_interp = cdf_interp(x_grid)
            F_ref = cdf_ref_interp(x_grid)
            integrand = np.abs(F_interp - F_ref)
            wd1 = np.trapezoid(integrand, x_grid)

            def crps_one_sample(y):
                x = x_grid
                F = cdf_interp(x)
                Heaviside = (x >= y).astype(float)
                integrand = (F - Heaviside) ** 2
                return np.trapezoid(integrand, x)
            crps = np.mean([crps_one_sample(y) for y in YY_total_yuan_high])

            log_pdf_at_truth = kde.logpdf(YY_total_yuan_high)
            nll = -log_pdf_at_truth.mean()

            uq_index.append(
                {'Dataset': k, 'Fold': kk, 'Model': name, 'x_grid': x_grid, 'pdf_model': pdf_model, 'cdf_model': cdf_model,
                 'JS': js_div, 'WD1': wd1, 'CRPS': crps, 'NLL': nll})

UQ_index = pd.DataFrame(uq_index)

pkl_file = os.path.join(SAVE_DIR, 'UQ_index.pkl')
with open(pkl_file, 'wb') as f:
    pickle.dump(UQ_index, f)
pkl_file = os.path.join(SAVE_DIR, 'UQ_index.pkl')
with open(pkl_file, 'rb') as f:
    UQ_index = pickle.load(f)


metrics = ['JS', 'WD1', 'CRPS', 'NLL']
stats_dict = {}
for metric in metrics:
    grouped = UQ_index.groupby(['Model'])[metric].agg(['mean', 'std', 'count']).reset_index()
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['ci'] = grouped.apply(lambda row: row['se'] * stats.t.ppf(1 - alpha / 2, row['count'] - 1) if row['count'] > 1 else 0, axis=1)
    stats_dict[metric] = grouped


metrics = ['JS', 'WD1']
metric_titles = {'JS': 'JS divergence', 'WD1': 'Wasserstein-1', 'CRPS': 'CRPS', 'NLL': 'NLL'}
metric_ylabels = {'JS': 'JS divergence (95% CI)', 'WD1': 'Wasserstein-1 (95% CI)', 'CRPS': 'CRPS (95% CI)', 'NLL': 'NLL (95% CI)'}

fig, axes = plt.subplots(
    ncols=len(metrics),
    figsize=(3.4, 1.8),
)

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    df_stat = stats_dict[metric].copy()

    df_stat = df_stat.sort_values('mean', ascending=False).reset_index(drop=True)

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

    ax.yaxis.set_label_coords(-0.15, 0.5)

    y_min, y_max = ax.get_ylim()
    yyy = np.linspace(0.000001, y_max, 5)
    ax.set_yticks(yyy)

    if metric == 'JS':
        order = -1
    elif metric == 'WD1':
        order = -3

    ax.yaxis.set_major_formatter(OOMFormatter(order=order, fformat="%.1f"))
    ax.yaxis.offsetText.set_x(-0.09)

    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    if metric == 'JS':
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
                         loc='upper right',
                         frameon=True,
                         ncol=2,
                         columnspacing=0.5,
                         handletextpad=0.3,
                         handlelength=0.8,
                         fontsize=6
                         )
        leg1.get_frame().set_linewidth(0.75)

plt.tight_layout(pad=0.01)
plt.subplots_adjust(left=0.05, right=0.97, top=0.85, bottom=0.30, wspace=0.30)

for ext in ['png', 'tiff', 'pdf']:
        output_path = os.path.join(SAVE_DIR, f'metric_uq.{ext}')
        plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)

plt.show()



N_low0 = 50
N_high0 = 25
random_seeds_train = np.array([111, 222, 333])
N_low_list = np.zeros((3,len(stop0)), dtype=int)
N_high_list = np.zeros((3,len(stop0)), dtype=int)
Unique_low_AMFTabPFN = [None for _ in range(3)]
Unique_high_AMFTabPFN = [None for _ in range(3)]
Simulation_low_AMFTabPFN = np.zeros((3,len(stop0)), dtype=int)
Simulation_high_AMFTabPFN = np.zeros((3,len(stop0)), dtype=int)
for j in range(3):
    random_seeds = random_seeds_train[j]
    np.random.seed(random_seeds)
    N = 1000
    indices_low = np.random.choice(N, size=N_low0, replace=False)
    indices_high = np.random.choice(N, size=N_high0, replace=False)
    result_low = []
    result_high = []
    for k in range(len(stop0)):
        Results_ONERAM6_list111 = Results_ONERAM6_list[j][:stop[j, k]]
        index1 = Results_ONERAM6_list111.query("Index_fidelity == 0")['Index_max'].to_numpy()
        index_low = np.concatenate([indices_low, index1])
        index2 = Results_ONERAM6_list111.query("Index_fidelity == 1")['Index_max'].to_numpy()
        index_high = np.concatenate([indices_high, index2])
        unique_count_low = len(np.unique(index_low))
        unique_count_high = len(np.unique(index_high))
        _, idx = np.unique(index_low, return_index=True)
        unique_values_low = index_low[np.sort(idx)]
        _, idx = np.unique(index_high, return_index=True)
        unique_values_high = index_high[np.sort(idx)]
        N_low_list[j, k] = unique_count_low
        N_high_list[j, k] = unique_count_high
        time111 = np.sum(Time_low[unique_values_low])
        time222 = np.sum(Time_high[unique_values_high])
        Simulation_low_AMFTabPFN[j, k] = time111
        Simulation_high_AMFTabPFN[j, k] = time222
        result_low.append(unique_values_low)
        result_high.append(unique_values_high)
    Unique_low_AMFTabPFN[j] = result_low
    Unique_high_AMFTabPFN[j] = result_high

Unique_low_AMFGPR = [None for _ in range(3)]
Unique_high_AMFGPR = [None for _ in range(3)]
Simulation_low_AMFGPR = np.zeros((3,len(stop0)), dtype=int)
Simulation_high_AMFGPR = np.zeros((3,len(stop0)), dtype=int)
for j in range(3):
    random_seeds = random_seeds_train[j]
    np.random.seed(random_seeds)
    N = 1000
    indices_low = np.random.choice(N, size=N_low0, replace=False)
    indices_high = np.random.choice(N, size=N_high0, replace=False)
    result_low = []
    result_high = []
    for k in range(len(stop0)):
        Results_ONERAM6_list111 = Results_ONERAM6_list_AMFGPR[j][:stop[j, k]]
        index1 = Results_ONERAM6_list111.query("Index_fidelity == 0")['Index_max'].to_numpy()
        index_low = np.concatenate([indices_low, index1])
        index2 = Results_ONERAM6_list111.query("Index_fidelity == 1")['Index_max'].to_numpy()
        index_high = np.concatenate([indices_high, index2])
        unique_count_low = len(np.unique(index_low))
        unique_count_high = len(np.unique(index_high))
        _, idx = np.unique(index_low, return_index=True)
        unique_values_low = index_low[np.sort(idx)]
        _, idx = np.unique(index_high, return_index=True)
        unique_values_high = index_high[np.sort(idx)]
        time111 = np.sum(Time_low[unique_values_low])
        time222 = np.sum(Time_high[unique_values_high])
        Simulation_low_AMFGPR[j, k] = time111
        Simulation_high_AMFGPR[j, k] = time222
        result_low.append(unique_values_low)
        result_high.append(unique_values_high)
    Unique_low_AMFGPR[j] = result_low
    Unique_high_AMFGPR[j] = result_high

Unique_low_ANMFGPR = [None for _ in range(3)]
Unique_high_ANMFGPR = [None for _ in range(3)]
Simulation_low_ANMFGPR = np.zeros((3,len(stop0)), dtype=int)
Simulation_high_ANMFGPR = np.zeros((3,len(stop0)), dtype=int)
for j in range(3):
    random_seeds = random_seeds_train[j]
    np.random.seed(random_seeds)
    N = 1000
    indices_low = np.random.choice(N, size=N_low0, replace=False)
    indices_high = np.random.choice(N, size=N_high0, replace=False)
    result_low = []
    result_high = []
    for k in range(len(stop0)):
        Results_ONERAM6_list111 = Results_ONERAM6_list_ANMFGPR[j][:stop[j, k]]
        index1 = Results_ONERAM6_list111.query("Index_fidelity == 0")['Index_max'].to_numpy()
        index_low = np.concatenate([indices_low, index1])
        index2 = Results_ONERAM6_list111.query("Index_fidelity == 1")['Index_max'].to_numpy()
        index_high = np.concatenate([indices_high, index2])
        unique_count_low = len(np.unique(index_low))
        unique_count_high = len(np.unique(index_high))
        _, idx = np.unique(index_low, return_index=True)
        unique_values_low = index_low[np.sort(idx)]
        _, idx = np.unique(index_high, return_index=True)
        unique_values_high = index_high[np.sort(idx)]
        time111 = np.sum(Time_low[unique_values_low])
        time222 = np.sum(Time_high[unique_values_high])
        Simulation_low_ANMFGPR[j, k] = time111
        Simulation_high_ANMFGPR[j, k] = time222
        result_low.append(unique_values_low)
        result_high.append(unique_values_high)
    Unique_low_ANMFGPR[j] = result_low
    Unique_high_ANMFGPR[j] = result_high

Unique_low = [None for _ in range(3)]
Unique_high = [None for _ in range(3)]
Simulation_low = np.zeros((3,len(stop0)), dtype=int)
Simulation_high = np.zeros((3,len(stop0)), dtype=int)
for j in range(3):
    result_low = []
    result_high = []
    for k in range(len(stop0)):
        random_seeds = random_seeds_train[j]
        np.random.seed(random_seeds)
        N_low = N_low_list[j, k]
        N_high = N_high_list[j, k]
        N = 1000
        indices_low = np.random.choice(N, size=N_low, replace=False)
        indices_high = np.random.choice(N, size=N_high, replace=False)
        time111 = np.sum(Time_low[indices_low])
        time222 = np.sum(Time_high[indices_high])
        Simulation_low[j, k] = time111
        Simulation_high[j, k] = time222
        result_low.append(indices_low)
        result_high.append(indices_high)
    Unique_low[j] = result_low
    Unique_high[j] = result_high



metric_index = []
for k in range(3):
    for kk in range(len(stop0)):
        Time_AMFTabPFN = np.sum(Results_ONERAM6_list[k].loc[0:stop[k, kk], ["Time_Train", "Time_Pred"]].to_numpy().astype(float)) + Simulation_low_AMFTabPFN[k, kk] + Simulation_high_AMFTabPFN[k, kk]
        Time_MFTabPFN = np.sum(Results_ONERAM6_list_ML[k][kk].query("Model == 'MFTabPFN'")[["Time_Train", "Time_Pred"]].iloc[0].values) + Simulation_low[k, kk] + Simulation_high[k, kk]
        Time_TabPFN_H = np.sum(Results_ONERAM6_list_ML[k][kk].query("Model == 'TabPFN-High'")[["Time_Train", "Time_Pred"]].iloc[0].values) + Simulation_low[k, kk] + Simulation_high[k, kk]
        Time_TabPFN_M = np.sum(Results_ONERAM6_list_ML[k][kk].query("Model == 'TabPFN-Multi'")[["Time_Train", "Time_Pred"]].iloc[0].values) + Simulation_low[k, kk] + Simulation_high[k, kk]
        Time_AutoGluon_H = np.sum(Results_ONERAM6_list_ML[k][kk].query("Model == 'AutoGluon-High'")[["Time_Train", "Time_Pred"]].iloc[0].values) + Simulation_low[k, kk] + Simulation_high[k, kk]
        Time_AutoGluon_M = np.sum(Results_ONERAM6_list_ML[k][kk].query("Model == 'AutoGluon-Multi'")[["Time_Train", "Time_Pred"]].iloc[0].values) + Simulation_low[k, kk] + Simulation_high[k, kk]
        Time_AMFGPR = np.sum(Results_ONERAM6_list_AMFGPR[k].loc[0:stop[k, kk], ["Time_Train", "Time_Pred"]].to_numpy().astype(float)) + Simulation_low_AMFGPR[k, kk] + Simulation_high_AMFGPR[k, kk]
        Time_ANMFGPR = np.sum(Results_ONERAM6_list_ANMFGPR[k].loc[0:stop[k, kk], ["Time_Train", "Time_Pred"]].to_numpy().astype(float)) + Simulation_low_ANMFGPR[k, kk] + Simulation_high_ANMFGPR[k, kk]
        Time_MFGPR = np.sum(Results_ONERAM6_list_MFGPR[k][kk].loc[0, ["Time_Train", "Time_Pred"]].to_numpy().astype(float)) + Simulation_low[k, kk] + Simulation_high[k, kk]
        Time_NMFGPR = np.sum(Results_ONERAM6_list_NMFGPR[k][kk].loc[0, ["Time_Train", "Time_Pred"]].to_numpy().astype(float)) + Simulation_low[k, kk] + Simulation_high[k, kk]
        Time_MAHNN = np.sum(Results_ONERAM6_list_MAHNN[k][kk].loc[0, ["Time_Train", "Time_Pred"]].to_numpy().astype(float)) + Simulation_low[k, kk] + Simulation_high[k, kk]
        Time_TLFNO = np.sum(Results_ONERAM6_list_TLFNO[k][kk].loc[0, ["Time_Train", "Time_Pred"]].to_numpy().astype(float)) + Simulation_low[k, kk] + Simulation_high[k, kk]

        Metric_AMFTabPFN = Time_AMFTabPFN
        Metric_MFTabPFN = Time_MFTabPFN
        Metric_TabPFN_H = Time_TabPFN_H
        Metric_TabPFN_M = Time_TabPFN_M
        Metric_AutoGluon_H = Time_AutoGluon_H
        Metric_AutoGluon_M = Time_AutoGluon_M
        Metric_AMFGPR = Time_AMFGPR
        Metric_ANMFGPR = Time_ANMFGPR
        Metric_MFGPR = Time_MFGPR
        Metric_NMFGPR = Time_NMFGPR
        Metric_MAHNN = Time_MAHNN
        Metric_TLFNO = Time_TLFNO

        model_metrics = [Metric_AMFTabPFN, Metric_MFTabPFN, Metric_TabPFN_H, Metric_TabPFN_M,
                        Metric_AutoGluon_H, Metric_AutoGluon_M, Metric_AMFGPR, Metric_ANMFGPR,
                        Metric_MFGPR, Metric_NMFGPR, Metric_MAHNN, Metric_TLFNO]

        for name, data_arr in zip(models, model_metrics):
            metric_index.append(
                {'Dataset': k, 'Fold': kk, 'Model': name, 'Time': data_arr})

METRIC_index = pd.DataFrame(metric_index)



Total_Time_Simulation = np.sum(Time_high)
time_columns = {
    'Total time': 'Time'
}
fig, ax = plt.subplots(1, 1, figsize=(3.4, 1.8))
order = METRIC_index.groupby('Model')['Time'].median().sort_values().index.tolist()
order = list(order) + ['Simulation']
datasets = []
for m in order:
    if m == 'Simulation':
        datasets.append([Total_Time_Simulation])
    else:
        datasets.append(METRIC_index[METRIC_index['Model'] == m]['Time'].dropna().values)
positions = np.arange(len(order)) + 1

violin_positions = positions[:-1]
violin_datasets = datasets[:-1]

parts = ax.violinplot(violin_datasets,
                      positions=violin_positions,
                      widths=0.85,
                      showmeans=False,
                      showextrema=False,
                      showmedians=False)

for pc, model_name in zip(parts['bodies'], order[:-1]):
    color = model_colors.get(model_name, '#808080')
    pc.set_facecolor(color)
    pc.set_edgecolor('none')
    pc.set_alpha(0.5)

for i, model_name in enumerate(order):
    color = model_colors.get(model_name, '#808080')
    y = datasets[i]

    if model_name == 'Simulation':
        ax.hlines(y=Total_Time_Simulation,
                  xmin=positions[i] - 0.2, xmax=positions[i] + 0.2,
                  colors='red',
                  linewidth=1.5,
                  label='Simulation (total time)')
    else:
        ax.boxplot([y], positions=[positions[i]], widths=0.4,
                   patch_artist=True,
                   boxprops=dict(facecolor=color, edgecolor=color, linewidth=0),
                   whiskerprops=dict(color=color, linewidth=1.5),
                   capprops=dict(color=color, linewidth=1.5),
                   medianprops=dict(color='black', linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor=color,
                                   markeredgecolor=color, markersize=1, alpha=1.0))

ax.set_title('Total time')
ax.set_xlabel('')
ax.set_ylabel('Time (seconds)')

ax.set_xticks(positions)
ax.set_xticklabels(order, rotation=60, ha='right', rotation_mode='anchor')

ax.tick_params(axis='x', which='minor', length=0)
ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True)

ax.set_yticks([0.00001, 50000, 100000, 150000, 200000])
ax.set_yticklabels([0, 50000, 100000, 150000, 200000])
ax.set_ylim(0, 200000)

order = 5
ax.yaxis.set_major_formatter(OOMFormatter(order=order, fformat="%.1f"))
ax.yaxis.offsetText.set_x(-0.03)

legend_elements = [
    Patch(facecolor='#1f77b4', alpha=0.5, edgecolor='none',
          label='Violin plot: data distribution'),
    Patch(facecolor='#1f77b4', edgecolor='none', alpha=1.0,
          label='Box: 25th–75th percentile (IQR)'),
    Line2D([0], [0], color='#1f77b4', lw=1.5, label='Upper and lower whiskers'),
    Line2D([0], [0], color='black', lw=1.5, label='Median time'),
    Line2D([0], [0], color='red', lw=1.5, label='Simulation time'),
]

leg1 = ax.legend(handles=legend_elements,
                 loc='upper left',
                 ncol=2,
                 frameon=True,
                 columnspacing=1.0,
                 handletextpad=0.5,
                 handlelength=0.8,
                 fontsize=6)

leg1.get_frame().set_linewidth(0.75)

plt.tight_layout(pad=0.01)
plt.subplots_adjust(left=0.05, right=0.97, top=0.85, bottom=0.30)

for ext in ['png', 'tiff', 'pdf']:
    output_path = os.path.join(SAVE_DIR, f'time_total_with_simulation.{ext}')
    plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)

plt.show()




metrics_info = [
    ('Time', 'Time')
]

all_stats_rows = []
order = METRIC_index.groupby('Model')['Time'].median().sort_values().index.tolist()
model_order = METRIC_index.groupby('Model')['Time'].median().sort_values().index.tolist()

for metric_col, metric_name in metrics_info:
    for model in model_order:
        data_series = METRIC_index[METRIC_index['Model'] == model][metric_col].dropna()
        n_samples = len(data_series)

        if n_samples == 0:
            all_stats_rows.append({
                'Metric': metric_name,
                'Model': model,
                'Median': None,
                'Q1': None,
                'Q3': None,
                'Lower_Whisker': None,
                'Upper_Whisker': None,
                'N': 0
            })
            continue
        stats = boxplot_stats(data_series)[0]

        all_stats_rows.append({
            'Metric': metric_name,
            'Model': model,
            'Median': round(stats['med'], 4),
            'Q1': round(stats['q1'], 4),
            'Q3': round(stats['q3'], 4),
            'Lower_Whisker': round(stats['whislo'], 4),
            'Upper_Whisker': round(stats['whishi'], 4),
            'N': n_samples
        })

stats_df = pd.DataFrame(all_stats_rows)

