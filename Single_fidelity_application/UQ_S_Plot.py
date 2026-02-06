import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import scipy.stats as stats
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from pathlib import Path

SAVE_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "Single-fidelity" / "Synthetic" / "UQ"

pkl_file = os.path.join(SAVE_DIR, 'YY_real_prediction.pkl')
with open(pkl_file, 'rb') as f:
    YY_real_prediction = pickle.load(f)
pkl_file = os.path.join(SAVE_DIR, 'YY_index.pkl')
with open(pkl_file, 'rb') as f:
    YY_index = pickle.load(f)


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

n_performances = len(YY_real_prediction)
n_simulations = len(YY_real_prediction[0])
JS_results = [[None for _ in range(n_simulations)] for _ in range(n_performances)]
Bandwidth = [[None for _ in range(n_simulations)] for _ in range(n_performances)]
NLL_results  = [[None for _ in range(n_simulations)] for _ in range(n_performances)]
CRPS_results = [[None for _ in range(n_simulations)] for _ in range(n_performances)]
WD_results = [[None for _ in range(n_simulations)] for _ in range(n_performances)]
PDF_dict = [[None for _ in range(n_simulations)] for _ in range(n_performances)]
CDF_dict = [[None for _ in range(n_simulations)] for _ in range(n_performances)]
X_grid = [[None for _ in range(n_simulations)] for _ in range(n_performances)]
ALL_data = [[None for _ in range(n_simulations)] for _ in range(n_performances)]
for j in range(0, n_performances):
    for k in range(n_simulations):
        YY_total_yuan         = np.ravel(YY_real_prediction[j][k].loc[0, 'Performance'])
        Y_test_prediction_MFTabPFN = np.ravel(YY_real_prediction[j][k].loc[2, 'Performance'])
        Y_test_prediction_TabPFN   = np.ravel(YY_real_prediction[j][k].loc[1, 'Performance'])
        Y_test_prediction_AutoGluon = np.ravel(YY_real_prediction[j][k].loc[3, 'Performance'])

        all_data = np.concatenate([YY_total_yuan, Y_test_prediction_MFTabPFN, Y_test_prediction_TabPFN, Y_test_prediction_AutoGluon])
        x_grid = np.linspace(all_data.min() - 0.05 * np.ptp(all_data), all_data.max() + 0.05 * np.ptp(all_data),5000)

        kde_ref = gaussian_kde(YY_total_yuan)
        pdf_ref = kde_ref(x_grid)
        pdf_ref = pdf_ref / np.trapezoid(pdf_ref, x_grid)
        cdf_ref = cumulative_trapezoid(pdf_ref, x_grid, initial=0)
        cdf_ref = cdf_ref / cdf_ref[-1]

        covariance_matrix = kde_ref.covariance
        variance = covariance_matrix[0, 0]
        reference_bandwidth = np.sqrt(variance)
        model_arrays = [Y_test_prediction_MFTabPFN, Y_test_prediction_TabPFN, Y_test_prediction_AutoGluon]

        pdf_dict = {'Reference': (x_grid, pdf_ref)}
        cdf_dict = {'Reference': (x_grid, cdf_ref)}
        bandwidth = {'Reference': reference_bandwidth}
        js_results_this = []
        nll_results_this = []
        crps_results_this = []
        wd_results_this = []
        for name, data_arr in zip(models, model_arrays):
            if len(data_arr) == 0:
                continue
            kde = gaussian_kde(data_arr)
            pdf_model = kde(x_grid)
            pdf_model = pdf_model / np.trapezoid(pdf_model, x_grid)
            cdf_model = cumulative_trapezoid(pdf_model, x_grid, initial=0)
            cdf_model = cdf_model / cdf_model[-1]
            pdf_dict[name] = (x_grid, pdf_model)
            cdf_dict[name] = (x_grid, cdf_model)
            bandwidth[name] = np.sqrt(kde.covariance[0, 0])

            # ───────────────────1. JS ───────────────────
            epsilon = 1e-12
            P = pdf_ref + epsilon
            Q = pdf_model + epsilon
            M = 0.5 * (P + Q)
            def js_divergence(p, q, m, x_grid):
                kl_pm = np.sum(p * np.log(p / m)) * (x_grid[1] - x_grid[0])
                kl_qm = np.sum(q * np.log(q / m)) * (x_grid[1] - x_grid[0])
                return 0.5 * (kl_pm + kl_qm)
            js_div = js_divergence(P, Q, M, x_grid)

            # ─────────────────── 2. NLL ───────────────────
            log_pdf_at_truth = kde.logpdf(YY_total_yuan)
            nll = -log_pdf_at_truth.mean()

            # ─────────────────── 3. CRPS ───────────────────
            cdf_interp = interp1d(x_grid, cdf_model, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))
            def crps_one_sample(y):
                x = x_grid
                F = cdf_interp(x)
                Heaviside = (x >= y).astype(float)
                integrand = (F - Heaviside) ** 2
                return np.trapezoid(integrand, x)

            crps = np.mean([crps_one_sample(y) for y in YY_total_yuan])

            # ─────────────────── 4. Wasserstein-1 ───────────────────

            cdf_ref_interp = interp1d(x_grid, cdf_ref, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))
            F_interp = cdf_interp(x_grid)
            F_ref = cdf_ref_interp(x_grid)
            integrand = np.abs(F_interp - F_ref)
            wd1 = np.trapezoid(integrand, x_grid)

            print(f"Case n={(j + 1) * 100} | Fold {k + 1:02d} | {name:10s} | JS = {js_div:.5f} | NLL = {nll:.5f} | CRPS = {crps:.5f} | WD = {wd1:.5f}")
            js_results_this.append({'Dataset': j + 1, 'Fold': k + 1, 'Model': name, 'JS_Divergence': float(js_div)})
            nll_results_this.append({'Dataset': j + 1, 'Fold': k + 1, 'Model': name, 'NLL': float(nll)})
            crps_results_this.append({'Dataset': j + 1, 'Fold': k + 1, 'Model': name, 'CRPS': float(crps)})
            wd_results_this.append({'Dataset': j + 1, 'Fold': k + 1, 'Model': name, 'WD': float(wd1)})

        JS_results[j][k] = js_results_this
        NLL_results[j][k] = nll_results_this
        CRPS_results[j][k] = crps_results_this
        WD_results[j][k] = wd_results_this
        Bandwidth[j][k] = bandwidth
        PDF_dict[j][k] = pdf_dict
        CDF_dict[j][k] = cdf_dict
        X_grid[j][k] = x_grid
        ALL_data[j][k] = all_data


def flatten_results(results_2d):
    flat = []
    for j in range(n_performances):
        for k in range(n_simulations):
            flat.extend(results_2d[j][k])
    return pd.DataFrame(flat)
df_js   = flatten_results(JS_results)
df_nll  = flatten_results(NLL_results)
df_crps = flatten_results(CRPS_results)
df_wd = flatten_results(WD_results)
df_all = df_js.merge(df_nll, on=['Dataset','Fold','Model']).merge(df_crps, on=['Dataset','Fold','Model']).merge(df_wd, on=['Dataset','Fold','Model'])

confidence = 0.95
alpha = 1 - confidence
metrics = ['JS_Divergence', 'NLL', 'CRPS', 'WD']
grouped = df_all.groupby(['Dataset', 'Model'])[metrics].agg(['mean', 'std', 'count']).reset_index()
grouped.columns = ['_'.join(col).strip() if col[1] != '' else col[0] for col in grouped.columns.values]
for m in metrics:
    grouped[f'{m}_se'] = grouped[f'{m}_std'] / np.sqrt(grouped[f'{m}_count'])
    grouped[f'{m}_ci'] = grouped[f'{m}_se'] * stats.t.ppf(1 - alpha/2, grouped[f'{m}_count'] - 1)
print("\n=== Final Summary ===")
print(grouped)

metric_labels = {
    'JS_Divergence': 'JS divergence (95% CI)',
    'NLL': 'NLL (95% CI)',
    'CRPS': 'CRPS (95% CI)',
    'WD': 'Wasserstein-1 (95% CI)'
}
metric_labels_title = {
    'JS_Divergence': 'JS divergence',
    'NLL': 'NLL',
    'CRPS': 'CRPS',
    'WD': 'Wasserstein-1'
}

datasets = sorted(grouped['Dataset'].unique())
x_pos = np.arange(len(datasets))
dataset_labels = [f'{d * 100}' for d in datasets]

plt.rcParams['xtick.major.pad'] = 1.5
plt.rcParams['ytick.major.pad'] = 0.8

metrics = ['JS_Divergence', 'WD']

fig, axes = plt.subplots(1, 2, figsize=(6.8, 1.65), sharex=True)

for col, metric in enumerate(metrics):
    ax = axes[col]

    for i, model in enumerate(models):
        model_data = grouped[grouped['Model'] == model]
        model_data = model_data.set_index('Dataset').loc[datasets].reset_index()

        means = model_data[f'{metric}_mean'].values
        cis = model_data[f'{metric}_ci'].values

        label = model_labels[i] if col == 0 else None

        ax.plot(x_pos, means, label=label,
                color=model_colors[model], linewidth=1.5,
                marker=model_markers[model], markersize=3)

        ax.fill_between(x_pos, means - cis, means + cis,
                        color=model_colors[model], alpha=0.2, edgecolor='none')

    ax.set_yscale('log')
    ax.set_title(metric_labels_title[metric])
    ax.set_ylabel(metric_labels[metric])
    ax.set_xlabel('PF (feature dimension n)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dataset_labels)
    ax.set_xlim(-0.15, 4.15)
    ax.grid(True, axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
    if metric == 'JS_Divergence':
        ax.set_yticks([0.0001, 0.001, 0.01, 0.1, 1])
        ax.set_yticklabels([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
        ax.set_ylim(0.0001, 1)
    elif metric == 'WD':
        ax.set_yticks([0.1, 1, 10, 100, 1000])
        ax.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$'])
        ax.set_ylim(0.1/2, 10000/5)

    if metric == 'JS_Divergence':
        ax.legend(loc='lower center', ncol=1, bbox_to_anchor=(0.5, -0.05),
                  columnspacing=1.0,
                  labelspacing=0.15,
                  handletextpad=0.4,
                  handlelength=1.2,
                  frameon=False,
                  fontsize=6)

plt.tight_layout(pad=0.01)
plt.subplots_adjust(wspace=0.20)

for ext in ['png', 'tiff', 'pdf']:
    output_path = os.path.join(SAVE_DIR, f'UQ_metrics_JS_WD.{ext}')
    plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)

plt.show()
plt.close(fig)




x_formatter = FormatStrFormatter('%.0f')
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


for k in range(n_simulations):
    fig = plt.figure(figsize=(6.8, 4.0))
    gs = fig.add_gridspec(3, 5)

    plt.rcParams['xtick.major.pad'] = 1.5
    plt.rcParams['ytick.major.pad'] = 0.8

    for j in range(n_performances):
        YY_total_yuan = np.ravel(YY_real_prediction[j][k].loc[0, 'Performance'])
        Y_pred_MFTabPFN = np.ravel(YY_real_prediction[j][k].loc[2, 'Performance'])
        Y_pred_TabPFN = np.ravel(YY_real_prediction[j][k].loc[1, 'Performance'])
        Y_pred_AutoGluon = np.ravel(YY_real_prediction[j][k].loc[3, 'Performance'])

        all_data = np.concatenate([YY_total_yuan, Y_pred_MFTabPFN, Y_pred_TabPFN, Y_pred_AutoGluon])
        x_min, x_max = all_data.min(), all_data.max()
        x_range = x_max - x_min if x_max > x_min else 1.0

        pdf_dict = PDF_dict[j][k]
        cdf_dict = CDF_dict[j][k]

        # ================== PDF ==================
        ax_pdf = fig.add_subplot(gs[0, j])
        bins = min(30, max(10, len(YY_total_yuan) // 10))

        for model in models:
            x_pdf, y_pdf = pdf_dict[model]
            if model == 'MFTabPFN':
                zorder = 3
                linestyle = '-'
            elif model == 'TabPFN':
                zorder = 2
                linestyle = '-'
            elif model == 'AutoGluon':
                zorder = 1
                linestyle = '-'
            ax_pdf.plot(x_pdf, y_pdf, color=model_colors[model], linewidth=1.5, linestyle=linestyle,
                        label=model if j == 0 else None, zorder=zorder)

        ax_pdf.hist(YY_total_yuan, bins=bins, density=True, alpha=0.4, color='#AED6F1',
                    edgecolor='black', label='Reference' if j == 0 else None, zorder=0)

        ax_pdf.set_title(f'PF (n={(j + 1) * 100})')
        ax_pdf.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
        ax_pdf.set_xticks([x_min, x_max])
        ax_pdf.xaxis.set_major_formatter(x_formatter)
        y_min, y_max = ax_pdf.get_ylim()
        ax_pdf.set_yticks([0, y_max])
        ax_pdf.set_yticklabels(['0', f'{y_max:.1f}'])

        if j == 2:
            order = -4
        elif j == 4:
            order = -2
        else:
            order = -1
        ax_pdf.yaxis.set_major_formatter(OOMFormatter(order=order, fformat="%.1f"))

        ax_pdf.yaxis.offsetText.set_x(-0.12)

        if j == 0:
            ax_pdf.text(-0.10, 0.5, 'PDF', transform=ax_pdf.transAxes, ha='center', va='center', rotation=90, fontsize=7)
        ax_pdf.text(0.5, -0.04, 'Value', transform=ax_pdf.transAxes, ha='center', va='top', fontsize=7)

        # ================== CDF ==================
        ax_cdf = fig.add_subplot(gs[1, j])
        for model in models:
            x_cdf, y_cdf = cdf_dict[model]
            if model == 'MFTabPFN':
                zorder = 3
                linestyle = '-'
            elif model == 'TabPFN':
                zorder = 2
                linestyle = '-'
            elif model == 'AutoGluon':
                zorder = 1
                linestyle = '-'
            ax_cdf.plot(x_cdf, y_cdf, color=model_colors[model], linewidth=1.5, linestyle=linestyle,
                        label=model if j == 0 else None, zorder=zorder)

        x_ref, y_ref = cdf_dict['Reference']
        ax_cdf.plot(x_ref, y_ref, color='black', linewidth=1.5, alpha=1.0, linestyle='-',
                    label='Reference' if j == 0 else None, zorder=0)

        ax_cdf.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
        ax_cdf.set_xticks([x_min, x_max])
        ax_cdf.xaxis.set_major_formatter(x_formatter)

        ax_cdf.set_ylim(-0.02, 1.02)
        ax_cdf.set_yticks([0.0, 1.0])
        if j == 0:
            ax_cdf.text(-0.10, 0.5, 'CDF', transform=ax_cdf.transAxes, ha='center', va='center', rotation=90, fontsize=7)
        ax_cdf.text(0.5, -0.04, 'Value', transform=ax_cdf.transAxes, ha='center', va='top', fontsize=7)

        # ================== Prediction ==================
        ax_scatter = fig.add_subplot(gs[2, j])
        norm = lambda arr: (arr - x_min) / x_range

        ax_scatter.plot([0, 1], [0, 1], '--', color='gray', lw=1.0, zorder=0)
        ax_scatter.scatter(norm(YY_total_yuan), norm(Y_pred_MFTabPFN),
                           color=model_colors['MFTabPFN'], s=2, alpha=1.0, marker=model_markers['MFTabPFN'],
                           label='MFTabPFN' if j == 0 else None, zorder=3)
        ax_scatter.scatter(norm(YY_total_yuan), norm(Y_pred_TabPFN),
                           color=model_colors['TabPFN'], s=2, alpha=1.0, marker=model_markers['TabPFN'], label='TabPFN' if j == 0 else None, zorder=2)
        ax_scatter.scatter(norm(YY_total_yuan), norm(Y_pred_AutoGluon),
                           color=model_colors['AutoGluon'], s=2, alpha=1.0, marker=model_markers['AutoGluon'],
                           label='AutoGluon' if j == 0 else None, zorder=1)

        ax_scatter.set_xlim(-0.02, 1.02)
        ax_scatter.set_ylim(-0.02, 1.02)
        ax_scatter.set_xticks([0.0, 1.0])
        ax_scatter.set_yticks([0.0, 1.0])
        if j == 0:
            ax_scatter.text(-0.10, 0.5, 'Prediction', transform=ax_scatter.transAxes, ha='center', va='center', rotation=90, fontsize=7)

        ax_scatter.text(0.5, -0.04, 'Reference', transform=ax_scatter.transAxes, ha='center', va='top', fontsize=7)

        if j == 0:
            ax_pdf.legend(bbox_to_anchor=(0.74, 1.04), loc='upper center', ncol=1,
                          columnspacing=1.0,
                          labelspacing=0.15,
                          handletextpad=0.4,
                          handlelength=0.7,
                          frameon=False,
                          fontsize=6)
            ax_cdf.legend(bbox_to_anchor=(0.74, 0.47), loc='upper center', ncol=1,
                          columnspacing=1.0,
                          labelspacing=0.15,
                          handletextpad=0.4,
                          handlelength=0.7,
                          frameon=False,
                          fontsize=6)
            ax_scatter.legend(bbox_to_anchor=(0.28, 1.04), loc='upper center', ncol=1,
                              columnspacing=1.0,
                              labelspacing=0.15,
                              handletextpad=0.4,
                              handlelength=0.7,
                              frameon=False,
                              fontsize=6)

    plt.tight_layout(pad=0.01)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.20, hspace=0.25)

    for ext in ['png', 'tiff', 'pdf']:
            output_path = os.path.join(SAVE_DIR, f'fold_{k+1}_comparison.{ext}')
            plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)

    plt.show()

