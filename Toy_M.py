from tabpfn import TabPFNRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from MLP_MFTabPFN_M import MLP_M
from pathlib import Path
from TabPFN_model import TabPFN_model_main
import torch
if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')

def f_low(x):
    ff = x[:, 0] ** 2 * np.cos(x[:, 0] * 20)
    return ff
def f_high(x):
    ff = f_low(x) + x[:, 0] ** 3
    return ff

N_low = 40
N_high = 20
N_low0 = N_low
N_high0 = N_high
N = 200
nx = 1

XX_train_low = np.linspace(0, 1, N_low).reshape(-1, 1)
XX_train_high = np.linspace(0, 1, N_high).reshape(-1, 1)
XX_total = np.linspace(0, 1, N).reshape(-1, 1)
XX_train_yuan_low = XX_train_low
XX_train_yuan_high = XX_train_high
XX_total_yuan = XX_total
YY_train_yuan_low = f_low(XX_train_yuan_low)
YY_train_yuan_high = f_high(XX_train_yuan_high)
YY_total_yuan_low = f_low(XX_total_yuan)
YY_total_yuan_high = f_high(XX_total_yuan)

XX_train_yuan = np.concatenate((XX_train_yuan_low, XX_train_yuan_high), axis=0)
YY_train_yuan = np.concatenate((YY_train_yuan_low, YY_train_yuan_high), axis=0)

scaler = StandardScaler()
xx_train = scaler.fit_transform(XX_train_yuan)
xx_total = scaler.transform(XX_total_yuan)
scaler_Y = StandardScaler()
yy_train = scaler_Y.fit_transform(YY_train_yuan.reshape(-1, 1))
yy_total_high = scaler_Y.transform(YY_total_yuan_high.reshape(-1, 1))
yy_total_low = scaler_Y.transform(YY_total_yuan_low.reshape(-1, 1))
Mu_scaler = scaler_Y.mean_
Sigma_scaler = np.sqrt(scaler_Y.var_)

INPUT = torch.tensor(xx_train, dtype=torch.float)
OUTPUT = torch.tensor(yy_train, dtype=torch.float)

input_TabPFN = np.min([np.max([20, nx]), 500])
hidden_channels = np.max([128, 2 * nx])
n_layer = 3
activate_function = 'tanh'# 'tanh'; 'sigmoid'; 'relu'
lr = 0.001
weight_decay = 1e-3
epochs = 100
bili = 1.0
task_type = "regression"
batch = 256

##########################################################################Active MFTabPFN
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

model_MLP = MLP_M(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay, epochs, hidden_channels, activate_function, batch, N_low, N_high)
model_MLP.eval()
YY_train_middle = model_MLP(torch.tensor(xx_train[N_low:, :], dtype=torch.float))
YY_test_middle = model_MLP(torch.tensor(xx_total, dtype=torch.float))

if nx > 500:
    reg_initial = TabPFNRegressor(n_estimators=8, random_state=42, ignore_pretraining_limits=True)
else:
    reg_initial = TabPFNRegressor(n_estimators=8, random_state=42)
reg_initial.fit(xx_train[:N_low, :], yy_train[:N_low].ravel())
TabPFN_prediction_initial_full = reg_initial.predict(xx_total, output_type="full")
TabPFN_prediction_initial = TabPFN_prediction_initial_full[f"mean"].reshape(-1, 1)
TabPFN_prediction_initial_yuan = TabPFN_prediction_initial * Sigma_scaler + Mu_scaler
TabPFN_prediction_tensor_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
Variance_initial = TabPFN_prediction_initial_full[f"variance"].reshape(-1, 1)
Prediction_sigma_initial = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)
lower_sigma_initial = TabPFN_prediction_initial_yuan.ravel() - 1.0 * Prediction_sigma_initial
upper_sigma_initial = TabPFN_prediction_initial_yuan.ravel() + 1.0 * Prediction_sigma_initial
if nx > 500:
    reg_initial0 = TabPFNRegressor(n_estimators=8, random_state=42, ignore_pretraining_limits=True)
else:
    reg_initial0 = TabPFNRegressor(n_estimators=8, random_state=42)
reg_initial0.fit(xx_train[:N_low, :], yy_train[:N_low].ravel())
TabPFN_prediction_initial0 = reg_initial0.predict(xx_train[N_low:, :]).reshape(-1, 1)
TabPFN_prediction_tensor_initial0 = torch.tensor(TabPFN_prediction_initial0, dtype=torch.float)
save_path_to_fine_tuned_model = (Path(__file__).parent / f"tabpfn_model_{task_type}.ckpt")
TabPFN_prediction_tensor, Lower, Upper, Variance = TabPFN_model_main(
    path_to_base_model="auto",
    save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
    X_train=YY_train_middle,
    y_train=OUTPUT[N_low:] - model_MLP.afa * TabPFN_prediction_tensor_initial0.to('cpu'),
    X_test=YY_test_middle,
    n_classes=None,
    categorical_features_index=None,
    task_type=task_type,
    # device="cuda" if torch.cuda.is_available() else "cpu",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)
Y_test_prediction_delt = TabPFN_prediction_tensor.detach().cpu().numpy() * Sigma_scaler
Y_test_prediction_low_afa = model_MLP.afa.detach().cpu().numpy() * TabPFN_prediction_tensor_initial.detach().cpu().numpy() * Sigma_scaler + Mu_scaler
Y_test_prediction = Y_test_prediction_delt + Y_test_prediction_low_afa
Prediction_sigma = np.sqrt(Variance.detach().cpu().numpy().ravel() * Sigma_scaler ** 2)
lower_sigma_delt = Y_test_prediction_delt.ravel() - 1.0 * Prediction_sigma
upper_sigma_delt = Y_test_prediction_delt.ravel() + 1.0 * Prediction_sigma
Prediction_total_sigma = np.sqrt(model_MLP.afa.detach().cpu().numpy() ** 2 * Prediction_sigma_initial ** 2 + Prediction_sigma ** 2)
lower_sigma = Y_test_prediction.ravel() - 1.0 * Prediction_total_sigma
upper_sigma = Y_test_prediction.ravel() + 1.0 * Prediction_total_sigma
rmse = 1 - root_mean_squared_error(YY_total_yuan_high, Y_test_prediction) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
mae = 1 - mean_absolute_error(YY_total_yuan_high, Y_test_prediction) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
r2 = r2_score(YY_total_yuan_high, Y_test_prediction)

fig, ax = plt.subplots(figsize=(4, 3), sharex=True)
ax.plot(XX_total_yuan, YY_total_yuan_low, linestyle='--', color=model_colors['TabPFN'],
        label='Reference low-fidelity solution', zorder=1)
ax.plot(XX_total_yuan, TabPFN_prediction_initial_yuan, linestyle='-', color=model_colors['TabPFN'],
        label='Predicted low-fidelity solution', zorder=1)
ax.fill_between(XX_total_yuan.ravel(), lower_sigma_initial, upper_sigma_initial, color=model_colors['TabPFN'],
                edgecolor='none', alpha=0.4, zorder=1)
ax.plot(XX_total_yuan, YY_total_yuan_high, linestyle='--', color=model_colors['MFTabPFN'],
        label='Reference high-fidelity solution', zorder=2)
ax.plot(XX_total_yuan, Y_test_prediction, linestyle='-', color=model_colors['MFTabPFN'],
        label='Predicted high-fidelity solution', zorder=2)
ax.fill_between(XX_total_yuan.ravel(), lower_sigma, upper_sigma, color=model_colors['MFTabPFN'], edgecolor='none',
                alpha=0.4, zorder=2)
ax.scatter(XX_train_yuan_high, YY_train_yuan_high, color=model_colors['MFTabPFN'],
           marker=model_markers['MFTabPFN'], s=35, label="High-fidelity training points", zorder=2)
ax.scatter(XX_train_yuan_low, YY_train_yuan_low, color=model_colors['TabPFN'],
           marker=model_markers['TabPFN'], s=35, label="Low-fidelity training points", zorder=1)
ax.set_xlim(-0.025, 1.025)
y_data = np.concatenate([
    YY_total_yuan_low, TabPFN_prediction_initial_yuan.ravel(), lower_sigma_initial, upper_sigma_initial,
    YY_total_yuan_high, Y_test_prediction.ravel(), lower_sigma, upper_sigma,
])
y_min, y_max = np.min(y_data), np.max(y_data)
ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
y_ticks = np.linspace(y_min, y_max, 5)
ax.set_yticks(y_ticks)
ax.set_xlabel('Input', fontsize=10)
ax.set_ylabel('Output', fontsize=10)
ax.xaxis.grid(True)
ax.yaxis.grid(True)
handles, labels = [], []
h, l = ax.get_legend_handles_labels()
handles.extend(h)
labels.extend(l)
plt.legend()
plt.tight_layout()
plt.subplots_adjust(right=0.9)
plt.show(block=True)
plt.close(fig)
