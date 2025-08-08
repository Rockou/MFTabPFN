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
from MLP_MFTabPFN_S import MLP_S
from pathlib import Path
from TabPFN_model import TabPFN_model_main
import torch
if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')

def toy(x):
    y = x[:, 0] * np.sin(x[:, 0] * 5) * np.cos(x[:, 0]) + x[:, 0]
    return y

nx = 1
N0 = 80
N_test = 500
XX_train_yuan = np.linspace(0, 20, N0).reshape(-1, 1)
XX_total_yuan = np.linspace(0, 20, N_test).reshape(-1, 1)
YY_train_yuan = toy(XX_train_yuan).reshape(-1, 1)
YY_total_yuan = toy(XX_total_yuan).reshape(-1, 1)

scaler = StandardScaler()
xx_train = scaler.fit_transform(XX_train_yuan)
xx_total = scaler.transform(XX_total_yuan)
scaler_Y = StandardScaler()
yy_train = scaler_Y.fit_transform(YY_train_yuan.reshape(-1, 1))
Mu_scaler = scaler_Y.mean_
Sigma_scaler = np.sqrt(scaler_Y.var_)

INPUT = torch.tensor(xx_train, dtype=torch.float)
OUTPUT = torch.tensor(yy_train, dtype=torch.float)

input_TabPFN = np.min([np.max([20, nx]), 500])
n_layer = 3
hidden_channels = np.max([128, 2 * nx])
activate_function = 'tanh'  # 'tanh'; 'sigmoid'; 'relu'
lr = 0.001
weight_decay = 1e-3
epochs = 100
bili = 1.0
task_type = "regression"
batch = 256

##########################################################################TabPFN
if nx > 500:
    reg = TabPFNRegressor(n_estimators=8, random_state=42, ignore_pretraining_limits=True)
else:
    reg = TabPFNRegressor(n_estimators=8, random_state=42)
reg.fit(xx_train, yy_train.ravel())
YY_test_prediction_initial = reg.predict(xx_total, output_type="full")
Y_test_prediction_initial = YY_test_prediction_initial[f"mean"] * Sigma_scaler + Mu_scaler
rmse = 1 - root_mean_squared_error(YY_total_yuan, Y_test_prediction_initial) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
mae = 1 - mean_absolute_error(YY_total_yuan, Y_test_prediction_initial) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
r2 = r2_score(YY_total_yuan, Y_test_prediction_initial)
Y_test_prediction_TabPFN = Y_test_prediction_initial
##########################################################################MFTabPFN
model_MLP = MLP_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay, epochs, hidden_channels, activate_function, batch)
model_MLP.eval()
YY_train_middle = model_MLP(torch.tensor(xx_train, dtype=torch.float))
YY_test_middle = model_MLP(torch.tensor(xx_total, dtype=torch.float))

if nx > 500:
    reg_initial = TabPFNRegressor(n_estimators=8, random_state=42, ignore_pretraining_limits=True)
else:
    reg_initial = TabPFNRegressor(n_estimators=8, random_state=42)
reg_initial.fit(INPUT.detach().numpy(), OUTPUT.detach().numpy().ravel())
TabPFN_prediction_initial_full = reg_initial.predict(xx_total, output_type="full")
TabPFN_prediction_initial = TabPFN_prediction_initial_full[f"mean"].reshape(-1, 1)
TabPFN_prediction_tensor_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)

if nx > 500:
    reg_initial0 = TabPFNRegressor(n_estimators=8, random_state=42, ignore_pretraining_limits=True)
else:
    reg_initial0 = TabPFNRegressor(n_estimators=8, random_state=42)
reg_initial0.fit(INPUT.detach().numpy(), OUTPUT.detach().numpy().ravel())
TabPFN_prediction_initial0 = reg_initial0.predict(INPUT.detach().numpy()).reshape(-1, 1)
TabPFN_prediction_tensor_initial0 = torch.tensor(TabPFN_prediction_initial0, dtype=torch.float)

save_path_to_fine_tuned_model = (Path(__file__).parent / f"tabpfn_model_{task_type}.ckpt")
TabPFN_prediction_tensor, Lower, Upper, Variance = TabPFN_model_main(
    path_to_base_model="auto",
    save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
    X_train=YY_train_middle,
    y_train=OUTPUT - TabPFN_prediction_tensor_initial0.to('cpu'),
    X_test=YY_test_middle,
    n_classes=None,
    categorical_features_index=None,
    task_type=task_type,
    # device="cuda" if torch.cuda.is_available() else "cpu",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)
Y_test_prediction = (TabPFN_prediction_tensor.detach().cpu().numpy() + TabPFN_prediction_tensor_initial.detach().cpu().numpy()) * Sigma_scaler + Mu_scaler
rmse = 1 - root_mean_squared_error(YY_total_yuan, Y_test_prediction) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
mae = 1 - mean_absolute_error(YY_total_yuan, Y_test_prediction) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
r2 = r2_score(YY_total_yuan, Y_test_prediction)

plt.figure(1)
plt.plot(XX_total_yuan, Y_test_prediction, linestyle='-', color='b', linewidth=2, label='MFTabPFN')
plt.plot(XX_total_yuan, Y_test_prediction_TabPFN, linestyle='--', color='g', linewidth=2, label='TabPFN')
plt.plot(XX_total_yuan, YY_total_yuan, linestyle=':', color='k', linewidth=2, label='Reference')
plt.scatter(XX_train_yuan, YY_train_yuan, color='r', s=50, alpha=0.6, label='Training points')
plt.xlabel('Input', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show(block=True)
