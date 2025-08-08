from tabpfn import TabPFNRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import scipy.stats as stats
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from RCNN_MFTabPFN_M import RCNN_M
from pathlib import Path
from TabPFN_model import TabPFN_model_main
from autogluon.tabular import TabularPredictor
import pickle
import torch
if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')

def UtoX(u):
    x = np.zeros_like(u)
    for i in range(len(mu)):
        if DisType[i] == 'normal':
            x[:, i] = u[:, i] * sigma[i] + mu[i]
        elif DisType[i] == 'uniform':
            a = mu[i] - np.sqrt(3) * sigma[i]
            b = mu[i] + np.sqrt(3) * sigma[i]
            x[:, i] = (b - a) * stats.norm.cdf(u[:, i]) + a
    return x
def XtoU(x):
    u = np.zeros_like(x)
    for i in range(len(mu)):
        if DisType[i] == 'normal':
            u[:, i] = (x[:, i] - mu[i]) / sigma[i]
        elif DisType[i] == 'uniform':
            a = mu[i] - np.sqrt(3) * sigma[i]
            b = mu[i] + np.sqrt(3) * sigma[i]
            u[:, i] = stats.norm.ppf((x[:, i] - a) / (b - a))
    return u

WORK_DIR = "./Datasets/ONERA_M6/Drag"

pkl_file = os.path.join(WORK_DIR, 'XX_total_yuan.pkl')
with open(pkl_file, 'rb') as f:
    XX_total_yuan = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'CD_total_high.pkl')
with open(pkl_file, 'rb') as f:
    CD_total_high = pickle.load(f)
pkl_file = os.path.join(WORK_DIR, 'CD_total_low.pkl')
with open(pkl_file, 'rb') as f:
    CD_total_low = pickle.load(f)

mu = np.full(198, 0)
sigma = np.full(198, 0.04 / np.sqrt(12))
DisType = ['uniform'] * 198
nx = mu.shape[0]
N_low = 50
N_high = 20
N = 1000
N_low0 = N_low
N_high0 = N_high

np.random.seed(123)
indices_low = np.random.choice(N, size=N_low, replace=False)
indices_high = np.random.choice(N, size=N_high, replace=False)
XX_train_yuan_low = XX_total_yuan[indices_low, :]
XX_train_yuan_high = XX_total_yuan[indices_high, :]

YY_train_yuan_low = CD_total_low[indices_low, :].ravel()
YY_train_yuan_high = CD_total_high[indices_high, :].ravel()
YY_total_yuan_low = CD_total_low[:N, :].ravel()
YY_total_yuan_high = CD_total_high[:N, :].ravel()

XX_train_yuan = np.concatenate((XX_train_yuan_low, XX_train_yuan_high), axis=0)
YY_train_yuan = np.concatenate((YY_train_yuan_low, YY_train_yuan_high), axis=0)

scaler = StandardScaler()
xx_train = scaler.fit_transform(XX_train_yuan)
xx_total = scaler.transform(XX_total_yuan[:N, :])
scaler_Y = StandardScaler()#
yy_train = scaler_Y.fit_transform(YY_train_yuan.reshape(-1, 1))
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
U = 300
Cost_low = 1
Cost_high = 3
Stop = np.zeros((U, 1), dtype=float)
Stop0 = 0.025
Index_fidelity = np.zeros((U, 1), dtype=float)
Uncertainty_max = np.zeros((U, 1), dtype=float)
Index_max_low = np.zeros((U, 1), dtype=float)
Index_max_high = np.zeros((U, 1), dtype=float)
RMSE = np.zeros((U, 1), dtype=float)
MAE = np.zeros((U, 1), dtype=float)
R2 = np.zeros((U, 1), dtype=float)
m_low = 0
m_high = 0

bins = min(30, max(10, len(YY_total_yuan_high) // 10))
hist, bin_edges = np.histogram(YY_total_yuan_high, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
mcs_pdf = hist
JS = np.zeros((U, 1), dtype=float)
for i in range(U):
    print(i)
    if i > 0:
        Y_test_prediction_iteration = Y_test_prediction.copy()
        Y_test_prediction_low_afa_iteration = Y_test_prediction_low_afa.copy()
        Y_test_prediction_delt_iteration = Y_test_prediction_delt.copy()
    model_RCNN = RCNN_M(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay, epochs, hidden_channels, activate_function, batch, N_low, N_high)
    model_RCNN.eval()
    YY_train_middle = model_RCNN(torch.tensor(xx_train[N_low:, :], dtype=torch.float))
    YY_test_middle = model_RCNN(torch.tensor(xx_total, dtype=torch.float))

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
    lower_sigma_initial = TabPFN_prediction_initial_yuan.ravel() - 1.96 * Prediction_sigma_initial
    upper_sigma_initial = TabPFN_prediction_initial_yuan.ravel() + 1.96 * Prediction_sigma_initial

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
        y_train=OUTPUT[N_low:] - model_RCNN.afa * TabPFN_prediction_tensor_initial0.to('cpu'),
        X_test=YY_test_middle,
        n_classes=None,
        categorical_features_index=None,
        task_type=task_type,
        # device="cuda" if torch.cuda.is_available() else "cpu",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    Y_test_prediction_delt = TabPFN_prediction_tensor.detach().cpu().numpy() * Sigma_scaler
    Y_test_prediction_low_afa = model_RCNN.afa.detach().cpu().numpy() * TabPFN_prediction_tensor_initial.detach().cpu().numpy() * Sigma_scaler + Mu_scaler
    Y_test_prediction = Y_test_prediction_delt + Y_test_prediction_low_afa
    Prediction_sigma = np.sqrt(Variance.detach().cpu().numpy().ravel() * Sigma_scaler ** 2)
    lower_sigma_delt = Y_test_prediction_delt.ravel() - 1.96 * Prediction_sigma
    upper_sigma_delt = Y_test_prediction_delt.ravel() + 1.96 * Prediction_sigma
    Prediction_total_sigma = np.sqrt(model_RCNN.afa.detach().cpu().numpy() ** 2 * Prediction_sigma_initial ** 2 + Prediction_sigma ** 2)
    lower_sigma = Y_test_prediction.ravel() - 1.96 * Prediction_total_sigma
    upper_sigma = Y_test_prediction.ravel() + 1.96 * Prediction_total_sigma
    rmse = 1 - root_mean_squared_error(YY_total_yuan_high, Y_test_prediction) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
    mae = 1 - mean_absolute_error(YY_total_yuan_high, Y_test_prediction) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
    r2 = r2_score(YY_total_yuan_high, Y_test_prediction)
    RMSE[i] = rmse
    MAE[i] = mae
    R2[i] = r2

    all_data = np.concatenate([YY_total_yuan_high, Y_test_prediction.ravel()])
    x_min, x_max = np.min(all_data), np.max(all_data)
    x_grid = np.linspace(x_min, x_max, 1000)
    mcs_pdf_grid = np.interp(x_grid, bin_centers, mcs_pdf, left=0, right=0)
    x_pdf, y_pdf = sns.kdeplot(data=Y_test_prediction).get_lines()[0].get_data()
    plt.close()
    model_pdf_grid = np.interp(x_grid, x_pdf, y_pdf, left=0, right=0)
    mcs_pdf_grid_norm = mcs_pdf_grid / np.trapz(mcs_pdf_grid, x_grid)
    model_pdf_grid_norm = model_pdf_grid / np.trapz(model_pdf_grid, x_grid)
    m = 0.5 * (mcs_pdf_grid_norm + model_pdf_grid_norm)
    kl_p_m = entropy(mcs_pdf_grid_norm + 1e-10, m + 1e-10)
    kl_q_m = entropy(model_pdf_grid_norm + 1e-10, m + 1e-10)
    js_div = 0.5 * (kl_p_m + kl_q_m)
    JS[i] = js_div

    if i > 0:
        Stop[i] = np.mean(np.abs(Y_test_prediction - Y_test_prediction_iteration))/(np.max(Y_test_prediction) - np.min(Y_test_prediction))
        # print("Convergence Criterion:", Stop[i])
        if i >= 2:
            if (Stop[i] + Stop[i-1])/2 <= Stop0:
                break

    Prediction_sigma_initial_mean = np.mean(model_RCNN.afa.detach().cpu().numpy() * Prediction_sigma_initial)
    Prediction_sigma_mean = np.mean(Prediction_sigma)

    distances_low = np.sum(np.abs(xx_total[:, np.newaxis] - xx_train[:N_low, :]), axis=2)
    min_distances_low = np.min(distances_low, axis=1)
    distances_high = np.sum(np.abs(xx_total[:, np.newaxis] - xx_train[N_low:, :]), axis=2)
    min_distances_high = np.min(distances_high, axis=1)

    if i < 10:
        Prediction_sigma_stack = np.stack((min_distances_low * model_RCNN.afa.detach().cpu().numpy() * Prediction_sigma_initial * Prediction_sigma_initial_mean / Cost_low, min_distances_high * Prediction_sigma * Prediction_sigma_mean / Cost_high), axis=1)
    else:
        Prediction_sigma_stack = np.stack((model_RCNN.afa.detach().cpu().numpy() * Prediction_sigma_initial * Prediction_sigma_initial_mean / Cost_low, Prediction_sigma * Prediction_sigma_mean / Cost_high),axis=1)

    max_value = np.max(Prediction_sigma_stack)
    max_indices = np.where(Prediction_sigma_stack == max_value)
    index_fidelity = max_indices[1][0]
    index_max = max_indices[0][0]
    if index_fidelity == 0:
        Index_max_low[m_low] = index_max
        m_low = m_low + 1
    else:
        Index_max_high[m_high] = index_max
        m_high = m_high + 1
    Uncertainty_max[i] = max_value
    Index_fidelity[i] = index_fidelity

    XX_new = XX_total_yuan[index_max]
    if index_fidelity == 0:
        XX_train_yuan_low = np.concatenate((XX_train_yuan_low, XX_new.reshape(1, -1)), axis=0)
        YY_new = YY_total_yuan_low[index_max]
        YY_train_yuan_low = np.concatenate((YY_train_yuan_low, YY_new.ravel()), axis=0)
        N_low = N_low + 1
    elif index_fidelity == 1:
        XX_train_yuan_high = np.concatenate((XX_train_yuan_high, XX_new.reshape(1, -1)), axis=0)
        YY_new = YY_total_yuan_high[index_max]
        YY_train_yuan_high = np.concatenate((YY_train_yuan_high, YY_new.ravel()), axis=0)
        N_high = N_high + 1

    XX_train_yuan = np.concatenate((XX_train_yuan_low, XX_train_yuan_high), axis=0)
    YY_train_yuan = np.concatenate((YY_train_yuan_low, YY_train_yuan_high), axis=0)

    scaler = StandardScaler()
    xx_train = scaler.fit_transform(XX_train_yuan)
    xx_total = scaler.transform(XX_total_yuan)
    scaler_Y = StandardScaler()
    yy_train = scaler_Y.fit_transform(YY_train_yuan.reshape(-1, 1))
    Mu_scaler = scaler_Y.mean_
    Sigma_scaler = np.sqrt(scaler_Y.var_)

    INPUT = torch.tensor(xx_train, dtype=torch.float)
    OUTPUT = torch.tensor(yy_train, dtype=torch.float)

Y_test_prediction_Ad_MFTabPFN = Y_test_prediction
N_low_repeat = N_low
N_high_repeat = N_high

# with open('i_CD.pkl', 'wb') as f:
#     pickle.dump(i, f)
# with open('Y_test_prediction_Ad_MFTabPFN_CD.pkl', 'wb') as f:
#     pickle.dump(Y_test_prediction_Ad_MFTabPFN, f)
# with open('YY_total_yuan_high_CD.pkl', 'wb') as f:
#     pickle.dump(YY_total_yuan_high, f)
# with open('RMSE_Ad_MFTabPFN_CD.pkl', 'wb') as f:
#     pickle.dump(RMSE, f)
# with open('MAE_Ad_MFTabPFN_CD.pkl', 'wb') as f:
#     pickle.dump(MAE, f)
# with open('R2_Ad_MFTabPFN_CD.pkl', 'wb') as f:
#     pickle.dump(R2, f)
# with open('JS_Ad_MFTabPFN_CD.pkl', 'wb') as f:
#     pickle.dump(JS, f)
# with open('Index_fidelity_Ad_MFTabPFN_CD.pkl', 'wb') as f:
#     pickle.dump(Index_fidelity, f)

array_flat = Index_max_low[:m_low].flatten()
values, counts = np.unique(array_flat, return_counts=True)
repeated_values = values[counts > 1]
repeated_counts = counts[counts > 1]
if len(repeated_values) != 0:
    repeated_counts_sum = np.sum(repeated_counts) - len(repeated_counts)
    N_low = N_low - repeated_counts_sum

# with open('N_low_call_CD.pkl', 'wb') as f:
#     pickle.dump(N_low, f)
# with open('repeated_values_low_CD.pkl', 'wb') as f:
#     pickle.dump(repeated_values, f)
# with open('repeated_counts_low_CD.pkl', 'wb') as f:
#     pickle.dump(repeated_counts, f)

array_flat = Index_max_high[:m_high].flatten()
values, counts = np.unique(array_flat, return_counts=True)
repeated_values = values[counts > 1]
repeated_counts = counts[counts > 1]
if len(repeated_values) != 0:
    repeated_counts_sum = np.sum(repeated_counts) - len(repeated_counts)
    N_high = N_high - repeated_counts_sum

# with open('N_high_call_CD.pkl', 'wb') as f:
#     pickle.dump(N_high, f)
# with open('repeated_values_high_CD.pkl', 'wb') as f:
#     pickle.dump(repeated_values, f)
# with open('repeated_counts_high_CD.pkl', 'wb') as f:
#     pickle.dump(repeated_counts, f)
######################################TabPFN-high
np.random.seed(123)
indices_low = np.random.choice(N, size=N_low, replace=False)
indices_high = np.random.choice(N, size=N_high, replace=False)
XX_train_yuan_low = XX_total_yuan[indices_low, :]
XX_train_yuan_high = XX_total_yuan[indices_high, :]

YY_train_yuan_low = CD_total_low[indices_low, :].ravel()
YY_train_yuan_high = CD_total_high[indices_high, :].ravel()
YY_total_yuan_low = CD_total_low[:N, :].ravel()
YY_total_yuan_high = CD_total_high[:N, :].ravel()

XX_train_yuan = np.concatenate((XX_train_yuan_low, XX_train_yuan_high), axis=0)
YY_train_yuan = np.concatenate((YY_train_yuan_low, YY_train_yuan_high), axis=0)

scaler = StandardScaler()
xx_train = scaler.fit_transform(XX_train_yuan)
xx_total = scaler.transform(XX_total_yuan[:N, :])
scaler_Y = StandardScaler()
yy_train = scaler_Y.fit_transform(YY_train_yuan.reshape(-1, 1))
Mu_scaler = scaler_Y.mean_
Sigma_scaler = np.sqrt(scaler_Y.var_)

if nx > 500:
    reg = TabPFNRegressor(n_estimators=8, random_state=42, ignore_pretraining_limits=True)
else:
    reg = TabPFNRegressor(n_estimators=8, random_state=42)
reg.fit(xx_train[N_low:, :], yy_train[N_low:].ravel())#high
YY_test_prediction_initial = reg.predict(xx_total, output_type="full")
Y_test_prediction_initial = YY_test_prediction_initial[f"mean"] * Sigma_scaler + Mu_scaler

Y_test_prediction_TabPFN_high = Y_test_prediction_initial
rmse = 1 - root_mean_squared_error(YY_total_yuan_high, Y_test_prediction_TabPFN_high) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
mae = 1 - mean_absolute_error(YY_total_yuan_high, Y_test_prediction_TabPFN_high) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
r2 = r2_score(YY_total_yuan_high, Y_test_prediction_TabPFN_high)

all_data = np.concatenate([YY_total_yuan_high, Y_test_prediction_TabPFN_high.ravel()])
x_min, x_max = np.min(all_data), np.max(all_data)
x_grid = np.linspace(x_min, x_max, 1000)
mcs_pdf_grid = np.interp(x_grid, bin_centers, mcs_pdf, left=0, right=0)
x_pdf, y_pdf = sns.kdeplot(data=Y_test_prediction_TabPFN_high).get_lines()[0].get_data()
plt.close()
model_pdf_grid = np.interp(x_grid, x_pdf, y_pdf, left=0, right=0)
mcs_pdf_grid_norm = mcs_pdf_grid / np.trapz(mcs_pdf_grid, x_grid)
model_pdf_grid_norm = model_pdf_grid / np.trapz(model_pdf_grid, x_grid)
m = 0.5 * (mcs_pdf_grid_norm + model_pdf_grid_norm)
kl_p_m = entropy(mcs_pdf_grid_norm + 1e-10, m + 1e-10)
kl_q_m = entropy(model_pdf_grid_norm + 1e-10, m + 1e-10)
js_div = 0.5 * (kl_p_m + kl_q_m)

# with open('Y_test_prediction_TabPFN_high_CD.pkl', 'wb') as f:
#     pickle.dump(Y_test_prediction_TabPFN_high, f)
# with open('RMSE_TabPFN_high_CD.pkl', 'wb') as f:
#     pickle.dump(rmse, f)
# with open('MAE_TabPFN_high_CD.pkl', 'wb') as f:
#     pickle.dump(mae, f)
# with open('R2_TabPFN_high_CD.pkl', 'wb') as f:
#     pickle.dump(r2, f)
# with open('JS_TabPFN_high_CD.pkl', 'wb') as f:
#     pickle.dump(js_div, f)

######################################TabPFN-low
if nx > 500:
    reg = TabPFNRegressor(n_estimators=8, random_state=42, ignore_pretraining_limits=True)
else:
    reg = TabPFNRegressor(n_estimators=8, random_state=42)
reg.fit(xx_train[:N_low, :], yy_train[:N_low].ravel())#high
YY_test_prediction_initial = reg.predict(xx_total, output_type="full")
Y_test_prediction_initial = YY_test_prediction_initial[f"mean"] * Sigma_scaler + Mu_scaler

Y_test_prediction_TabPFN_low = Y_test_prediction_initial
rmse = 1 - root_mean_squared_error(YY_total_yuan_high, Y_test_prediction_TabPFN_low) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
mae = 1 - mean_absolute_error(YY_total_yuan_high, Y_test_prediction_TabPFN_low) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
r2 = r2_score(YY_total_yuan_high, Y_test_prediction_TabPFN_low)

all_data = np.concatenate([YY_total_yuan_high, Y_test_prediction_TabPFN_low.ravel()])
x_min, x_max = np.min(all_data), np.max(all_data)
x_grid = np.linspace(x_min, x_max, 1000)
mcs_pdf_grid = np.interp(x_grid, bin_centers, mcs_pdf, left=0, right=0)
x_pdf, y_pdf = sns.kdeplot(data=Y_test_prediction_TabPFN_low).get_lines()[0].get_data()
plt.close()
model_pdf_grid = np.interp(x_grid, x_pdf, y_pdf, left=0, right=0)
mcs_pdf_grid_norm = mcs_pdf_grid / np.trapz(mcs_pdf_grid, x_grid)
model_pdf_grid_norm = model_pdf_grid / np.trapz(model_pdf_grid, x_grid)
m = 0.5 * (mcs_pdf_grid_norm + model_pdf_grid_norm)
kl_p_m = entropy(mcs_pdf_grid_norm + 1e-10, m + 1e-10)
kl_q_m = entropy(model_pdf_grid_norm + 1e-10, m + 1e-10)
js_div = 0.5 * (kl_p_m + kl_q_m)

# with open('Y_test_prediction_TabPFN_low_CD.pkl', 'wb') as f:
#     pickle.dump(Y_test_prediction_TabPFN_low, f)
# with open('RMSE_TabPFN_low_CD.pkl', 'wb') as f:
#     pickle.dump(rmse, f)
# with open('MAE_TabPFN_low_CD.pkl', 'wb') as f:
#     pickle.dump(mae, f)
# with open('R2_TabPFN_low_CD.pkl', 'wb') as f:
#     pickle.dump(r2, f)
# with open('JS_TabPFN_low_CD.pkl', 'wb') as f:
#     pickle.dump(js_div, f)
######################################AutoGluon-high
TRAIN_Autogluon = pd.DataFrame(xx_train[N_low:, :], columns=[f'feature_{i}' for i in range(nx)])
TRAIN_Autogluon['target'] = yy_train[N_low:]
predictor = TabularPredictor(
    label="target",
    problem_type="regression",
    path="autogluon_model"
)
predictor.fit(train_data=TRAIN_Autogluon, time_limit=600)
TEST_Autogluon = pd.DataFrame(xx_total, columns=[f'feature_{i}' for i in range(nx)])
Y_test_prediction_initial = predictor.predict(TEST_Autogluon).to_numpy() * Sigma_scaler + Mu_scaler
Y_test_prediction_AutoGluon_high = Y_test_prediction_initial
rmse = 1 - root_mean_squared_error(YY_total_yuan_high, Y_test_prediction_initial) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
mae = 1 - mean_absolute_error(YY_total_yuan_high, Y_test_prediction_initial) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
r2 = r2_score(YY_total_yuan_high, Y_test_prediction_initial)

all_data = np.concatenate([YY_total_yuan_high, Y_test_prediction_AutoGluon_high.ravel()])
x_min, x_max = np.min(all_data), np.max(all_data)
x_grid = np.linspace(x_min, x_max, 1000)
mcs_pdf_grid = np.interp(x_grid, bin_centers, mcs_pdf, left=0, right=0)
x_pdf, y_pdf = sns.kdeplot(data=Y_test_prediction_AutoGluon_high).get_lines()[0].get_data()
plt.close()
model_pdf_grid = np.interp(x_grid, x_pdf, y_pdf, left=0, right=0)
mcs_pdf_grid_norm = mcs_pdf_grid / np.trapz(mcs_pdf_grid, x_grid)
model_pdf_grid_norm = model_pdf_grid / np.trapz(model_pdf_grid, x_grid)
m = 0.5 * (mcs_pdf_grid_norm + model_pdf_grid_norm)
kl_p_m = entropy(mcs_pdf_grid_norm + 1e-10, m + 1e-10)
kl_q_m = entropy(model_pdf_grid_norm + 1e-10, m + 1e-10)
js_div = 0.5 * (kl_p_m + kl_q_m)

# with open('Y_test_prediction_AutoGluon_high_CD.pkl', 'wb') as f:
#     pickle.dump(Y_test_prediction_AutoGluon_high, f)
# with open('RMSE_AutoGluon_high_CD.pkl', 'wb') as f:
#     pickle.dump(rmse, f)
# with open('MAE_AutoGluon_high_CD.pkl', 'wb') as f:
#     pickle.dump(mae, f)
# with open('R2_AutoGluon_high_CD.pkl', 'wb') as f:
#     pickle.dump(r2, f)
# with open('JS_AutoGluon_high_CD.pkl', 'wb') as f:
#     pickle.dump(js_div, f)

######################################AutoGluon-low
TRAIN_Autogluon = pd.DataFrame(xx_train[:N_low, :], columns=[f'feature_{i}' for i in range(nx)])
TRAIN_Autogluon['target'] = yy_train[:N_low]
predictor = TabularPredictor(
    label="target",
    problem_type="regression",
    path="autogluon_model"
)
predictor.fit(train_data=TRAIN_Autogluon, time_limit=600)
TEST_Autogluon = pd.DataFrame(xx_total, columns=[f'feature_{i}' for i in range(nx)])
Y_test_prediction_initial = predictor.predict(TEST_Autogluon).to_numpy() * Sigma_scaler + Mu_scaler
Y_test_prediction_AutoGluon_low = Y_test_prediction_initial
rmse = 1 - root_mean_squared_error(YY_total_yuan_high, Y_test_prediction_initial) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
mae = 1 - mean_absolute_error(YY_total_yuan_high, Y_test_prediction_initial) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
r2 = r2_score(YY_total_yuan_high, Y_test_prediction_initial)

all_data = np.concatenate([YY_total_yuan_high, Y_test_prediction_AutoGluon_low.ravel()])
x_min, x_max = np.min(all_data), np.max(all_data)
x_grid = np.linspace(x_min, x_max, 1000)
mcs_pdf_grid = np.interp(x_grid, bin_centers, mcs_pdf, left=0, right=0)
x_pdf, y_pdf = sns.kdeplot(data=Y_test_prediction_AutoGluon_low).get_lines()[0].get_data()
plt.close()
model_pdf_grid = np.interp(x_grid, x_pdf, y_pdf, left=0, right=0)
mcs_pdf_grid_norm = mcs_pdf_grid / np.trapz(mcs_pdf_grid, x_grid)
model_pdf_grid_norm = model_pdf_grid / np.trapz(model_pdf_grid, x_grid)
m = 0.5 * (mcs_pdf_grid_norm + model_pdf_grid_norm)
kl_p_m = entropy(mcs_pdf_grid_norm + 1e-10, m + 1e-10)
kl_q_m = entropy(model_pdf_grid_norm + 1e-10, m + 1e-10)
js_div = 0.5 * (kl_p_m + kl_q_m)

# with open('Y_test_prediction_AutoGluon_low_CD.pkl', 'wb') as f:
#     pickle.dump(Y_test_prediction_AutoGluon_low, f)
# with open('RMSE_AutoGluon_low_CD.pkl', 'wb') as f:
#     pickle.dump(rmse, f)
# with open('MAE_AutoGluon_low_CD.pkl', 'wb') as f:
#     pickle.dump(mae, f)
# with open('R2_AutoGluon_low_CD.pkl', 'wb') as f:
#     pickle.dump(r2, f)
# with open('JS_AutoGluon_low_CD.pkl', 'wb') as f:
#     pickle.dump(js_div, f)
