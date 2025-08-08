from tabpfn import TabPFNRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
from autogluon.tabular import TabularPredictor
import pickle
import torch
if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')

WORK_DIR = "./Datasets/DLR_F4"

pkl_file1 = os.path.join(WORK_DIR, 'Simulation_lower.pkl')
with open(pkl_file1, 'rb') as f:
    Simulation_lower = pickle.load(f)
pkl_file2 = os.path.join(WORK_DIR, 'Simulation_upper.pkl')
with open(pkl_file2, 'rb') as f:
    Simulation_upper = pickle.load(f)
pkl_file3 = os.path.join(WORK_DIR, 'Experiment_lower.pkl')
with open(pkl_file3, 'rb') as f:
    Experiment_lower = pickle.load(f)
pkl_file4 = os.path.join(WORK_DIR, 'Experiment_upper.pkl')
with open(pkl_file4, 'rb') as f:
    Experiment_upper = pickle.load(f)

MA = np.array([0.75])
AFA = np.array([0.18])
Y_TARGET = np.array([0.185, 0.331, 0.512, 0.844])

training_simulation_index = [0, 1, 2, 3]
training_experiment_index = [0, 1, 2, 3]
testing_index = [0, 1, 2, 3]

Case = 1 # 1: training data ratio 20%; 2: training data ratio 40%; 3: training data ratio 60%; 4: training data ratio 80%
if Case == 1:
    Training_bili = 0.20
elif Case == 2:
    Training_bili = 0.40
elif Case == 3:
    Training_bili = 0.60
elif Case == 4:
    Training_bili = 0.80

def sample_rows(df, fraction=0.2, case=1):
    total_rows = len(df)
    sample_size = int(np.ceil(total_rows * fraction))
    if case == 1:
        selected_indices = np.linspace(0, total_rows - 1, sample_size, dtype=int)
    else:
        selected_indices = np.linspace(0, total_rows - 2, sample_size, dtype=int)
    return df.iloc[selected_indices]

def sample_complement(df, fraction=0.2, case=1):
    total_rows = len(df)
    sample_size = int(np.ceil(total_rows * fraction))
    if case == 1:
        selected_indices = np.linspace(0, total_rows - 1, sample_size, dtype=int)
    else:
        selected_indices = np.linspace(0, total_rows - 2, sample_size, dtype=int)
    all_indices = np.arange(total_rows)
    complement_indices = np.setdiff1d(all_indices, selected_indices)
    return df.iloc[complement_indices]

lower_simulation_data = {}
upper_simulation_data = {}
for i in range(0, len(training_simulation_index)):
    Ma = MA[0]
    afa = AFA[0]
    target = Y_TARGET[training_simulation_index[i]]
    lower_simulation = Simulation_lower[f'Simulation_m{Ma:.4f}_a{afa:.3f}_lower.dat']
    upper_simulation = Simulation_upper[f'Simulation_m{Ma:.4f}_a{afa:.3f}_upper.dat']
    lower_simulation_data[f'Simulation_m{Ma:.4f}_a{afa:.3f}_yb{target:.3f}_lower.dat'] = lower_simulation[lower_simulation['y/b']==target]
    upper_simulation_data[f'Simulation_m{Ma:.4f}_a{afa:.3f}_yb{target:.3f}_upper.dat'] = upper_simulation[upper_simulation['y/b']==target]
lower_simulation_train = pd.concat(lower_simulation_data.values(), ignore_index=True)
upper_simulation_train = pd.concat(upper_simulation_data.values(), ignore_index=True)
lower_simulation_train.insert(loc=4, column='c', value=-1)
upper_simulation_train.insert(loc=4, column='c', value=1)
simulation_train = pd.concat([lower_simulation_train, upper_simulation_train], ignore_index=True)

lower_experiment_data = {}
upper_experiment_data = {}
for i in range(0, len(training_experiment_index)):
    Ma = MA[0]
    afa = AFA[0]
    target = Y_TARGET[training_experiment_index[i]]
    lower_experiment = Experiment_lower[f'Experiment_m{Ma:.4f}_a{afa:.3f}_lower.dat']
    upper_experiment = Experiment_upper[f'Experiment_m{Ma:.4f}_a{afa:.3f}_upper.dat']
    lower_lower = lower_experiment[lower_experiment['y/b']==target]
    lower_lower_bili = sample_rows(lower_lower, fraction=Training_bili, case=2)
    lower_experiment_data[f'Experiment_m{Ma:.4f}_a{afa:.3f}_yb{target:.3f}_lower.dat'] = lower_lower_bili
    upper_upper = upper_experiment[upper_experiment['y/b']==target]
    upper_upper_bili = sample_rows(upper_upper, fraction=Training_bili)
    upper_experiment_data[f'Experiment_m{Ma:.4f}_a{afa:.3f}_yb{target:.3f}_upper.dat'] = upper_upper_bili
lower_experiment_train = pd.concat(lower_experiment_data.values(), ignore_index=True)
upper_experiment_train = pd.concat(upper_experiment_data.values(), ignore_index=True)
lower_experiment_train.insert(loc=4, column='c', value=-1)
upper_experiment_train.insert(loc=4, column='c', value=1)
experiment_train = pd.concat([lower_experiment_train, upper_experiment_train], ignore_index=True)

lower_experiment_data1 = {}
upper_experiment_data1 = {}
for i in range(0, len(testing_index)):
    Ma = MA[0]
    afa = AFA[0]
    target = Y_TARGET[testing_index[i]]
    lower_experiment1 = Experiment_lower[f'Experiment_m{Ma:.4f}_a{afa:.3f}_lower.dat']
    upper_experiment1 = Experiment_upper[f'Experiment_m{Ma:.4f}_a{afa:.3f}_upper.dat']
    lower_lower1 = lower_experiment1[lower_experiment1['y/b']==target]
    lower_lower_bili1 = sample_complement(lower_lower1, fraction=Training_bili, case=2)
    lower_experiment_data1[f'Experiment_m{Ma:.4f}_a{afa:.3f}_yb{target:.3f}_lower.dat'] = lower_lower_bili1
    upper_upper1 = upper_experiment1[upper_experiment1['y/b']==target]
    upper_upper_bili1 = sample_complement(upper_upper1, fraction=Training_bili)
    upper_experiment_data1[f'Experiment_m{Ma:.4f}_a{afa:.3f}_yb{target:.3f}_upper.dat'] = upper_upper_bili1
lower_experiment_test = pd.concat(lower_experiment_data1.values(), ignore_index=True)
upper_experiment_test = pd.concat(upper_experiment_data1.values(), ignore_index=True)
lower_experiment_test.insert(loc=4, column='c', value=-1)
upper_experiment_test.insert(loc=4, column='c', value=1)
experiment_test = pd.concat([lower_experiment_test, upper_experiment_test], ignore_index=True)

N_low = simulation_train.shape[0]
N_high = experiment_train.shape[0]
N = experiment_test.shape[0]

XX_train_yuan_low = simulation_train.values[:, 2:-1]
YY_train_yuan_low = simulation_train.values[:, -1]
XX_train_yuan_high = experiment_train.values[:, 2:-1]
YY_train_yuan_high = experiment_train.values[:, -1]
XX_total_yuan = experiment_test.values[:, 2:-1]
YY_total_yuan_high = experiment_test.values[:, -1]

XX_train_yuan = np.concatenate((XX_train_yuan_low, XX_train_yuan_high), axis=0)
YY_train_yuan = np.concatenate((YY_train_yuan_low, YY_train_yuan_high), axis=0)

nx = XX_train_yuan_low.shape[1]

scaler = StandardScaler()
xx_train = scaler.fit_transform(XX_train_yuan)
xx_total = scaler.transform(XX_total_yuan)
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

##########################################################################TabPFN
if nx > 500:
    reg = TabPFNRegressor(n_estimators=8, random_state=42, ignore_pretraining_limits=True)
else:
    reg = TabPFNRegressor(n_estimators=8, random_state=42)
reg.fit(xx_train[N_low:, :], yy_train[N_low:].ravel())#high
YY_test_prediction_initial = reg.predict(xx_total, output_type="full")
Y_test_prediction_initial = YY_test_prediction_initial[f"mean"] * Sigma_scaler + Mu_scaler
lower_initial = YY_test_prediction_initial[f"quantiles"][0] * Sigma_scaler + Mu_scaler
upper_initial = YY_test_prediction_initial[f"quantiles"][8] * Sigma_scaler + Mu_scaler
rmse = 1 - root_mean_squared_error(YY_total_yuan_high, Y_test_prediction_initial) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
mae = 1 - mean_absolute_error(YY_total_yuan_high, Y_test_prediction_initial) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
r2 = r2_score(YY_total_yuan_high, Y_test_prediction_initial)
print("TabPFN RMSE:", rmse)
print("TabPFN MAE:", mae)
print("TabPFN R2:", r2)
Y_test_prediction_TabPFN = Y_test_prediction_initial.copy()
YY_test_prediction_initial = reg.predict(xx_train[:N_low, :], output_type="full")
Y_test_prediction_initial = YY_test_prediction_initial[f"mean"] * Sigma_scaler + Mu_scaler
Variance_initial = YY_test_prediction_initial[f"variance"].reshape(-1, 1)
Prediction_sigma_initial = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)
LOWER_SIGMA_TabPFN = Y_test_prediction_initial.ravel() - 1.00 * Prediction_sigma_initial
UPPER_SIGMA_TabPFN = Y_test_prediction_initial.ravel() + 1.00 * Prediction_sigma_initial
Y_test_prediction_TabPFN_PLOT = Y_test_prediction_initial.copy()
##########################################################################AutoGluon
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
rmse = 1 - root_mean_squared_error(YY_total_yuan_high, Y_test_prediction_initial) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
mae = 1 - mean_absolute_error(YY_total_yuan_high, Y_test_prediction_initial) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
r2 = r2_score(YY_total_yuan_high, Y_test_prediction_initial)
print("Autogluon RMSE:", rmse)
print("Autogluon MAE:", mae)
print("Autogluon R2:", r2)
Y_test_prediction_AutoGluon = Y_test_prediction_initial.copy()
TEST_Autogluon = pd.DataFrame(xx_train[:N_low, :], columns=[f'feature_{i}' for i in range(nx)])
Y_test_prediction_initial = predictor.predict(TEST_Autogluon).to_numpy() * Sigma_scaler + Mu_scaler
Y_test_prediction_AutoGluon_PLOT = Y_test_prediction_initial.copy()
##########################################################################MFTabPFN
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
TabPFN_prediction_tensor_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
Lower_initial = TabPFN_prediction_initial_full[f"quantiles"][0].reshape(-1, 1)
Upper_initial = TabPFN_prediction_initial_full[f"quantiles"][8].reshape(-1, 1)
Variance_initial = TabPFN_prediction_initial_full[f"variance"].reshape(-1, 1)
Prediction_sigma_initial = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)
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
Y_test_prediction = (TabPFN_prediction_tensor.detach().cpu().numpy() + model_MLP.afa.detach().cpu().numpy() * TabPFN_prediction_tensor_initial.detach().cpu().numpy()) * Sigma_scaler + Mu_scaler
lower = (Lower.detach().cpu().numpy().ravel()+ model_MLP.afa.detach().cpu().numpy() * TabPFN_prediction_tensor_initial.detach().cpu().numpy().ravel()) * Sigma_scaler + Mu_scaler
upper = (Upper.detach().cpu().numpy().ravel()+ model_MLP.afa.detach().cpu().numpy() * TabPFN_prediction_tensor_initial.detach().cpu().numpy().ravel())  * Sigma_scaler + Mu_scaler
Prediction_sigma = np.sqrt(Variance.detach().cpu().numpy().ravel() * Sigma_scaler ** 2)
lower_sigma = Y_test_prediction.ravel() - 3 * Prediction_sigma
upper_sigma = Y_test_prediction.ravel() + 3 * Prediction_sigma
Prediction_total_sigma2 = model_MLP.afa.detach().cpu().numpy() ** 2 * Prediction_sigma_initial ** 2 + Prediction_sigma ** 2
LOWER_SIGMA = Y_test_prediction.ravel() - 1.96 * np.sqrt(Prediction_total_sigma2)
UPPER_SIGMA = Y_test_prediction.ravel() + 1.96 * np.sqrt(Prediction_total_sigma2)
rmse = 1 - root_mean_squared_error(YY_total_yuan_high, Y_test_prediction) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
mae = 1 - mean_absolute_error(YY_total_yuan_high, Y_test_prediction) / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
r2 = r2_score(YY_total_yuan_high, Y_test_prediction)
print("MFTabPFN RMSE:", rmse)
print("MFTabPFN MAE:", mae)
print("MFTabPFN R2:", r2)
Y_test_prediction_MFTabPFN = Y_test_prediction.copy()
####For plot figure
YY_test_middle = model_MLP(torch.tensor(xx_train[:N_low, :], dtype=torch.float))
if nx > 500:
    reg_initial = TabPFNRegressor(n_estimators=8, random_state=42, ignore_pretraining_limits=True)
else:
    reg_initial = TabPFNRegressor(n_estimators=8, random_state=42)
reg_initial.fit(xx_train[:N_low, :], yy_train[:N_low].ravel())
TabPFN_prediction_initial_full = reg_initial.predict(xx_train[:N_low, :], output_type="full")
TabPFN_prediction_initial = TabPFN_prediction_initial_full[f"mean"].reshape(-1, 1)
TabPFN_prediction_tensor_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
Lower_initial = TabPFN_prediction_initial_full[f"quantiles"][0].reshape(-1, 1)
Upper_initial = TabPFN_prediction_initial_full[f"quantiles"][8].reshape(-1, 1)
Variance_initial = TabPFN_prediction_initial_full[f"variance"].reshape(-1, 1)
Prediction_sigma_initial = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)
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
    n_classes=None,  # fen lei
    categorical_features_index=None,
    task_type=task_type,
    # device="cuda" if torch.cuda.is_available() else "cpu",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)
Y_test_prediction = (TabPFN_prediction_tensor.detach().cpu().numpy() + model_MLP.afa.detach().cpu().numpy() * TabPFN_prediction_tensor_initial.detach().cpu().numpy()) * Sigma_scaler + Mu_scaler
lower = (Lower.detach().cpu().numpy().ravel()+ model_MLP.afa.detach().cpu().numpy() * TabPFN_prediction_tensor_initial.detach().cpu().numpy().ravel()) * Sigma_scaler + Mu_scaler
upper = (Upper.detach().cpu().numpy().ravel()+ model_MLP.afa.detach().cpu().numpy() * TabPFN_prediction_tensor_initial.detach().cpu().numpy().ravel())  * Sigma_scaler + Mu_scaler
Prediction_sigma = np.sqrt(Variance.detach().cpu().numpy().ravel() * Sigma_scaler ** 2)
Prediction_total_sigma2 = model_MLP.afa.detach().cpu().numpy() ** 2 * Prediction_sigma_initial ** 2 + Prediction_sigma ** 2
LOWER_SIGMA = Y_test_prediction.ravel() - 1.00 * np.sqrt(Prediction_total_sigma2)
UPPER_SIGMA = Y_test_prediction.ravel() + 1.00 * np.sqrt(Prediction_total_sigma2)
Y_test_prediction_MFTabPFN_PLOT = Y_test_prediction.copy()

# SAVE_DIR = './Datasets/DLR_F4/Ratio'
# pkl_file = os.path.join(SAVE_DIR, f'Training_bili.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(Training_bili, f)
# pkl_file = os.path.join(SAVE_DIR, f'MA.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(MA, f)
# pkl_file = os.path.join(SAVE_DIR, f'AFA.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(AFA, f)
# pkl_file = os.path.join(SAVE_DIR, f'Y_TARGET.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(Y_TARGET, f)
# pkl_file = os.path.join(SAVE_DIR, f'XX_total_yuan.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(XX_total_yuan, f)
# pkl_file = os.path.join(SAVE_DIR, f'XX_train_yuan_low.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(XX_train_yuan_low, f)
# pkl_file = os.path.join(SAVE_DIR, f'XX_train_yuan_high.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(XX_train_yuan_high, f)
# pkl_file = os.path.join(SAVE_DIR, f'YY_train_yuan_high.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(YY_train_yuan_high, f)
# pkl_file = os.path.join(SAVE_DIR, f'YY_total_yuan_high.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(YY_total_yuan_high, f)
# pkl_file = os.path.join(SAVE_DIR, f'YY_train_yuan_low.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(YY_train_yuan_low, f)
# pkl_file = os.path.join(SAVE_DIR, f'Y_test_prediction_AutoGluon_PLOT.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(Y_test_prediction_AutoGluon_PLOT, f)
# pkl_file = os.path.join(SAVE_DIR, f'Y_test_prediction_TabPFN_PLOT.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(Y_test_prediction_TabPFN_PLOT, f)
# pkl_file = os.path.join(SAVE_DIR, f'LOWER_SIGMA_TabPFN.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(LOWER_SIGMA_TabPFN, f)
# pkl_file = os.path.join(SAVE_DIR, f'UPPER_SIGMA_TabPFN.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(UPPER_SIGMA_TabPFN, f)
# pkl_file = os.path.join(SAVE_DIR, f'Y_test_prediction_MFTabPFN_PLOT.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(Y_test_prediction_MFTabPFN_PLOT, f)
# pkl_file = os.path.join(SAVE_DIR, f'LOWER_SIGMA.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(LOWER_SIGMA, f)
# pkl_file = os.path.join(SAVE_DIR, f'UPPER_SIGMA.pkl')
# with open(pkl_file, 'wb') as f:
#     pickle.dump(UPPER_SIGMA, f)
