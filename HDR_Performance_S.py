from tabpfn import TabPFNRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import scipy.stats as stats
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from RCNN_MFTabPFN_S import RCNN_S
from ANN_model import SingleANN_PM
from pathlib import Path
from TabPFN_model import TabPFN_model_main
from autogluon.tabular import TabularPredictor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import torch
if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')

SAVE_DIR = './Datasets/Synthetic/HDR_fixed_data_size'
def dis(Case):
    if Case == 0:
        mu = np.full(100, 0.5)
        sigma = np.full(100, 1/np.sqrt(12))
        DisType = ['uniform'] * 100
    elif Case == 1:
        mu = np.full(200, 0.5)
        sigma = np.full(200, 1/np.sqrt(12))
        DisType = ['uniform'] * 200
    elif Case == 2:
        mu = np.full(300, 0.5)
        sigma = np.full(300, 1/np.sqrt(12))
        DisType = ['uniform'] * 300
    elif Case == 3:
        mu = np.full(400, 0.5)
        sigma = np.full(400, 1/np.sqrt(12))
        DisType = ['uniform'] * 400
    elif Case == 4:
        mu = np.full(500, 0.5)
        sigma = np.full(500, 1/np.sqrt(12))
        DisType = ['uniform'] * 500
    elif Case == 5:
        mu = np.full(600, 0.5)
        sigma = np.full(600, 1/np.sqrt(12))
        DisType = ['uniform'] * 600
    elif Case == 6:
        mu = np.full(700, 0.5)
        sigma = np.full(700, 1/np.sqrt(12))
        DisType = ['uniform'] * 700
    elif Case == 7:
        mu = np.full(800, 0.5)
        sigma = np.full(800, 1/np.sqrt(12))
        DisType = ['uniform'] * 800
    elif Case == 8:
        mu = np.full(900, 0.5)
        sigma = np.full(900, 1/np.sqrt(12))
        DisType = ['uniform'] * 900
    elif Case == 9:
        mu = np.full(1000, 0.5)
        sigma = np.full(1000, 1/np.sqrt(12))
        DisType = ['uniform'] * 1000
    return mu, sigma, DisType

def func(x, Case):
    if Case == 0:
        ff = np.sum([(x[:, k] - 1) ** 4 for k in range(0, 50)], axis=0) + np.sum([np.sqrt(k + 1) * (x[:, k] - x[:, k - 1] ** 2) ** 2 for k in range(1, 100)], axis=0)
    elif Case == 1:
        ff = 100 * x[:, 0] + np.sum([x[:, k] ** 3 for k in range(1, 200)], axis=0)
    elif Case == 2:
        ff = np.sum([(x[:, k] - 1) ** 4 for k in range(0, 150)], axis=0) + np.sum([np.sqrt(k + 1) * (x[:, k] - x[:, k - 1] ** 2) ** 2 for k in range(1, 300)], axis=0)
    elif Case == 3:
        ff = 100 * x[:, 0] + np.sum([x[:, k] ** 3 for k in range(1, 400)], axis=0)
    elif Case == 4:
        ff = 10 * (x[:, 1] - x[:, 0] ** 2) ** 5 + np.sum([(x[:, k + 1] - x[:, k] ** 2) ** 5 for k in range(1, 499)], axis=0)
    elif Case == 5:
        ff = (x[:, 0] - 1) ** 2 + np.sum([(k + 1) * (2 * x[:, k] ** 2 - x[:, k - 1]) ** 2 for k in range(1, 600)], axis=0)
    elif Case == 6:
        ff = 10 * (x[:, 1] - x[:, 0] ** 2) ** 5 + np.sum([(x[:, k + 1] - x[:, k] ** 2) ** 5 for k in range(1, 699)], axis=0)
    elif Case == 7:
        ff = (x[:, 0] - 1) ** 2 + np.sum([(k + 1) * (2 * x[:, k] ** 2 - x[:, k - 1]) ** 2 for k in range(1, 800)], axis=0)
    elif Case == 8:
        ff = np.sum([x[:, k] for k in range(0, 900)], axis=0) + 20 * x[:, 0] ** 2 * x[:, 1] ** 2 + np.sum([x[:, k] ** 2 * x[:, k + 1] ** 2 for k in range(1, 899)], axis=0) - np.sum([np.sin(x[:, k]) * np.exp(x[:, k] - 2) for k in range(0, 900)], axis=0) - 10
    elif Case == 9:
        ff = np.sum([x[:, k] for k in range(0, 1000)], axis=0) + 20 * x[:, 0] ** 2 * x[:, 1] ** 2 + np.sum([x[:, k] ** 2 * x[:, k + 1] ** 2 for k in range(1, 999)], axis=0) - np.sum([np.sin(x[:, k]) * np.exp(x[:, k] - 2) for k in range(0, 1000)], axis=0) - 10
    return ff

def UtoX(u, mu, sigma, DisType):
    x = np.zeros_like(u)
    for i in range(len(mu)):
        if DisType[i] == 'normal':
            x[:, i] = u[:, i] * sigma[i] + mu[i]
        elif DisType[i] == 'uniform':
            a = mu[i] - np.sqrt(3) * sigma[i]
            b = mu[i] + np.sqrt(3) * sigma[i]
            x[:, i] = (b - a) * stats.norm.cdf(u[:, i]) + a
    return x
def XtoU(x, mu, sigma, DisType):
    u = np.zeros_like(x)
    for i in range(len(mu)):
        if DisType[i] == 'normal':
            u[:, i] = (x[:, i] - mu[i]) / sigma[i]
        elif DisType[i] == 'uniform':
            a = mu[i] - np.sqrt(3) * sigma[i]
            b = mu[i] + np.sqrt(3) * sigma[i]
            u[:, i] = stats.norm.ppf((x[:, i] - a) / (b - a))
    return u

n_simulations = 10
Results_simulation_list = [None for _ in range(n_simulations)]
for j in range(0, n_simulations):
    print(j)
    Case = j
    mu, sigma, DisType = dis(Case)
    nx = mu.shape[0]
    N_train = 3 * nx
    N = 1000
    np.random.seed(123)
    XX_total = np.random.normal(0, 1, size=(N, nx))
    np.random.seed(456)
    XX_train = np.random.normal(0, 1, size=(N_train, nx))

    XX_train_yuan = UtoX(XX_train, mu, sigma, DisType)
    XX_total_yuan = UtoX(XX_total, mu, sigma, DisType)
    YY_train_yuan = func(XX_train_yuan, Case)
    YY_total_yuan = func(XX_total_yuan, Case)

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
    results_list = []
    results_list.append({'Model': 'TabPFN', 'RMSE': rmse, 'R2': r2, 'MAE': mae})
    ##########################################################################ML
    models = {
        'SVR': SVR(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'ExtraTrees': ExtraTreesRegressor(random_state=42),
        'KNN': KNeighborsRegressor(),
        'Ridge': Ridge(),
        'LightGBM': LGBMRegressor(verbose=-1),
        'CatBoost': CatBoostRegressor(verbose=0),
        'XGBoost': XGBRegressor()
    }
    for name, model in models.items():
        model.fit(xx_train, yy_train.ravel())
        Y_test_prediction_initial = model.predict(xx_total) * Sigma_scaler + Mu_scaler
        rmse = 1 - root_mean_squared_error(YY_total_yuan, Y_test_prediction_initial) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        mae = 1 - mean_absolute_error(YY_total_yuan, Y_test_prediction_initial) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        r2 = r2_score(YY_total_yuan, Y_test_prediction_initial)
        results_list.append({'Model': name, 'RMSE': rmse, 'R2': r2, 'MAE': mae})
    ##########################################################################AutoGluon
    TRAIN_Autogluon = pd.DataFrame(xx_train, columns=[f'feature_{i}' for i in range(nx)])
    TRAIN_Autogluon['target'] = yy_train
    predictor = TabularPredictor(
        label="target",
        problem_type="regression",
        path="autogluon_model"
    )
    predictor.fit(train_data=TRAIN_Autogluon, time_limit=600)
    TEST_Autogluon = pd.DataFrame(xx_total, columns=[f'feature_{i}' for i in range(nx)])
    Y_test_prediction_initial = predictor.predict(TEST_Autogluon).to_numpy() * Sigma_scaler + Mu_scaler
    rmse = 1 - root_mean_squared_error(YY_total_yuan, Y_test_prediction_initial) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
    mae = 1 - mean_absolute_error(YY_total_yuan, Y_test_prediction_initial) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
    r2 = r2_score(YY_total_yuan, Y_test_prediction_initial)
    results_list.append({'Model': 'AutoGluon', 'RMSE': rmse, 'R2': r2, 'MAE': mae})
    ##########################################################################MFTabPFN
    model_RCNN = RCNN_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay, epochs, hidden_channels,activate_function, batch)
    model_RCNN.eval()
    YY_train_middle = model_RCNN(torch.tensor(xx_train, dtype=torch.float))
    YY_test_middle = model_RCNN(torch.tensor(xx_total, dtype=torch.float))

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
    results_list.append({'Model': 'MFTabPFN', 'RMSE': rmse, 'R2': r2, 'MAE': mae})
    ##########################################################################MLP
    epochs_ann = 500
    model_SingleANN = SingleANN_PM(INPUT, OUTPUT, bili, n_layer, lr, weight_decay, epochs_ann, hidden_channels,activate_function, batch)
    model_SingleANN.eval()
    ANN_prediction_tensor = model_SingleANN(torch.tensor(xx_total, dtype=torch.float))
    ANN_prediction = ANN_prediction_tensor.detach().cpu().numpy() * Sigma_scaler + Mu_scaler
    rmse = 1 - root_mean_squared_error(YY_total_yuan, ANN_prediction) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
    mae = 1 - mean_absolute_error(YY_total_yuan, ANN_prediction) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
    r2 = r2_score(YY_total_yuan, ANN_prediction)
    results_list.append({'Model': 'ANN', 'RMSE': rmse, 'R2': r2, 'MAE': mae})
    Results_simulation_list[j] = pd.DataFrame(results_list)
    # with open('Results_simulation_list.pkl', 'wb') as f:
    #     pickle.dump(Results_simulation_list, f)
    # print("Results_simulation_list is saved to 'Results_simulation_list.pkl'")
