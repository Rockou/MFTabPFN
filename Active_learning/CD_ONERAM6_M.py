import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tabpfn import TabPFNRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from RCNN_MFTabPFN_M import RCNN_M
from MLP_MFTabPFN_M import MLP_M
from pathlib import Path
from TabPFN_model import TabPFN_model_main
import pickle
import time
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import gc
import torch
if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')

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

def PREPROCESSOR(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        ]), cat_cols)
    ], remainder="drop")
    return preprocessor

mu = np.full(198, 0)
sigma = np.full(198, 0.04 / np.sqrt(12))
DisType = ['uniform'] * 198
nx = mu.shape[0]

n_simulations = 3
Results_ONERAM6_list = [None for _ in range(n_simulations)]
Results_ONERAM6_prediction = [None for _ in range(n_simulations)]
random_seeds_train = np.array([111, 222, 333])

for j in range(0, n_simulations):
    N_low = 50
    N_high = 25
    N = 1000
    random_seeds = random_seeds_train[j]
    np.random.seed(random_seeds)
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

    xx_train1 = pd.DataFrame(XX_train_yuan)
    xx_total1 = pd.DataFrame(XX_total_yuan)
    preprocessor = PREPROCESSOR(xx_train1)
    xx_train1 = preprocessor.fit_transform(xx_train1)
    xx_total1 = preprocessor.transform(xx_total1)

    xx_train = xx_train1
    xx_total = xx_total1
    scaler_Y = StandardScaler()
    yy_train = scaler_Y.fit_transform(YY_train_yuan.reshape(-1, 1))
    Mu_scaler = scaler_Y.mean_
    Sigma_scaler = np.sqrt(scaler_Y.var_)

    INPUT = torch.tensor(xx_train, dtype=torch.float)
    OUTPUT = torch.tensor(yy_train, dtype=torch.float)

    input_TabPFN = np.min([np.max([20, nx]), 500])
    hidden_channels = np.max([128, 2 * nx])
    n_layer = 3
    activate_function = 'tanh'
    lr = 0.001
    weight_decay = 1e-3
    epochs = 100
    bili = 1.0
    task_type = "regressor"
    batch = 256

    ##########################################################################AMFTabPFN
    U = 300
    Cost_low = 1
    Cost_high = 1
    Stop = np.zeros((U, 1), dtype=float)
    Stop0 = 0.04
    m_low = 0
    m_high = 0

    Index_low_list = np.zeros(N, dtype=int)
    Index_low_list[:N_low] = indices_low
    Index_high_list = np.zeros(N, dtype=int)
    Index_high_list[:N_high] = indices_high

    results_list = []
    results_prediction = []
    for i in range(U):
        print(i)
        if i > 0:
            Y_test_prediction_iteration = Y_test_prediction.copy()
            Y_test_prediction_low_afa_iteration = Y_test_prediction_low_afa.copy()
            Y_test_prediction_delt_iteration = Y_test_prediction_delt.copy()

        criterion = torch.nn.MSELoss()
        reg = TabPFNRegressor(device="cuda", random_state=random_seeds)
        start_time = time.perf_counter()
        reg.fit(xx_train[:N_low, :], yy_train[:N_low].ravel())
        TabPFN_prediction_initial = reg.predict(xx_train[N_low:, :]).reshape(-1, 1)
        YY_train_prediction_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
        loss_train = criterion(OUTPUT[N_low:], YY_train_prediction_initial)
        TabPFN_prediction_initial11111 = TabPFN_prediction_initial
        alpha_index = 0
        alpha = 1
        if loss_train > 0.01:
            if nx < 100:
                model_RCNN = MLP_M(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay, epochs,
                                   hidden_channels, activate_function, batch, N_low, N_high, TabPFN_prediction_initial11111,
                                   random_seeds, alpha, alpha_index)
            else:
                model_RCNN = RCNN_M(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay, epochs,
                                    hidden_channels, activate_function, batch, N_low, N_high,
                                    TabPFN_prediction_initial11111,
                                    random_seeds, alpha, alpha_index)
            train_time = time.perf_counter() - start_time
            model_RCNN.eval()
            start_time = time.perf_counter()
            YY_train_middle = model_RCNN(torch.tensor(xx_train[N_low:, :], dtype=torch.float))
            YY_test_middle = model_RCNN(torch.tensor(xx_total, dtype=torch.float))
            TabPFN_prediction_initial_full = reg.predict(xx_total, output_type="full")
            TabPFN_prediction_initial = TabPFN_prediction_initial_full[f"mean"].reshape(-1, 1)
            Variance_initial = TabPFN_prediction_initial_full[f"variance"].reshape(-1, 1)
            Prediction_sigma_initial = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)
            TabPFN_prediction_tensor_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
            TabPFN_prediction_initial0 = reg.predict(xx_train[N_low:, :]).reshape(-1, 1)
            TabPFN_prediction_tensor_initial0 = torch.tensor(TabPFN_prediction_initial0, dtype=torch.float)
            save_path_to_fine_tuned_model = (Path(__file__).parent / f"tabpfn-v2-{task_type}.ckpt")
            TabPFN_prediction_tensor, Lower, Upper, Variance = TabPFN_model_main(
                path_to_base_model="auto",
                save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
                X_train=YY_train_middle,
                y_train=OUTPUT[N_low:] - model_RCNN.afa * TabPFN_prediction_tensor_initial0.to('cpu'),
                X_test=YY_test_middle,
                n_classes=None,
                categorical_features_index=None,
                task_type=task_type,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            Y_test_prediction_delt = TabPFN_prediction_tensor.detach().cpu().numpy() * Sigma_scaler
            Y_test_prediction_low_afa = model_RCNN.afa.detach().cpu().numpy() * TabPFN_prediction_tensor_initial.detach().cpu().numpy() * Sigma_scaler + Mu_scaler
            Y_test_prediction = (TabPFN_prediction_tensor.detach().cpu().numpy() + model_RCNN.afa.detach().cpu().numpy() * TabPFN_prediction_tensor_initial.detach().cpu().numpy()) * Sigma_scaler + Mu_scaler
            pred_time = time.perf_counter() - start_time
            Prediction_sigma = np.sqrt(Variance.detach().cpu().numpy().ravel() * Sigma_scaler ** 2)
            Prediction_total_sigma = np.sqrt(model_RCNN.afa.detach().cpu().numpy() ** 2 * Prediction_sigma_initial ** 2 + Prediction_sigma ** 2)
            rmse = root_mean_squared_error(YY_total_yuan_high, Y_test_prediction)
            mae = mean_absolute_error(YY_total_yuan_high, Y_test_prediction)
            r2 = r2_score(YY_total_yuan_high, Y_test_prediction)
            rmse_normalized = 1 - rmse / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
            mae_normalized = 1 - mae / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
            Y_test_prediction_MFTabPFN = Y_test_prediction.copy()
            Y_test_prediction_MFTabPFN_low = TabPFN_prediction_initial.ravel() * Sigma_scaler + Mu_scaler
        else:
            reg.fit(xx_train, yy_train.ravel())
            train_time = time.perf_counter() - start_time
            start_time = time.perf_counter()
            YY_test_prediction_initial = reg.predict(xx_total)
            pred_time = time.perf_counter() - start_time
            Y_test_prediction = YY_test_prediction_initial * Sigma_scaler + Mu_scaler
            rmse = root_mean_squared_error(YY_total_yuan_high, Y_test_prediction)
            mae = mean_absolute_error(YY_total_yuan_high, Y_test_prediction)
            r2 = r2_score(YY_total_yuan_high, Y_test_prediction)
            rmse_normalized = 1 - rmse / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
            mae_normalized = 1 - mae / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
            Y_test_prediction_MFTabPFN = Y_test_prediction.copy()
            Y_test_prediction_MFTabPFN_low = Y_test_prediction_MFTabPFN

        if i > 0:
            Stop[i] = np.mean(np.abs(Y_test_prediction - Y_test_prediction_iteration))/(np.max(Y_test_prediction) - np.min(Y_test_prediction))
            # if Stop[i] <= Stop0:
            #     break

        Prediction_sigma_initial_mean = np.mean(model_RCNN.afa.detach().cpu().numpy() * Prediction_sigma_initial)
        Prediction_sigma_mean = np.mean(Prediction_sigma)

        distances_low = np.sum(np.abs(xx_total[:, np.newaxis] - xx_train[:N_low, :]), axis=2)
        min_distances_low = np.min(distances_low, axis=1)
        distances_high = np.sum(np.abs(xx_total[:, np.newaxis] - xx_train[N_low:, :]), axis=2)
        min_distances_high = np.min(distances_high, axis=1)

        if i<25:
            Prediction_sigma_stack = np.stack((min_distances_low * model_RCNN.afa.detach().cpu().numpy() * Prediction_sigma_initial * Prediction_sigma_initial_mean / Cost_low, min_distances_high * Prediction_sigma * Prediction_sigma_mean / Cost_high), axis=1)
        else:
            Prediction_sigma_stack = np.stack((model_RCNN.afa.detach().cpu().numpy() * Prediction_sigma_initial * Prediction_sigma_initial_mean / Cost_low, Prediction_sigma * Prediction_sigma_mean / Cost_high),axis=1)

        Prediction_sigma_stack[Index_low_list[:N_low], 0] = 0
        Prediction_sigma_stack[Index_high_list[:N_high], 1] = 0

        max_value = np.max(Prediction_sigma_stack)
        max_indices = np.where(Prediction_sigma_stack == max_value)
        index_fidelity = max_indices[1][0]
        index_max = max_indices[0][0]

        if index_fidelity == 0:
            Index_low_list[N_low] = index_max
        elif index_fidelity == 1:
            Index_high_list[N_high] = index_max

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

        results_list.append(
            {'Model': 'MFTabPFN', 'RMSE': rmse, 'R2': r2, 'MAE': mae, 'RMSE_N': rmse_normalized, 'MAE_N': mae_normalized,
             'Time_Train': train_time, 'Time_Pred': pred_time, 'Uncertainty_max': max_value, 'Index_fidelity': index_fidelity,
             'Index_max': index_max, 'N_low': N_low, 'N_high': N_high, 'Stop': Stop[i], 'Alpha': model_RCNN.afa.detach().cpu().numpy()})

        results_prediction.append({'Model': 'MFTabPFN', 'Performance': Y_test_prediction_MFTabPFN, 'Performance_low': Y_test_prediction_MFTabPFN_low,
                                   'Prediction_sigma_initial': Prediction_sigma_initial, 'Prediction_sigma': Prediction_sigma, 'Prediction_total_sigma': Prediction_total_sigma})

        XX_train_yuan = np.concatenate((XX_train_yuan_low, XX_train_yuan_high), axis=0)
        YY_train_yuan = np.concatenate((YY_train_yuan_low, YY_train_yuan_high), axis=0)

        xx_train1 = pd.DataFrame(XX_train_yuan)
        xx_total1 = pd.DataFrame(XX_total_yuan)
        preprocessor = PREPROCESSOR(xx_train1)
        xx_train1 = preprocessor.fit_transform(xx_train1)
        xx_total1 = preprocessor.transform(xx_total1)

        xx_train = xx_train1
        xx_total = xx_total1
        scaler_Y = StandardScaler()
        yy_train = scaler_Y.fit_transform(YY_train_yuan.reshape(-1, 1))
        Mu_scaler = scaler_Y.mean_
        Sigma_scaler = np.sqrt(scaler_Y.var_)

        INPUT = torch.tensor(xx_train, dtype=torch.float)
        OUTPUT = torch.tensor(yy_train, dtype=torch.float)

        del reg, model_RCNN
        torch.cuda.empty_cache()
        gc.collect()

    Results_ONERAM6_list[j] = pd.DataFrame(results_list)
    Results_ONERAM6_prediction[j] = pd.DataFrame(results_prediction)
    with open('Results_ONERAM6_list.pkl', 'wb') as f:
        pickle.dump(Results_ONERAM6_list, f)
    with open('Results_ONERAM6_prediction.pkl', 'wb') as f:
        pickle.dump(Results_ONERAM6_prediction, f)

    del xx_train, xx_total, yy_train
    del INPUT, OUTPUT, results_list, results_prediction
    del preprocessor, scaler_Y
    gc.collect()
    torch.cuda.empty_cache()


