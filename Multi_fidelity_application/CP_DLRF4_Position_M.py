import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib
matplotlib.use('TkAgg')
from tabpfn import TabPFNRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from RCNN_MFTabPFN_M import RCNN_M
from MLP_MFTabPFN_M import MLP_M
from pathlib import Path
from TabPFN_model import TabPFN_model_main
from autogluon.tabular import TabularPredictor
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import time
import torch
if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')


WORK_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "Multi-fidelity" / "DLRF4" / "DLR_F4_Data"

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


MA = np.array([0.75])
AFA = np.array([0.18])
Y_TARGET = np.array([0.185, 0.331, 0.512, 0.844])
training_simulation_index = [0, 1, 2, 3]

n_performances = 4
n_simulations = 3
Results_DLRF4_list = [[None for _ in range(n_simulations)] for _ in range(n_performances)]
Results_DLRF4_prediction = [[None for _ in range(n_simulations)] for _ in range(n_performances)]
Results_DLRF4_input = [None for _ in range(n_performances)]
random_seeds_train = np.array([111, 222, 333])

time_limit1 = 3600

for j in range(0, n_performances):
    Case = j + 1 # 1: testing data at y/b=0.185; 2: testing data at y/b=0.331; 3: testing data at y/b=0.512; 4: testing data at y/b=0.844
    if Case == 1:
        training_experiment_index = [1, 2, 3]
        testing_index = [0]
    elif Case == 2:
        training_experiment_index = [0, 2, 3]
        testing_index = [1]
    elif Case == 3:
        training_experiment_index = [0, 1, 3]
        testing_index = [2]
    elif Case == 4:
        training_experiment_index = [0, 1, 2]
        testing_index = [3]

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
        lower_experiment_data[f'Experiment_m{Ma:.4f}_a{afa:.3f}_yb{target:.3f}_lower.dat'] = lower_experiment[lower_experiment['y/b']==target]
        upper_experiment_data[f'Experiment_m{Ma:.4f}_a{afa:.3f}_yb{target:.3f}_upper.dat'] = upper_experiment[upper_experiment['y/b']==target]
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
        lower_experiment_data1[f'Experiment_m{Ma:.4f}_a{afa:.3f}_yb{target:.3f}_lower.dat'] = lower_experiment1[lower_experiment1['y/b']==target]
        upper_experiment_data1[f'Experiment_m{Ma:.4f}_a{afa:.3f}_yb{target:.3f}_upper.dat'] = upper_experiment1[upper_experiment1['y/b']==target]
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
    n_layer = 2
    activate_function = 'tanh'# 'tanh'; 'sigmoid'; 'relu'
    lr = 0.001
    weight_decay = 1e-3
    epochs = 100
    bili = 1.0
    task_type = "regressor"
    batch = 256

    results_input = []
    results_input.append({'testing_index': testing_index, 'MA': MA, 'AFA': AFA, 'Y_TARGET': Y_TARGET,
                          'XX_total_yuan': XX_total_yuan, 'XX_train_yuan_low': XX_train_yuan_low,
                          'XX_train_yuan_high': XX_train_yuan_high, 'YY_train_yuan_low': YY_train_yuan_low,
                          'YY_train_yuan_high': YY_train_yuan_high, 'YY_total_yuan_high': YY_total_yuan_high})
    for k in range(n_simulations):
        print(j * 10 + k)
        random_seeds = random_seeds_train[k]
        np.random.seed(random_seeds)
        ##########################################################################TabPFN-high
        reg = TabPFNRegressor(device="cuda", random_state=random_seeds)
        start_time = time.perf_counter()
        reg.fit(xx_train[N_low:, :], yy_train[N_low:].ravel())
        train_time_default = time.perf_counter() - start_time
        qidong = reg.predict(xx_total)
        start_time = time.perf_counter()
        YY_test_prediction_initial = reg.predict(xx_total)
        pred_time_default = time.perf_counter() - start_time
        Y_test_prediction_initial_default = YY_test_prediction_initial * Sigma_scaler + Mu_scaler
        rmse_default = root_mean_squared_error(YY_total_yuan_high, Y_test_prediction_initial_default)
        mae_default = mean_absolute_error(YY_total_yuan_high, Y_test_prediction_initial_default)
        r2_default = r2_score(YY_total_yuan_high, Y_test_prediction_initial_default)
        rmse_normalized_default = 1 - rmse_default / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
        mae_normalized_default = 1 - mae_default / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
        results_list = []
        results_list.append({'Model': 'TabPFN-High', 'RMSE': rmse_default, 'R2': r2_default, 'MAE': mae_default, 'RMSE_N': rmse_normalized_default, 'MAE_N': mae_normalized_default, 'Time_Train': train_time_default, 'Time_Pred': pred_time_default})
        Y_test_prediction_TabPFN_high = Y_test_prediction_initial_default.copy()
        #########for plot
        YY_test_prediction_initial = reg.predict(xx_train[:N_low, :], output_type="full")
        Y_test_prediction_initial = YY_test_prediction_initial[f"mean"] * Sigma_scaler + Mu_scaler
        Variance_initial = YY_test_prediction_initial[f"variance"].reshape(-1, 1)
        Prediction_sigma_initial = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)
        LOWER_SIGMA_TabPFN_high = Y_test_prediction_initial.ravel() - 1.00 * Prediction_sigma_initial
        UPPER_SIGMA_TabPFN_high = Y_test_prediction_initial.ravel() + 1.00 * Prediction_sigma_initial
        Y_test_prediction_TabPFN_high_PLOT = Y_test_prediction_initial.copy()
        results_prediction = []
        results_prediction.append({'Model': 'TabPFN-High', 'Y_test_prediction': Y_test_prediction_TabPFN_high, 'Y_test_prediction_PLOT': Y_test_prediction_TabPFN_high_PLOT,
                                   'LOWER_SIGMA': LOWER_SIGMA_TabPFN_high, 'UPPER_SIGMA': UPPER_SIGMA_TabPFN_high})
        ##########################################################################TabPFN-multi
        reg = TabPFNRegressor(device="cuda", random_state=random_seeds)
        start_time = time.perf_counter()
        reg.fit(xx_train, yy_train.ravel())
        train_time_default = time.perf_counter() - start_time
        qidong = reg.predict(xx_total)
        start_time = time.perf_counter()
        YY_test_prediction_initial = reg.predict(xx_total)
        pred_time_default = time.perf_counter() - start_time
        Y_test_prediction_initial_default = YY_test_prediction_initial * Sigma_scaler + Mu_scaler
        rmse_default = root_mean_squared_error(YY_total_yuan_high, Y_test_prediction_initial_default)
        mae_default = mean_absolute_error(YY_total_yuan_high, Y_test_prediction_initial_default)
        r2_default = r2_score(YY_total_yuan_high, Y_test_prediction_initial_default)
        rmse_normalized_default = 1 - rmse_default / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
        mae_normalized_default = 1 - mae_default / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
        results_list.append({'Model': 'TabPFN-Multi', 'RMSE': rmse_default, 'R2': r2_default, 'MAE': mae_default, 'RMSE_N': rmse_normalized_default, 'MAE_N': mae_normalized_default, 'Time_Train': train_time_default, 'Time_Pred': pred_time_default})
        Y_test_prediction_TabPFN_multi = Y_test_prediction_initial_default.copy()
        #########for plot
        YY_test_prediction_initial = reg.predict(xx_train[:N_low, :], output_type="full")
        Y_test_prediction_initial = YY_test_prediction_initial[f"mean"] * Sigma_scaler + Mu_scaler
        Variance_initial = YY_test_prediction_initial[f"variance"].reshape(-1, 1)
        Prediction_sigma_initial = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)
        LOWER_SIGMA_TabPFN_multi = Y_test_prediction_initial.ravel() - 1.00 * Prediction_sigma_initial
        UPPER_SIGMA_TabPFN_multi = Y_test_prediction_initial.ravel() + 1.00 * Prediction_sigma_initial
        Y_test_prediction_TabPFN_multi_PLOT = Y_test_prediction_initial.copy()
        results_prediction.append({'Model': 'TabPFN-Multi', 'Y_test_prediction': Y_test_prediction_TabPFN_multi, 'Y_test_prediction_PLOT': Y_test_prediction_TabPFN_multi_PLOT,
                                   'LOWER_SIGMA': LOWER_SIGMA_TabPFN_multi, 'UPPER_SIGMA': UPPER_SIGMA_TabPFN_multi})
        ##########################################################################MFTabPFN
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
                                    hidden_channels, activate_function, batch, N_low, N_high, TabPFN_prediction_initial11111,
                                    random_seeds, alpha, alpha_index)
            train_time = time.perf_counter() - start_time
            model_RCNN.eval()
            start_time = time.perf_counter()
            YY_train_middle = model_RCNN(torch.tensor(xx_train[N_low:, :], dtype=torch.float))
            YY_test_middle = model_RCNN(torch.tensor(xx_total, dtype=torch.float))
            TabPFN_prediction_initial_full = reg.predict(xx_total)
            TabPFN_prediction_initial = TabPFN_prediction_initial_full.reshape(-1, 1)
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
            Y_test_prediction = (TabPFN_prediction_tensor.detach().cpu().numpy() + model_RCNN.afa.detach().cpu().numpy() * TabPFN_prediction_tensor_initial.detach().cpu().numpy()) * Sigma_scaler + Mu_scaler
            pred_time = time.perf_counter() - start_time
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
        results_list.append({'Model': 'MFTabPFN', 'RMSE': rmse, 'R2': r2, 'MAE': mae, 'RMSE_N': rmse_normalized, 'MAE_N': mae_normalized, 'Time_Train': train_time, 'Time_Pred': pred_time})
        #########for plot
        if loss_train > 0.01:
            model_RCNN.eval()
            YY_test_middle = model_RCNN(torch.tensor(xx_train[:N_low, :], dtype=torch.float))
            TabPFN_prediction_initial_full = reg.predict(xx_train[:N_low, :], output_type="full")
            TabPFN_prediction_initial = TabPFN_prediction_initial_full[f"mean"].reshape(-1, 1)
            TabPFN_prediction_tensor_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
            Variance_initial = TabPFN_prediction_initial_full[f"variance"].reshape(-1, 1)
            Prediction_sigma_initial = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)
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
            Y_test_prediction = (TabPFN_prediction_tensor.detach().cpu().numpy() + model_RCNN.afa.detach().cpu().numpy() * TabPFN_prediction_tensor_initial.detach().cpu().numpy()) * Sigma_scaler + Mu_scaler
            Prediction_sigma = np.sqrt(Variance.detach().cpu().numpy().ravel() * Sigma_scaler ** 2)
            Prediction_total_sigma2 = model_RCNN.afa.detach().cpu().numpy() ** 2 * Prediction_sigma_initial ** 2 + Prediction_sigma ** 2
            LOWER_SIGMA = Y_test_prediction.ravel() - 1.00 * np.sqrt(Prediction_total_sigma2)
            UPPER_SIGMA = Y_test_prediction.ravel() + 1.00 * np.sqrt(Prediction_total_sigma2)
            Y_test_prediction_MFTabPFN_PLOT = Y_test_prediction.copy()
        else:
            YY_test_prediction_initial = reg.predict(xx_train[:N_low, :], output_type="full")
            Y_test_prediction = YY_test_prediction_initial[f"mean"] * Sigma_scaler + Mu_scaler
            Variance = YY_test_prediction_initial[f"variance"].reshape(-1, 1)
            Prediction_sigma = np.sqrt(Variance.ravel() * Sigma_scaler ** 2)
            LOWER_SIGMA = Y_test_prediction.ravel() - 1.00 * Prediction_sigma
            UPPER_SIGMA = Y_test_prediction.ravel() + 1.00 * Prediction_sigma
            Y_test_prediction_MFTabPFN_PLOT = Y_test_prediction.copy()
        results_prediction.append({'Model': 'MFTabPFN', 'Y_test_prediction': Y_test_prediction_MFTabPFN, 'Y_test_prediction_PLOT': Y_test_prediction_MFTabPFN_PLOT,
                                   'LOWER_SIGMA': LOWER_SIGMA, 'UPPER_SIGMA': UPPER_SIGMA,
                                   'Y_test_prediction_low': Y_test_prediction_MFTabPFN_low})
        ##########################################################################AutoGluon-high
        TRAIN_Autogluon = pd.DataFrame(xx_train[N_low:, :], columns=[f'feature_{i}' for i in range(nx)])
        TRAIN_Autogluon['target'] = yy_train[N_low:]
        TEST_Autogluon = pd.DataFrame(xx_total, columns=[f'feature_{i}' for i in range(nx)])
        start_time = time.perf_counter()
        predictor = TabularPredictor(
            label="target",
            problem_type="regression",
        )
        predictor.fit(train_data=TRAIN_Autogluon, time_limit=time_limit1, presets='best_quality')
        train_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        Y_test_prediction_initial = predictor.predict(TEST_Autogluon).to_numpy() * Sigma_scaler + Mu_scaler
        pred_time = time.perf_counter() - start_time
        rmse = root_mean_squared_error(YY_total_yuan_high, Y_test_prediction_initial)
        mae = mean_absolute_error(YY_total_yuan_high, Y_test_prediction_initial)
        r2 = r2_score(YY_total_yuan_high, Y_test_prediction_initial)
        rmse_normalized = 1 - rmse / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
        mae_normalized = 1 - mae / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
        results_list.append({'Model': 'AutoGluon-High', 'RMSE': rmse, 'R2': r2, 'MAE': mae, 'RMSE_N': rmse_normalized, 'MAE_N': mae_normalized, 'Time_Train': train_time, 'Time_Pred': pred_time})
        Y_test_prediction_AutoGluon_high = Y_test_prediction_initial.copy()
        #########for plot
        TEST_Autogluon = pd.DataFrame(xx_train[:N_low, :], columns=[f'feature_{i}' for i in range(nx)])
        Y_test_prediction_initial = predictor.predict(TEST_Autogluon).to_numpy() * Sigma_scaler + Mu_scaler
        Y_test_prediction_AutoGluon_high_PLOT = Y_test_prediction_initial.copy()
        results_prediction.append({'Model': 'AutoGluon-High', 'Y_test_prediction': Y_test_prediction_AutoGluon_high, 'Y_test_prediction_PLOT': Y_test_prediction_AutoGluon_high_PLOT})
        ##########################################################################AutoGluon-multi
        TRAIN_Autogluon = pd.DataFrame(xx_train, columns=[f'feature_{i}' for i in range(nx)])
        TRAIN_Autogluon['target'] = yy_train
        TEST_Autogluon = pd.DataFrame(xx_total, columns=[f'feature_{i}' for i in range(nx)])
        start_time = time.perf_counter()
        predictor = TabularPredictor(
            label="target",
            problem_type="regression",
        )
        predictor.fit(train_data=TRAIN_Autogluon, time_limit=time_limit1, presets='best_quality')
        train_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        Y_test_prediction_initial = predictor.predict(TEST_Autogluon).to_numpy() * Sigma_scaler + Mu_scaler
        pred_time = time.perf_counter() - start_time
        rmse = root_mean_squared_error(YY_total_yuan_high, Y_test_prediction_initial)
        mae = mean_absolute_error(YY_total_yuan_high, Y_test_prediction_initial)
        r2 = r2_score(YY_total_yuan_high, Y_test_prediction_initial)
        rmse_normalized = 1 - rmse / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
        mae_normalized = 1 - mae / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
        results_list.append({'Model': 'AutoGluon-Multi', 'RMSE': rmse, 'R2': r2, 'MAE': mae, 'RMSE_N': rmse_normalized, 'MAE_N': mae_normalized, 'Time_Train': train_time, 'Time_Pred': pred_time})
        Y_test_prediction_AutoGluon_multi = Y_test_prediction_initial.copy()
        #########for plot
        TEST_Autogluon = pd.DataFrame(xx_train[:N_low, :], columns=[f'feature_{i}' for i in range(nx)])
        Y_test_prediction_initial = predictor.predict(TEST_Autogluon).to_numpy() * Sigma_scaler + Mu_scaler
        Y_test_prediction_AutoGluon_multi_PLOT = Y_test_prediction_initial.copy()
        results_prediction.append({'Model': 'AutoGluon-Multi', 'Y_test_prediction': Y_test_prediction_AutoGluon_multi, 'Y_test_prediction_PLOT': Y_test_prediction_AutoGluon_multi_PLOT})

        Results_DLRF4_list[j][k] = pd.DataFrame(results_list)
        Results_DLRF4_prediction[j][k] = pd.DataFrame(results_prediction)
        Results_DLRF4_input[j] = pd.DataFrame(results_input)
        with open('Results_DLRF4_list.pkl', 'wb') as f:
            pickle.dump(Results_DLRF4_list, f)
        with open('Results_DLRF4_prediction.pkl', 'wb') as f:
            pickle.dump(Results_DLRF4_prediction, f)
        with open('Results_DLRF4_input.pkl', 'wb') as f:
            pickle.dump(Results_DLRF4_input, f)

