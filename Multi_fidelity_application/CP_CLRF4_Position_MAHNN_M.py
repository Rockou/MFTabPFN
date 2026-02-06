import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from ANN_model import SingleANN_M
from ANN_NoAct_model import SingleANN_NoAct_M
from RBFNN_model import SingleRBFNN_M
from pathlib import Path
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
Results_DLRF4_list_MAHNN = [[None for _ in range(n_simulations)] for _ in range(n_performances)]
Results_DLRF4_prediction_MAHNN = [[None for _ in range(n_simulations)] for _ in range(n_performances)]
Results_DLRF4_input_MAHNN = [None for _ in range(n_performances)]
random_seeds_train = np.array([111, 222, 333])

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

    hidden_channels1 = 30
    hidden_channels2 = 20
    n_layer = 2
    n_layer1 = 3
    activate_function = 'tanh'# 'tanh'; 'sigmoid'; 'relu'
    lr = 0.001
    weight_decay = 1e-3
    epochs = 1000
    bili = 1.0
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
        ##########################################################################MA-HNN
        start_time = time.perf_counter()
        model_LOW = SingleANN_M(INPUT[:N_low, :], OUTPUT[:N_low], bili, n_layer1, lr, weight_decay, epochs, hidden_channels1, activate_function, batch, random_seeds)
        model_LOW.eval()
        YY_train_middle1 = model_LOW(INPUT[N_low:, :]).detach().cpu().numpy()
        input_lin = np.concatenate([xx_train[N_low:, :], YY_train_middle1], axis=1)
        INPUT_LIN = torch.tensor(input_lin, dtype=torch.float)
        model_LIN = SingleANN_NoAct_M(INPUT_LIN, OUTPUT[N_low:], bili, n_layer, lr, weight_decay, epochs, hidden_channels1, batch, random_seeds)
        model_LIN.eval()
        YY_train_middle2 = model_LIN(INPUT_LIN).detach().cpu().numpy()
        OUTPUT_REM = OUTPUT[N_low:] - torch.tensor(YY_train_middle2, dtype=torch.float)
        model_REM = SingleRBFNN_M(INPUT[N_low:, :], OUTPUT_REM, bili, hidden_channels2, lr, weight_decay, epochs, batch, random_seeds)
        train_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        model_REM.eval()
        YY_train_middle_test1 = model_LOW(torch.tensor(xx_total, dtype=torch.float)).detach().cpu().numpy()
        input_lin_test = np.concatenate([torch.tensor(xx_total, dtype=torch.float), YY_train_middle_test1], axis=1)
        INPUT_LIN_test = torch.tensor(input_lin_test, dtype=torch.float)
        YY_train_middle_test2 = model_LIN(INPUT_LIN_test).detach().cpu().numpy()
        YY_train_middle_test3 = model_REM(torch.tensor(xx_total, dtype=torch.float)).detach().cpu().numpy()
        Y_test_prediction = (YY_train_middle_test2 + YY_train_middle_test3) * Sigma_scaler + Mu_scaler
        pred_time = time.perf_counter() - start_time
        Y_test_prediction_low = YY_train_middle_test1 * Sigma_scaler + Mu_scaler
        rmse = root_mean_squared_error(YY_total_yuan_high, Y_test_prediction)
        mae = mean_absolute_error(YY_total_yuan_high, Y_test_prediction)
        r2 = r2_score(YY_total_yuan_high, Y_test_prediction)
        rmse_normalized = 1 - rmse / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
        mae_normalized = 1 - mae / (np.max(YY_total_yuan_high) - np.min(YY_total_yuan_high))
        results_list = []
        results_list.append({'Model': 'MA-HNN', 'RMSE': rmse, 'R2': r2, 'MAE': mae, 'RMSE_N': rmse_normalized, 'MAE_N': mae_normalized, 'Time_Train': train_time, 'Time_Pred': pred_time})
        Y_test_prediction_MAHNN = Y_test_prediction.copy()
        Y_test_prediction_MAHNN_low = Y_test_prediction_low.copy()
        #########for plot
        YY_train_middle_test1 = model_LOW(torch.tensor(xx_train[:N_low, :], dtype=torch.float)).detach().cpu().numpy()
        input_lin_test = np.concatenate([torch.tensor(xx_train[:N_low, :], dtype=torch.float), YY_train_middle_test1], axis=1)
        INPUT_LIN_test = torch.tensor(input_lin_test, dtype=torch.float)
        YY_train_middle_test2 = model_LIN(INPUT_LIN_test).detach().cpu().numpy()
        YY_train_middle_test3 = model_REM(torch.tensor(xx_train[:N_low, :], dtype=torch.float)).detach().cpu().numpy()
        Y_test_prediction = (YY_train_middle_test2 + YY_train_middle_test3) * Sigma_scaler + Mu_scaler
        Y_test_prediction_MAHNN_PLOT = Y_test_prediction.copy()
        results_prediction = []
        results_prediction.append({'Model': 'MA-HNN', 'Y_test_prediction': Y_test_prediction_MAHNN, 'Y_test_prediction_PLOT': Y_test_prediction_MAHNN_PLOT,
                                   'Y_test_prediction_low': Y_test_prediction_MAHNN_low})

        Results_DLRF4_list_MAHNN[j][k] = pd.DataFrame(results_list)
        Results_DLRF4_prediction_MAHNN[j][k] = pd.DataFrame(results_prediction)
        Results_DLRF4_input_MAHNN[j] = pd.DataFrame(results_input)
        with open('Results_DLRF4_list_MAHNN.pkl', 'wb') as f:
            pickle.dump(Results_DLRF4_list_MAHNN, f)
        with open('Results_DLRF4_prediction_MAHNN.pkl', 'wb') as f:
            pickle.dump(Results_DLRF4_prediction_MAHNN, f)
        with open('Results_DLRF4_input_MAHNN.pkl', 'wb') as f:
            pickle.dump(Results_DLRF4_input_MAHNN, f)

