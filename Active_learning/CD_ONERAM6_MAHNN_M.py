import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
from ANN_model import SingleANN_M
from ANN_NoAct_model import SingleANN_NoAct_M
from RBFNN_model import SingleRBFNN_M
from pathlib import Path
import pickle
import time
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
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

SAVE_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "Multi-fidelity" / "ONERAM6" / "Result"
pkl_file = os.path.join(SAVE_DIR, 'Results_ONERAM6_list.pkl')
with open(pkl_file, 'rb') as f:
    Results_ONERAM6_list = pickle.load(f)
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

n_stop = len(stop0)
n_simulations = 3
Results_ONERAM6_list_MAHNN = [[None for _ in range(n_simulations)] for _ in range(n_stop)]
Results_ONERAM6_prediction_MAHNN = [[None for _ in range(n_simulations)] for _ in range(n_stop)]
random_seeds_train = np.array([111, 222, 333])

N_low_list = np.zeros((3,len(stop0)), dtype=int)
N_high_list = np.zeros((3,len(stop0)), dtype=int)
for j in range(0, n_simulations):
    random_seeds = random_seeds_train[j]
    np.random.seed(random_seeds)
    N_low1 = 50
    N_high1 = 25
    N = 1000
    indices_low = np.random.choice(N, size=N_low1, replace=False)
    indices_high = np.random.choice(N, size=N_high1, replace=False)
    for k in range(0, n_stop):
        Results_ONERAM6_list111 = Results_ONERAM6_list[j][:stop[j, k]]
        index1 = Results_ONERAM6_list111.query("Index_fidelity == 0")['Index_max'].to_numpy()
        index_low = np.concatenate([indices_low, index1])
        index2 = Results_ONERAM6_list111.query("Index_fidelity == 1")['Index_max'].to_numpy()
        index_high = np.concatenate([indices_high, index2])
        unique_count_low = len(np.unique(index_low))
        unique_count_high = len(np.unique(index_high))
        N_low_list[j, k] = unique_count_low
        N_high_list[j, k] = unique_count_high

for j in range(0, n_simulations):
    for k in range(0, n_stop):
        print(j * 10 + k)
        N_low = N_low_list[j, k]
        N_high = N_high_list[j, k]
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

        hidden_channels1 = 30
        hidden_channels2 = 20
        n_layer = 2
        n_layer1 = 3
        activate_function = 'tanh'
        lr = 0.001
        weight_decay = 1e-3
        epochs = 1000
        bili = 1.0
        batch = 256

        ##########################################################################MA-HNN
        start_time = time.perf_counter()
        model_LOW = SingleANN_M(INPUT[:N_low, :], OUTPUT[:N_low], bili, n_layer1, lr, weight_decay, epochs,
                                hidden_channels1, activate_function, batch, random_seeds)
        model_LOW.eval()
        YY_train_middle1 = model_LOW(INPUT[N_low:, :]).detach().cpu().numpy()
        input_lin = np.concatenate([xx_train[N_low:, :], YY_train_middle1], axis=1)
        INPUT_LIN = torch.tensor(input_lin, dtype=torch.float)
        model_LIN = SingleANN_NoAct_M(INPUT_LIN, OUTPUT[N_low:], bili, n_layer, lr, weight_decay, epochs,
                                      hidden_channels1, batch, random_seeds)
        model_LIN.eval()
        YY_train_middle2 = model_LIN(INPUT_LIN).detach().cpu().numpy()
        OUTPUT_REM = OUTPUT[N_low:] - torch.tensor(YY_train_middle2, dtype=torch.float)
        model_REM = SingleRBFNN_M(INPUT[N_low:, :], OUTPUT_REM, bili, hidden_channels2, lr, weight_decay, epochs, batch,
                                  random_seeds)
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
        results_list = []
        results_list.append(
            {'Model': 'MA-HNN', 'Time_Train': train_time, 'Time_Pred': pred_time, 'N_low': N_low, 'N_high': N_high, 'Stop': stop0[k]})
        Y_test_prediction_MAHNN = Y_test_prediction.copy()
        Y_test_prediction_MAHNN_low = Y_test_prediction_low.copy()
        results_prediction = []
        results_prediction.append({'Model': 'MA-HNN', 'Y_test_prediction': Y_test_prediction_MAHNN,
                                   'Y_test_prediction_low': Y_test_prediction_MAHNN_low})

        Results_ONERAM6_list_MAHNN[j][k] = pd.DataFrame(results_list)
        Results_ONERAM6_prediction_MAHNN[j][k] = pd.DataFrame(results_prediction)
        with open('Results_ONERAM6_list_MAHNN.pkl', 'wb') as f:
            pickle.dump(Results_ONERAM6_list_MAHNN, f)
        with open('Results_ONERAM6_prediction_MAHNN.pkl', 'wb') as f:
            pickle.dump(Results_ONERAM6_prediction_MAHNN, f)

