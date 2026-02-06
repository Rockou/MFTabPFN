import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tabpfn import TabPFNRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
from RCNN_MFTabPFN_M import RCNN_M
from MLP_MFTabPFN_M import MLP_M
from pathlib import Path
from TabPFN_model import TabPFN_model_main
from autogluon.tabular import TabularPredictor
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
Results_ONERAM6_list_ML = [[None for _ in range(n_simulations)] for _ in range(n_stop)]
Results_ONERAM6_prediction_ML = [[None for _ in range(n_simulations)] for _ in range(n_stop)]
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
        results_list = []
        results_list.append({'Model': 'TabPFN-High', 'Time_Train': train_time_default, 'Time_Pred': pred_time_default})
        Y_test_prediction_TabPFN_high = Y_test_prediction_initial_default.copy()
        results_prediction = []
        results_prediction.append({'Model': 'TabPFN-High', 'Y_test_prediction': Y_test_prediction_TabPFN_high})
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
        results_list.append({'Model': 'TabPFN-Multi', 'Time_Train': train_time_default, 'Time_Pred': pred_time_default})
        Y_test_prediction_TabPFN_multi = Y_test_prediction_initial_default.copy()
        results_prediction.append({'Model': 'TabPFN-Multi', 'Y_test_prediction': Y_test_prediction_TabPFN_multi})
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
                                    hidden_channels, activate_function, batch, N_low, N_high,
                                    TabPFN_prediction_initial11111,
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
            Y_test_prediction_MFTabPFN = Y_test_prediction.copy()
            Y_test_prediction_MFTabPFN_low = TabPFN_prediction_initial.ravel() * Sigma_scaler + Mu_scaler
        else:
            reg.fit(xx_train, yy_train.ravel())
            train_time = time.perf_counter() - start_time
            start_time = time.perf_counter()
            YY_test_prediction_initial = reg.predict(xx_total)
            pred_time = time.perf_counter() - start_time
            Y_test_prediction = YY_test_prediction_initial * Sigma_scaler + Mu_scaler
            Y_test_prediction_MFTabPFN = Y_test_prediction.copy()
            Y_test_prediction_MFTabPFN_low = Y_test_prediction_MFTabPFN
        results_list.append(
            {'Model': 'MFTabPFN', 'Time_Train': train_time, 'Time_Pred': pred_time, 'N_low': N_low, 'N_high': N_high, 'Stop': stop0[k], 'Alpha': model_RCNN.afa.detach().cpu().numpy()})
        results_prediction.append({'Model': 'MFTabPFN', 'Y_test_prediction': Y_test_prediction_MFTabPFN,
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
        predictor.fit(train_data=TRAIN_Autogluon, time_limit=3600, presets='best_quality')
        train_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        Y_test_prediction_initial = predictor.predict(TEST_Autogluon).to_numpy() * Sigma_scaler + Mu_scaler
        pred_time = time.perf_counter() - start_time
        results_list.append({'Model': 'AutoGluon-High', 'Time_Train': train_time, 'Time_Pred': pred_time})
        Y_test_prediction_AutoGluon_high = Y_test_prediction_initial.copy()
        results_prediction.append({'Model': 'AutoGluon-High', 'Y_test_prediction': Y_test_prediction_AutoGluon_high})
        ##########################################################################AutoGluon-multi
        TRAIN_Autogluon = pd.DataFrame(xx_train, columns=[f'feature_{i}' for i in range(nx)])
        TRAIN_Autogluon['target'] = yy_train
        TEST_Autogluon = pd.DataFrame(xx_total, columns=[f'feature_{i}' for i in range(nx)])
        start_time = time.perf_counter()
        predictor = TabularPredictor(
            label="target",
            problem_type="regression",
        )
        predictor.fit(train_data=TRAIN_Autogluon, time_limit=3600, presets='best_quality')
        train_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        Y_test_prediction_initial = predictor.predict(TEST_Autogluon).to_numpy() * Sigma_scaler + Mu_scaler
        pred_time = time.perf_counter() - start_time
        results_list.append({'Model': 'AutoGluon-Multi', 'Time_Train': train_time, 'Time_Pred': pred_time})
        Y_test_prediction_AutoGluon_multi = Y_test_prediction_initial.copy()
        results_prediction.append({'Model': 'AutoGluon-Multi', 'Y_test_prediction': Y_test_prediction_AutoGluon_multi})

        Results_ONERAM6_list_ML[j][k] = pd.DataFrame(results_list)
        Results_ONERAM6_prediction_ML[j][k] = pd.DataFrame(results_prediction)
        with open('Results_ONERAM6_list_ML.pkl', 'wb') as f:
            pickle.dump(Results_ONERAM6_list_ML, f)
        with open('Results_ONERAM6_prediction_ML.pkl', 'wb') as f:
            pickle.dump(Results_ONERAM6_prediction_ML, f)

