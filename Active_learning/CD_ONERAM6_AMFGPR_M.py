import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
from pathlib import Path
import pickle
import time
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import GPy
import emukit.multi_fidelity
import emukit.test_functions
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel

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

n_simulations = 3
Results_ONERAM6_list_AMFGPR = [None for _ in range(n_simulations)]
Results_ONERAM6_prediction_AMFGPR = [None for _ in range(n_simulations)]
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

    ##########################################################################
    U = 300
    Cost_low = 1
    Cost_high = 1
    Stop = np.zeros((U, 1), dtype=float)
    Stop0 = 0.025
    m_low = 0
    m_high = 0

    Index_fidelity = Results_ONERAM6_list[j].loc[:, "Index_fidelity"].values

    Index_low_list = np.zeros(N, dtype=int)
    Index_low_list[:N_low] = indices_low
    Index_high_list = np.zeros(N, dtype=int)
    Index_high_list[:N_high] = indices_high

    results_list = []
    results_prediction = []
    for i in range(U):
        print(i)

        ##########################################################################Linear MFGP
        X_train, Y_train = convert_xy_lists_to_arrays([xx_train[:N_low, :], xx_train[N_low:, :]], [yy_train[:N_low], yy_train[N_low:]])
        kernels = [GPy.kern.RBF(nx), GPy.kern.RBF(nx)]
        lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
        gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2)
        lin_mf_model = model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=5)
        start_time = time.perf_counter()
        lin_mf_model.optimize()
        train_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        X_total = np.hstack((xx_total, np.ones((xx_total.shape[0], 1))))
        hf_mean_lin_mf_model, hf_var_lin_mf_model = lin_mf_model.predict(X_total)
        pred_time = time.perf_counter() - start_time
        Y_test_prediction = hf_mean_lin_mf_model.ravel() * Sigma_scaler + Mu_scaler
        Prediction_sigma = np.sqrt(hf_var_lin_mf_model.ravel() * Sigma_scaler ** 2)
        X_total = np.hstack((xx_total, np.zeros((xx_total.shape[0], 1))))
        lf_mean_lin_mf_model, lf_var_lin_mf_model = lin_mf_model.predict(X_total)
        Y_test_prediction_low = lf_mean_lin_mf_model.ravel() * Sigma_scaler + Mu_scaler
        Prediction_sigma_initial = np.sqrt(lf_var_lin_mf_model.ravel() * Sigma_scaler ** 2)
        Y_test_prediction_LinearMFGP = Y_test_prediction.copy()
        Y_test_prediction_LinearMFGP_low = Y_test_prediction_low.copy()

        Prediction_sigma_initial[Index_low_list[:N_low]] = 0
        Prediction_sigma[Index_high_list[:N_high]] = 0

        index_fidelity = Index_fidelity[i]
        if index_fidelity == 0:
            index_max = np.argmax(Prediction_sigma_initial)
        elif index_fidelity == 1:
            index_max = np.argmax(Prediction_sigma)

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
            {'Model': 'LinearMFGP', 'Time_Train': train_time, 'Time_Pred': pred_time, 'Index_fidelity': index_fidelity,
             'Index_max': index_max, 'N_low': N_low, 'N_high': N_high})
        results_prediction.append({'Model': 'LinearMFGP', 'Performance': Y_test_prediction_LinearMFGP, 'Performance_low': Y_test_prediction_LinearMFGP_low,
                                   'Prediction_sigma_initial': Prediction_sigma_initial, 'Prediction_sigma': Prediction_sigma})

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

    Results_ONERAM6_list_AMFGPR[j] = pd.DataFrame(results_list)
    Results_ONERAM6_prediction_AMFGPR[j] = pd.DataFrame(results_prediction)
    with open('Results_ONERAM6_list_AMFGPR.pkl', 'wb') as f:
        pickle.dump(Results_ONERAM6_list_AMFGPR, f)
    with open('Results_ONERAM6_prediction_AMFGPR.pkl', 'wb') as f:
        pickle.dump(Results_ONERAM6_prediction_AMFGPR, f)

